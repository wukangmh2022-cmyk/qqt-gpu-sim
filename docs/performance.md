# 性能优化详解

两层的优化：**引擎层面**（怎么把成千上万局塞进 GPU 喂给 PPO——宏观架构决策）和
**算子层面**（单个算子怎么写才不拖后腿——微观实现技巧）。实测结果：
**DCU（国产海光，DTK 26.04）上 36~41k env-steps/s**（5632 env × 128 rollout，MLP），
12 小时 ≈ 1.6B 步 ≈ 180 万局。老 CUDA kernel 的更多细节见 `sim/cuda/bomber_kernels.cu`、
`bench/roofline.py`（训练主线已迁 torch 后端，CUDA 是未来提速选项）。

---

## 一、引擎层面（宏观）：高并发仿真的搭建

### 1. 并行度模型：N 局并行，不是 N 角色并行

一局里只有 2~4 个角色，把角色并行化最多拿到 4 路——毫无意义。
真正的并行度来自**同时跑几千个独立关卡**（`--num-envs 5632`）：

```
一个逻辑单元 = 一个完整关卡（含它内部的 2~4 个角色，串行处理）
GPU 一次推进 = 几千个关卡
```

所以 `BatchedSim` 的状态全是 `(N, ...)` 张量（位置 `(N,P,2)`、泡泡 `(N,H,W)`、掩码
`(N,P,5)`），**任何算子都是整批的张量运算，没有逐 env 的 Python 循环**——这才是
GPU 能吃满的关键（5632 个 env 同时计算，隐藏延迟、占满访存）。

### 2. 共享观测：每个 env 一份，不按角色复制

经典写法给每个角色各写一份视角（`(N, P, C, H, W)`）——P 份里 3 个通道字节完全相同、
位置通道也只是顺序不同，纯粹重复写。这里**每个 env 只存一份 `(N, C, H, W)`**
（默认 fp16），写入量减少 **84.7%**（13x13、4 人：24,336 B → 3,718 B）。
观测写入本来就是模拟器第一访存瓶颈，这一刀砍在最疼的地方。

### 3. 视角置换吸收进第一层权重：数据一个字节不搬

"我是谁"不再靠拷贝数据表达，而是把视角的**通道置换等价地施加在权重上**：
`sum_j view[j]·w[:,j] = sum_k shared[k]·w[:,inv[k]]`。MLP 是 `shared[0].weight` 的列块
重排（`inv_cols`），CNN 是 `conv0.weight` 的输入通道索引（`inv_perm`）——**观测数据
零拷贝**，代价只是每 pid 一个 14 元素的小索引表。`test_weight_perm_equals_data_gather`
钉住了与显式 gather 的逐位等价性。

### 4. fp16 观测：写入带宽 + 显存双省

观测通道值全在 [0,1]，fp16 的 10 位尾数远够用。省的是写入带宽（observe 是第一瓶颈）
和 rollout buffer 显存（`(T,N,C,H,W)` 直接减半 = 同样显存开更大 batch）。
网络入口 cast 一次到参数 dtype（MLP 首个 Linear 是 GEMM，开销可忽略）。

### 5. 内存守卫：启动前算峰值，拒绝启动而不是 swap 到死

PPO 单个 minibatch 的激活张量是唯一大线性项：`minibatch = num_envs × rollout_steps /
minibatches`。`train.py` 启动前按公式估算峰值（`estimate_peak_bytes`），超过可用内存的
`--max-mem-frac`（默认 55%）**打印改法、拒绝启动**（曾把 8GB 机器整机跑死机）。
运行中意外 OOM 也存盘后退出，不静默丢进度。

### 6. 训练吞吐实测（DCU）

| 阶段 | 对手构成 | 平均 sps |
|---|---|---|
| warmup 段（0~150M 步） | astar + greedy 规则 bot | 36,811 |
| 150M 之后 | 固定陪练 + 模型池 | 36,682 |
| 5x3 参照（纯网络对手） | 冻结网络 | 36,866 |

三段几乎一样——**astar 每 tick 算两场 Dijkstra 也不拖慢训练**：169 格的全张量
Bellman-Ford 在 5632 env 上就是几个 `(5632,169)` 的小 kernel，相对观察写入带宽和
网络 GEMM 是零头。

---

## 二、算子层面（微观）：单个算子怎么写才快

### 1. 危险图用 gather 而不是 scatter

直觉写法：遍历每个泡泡向四个方向 scatter 写"何时会被覆盖"——多个泡泡的火焰写到
同一格，需要 `atomicMax`。反过来：**每个格子自己向四个方向看出去 blast 格**，取
遇到的泡泡里最紧急的那个。读是共享的、写是独占的，**零 atomic、零 scratch buffer**。
代价是重复读 4×blast 格，但那些格子刚被邻居线程读过，全在 L1/L2 里。
（`sim/blast.py::danger_map`，这也是寻路 AI 和网络危险通道的同一份输出。）

### 2. 连锁爆炸：固定轮数同步迭代，无 early-exit

CPU 写连锁会自然写成 BFS 队列（动态内存、线程间不等长、需要 atomic）。
这里改成**定轮同步传播**：每一轮读上一轮的 `triggered` 写本轮的 `covered`，最多迭代
`max_chain=8` 轮。连锁结束后多算的空轮只产生空覆盖，结果与早退版**逐位一致**，
但**无 Python 分支 / 无 early-exit**（CUDA graph 兼容）——固定最坏执行时间，不会因为
某一局连锁特别长而全体等待。（`sim/blast.py::resolve_explosions`。）

### 3. SoA 布局，env 放最内层（CUDA）

```cpp
idx(cell, env) = cell * num_envs + env
```

同一个 warp 的 32 个线程同时访问"同一个格子、相邻 env"，地址连续 → 一次访存事务。
AoS（`env * n_cells + cell`）会让相邻线程地址差 169 个元素，带宽利用率掉到 1/32。
（当前训练用 torch 后端，此优化在 CUDA kernel 里，`sim/cuda/bomber_kernels.cu`。）

### 4. 合法动作掩码作用在 logits 上

掩码加 `-inf` 到 logits 而不是在概率上乘 0：后者在全掩码时得到全零分布，softmax 出
NaN。配合**因子化双头动作**（move × bomb 独立），`MOVE_IDLE` 和 `bomb=0` 恒合法 →
掩码整行不可能全 inf → 连 NaN 兜底分支都不需要。

### 5. 合法动作采样用 cumsum 技巧，无 CPU 同步

`sim/bots.py::_sample_legal`：`mask.float().cumsum(-1)` 把合法项映射成 [0, 合法数) 的
区间，均匀样本 `(u > cs).sum()` 即落在哪个动作——**全张量、零 CPU 同步**。
5632 env 的每 tick 采样不需要 .cpu() 往返（那种 `.item()` 每 tick 几十次会直接卡死训练，
见 `sim/blast.py` 头部注释）。

### 6. 渲染层：静态层整图缓存（试玩侧）

墙/砖只在爆炸后变化 → 预渲染成整图，10Hz 重渲一次，60fps 渲染帧直接 blit
（`play/duel.py::build_static`）；危险区/爆炸/泡泡/角色仍逐帧画。CJK 文本也按内容
缓存 surface，不再每帧 font.render（之前是掉帧源）。

---

## 三、老基准（bench/，训练主线已迁 torch 后端）

- `bench/throughput.py`：env-steps/s vs 并行关卡数
- `bench/roofline.py`：流水线分阶段计时 + 显存探测 + 瓶颈归因 + 训练预算推算
- `bench/theoretical.py`：按显卡 spec 反推理论上限（纸面参照系，非承诺）

结论（老实测）：**模拟器不再是瓶颈**（占训练总时间 <1%），RL 成本回到网络前向和
PPO 更新——所以训练主线用 torch 后端足够，CUDA kernel 是"要把模拟占比再压到 0"时的
未来选项。
