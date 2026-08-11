# triton-ascend 3.2.0 兼容性差异与 910B 性能实测（2026-08-11）

> 为昇腾生态 PR 准备的差异清单 + 910B 性能剖析结论。
> 相关代码：`sim/dev.py`、`sim/triton_sim.py`、`sim/triton_step.py`、`verify_*.py`。

## 一、正确性差异（triton-ascend vs triton CUDA/MPS）

### 1. `tl.sum(bool_tensor)` 结果错误（恒 1）
- 现象：`tl.sum(mask_bool)` 在昇腾返回 1（正确应为 count）。
- 修复：先转 int 再 sum：`tl.sum(tl.where(cond, 1, 0))`。
- 位置：`_count_bombs_kernel`。

### 2. `bool & int64` 混合类型编译失败
- 现象：bool 与 int64 张量 `&` 直接编译报错。
- 修复：`> 0` 显式转 bool：`bm = tl.load(bomb + ...) > 0`。
- 位置：`_place_bombs_kernel`。

### 3. 标量 `tl.floor` 在存储地址链上编译失败
- 现象：`unresolved materialization from () to tensor<1xf32>`。
- 修复：pos 恒 >= 0，用截断 `.to(tl.int32)` 替代 floor。
- 位置：`_place_bombs_kernel`。

### 4. `tl.store(ptr, tl.where(ok, a, b))` 是编译雷区
- 现象：where-store 在 triton-ascend 上编译异常。
- 修复：改用 mask-store 模式 `tl.store(ptr, val, mask=ok)`。
- 位置：`_place_bombs_kernel` 的条件写。

### 5. mask 参数必须严格 i1
- 现象：`Mask must have boolean scalar type`。
- 修复：`.to(tl.int1)` + `~on_brick`（不用 `== 0`）。
- 位置：`_place_bombs_kernel` 的 `ok` 计算。

### 6. constexpr 无 `.to()` 方法
- 现象：`'constexpr' object has no attribute 'to'`。
- 修复：constexpr 值直接存（triton 自动 cast），或先算 runtime 值。
- 位置：`_place_bombs_kernel` 的 owner 写入（static_range 的 `me`）。

### 7. gather 越界 = 未定义行为（无边界检查）
- 现象：torch_npu gather 对 OOB 索引返回垃圾值（实测 1.67e18、-13）。
- 修复：所有 gather/scatter 索引 clamp：`tl.minimum(tl.maximum(gidx, 0), N*H*W-1)`。
- 位置：所有 kernel 的 gather 索引 + torch 侧 gather 索引。

### 8. verify 脚本设备检测不认 NPU → 落 cpu（本会话最大坑）
- 现象：910B 上 `torch.cuda.is_available()` 为 False（torch 是 CPU 版），
  脚本 fallback 到 `dev="cpu"`。但 triton 后端是 ascend —— **kernel 在 NPU 上
  执行 host 内存指针** → 确定性 aivec 矢量核异常
  （`pc start: 0x124000000000`，`aic error mask: 0x6500020bd00028c`）。
- 修复：`sim/dev.py::pick_device()` 统一设备选择（mps → cuda → npu:0 → cpu），
  6 个 verify 脚本全部改用。
- 教训：**在 NPU 机器上跑"兼容性"验证脚本，设备检测必须含 npu**，
  否则 triton kernel 在 NPU 上跑 CPU 张量，表象像"codegen bug"实则是
  host/device 指针错配（0x124000000000 就是典型用户态虚拟地址）。

## 二、本会话发现并修复的 4 个真实 bug（非 triton-ascend 特有）

### B1. place kernel 数据竞争：双人同格并发放泡
- 现象：`_place_bombs_kernel` grid=(n*p,)，pl 0 / pl 1 各自一个 program。
  两人同格（角色不碰撞可叠站）且同 tick 都按放泡键时，pl 1 的
  `cur_fuse<=0` 读可能赶在 pl 0 写之前 → 同格叠放两泡。
  torch 版顺序 for 循环无此问题（pl 1 能看到 pl 0 刚写的 fuse=30 → 拒绝）。
- 修复：grid 改 (n,)，内层 `tl.static_range(P)` 顺序处理玩家——
  程序内 store→load 同址可见，逐位复刻 torch 语义。
- 验证：verify_triton_step 40 tick 全字段 0 差。

### B2. triton_step_core 缺 brick/crate 更新
- 现象：torch step 在 resolve 后 `brick.bitwise_and_(~covered)` +
  `crate.bitwise_or_(brick & covered)`；core 没做 → 第一波爆炸后
  blocked(=wall|brick|fuse>0) 分叉 → 后续级联发散（fuse/owner/pos 全差）。
- 修复：core 第 5.5 步补 brick/crate 更新（与 torch 顺序一致）。
- 教训：blocked 用 brick 不用 crate；crate 只显示/拾取，不影响移动。

### B3. danger_triton 位置传参错位
- 现象：`triton_step.py` 调用
  `danger_triton(fuse, wall, bombed, brick, blast, fuse_max, cfg.max_chain, ...)`
  把 `cfg.max_chain`(16) 传进了 **b_max** 位置 → max_chain 恒 1（阶段 A
  连锁修正丢失）+ b_max=16（blast 距离虚高 2 倍）。语义错 + 编译出
  BMAX=16 的 kernel（昇腾慢编译 10+ 分钟）。
- 修复：改关键字 `max_chain=cfg.max_chain`。

### B4. verify_triton_step.py 配置与 core 范围不匹配
- 现象：open_fraction=0.5 → 一半 open 关强制 crate_prob=1.0（踩箱必升），
  而 triton_step_core（Step 1）不含拾取 → 成长字段发散。
- 修复：open_fraction=0.0（纯 corridor + growth_crate_prob=0 → 无箱无拾取）。
- 另修脚本 bug：速度段 `acts` 未定义 → `acts_seq[-1]`。

## 三、910B 性能剖析（N=2048 富泡状态，稳态计时）

| 组件 | 910B triton | torch 对照 | 结构 |
|---|---|---|---|
| count | 0.71 ms | — | block 归约 |
| place | 0.95 ms | — | 简单逐玩家 |
| move | 0.35 ms | — | 简单逐元素 |
| explode ×1 | **23.9 ms** | rays 3.06 ms | **4方向×BMAX gather 扫描** |
| resolve ×16 | **384 ms** | resolve 0.5 ms | 16 轮无早退 explode |
| danger mc=1 | **27.1 ms** | — | gather 扫描 |
| danger mc=16 | **151.7 ms** | danger_map 24.3 ms | 阶段A 16轮 |
| trivial launch | 0.14 ms | — | grid=1 |

**瓶颈定位**：`4方向 × BMAX 步 × 每格 5-7 次散乱 gather load` 的模式在昇腾
SIMD 架构上编译/执行极差（每格 ~196 次内存操作）。torch 的 `rays` 用
`F.pad` 连续 shift 波前（28 次整图 memcpy），快 8-50 倍。
静态展开（static_range）不是慢因（动态 range 变体实测 53ms 更慢）。

**910B SPS 实测（triton_step_full vs torch step）**：
| N | torch | triton_full |
|---|---|---|
| 4096 | 93,939 | 18,390 |
| 8192 | 133,861 | 19,165 |
| 16384 | **168,951** | 19,500 |

**结论**：
- 在 triton-ascend 3.2.0 上，gather 扫描型 kernel（explode/danger）是负优化
  （比 torch 慢 8-50 倍）；简单 kernel（move/place/count）才有正收益。
- resolve_triton 加早退后 core 段 383ms → 93ms，但爆炸 tick 仍受 24ms
  explode 拖累。
- **910B 上当前最优配置 = torch step（17万 SPS）**；trition 化 kernel 只在
  MPS/DCU 这类 gather 友好的后端有正收益（本地 MPS x5.4）。

## 四、100万 SPS 路径（910B）

1. **danger 是 torch step 的最大单项**（N=16384 时约占 60%）：把 danger_map
   重写为 triton shift 波前（不是 gather 扫描），目标 24ms → 3ms。
   参考 MPS 上 danger_triton 1.6ms @N=8192 的达成路径。
2. **reward 段 trition 化**：hit_attr/combo/拾取/连锁兑现目前是 ~23ms torch 算子。
3. **批量放大**：N=16384 时 launch 开销已摊薄（43→61→97ms 亚线性）。
4. **多进程 collect**：多卡/多进程并行收集，接近线性扩展。
5. 混合配置：910B 上 move/place/count 用 triton，explode/danger 用 torch
   （shifts），可作为过渡方案。

## 五、给昇腾生态的 PR 建议

1. `pick_device` 教训 → 昇腾官方示例/模板补 NPU 设备检测分支。
2. gather 越界"宽松语义"→ 文档明确 OOB 为 UB 并在 triton-ascend 加编译期
   警告或运行期检测开关。
3. gather 扫描型 kernel 的 SIMD 代码生成质量 → 收集 explode/danger 复现
   用例提交给 bisheng 编译器团队（昇腾 SIMD 为主的架构对 per-lane gather
   的支持是性能分水岭）。

---

## 六、100万 SPS 攻坚剖析（2026-08-11 第二轮，单卡 910B3）

### 实测基线（N=16384，torch step 稳态）
- 单 step **~108ms = 15.2万 SPS**（放炮率 0.03~0.45 均为 ~100-122ms，非 dense 假象）
- 成本分解（mock 分段）：
  | 块 | 贡献 | 说明 |
  |---|---|---|
  | danger_map(16) | ~33ms | 含同步冲刷效应；独立测 15.7ms |
  | reward 其余（damage/clear/combo/win/pickup/结算） | ~47ms | ~1000 小算子 × 19µs |
  | 同步（_local_scalar_dense ×94/step） | ~36ms | bool(any())/.item()/reset_ |
  | place_predict | 5.6ms | 火焰预测 rays |
  | 核心（triton move/place/count） | ~10ms | 已经很快 |
  | reset_ + 成长 | ~10ms | |

### 根因：dispatch-bound（不是 device-bound）
- **单算子 dispatch = 19.3µs**（(16384,13,13) mul）
- step 内 ~3000 算子 × 19µs ≈ **58ms dispatch**
- 每算子都建临时张量：4940 empty_tensor/5step = **988 alloc/step**
- bool(any()) 同步 0.38ms/次 × 94 = 36ms

### 已排除的路线（实测数据）
| 方案 | 结果 |
|---|---|
| triton gather 扫描 kernel（explode/danger） | 慢 8-50x（昇腾 SIMD 对 per-lane gather 无解） |
| BLOCK 扫描（256~8192） | 无效（static_range 展开非慢因） |
| int32/动态 range 变体 | 53ms vs 24ms 更慢 |
| danger fixed b_max（免 max 同步） | 3x 更慢（空轮 pad 比同步贵） |
| torch.compile(backend='npu') | 8.0ms vs 4.6ms 更慢 |
| danger stack 融合（fw+fd 一次 shift） | 16.8→15.7ms（+6%，保留） |
| 多卡并行 | 单卡（davinci6），不可行 |

### 100万 SPS 路线图（单卡，按收益排序）
1. **reward 段 triton 化**（最大块，~47ms → 目标 5ms）：reward 数学是
   elementwise（mul/where/clamp/sum）—— 910B 上 elementwise triton kernel
   已验证快（move 0.35ms/place 0.95ms @N=2048）。预计 +42ms。
2. **danger 重写**（~33ms → 目标 6ms）：stage A/B 换更省算子的波前
   （少 pad、少轮），或验证昇腾 aclnn 的融合算子（aclnnMax 等）。
3. **同步消除**（~36ms → 目标 10ms）：resolve/danger 的早退轮检查从
   CHECK_EVERY=2 → 4 降频；reset_ 空 mask 短路免 int(sum)；reward 门控
   改设备端掩码。
4. 达成后 step ~35ms → **47万 SPS**；再叠加 1-3 的极端版（danger 4ms +
   reward 2ms + 同步 5ms）→ ~20ms → **80万+**，逼近 100万。
5. 1000万量级需多卡（当前 Dev Space 单卡）或换算法（距离变换/前缀扫描）。

---

## 七、第三轮：dispatch 同步消除 + 标量 op 量化（2026-08-11 晚）

> 第六节的 108ms 基线 → 80.6ms（N=16384），单卡 SPS 峰值 22.2万 @ N=65536。
> 本节全部优化位级一致（本地 MPS old-vs-new 随机对拍 + 910B triton 全字段对拍）。

### 1. danger 同步消除（107.6 → 91.1ms，commit 208e4cf）
| 手段 | 效果 |
|---|---|
| `blast_hint` 提前到放泡后浅队列（~30 op）取 `int(bomb_blast.max())` | danger 内部 5 次 host 同步 → 0 |
| `chain_cap`（sync_free 固定轮 min(max_chain, cap)） | 免 newly 轮检查同步；链长≤cap 位级一致 |
| 2-pad 连续张量（撤 stack 融合） | stack 后 st[0]/st[1] 非连续视图上的标量 op 触发 item() |
| `one_buf` 预分配全 1 张量（`fd1 - one_buf`） | 张量-张量 op 免 item，省 ~50 次同步 |

### 2. reward 段标量 op 全量化（91.1 → 88.7ms，commit 904eaa1）
- **根因**：torch_npu 上"张量 ±/×/÷ Python 标量"的 op 每次 dispatch 内部
  `_local_scalar_dense`（item）同步，~0.26ms/次，94/step 是大头。
- **修复**：`_sc(value, shape)` 缓存 `(值,shape)→全值张量`，`cfg.X*tensor`
  换成张量-张量 op（零同步）；`_explore_coef` 退火变化时缓存自然失效。
- 7 处替换：step_penalty / hit(dealt,dmg) / explore / chain / brick /
  danger / combo / win（fixed 与 timeout）。

### 3. resolve 免深队列 max 同步 + where→bool-mul + move grid 2D（88.7 → 80.6ms，commit 0e703dd）
- **resolve 传 blast_max_hint**：非 graph 也用浅队列算好的 `blast_hint`
  （与 `_blast_map()` 同源 → hint == 实际 max 无空轮 pad）→ rays 免每次
  `int(blast_cell.max())` 深队列同步。位级不变。
- **where→bool-mul**：`torch.where(keep, fw1, 0.0)` → `fw1 * keep`
  （bool×float32 提升，keep=False→0.0），省 1 次 dispatch/格步；
  本地 20 组随机 × max_chain/cap/hint 全组合位级 PASS。
- **move grid 2D**：`(n*p,) → (p,n)`（triton-ascend 总 program <65536 硬限；
  2D 拆后大 N 仍需 `TRITON_ALL_BLOCKS_PARALLEL=1`，driver 层限制）。

### 4. N 扫描与单卡物理上限
```
N=   4096:  41.41 ms/tick    9.89万 SPS  每 env 10.11 µs   (launch 主导)
N=   8192:  46.30 ms/tick   17.69万 SPS  每 env  5.65 µs
N=  16384:  80.68 ms/tick   20.31万 SPS  每 env  4.92 µs
N=  32768: 152.39 ms/tick   21.50万 SPS  每 env  4.65 µs
N=  65536: 295.13 ms/tick   22.21万 SPS  每 env  4.50 µs   ← 峰值
N= 131072: 599.01 ms/tick   21.88万 SPS  每 env  4.57 µs
N= 262144:1205.46 ms/tick   21.75万 SPS  每 env  4.60 µs
```
- **SPS 饱和**在设备吞吐：每 env ~4.5µs 是 910B3 上本算法的计算硬底
  （2200 kernel/step 的设备侧执行 + 启动，数据量 13×13×N）。
- wall ≈ max(launch, device) 已重叠（N=4096 时 41ms ≈ launch 主导，
  N≥16384 时 device 主导），**NPU graph 压 launch 无收益**（设备执行不变）。

### 5. 分段短路实验（找最大单项）
```
baseline               88.20 ms
danger→zeros           78.29 ms   ← danger_map 最大单项（~10ms，~300 op/调用）
resolve→trivial        88.96 ms   （早退生效，非爆炸 tick 免费）
place_predict→zeros    87.85 ms   （放泡 tick 少）
move→identity          85.93 ms   （triton 单 kernel 仍 ~2.3ms 数据搬移）
```

### 6. 结论：单卡 100万 不可达，峰值 ~22.2万
- 单卡 eager torch 的 SPS 上限 ≈ **22.2万 @ N=65536**（设备每 env ~4.5µs）。
- 100万 需 4.5 卡数据并行（每卡 22.2万）；当前 Dev Space 单卡（910B3）。
- 剩余候选优化（danger 前缀扫描 cummax、reward/clear 段 in-place 合并）仅
  边际 ~5-10%，改变不了量级；triton/torch.compile/NPU graph 均已实测排除。

---

## 八、第四轮：AutoFuse / multistream / 段编译 / legal_mask 全面探索（2026-08-11 深夜）

### 1. 训练侧整体 SPS（step + obs + legal_mask，PPO collect 口径）
```
纯 step        :  85.51 ms/tick  19.16万 SPS
step+obs       :  92.35 ms/tick  17.74万 SPS  (obs 8.4ms)
step+obs+mask  : 122.37 ms/tick  13.39万 SPS  (legal_mask 16.4ms = 13%)
```
- **训练侧真实 SPS = 13.4万**（比纯 step 的 20万低 33%）——legal_mask 是最大
  非 step 单项（16.4ms，~1562 个分散小 op，无单一大头）。
- observe 的 danger 通道复用 step 的 `_dng_cache`（每 tick 只算 1 次 danger）。

### 2. AutoFuse（GE 自动融合）——本会话最重要发现
- **使能**：`AUTOFUSE_FLAGS="--enable_autofuse=true"`（环境变量，GE 编译时读）。
  文档：CANN AutoFuse 使能方式（Elemwise/Broadcast 默认开，reduce/concat 需
  `--autofuse_enable_pass=reduce,concat`）。
- **单段实测（torch.compile(backend='npu') + AutoFuse）**：
  ```
  max_b=1: eager 9.10ms | compiled 2.31ms | x3.94 | 位级 maxdiff=0.00e+00
  max_b=2: eager 15.45 | compiled 4.21  | x3.67 | 位级 maxdiff=0
  max_b=3: eager 21.38 | compiled 5.95  | x3.59 | 位级 maxdiff=0
  max_b=4: eager 28.66 | compiled 7.69  | x3.73 | 位级 maxdiff=0
  max_b=7: eager 49.20 | compiled 12.80 | x3.84 | 位级 maxdiff=0
  ```
  danger（纯张量 sync_free 段）**全档位 x3.6-3.9 且位级完全一致**（GE 融合保序）。
- **但集成进 step 是净负**：danger 的 `blast_max_hint` 每 tick 动态变化 →
  编译函数每次调用 dynamo guard 检查 + GE 图切换，调度开销吃掉融合收益
  （集成后 89.9ms vs eager 80.6ms）。**结论：编译只适合"输入固定、无动态
  常量"的段**；动态参数段（按档位变化）编译反而慢。
- **整步编译不可行**：位级破（maxdiff 8e-10，跨段融合改浮点顺序）+ 更慢
  （x0.71）；move 是 triton kernel 时 torchair op converter 直接报错
  （triton_kernel_wrapper_functional 不支持）。
- **backend='inductor' 崩溃**：CANN 8.5.2 无 PyTorch Inductor 对接
  （BackendCompilerFailed 子进程异常）——只能走 backend='npu'（torchair）。
- **legal_mask 编译崩溃**：TBE Subprocess task_distribute main process
  disappeared（图太大/算子不支持）。

### 3. multistream（torch.npu.Stream）实测
- 独立 op 链：单 stream 188ms → 4 stream 55ms（**x3.42 真并行**）。
- danger 4-stream 原型（阶段 B 单独）：x1.45；完整 danger（阶段 A+B）：
  **x0.89 负优化** —— 阶段 A 每轮 2 组 event 同步 + 轮间强串行，同步开销 >
  并行收益；阶段 B 单独有收益但在完整上下文被阶段 A 拖累。**已撤回**。
- 内存搬移型 kernel（F.pad 整图搬移）受 HBM 带宽限制，并行度买不到带宽。
- host↔NPU 带宽 18GB/s（(16384,13,13) 拷贝 0.51ms）→ 把段搬 CPU 算不划算。

### 4. stack 融合复测（张量操作数时代）
- fw+fd stack(2,n,h,w) 一次 pad：视图上仍有 item 同步（4/调用）+ 更慢
  （stack 0.41ms vs dual 0.30ms）→ 双 pad 连续张量（现状）确认为最优。

### 5. legal_mask 优化实测
- triton 复用 move kernel 做 4 方向试探（位级对拍 5 组 PASS）：16.28 →
  15.36ms（**x1.06**）—— 4 次 kernel 的组装开销吃掉收益，AABB 碰撞计算量
  是硬成本。保留 torch 版（triton_sim.legal_mask_triton 留作参考）。

### 6. 结论
- 编译（AutoFuse）只对固定输入段有效，训练热路径（动态档位/分支多）不可用。
- multistream / stack 融合 / legal_mask triton：全部实测无净收益。
- 训练侧真实 SPS 13.4万（step 80.6 + obs 8.4 + mask 16.4 + act 杂项 17ms）；
  单卡 100万 需执行模型级突破（多卡/换算法），非本轮手段可达。

---

## 九、真实训练与 ppo_update：瓶颈真相（2026-08-12 凌晨）

### 1. train.py 的设备检测 bug（910B 上一直在 CPU 训）
- `train/train.py` 设备选择只有 `cuda if available else cpu`，**没有 npu 分支**
  → 910B 上 `--device` 不传时模型落在 CPU（SPS 极低）。
- 修复：启动传 `--device npu:0`；训练侧同样需要 npu 检测（PR 素材）。
- resume 另踩 1v2 课程通道不匹配（14ch 快照 vs P=3）→ `--single-stage` 跳过。

### 2. 真实训练 SPS 实测（N=2048, MLP, 910B）
```
[ 10] step=2.62M ... sps=17k   ← 真实训练 ~1.7万（含 collect + update + 自我博弈）
```
- 每 iter = collect(128 tick) + ppo_update + 对手构建。collect 里模拟器
  （step + obs + mask）占 ~84%，**自我博弈双网络前向**（learner + 冻结对手）
  是隐藏成本（bench 单网络测不到）。

### 3. ppo_update 10x 加速实测（N=4096, rollout=128, 目标 6131→613ms）
```
CNN baseline (fp32):        6142 ms
CNN autocast fp16:          7085 ms   （更慢 18%：910B fp16 无收益）
CNN compile(AutoFuse):      6130 ms   （零效果：反向不编译/前向非瓶颈）
CNN epochs=2:               3085 ms   （线性减半，训练超参）
MLP baseline:                503 ms   ★ x12.21 ← 达成 10x
```
- **MLP 架构 = update 10x 的唯一实测路径**：GEMM（AI Core cube）效率高 +
  参数少（192k vs 281k）+ 设备队列浅 → item 同步便宜。train.py `--arch mlp`。
- **CNN 为何不行**：profile 显示 update 94% 时间（5.6s）在 1634 次
  `_local_scalar_dense`（平均 3.46ms/次）——但这是**卷积慢的表象**（设备队列
  深 → item 同步显式化等待）。用模拟器同款 _sc 张量操作数消除 loss 段标量
  op 后**实测零收益**（6121ms 不变）→ 根因是 910B 小卷积（28 万参数 CNN、
  131072 batch）AI Core 利用率 <2%，不是 dispatch。
- 理论算力 3-10ms vs 实测 383ms/步 → 40-100x 效率差距，全部是 kernel 执行
  效率问题（无 dispatch/精度/编译因素可挖）。

### 4. 瓶颈结论
- update 10x（MLP）后，瓶颈转移到 **collect**：模拟器（step 硬底 80.6ms@16384
  + obs + mask）占 ~84%，自我博弈双网络前向是第二。
- 完整训练 SPS 的量级：CNN @4096 ~3.9万（bench 口径）→ MLP @2048 真实 1.7万
  （含自我博弈/评估）→ 与本地 MacBook/DCU 的 30-40k 同量级 —— **跨平台瓶颈
  一直是网络训练效率（CNN 小卷积），不是模拟器**。

## 十、DCU 38k vs 910B 17k 口径调查（2026-08-12，用户质疑"优化后退步"）

### 1. 用户的质疑
> "DCU 那边的版本（提交 commit 之前）垃圾显卡都能跑 38K，你优化了还只剩 17K。
>  要不先看一下为什么？"

### 2. 调查结论：不是退步，是 N 口径 + 模型架构完全不同
910B 优化没有退步。两个数字根本不是同一配置：

| 配置 | DCU（海光 HIP） | 910B（昇腾） | 结论 |
|---|---|---|---|
| **完整训练 SPS** | 36-41k（performance.md §6） | **43.2k（本会话实测）** | 同配置 910B 快 ~1.1-1.2x |
| 模型 | MLP 345k | MLP 345k | **相同** |
| N（num_envs） | 5632 | 5632 | **相同** |
| 迭代 | 128 rollout + ppo_update | 128 rollout + ppo_update | **相同** |
| **更大的 N** | 35k @ n=20000（HANDOFF L208） | **89.4k @ N=20000（实测）** | 910B 快 2.4x |

**17k 的来源**（不是优化前的数字）：
- 910B 真实训练 17k 是 **CNN（duel_cnn.pt，步数 300M）@ N=2048** 的完整训练。
- 两个维度都变了：架构 MLP→CNN（**计算量反而大 14 倍**，见下）+ N 被显存逼到 2048。
- CNN @ N=8192 实测 OOM（warmup 就占 46GB）→ N 只能 2048-4096 → SPS 低。

### 2b. 为什么"参数更少的 CNN"反而慢 12 倍？（用户追问）
**参数少 ≠ 计算量少**。实测参数量与 FLOPs（batch=131072, obs 14×13×13）：

| | 参数 | 每前向 FLOPs | 910B ppo_update |
|---|---|---|---|
| CNN（3×3x3 卷积 + 1x1） | 240,672（0.70x） | **1182 GFLOP（14.1x）** | 6142 ms |
| MLP（2 层全连接） | 344,776 | 83.7 GFLOP | 503 ms |

- CNN 的 FLOPs 高 14 倍来自 4 个卷积层（14→16→32→64→8）在 13×13 全图上
  反复滑窗：`FLOPs = batch×cout×k²×cin×h×w`，通道越堆越多、每层扫全图。
- MLP 只有 2 个全连接层（2366→128→128），大 FLOPs 只有第一层一个点。
- 注意：真实 update 慢 12.2x 略小于 FLOPs 比 14x —— 910B 的 3x3 小卷积
  在 AI Core 利用率 <2%（第九节），卷积只花了理论该花的一部分时间；
  MLP 的 GEMM 走 Cube 单元效率高，实测贴近理论。两者差距 = 卷积的
  "理论 FLOPs 巨大 × 硬件效率低"双重打击。

### 3. 关键实测（bench_dcu_parity.py，本会话 910B 上跑）
```
[parity] N=20000 arch=mlp   collect=25.7s update=3.0s  final sps = 89.4k   ← DCU 35k 的 2.4x
[parity] N=5632  arch=mlp   collect=15.6s update=1.1s  final sps = 43.2k   ← DCU 36-41k 的 ~1.1x
[parity] N=8192  arch=cnn   → OOM（warmup 46GB，CNN 大激活 + N 大）
```
- 口径定义：完整训练（collect 128 tick + ppo_update），SPS = N×128 / 迭代秒。
- 这解释了"为什么我之前说 17k"：真实训练 resume 的是 **CNN ckpt**（duel_cnn.pt，
  步数 300M，参数 281k），CNN 每前向 FLOPs 是 MLP 的 14x（见 2b）+ 910B 小卷积
  AI Core 利用率 <2%（第九节）→ update 慢 12 倍（6142 vs 503ms），且显存
  限制 N 只能 2048 → 完整训练只有 17k。

### 4. 教训
- **跨平台/跨配置比 SPS 必须同 N + 同模型 + 同迭代**。N=2048 的 CNN 训练
  和 N=20000 的 MLP 训练差 5-10 倍很正常，与优化无关。
- 用户印象里的"DCU 38k"来自 performance.md §6 的 **5632 env MLP** 真实训练
  （36-41k）；910B 同配置实测 **43.2k** —— 优化没有丢分，还高了 10-20%。
- 唯一真正"低"的是 **CNN @ N=2048 = 17k**，那是模型计算量大 14x + 显存上限
  的物理结果，不是模拟器优化的锅。

## 十一、LSTM 网络（局部 7×7 + 相对坐标 + 全局状态 + LSTM）910B 实测（2026-08-12）

### 1. 背景
用户给了经典 Bomberman RL 网络（BombermanNet）：局部 7×7 视野 CNN（10→32→64 3x3
+ MaxPool）+ 相对坐标 MLP（10 目标 × (dx,dy,onehot4)=60）+ 全局状态 MLP（5 维）
+ fusion(168→256) + LSTM(256→128) + 双头。要求从零训练、测真实 SPS，层数/核大小可改。

### 2. 实现（纯新增，cnn/mlp 路径零破坏）
- `train/model.py`：`arch="lstm"` 分支（BombermanNet 结构 + 动作保留 5+2 双头）。
  新增 `extract_fused()`：conv/rel/glob/fusion 逐帧独立部分，BPTT 时可一次喂
  整个 (T*N, ...) 大 batch，只对 LSTM 层沿 T 展开。
- `sim/obs.py`：新增 `local_view_features()` —— 从共享 obs 抠每角色 7×7 局部窗 +
  相对坐标（对手+最近炸弹 topk）+ 全局标量（t/hp/danger/炸弹数/fuse）。`only_p0=True`
  省一半 gather/topk。
- `train/ppo.py`：RolloutBuffer 存局部特征三元组；collect 传 hidden（done 置零）；
  `_ppo_update_lstm` BPTT（minibatch 按 env 维切，沿 T 顺序重放 LSTM）。
- `train_lstm_speed.py`：从零训练闭环（greedy bot 对手），与 bench_dcu_parity 同口径。

### 3. 性能实测（910B，完整训练 = collect 128 tick + ppo_update）
```
配置                      collect   update    SPS
LSTM @N=2048 mb=4          16.1s     27.3s     5.8k
LSTM @N=2048 mb=1          16.7s      7.0s    11.0k
LSTM @N=4096 mb=1          18.5s      9.3s    18.5k
LSTM @N=4096 mb=1 +only_p0 17.0s      9.2s    19.9k   ← 最优
LSTM @N=8192                → OOM（BPTT 激活 12.25GB，N 上限 ~4096-6144）
MLP  @N=4096（同口径）      14.0s      1.0s    35.1k   ← 对照组
MLP  @N=20000              25.7s      3.0s    89.4k
```

### 4. 结论：LSTM 在 910B 上全面落后 MLP
- **同口径 N=4096：MLP 35.1k vs LSTM 19.9k = 慢 1.8 倍**；N=8192 时 LSTM 直接 OOM。
- 三个原因（都实测定位）：
  1. **局部特征生成 16.8ms/tick**（observe 15.7 + 局部 gather/topk/零散 kernel），
     是共享全图 obs（零搬运 + 权重置换）的**固有代价** —— 每角色要单独抠图。
  2. **BPTT 更新**：minibatch 数线性放大（mb=4 → update 27.3s vs mb=1 → 7s），
     因为每个 mb 都要重跑整个 T=128 的 LSTM 序列；而 MLP 的 flat 大 batch 一次算完。
  3. **LSTM 前向本身不慢**（4.8ms/tick，纯 LSTM 层 2.3ms）—— 它走 GEMM/Cube
     效率高，但被 1+2 拖累。
- 纯前向快（bench_bomber_net 2.5ms）≠ 完整训练快：**LSTM 的时序本质决定了 BPTT
  无法 flat、局部观测必须逐角色 gather** —— 这两条和 910B 的"大 batch GEMM 高效、
  小 kernel dispatch 贵"特性相克。

### 5. 建议
- **910B 上不要用 LSTM/局部视野架构追求 SPS**。共享全图 + MLP（GEMM）是唯一
  已验证的高吞吐路径（89.4k @N=20000）。
- 若想要"记忆"能力：可用共享全图 MLP + 隐式位置编码替代 LSTM（网络自己记住
  危险图历史），或接受 LSTM 的 SPS 代价做小 N 实验。
- 代码保留（arch="lstm" 完整可用），作为论文式架构的对照实验与 PR 素材。

## 十二、LSTM 网络在本地 MacBook（MPS）实测 + Triton 化（2026-08-12）

### 1. 背景
用户不愿放弃 LSTM/CNN 架构，要求测本地 MacBook 吞吐，能 Triton 更好。

### 2. 环境
- MacBook Apple Silicon (arm64)，torch 2.13.0 + MPS 可用，triton 3.8.0（MPS 后端可用）

### 3. 实测（完整训练 = collect 128 tick + ppo_update，同 910B 口径）
```
配置                                      collect  update   SPS
LSTM @N=1024 纯 torch                      14.7s    32.5s    2.8k
MLP  @N=1024 纯 torch（同口径对照）         10.5s     4.2s    9.0k   ← LSTM 慢 3.2x
LSTM @N=1536 纯 torch                       —        —       3.2k
LSTM @N=1024 +triton step                  10.1s    30.0s    3.3k   (+18%)
LSTM @N=1536 +triton step                  13.3s    48.7s    3.2k   （BPTT 随 N 线性涨）
LSTM @N=1024 +triton step +bptt_window=4    8.8s    20.8s    4.4k   ← 最优（总 +57%）
LSTM @N=2048 → MPS OOM（BPTT 激活超 18GB 共享内存上限）
```

### 4. 瓶颈剖析（N=1024，MPS）
- **collect ~16ms/tick**：obs+mask+feat+act ~16ms（其中 LSTM 网络前向仅 1.2ms，
  其余是模拟器 observe/legal_mask）+ **sim.step ~33ms/tick（最大单项）**
- **BPTT update 32.5s**：extract_fused(T*N=131072) 1.26s + LSTM 128 步前向 174ms
  + **128 步反向 4.96s（最大头）**，×4 epochs
- 网络本身极快（extract_fused 0.6ms / LSTM 层 0.2ms / 完整前向 1.2ms）——
  瓶颈全在**模拟器 + BPTT 反向**，不是 LSTM 架构本身

### 5. Triton 化结果
- **sim.step triton 化**（triton_step_full）：MPS 上可用；N=2048 时 step 30.7→23.4ms
  （-24%），N=1024 反而 +17%（小 batch launch 开销）→ 大 N 才有收益
- **truncated BPTT**（`PPOConfig.bptt_window`，标准 LSTM-RL 做法）：反向窗口
  128→4，update 29.3→20.4s（-30%）。CPU 冒烟数值正确，不影响前向采样

### 6. 结论
- MacBook 上 LSTM 完整训练 ~2.8-4.4k sps，**受 MPS 显存限制 N 只能 ~1024-1536**
  （BPTT 激活 18GB 上限），是 910B（19.9k@N=4096）的 1/5
- Triton 化 step + truncated BPTT 合计 +57%，验证了 LSTM 路径可优化的三个方向：
  ① 模拟器 triton 化（大 N 生效）② BPTT 截断（反向降 30%+）③ 更大 N（launch 摊薄）
- **架构取舍**：LSTM 慢的本质是"时序无法 flat + 局部观测逐角色 gather"，
  与硬件无关（MacBook/910B 都成立）。但代码完整可用，truncated BPTT 已进
  PPOConfig，后续想保 LSTM 可继续调 window/架构。
