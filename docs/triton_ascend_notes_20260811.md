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
