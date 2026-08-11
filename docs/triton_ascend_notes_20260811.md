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
