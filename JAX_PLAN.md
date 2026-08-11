# JAX 全栈移植计划（B 路线）—— 50.7M FPS 路径

## 目标
把 Bomberman 训练的 **collect 全链路**（网络前向 + 128 tick 模拟）JIT 编译成一个 XLA 图，
消除 5145 kernel/tick 的 launch 瓶颈（POC 已验证：500 算子融合 179x、GEMM 18x、rays 2.6x）。

## 架构决策
- **collect 全 jax**：网络（jax 重写 MLP）+ 模拟 step（jax 重写）→ 128 tick 编译成 1 个 XLA 图（`lax.scan`）
- **PPO 更新留 torch**：网络参数 torch 持有（PPO 更新）→ 每迭代转 numpy→jax（345k 参数 ~10ms）
- **数据转换**：collect 产出 buffer（obs/act/logp/val/rew/done）一次性 jnp→torch（每迭代一次，非每 tick）
- **状态管理**：jax 纯函数（immutable pytree，`step(state, actions) -> (new_state, ...)`），XLA buffer donation 优化 in-place
- **随机性**：显式 RNG key（`jax.random.split`）——lax.scan 内合法（Generals.io 同款）
- **对手**：astar/hunter 等规则 bot 也 jax 化（或 collect 内嵌 jax 版 dijkstra）

## 关键风险（已验证/待验证）
- ✅ LD_PRELOAD libmpi 解决 hipfftMp 符号问题
- ✅ jax rays bitwise 一致（POC2）
- ⚠️ 128 tick XLA 图编译时间（预计分钟级，首次编译一次缓存）
- ⚠️ XLA 图显存（128 tick 中间张量；buffer donation 缓解）
- ⚠️ 动态控制流（爆炸链 while、reset）——lax.while_loop/固定轮 + 掩码

## 分阶段（每阶段 bitwise 验证 vs torch_sim）
- **S1 核心 step**：引信递减 + 放泡 + 爆炸链（resolve/rays）+ 移动（AABB）+ 清场 + 基础奖励
- **S2 完整 step**：+ 危险图（danger_map）+ place_predict 奖励 + hit_attr + combo + 统计 + 终局/reset
- **S3 网络 + collect**：jax ActorCritic（forward/act/logp/entropy）+ `lax.scan` 128 tick + 对手
- **S4 训练集成**：jax collect ↔ torch PPO 参数/数据转换 + 训练循环改造
- **S5 端到端**：训练 sps 实测 + 与 torch 基线对比 + 正确性回归

## 文件
- `sim/jax_sim.py`：jax 版模拟（step/observe/rays/danger/移动）
- `sim/jax_net.py`：jax 版 ActorCritic
- `train/jax_collect.py`：collect 扫描 + 转换
- 验证脚本：`opt_test/scripts/verify_jax_sim.py`（bitwise vs torch_sim）

## 当前进度
- 2026-08-10：JAX 安装完成（venv + LD_PRELOAD）；POC 验证（launch 179x/GEMM 18x/rays 2.6x bitwise 一致）
- 下一步：S1 核心 step（引信/放泡/爆炸/移动）
