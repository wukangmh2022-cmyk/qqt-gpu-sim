# jax_bomb_v2 — 下一波测试副本（不动正在跑的 jax_bomb）

基线 = 当前运行版（1cc8032 回退 + crate 档位奖励 0.15 + 半径 0.42），
叠加以下改动。启动：把 train_cmd.sh 的模块名换成 `python3 -m jax_bomb_v2.train_real`，
其余参数 + `--adv-top-frac 0.25 --ema-decay 0.999 --obs-quant --lsgd-bf16`。

## 改动清单
1. **方案 1（正确版）**：Actor 用 |优势| Top-25% 高信噪比样本，Critic 用均匀无偏样本
   （每个 minibatch 前半/后半），全批优势标准化。**三条路径（ppo_update /
   ppo_update_lsgd / ppo_update_gradsync）全部真实现**——修复了 ae88ee3 版
   LSGD 组合被忽略的死代码（本天审计发现的 Bug 1）。
2. **HL-Gauss 128 桶价值头，量程 ±20 / σ=1.5**：按 round-1 老奖励量程重配
   （|G|max ≈ 击杀10+命中7.5+超时2+吃箱；P0 的 ±1 量程配老奖励会静默软截断，
   这是"能不能用 128 桶"的关键——量程必须配奖励，不是要不要的问题）。
3. **无偏提速全家桶**：LSGD k=256 param（已在跑）、`--lsgd-bf16`（同步流量÷2，
   仅同步精度损失）、`--obs-quant`（obs uint8，显存÷4、精度 1/255）、
   Top-25% 使更新阶段样本量 ÷4（Actor 侧有偏是设计意图，Critic 保持无偏均匀）。
4. **EMA(0.999) 参数快照**（params_*_ema.pkl 与 raw 并存）。
5. **eval_every=0 除零守卫**（fa23611 同款）。
6. 已知并修复：ae88ee3 版 `ppo_update_lsgd` 组合死代码（本副本 LSGD 真过滤）、
   `n_mb_c` 定义缺失、hl_gauss 函数缺失（构建时补齐）。

## 刻意不改
- **patch 4 保持**（--patch 是现成 flag，patch 3 留作独立 A/B，不和过滤/量程混变量）
- **γ=0.995、老奖励、epochs 2、无课程**（与正在跑的基线同血统，对照干净）

## 与基线的对照实验设计
同种子同参数，仅切换 jax_bomb / jax_bomb_v2 + 上述 flag，对比：
锚点三胜率（it158/801/1450）、IDLE 分地形自杀率、vs Hunter 胜率、sps/iter。
