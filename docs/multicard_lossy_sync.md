# 多机 DCU 训练：有损同步（Local SGD）方案与关键 Debug 记录

> 人类知识沉淀：260B 参数训练跨多机 DCU 的通信问题、探索死路、最终方案、实测数据与部署清单。
> 更新：2026-08-19。关联代码：`jax_bomb/jax_train.py`（ppo_update_lsgd / ppo_update_gradsync）、
> `jax_bomb/multicard_train.py`（--lsgd-* 参数）、`scripts/scnet_model_train.sh`（LSGD 透传）。

## 1. 背景与目标

- 训练目标：260B 参数量级模型，3.5 天完成 → 需 **~860K sps**（env-steps/s）。
- 资源现实：
  - 单机 8 卡 = 304K sps（260B ≈ 9.9 天），但**平台要审批**，拿不到。
  - 最容易自由支配的配置：**10 个账号 × 10 台 notebook × 2 卡 = 20 卡**。
- 20 卡无损（每 minibatch 全量梯度 allreduce）不可行：通信 100s+/迭代（见下）。
- 结论：**有损同步（Local SGD）是唯一出路**；目标是通信开销 <10%、质量损耗尽可能小。

## 2. 为什么无损多机不可行（物理数学 + 实测）

- 模型 6.39M 参数 = 25.5MB fp32。一次全量 allreduce 在 N 卡 ring 下每卡移动 `2×S×(N−1)/N` 字节
  （2 卡 = 25.5MB，20 卡 = 48.5MB）。
- 每迭代 minibatch 数 = `(2×envs_per×num_steps/minibatch) × epochs`，生产配置 = **1024 个**。
- 每个 minibatch 一次全量梯度同步 → 20 卡时 **~50GB/迭代/卡** → 通信 100s+ 级，完全不可行。
- **关键澄清**：这 1024 次 allreduce 是**顺序依赖的优化步**（每步需要前一步结果），不是
  一次 pmean 的叶子碎片化。XLA 早已把单次 pmean 的叶子融合成一条大消息，所以"加大梯度桶
  把 512 次变 4 次"在单次 pmean 内做不到——要减次数只能**跳过依赖**（即 Local SGD）。

## 3. 探索过的死路（每条都有实测证据，别再走）

| 方案 | 实测结果 |
|---|---|
| RCCL 调优：`sysctl` 加大 TCP 缓冲 | 容器内不可用（`/proc/sys/net/core/rmem_max` 不存在，内核参数被屏蔽）|
| `RCCL_SOCKET_IFNAME=eth0` | 无效——RCCL 本来就选 eth0:172.31.x |
| `RCCL_BUFFSIZE=4MB` + `NCCL_SOCKET_NTHREADS=4` | **反而慢 2 倍**（244s/iter vs 131.5s）|
| 裸 TCP 带宽 | 3.37Gbps（420MB/s），RCCL socket 有效只有 ~250MB/s（协议开销 ~40ms/次 × 1024 次 ≈ 43s/迭代 的延迟地板）|
| 文件系统通信（NFS）| NFS 属性缓存 acdirmin=30 → 跨客户端文件可见性延迟 ~30s，不可行 |
| HPC 节点 + notebook 组合 | RCCL 要求**每对 worker 双向可达**，HPC 无法连回平台 overlay（172.31.x），`socketStartConnect` 失败 |
| 两 notebook 跨账号跨机 | **可行**（双向 172.31.x overlay），consistency PASS，是后续所有 A/B 的基础 |

**结论**：调优是死路，字节压缩（稀疏化/量化）也救不了——它只减字节不减同步次数，1024 次的
延迟地板（~43s/迭代）还在。必须**减少同步次数**。

## 4. 有损同步方案：Local SGD，两个模式

设计要点（`jax_train.py`）：

- **通信通道无损**：fp32 稠密 allreduce（可选 bf16 半精度减半），无稀疏化、无量化、无误差反馈。
  "有损"只来自 K 步的更新节奏差异（staleness），K 是质量旋钮。
- **param 模式（`--lsgd-mode param`，默认）**：minibatch 内零通信，每 K 步 `pmean(params)`
  （可选 `--lsgd-sync-state` 连 Adam 动量/方差一起平均，防本地漂移但流量 ×3）。
  保持每迭代 1024 次更新不变；代价是 K 步本地漂移。**无内存限制，任意 K**。
- **grad 模式（`--lsgd-mode grad`）**：每 K 个 minibatch 拼成一个 K× 大 batch，一次
  value_and_grad（`∇(1/K Σᵢ lossᵢ) = (1/K) Σᵢ ∇lossᵢ`，线性性）、一次 pmean、一次更新。
  **参数任何时刻逐位一致、零漂移**；`--lsgd-k 1` 时与无损路径**逐位一致**（实测 digest 相同）。
  代价：每迭代只有 n_mb/K 次更新（大 batch 效应）；**受激活内存限制**（见 Debug #3/#4）。
- 通信量：同步次数从 1024 降到 1024/K。K=256 → 4 次/迭代 ≈ 194MB（20 卡）/ 102MB（2 卡）。
- **评估（--eval-vs + --eval-every）**：两策略对打（`collect_rollout_two`）——当前策略 p0 vs
  冻结基线 p1，报 p0 胜率。基线支持 `{RANK}` 占位（各 rank 用自己那份 ckpt，无需跨机拷贝）。
  自包含、不依赖 torch。实测 vs 1 迭代前的自己 winrate≈0.49（语义正确）。
- **每迭代聚合统计**：`rew=`（平均回报）`ep_len=`（平均对局长度 = 1/每帧结束率），连续监控
  训练健康；短 rollout（< 一局长度）时 ep_len=nan 属正常。
- **掉线容错**：RCCL 是同步屏障，**单台掉线=全体卡死/崩溃**（无自动恢复）。降级路径 =
  检查点 + 重启：检测（watch 标 [疑似卡死]）→ 全杀 → 存活节点以 `WORLD_SIZE=N-1` 从最近
  ckpt 重启。续训校验已改为**按 per-replica 负载**（envs_per/mb_local，机器数变化只警告），
  掉线降级可直接接续（--num-envs/--minibatch 按新卡数重设）。

### 数学依据（用户确认的线性性论证）

`(g1+...+gN)/N = g1/N + ... + gN/N` —— allreduce（求平均）与求和可交换。攒 K 个 minibatch
再同步一次，传输的"平均梯度之和"与逐步同步是**同一个数学量**。但训练轨迹不完全相等：
逐步同步的梯度是在移动的参数上求的，攒批的在冻结参数上求（差 O(K·lr·曲率)），且 Adam 非线性。
grad 模式=大 batch 训练（batch ×K），param 模式=经典 Local SGD（参数漂移后拉齐）。

## 5. 关键 Debug 记录

1. **`--ff-factor` float shape bug（2026-08-19）**：argparse `type=float` 把 `--ff-factor 4`
   解析成 `4.0`，`ff_factor × embed = 1568.0` 变成浮点 shape → `jax.random.normal` 崩
   `TypeError: Shapes must be 1D sequences of concrete values`。生产默认不传该参数（默认值 int 4）
   所以没暴露。**修复**：`jax_net.py` 所有 shape 计算处包 `int()`（init_transformer 的 ff1/ff2、
   init_mlp_mixer 的 tm1/tm2/cm1/cm2，共 6 处）。
2. **scan 内逐 minibatch 累加梯度慢 2.4×**：grad 模式最初用"内层 scan 累加 g_acc"实现，
   DCU 上比 baseline 慢 2.4×（10.6s vs 4.35s，同样 512 次前向/反向）。闭包捕获外层 scan 循环
   变量改成 carry 显式传递**无效**（digest 逐位相同，性能不变）→ 根因是嵌套 scan + 累加 carry
   的 XLA 编译，不是闭包。**修复**：改大 batch 实现（一次大前向/反向），实测**比 baseline 快 17%**
   （大 GEMM 效率更高）。
3. **grad 模式激活内存 OOM**：大 batch 的激活内存随样本数线性增长，64GB 卡实测 131K 样本一批
   OOM（要 78GB）。**修复**：`GRAD_MAX_SAMPLES=65536` 护栏，超限报错并提示减小 K 或换 param
   模式。grad 模式适用 K ≤ ~32-64（视 minibatch）。
4. **Python 展开 sub-batch 不省内存**：把大批拆成 4×32K 的 Python 循环，仍 OOM（70.9GB）——
   XLA 不会在展开的多次调用间复用缓冲区。结论：内存安全只能靠护栏，不能靠拆包。
5. **grad k=1 与无损逐位一致（验证方法）**：单 notebook 2 卡（world=1），同 seed 跑
   `--lsgd-k 0` 与 `--lsgd-k 1 --lsgd-mode grad`，digest 完全相同
   （1.96905749511718750e+03）+ loss 相同（0.1179）→ grad 模式是现状无损路径的严格超集。
6. **notebook 卡数陷阱**：`jax.devices()` 可能显示 `['rocm:0','rocm:1']` 但
   `n_local=1`——因为 notebook **创建时设备数设了 1**。要 2 卡需创建时选 2 卡（用户确认可行）。
7. **RCCL 跨机一致性校验**：`multicard_train` 每迭代 all_gather 参数 digest，要求逐位一致
   （PASS）。LSGD 模式下迭代结束恰在同步边界后，digest 仍应 PASS——FAIL 说明 RCCL 异常。

## 6. 实测结果（跨机 A/B：2 notebook × 1 卡，同 seed，num_envs=4096/256 步/epochs=2）

| 配置 | iter 时间 | sps | 提速 | 通信/迭代 | 效率损耗 | loss@iter3 |
|---|---|---|---|---|---|---|
| baseline 无损（`--lsgd-k 0`）| 132.5s | 15.8K | 1× | ~106s | 80% | 0.4548 |
| grad k=32 | 30.6s | 68.6K | 4.3× | ~4.5s | 17% | (2.08)* |
| param k=128 | 26.1s | 80.4K | 5.1× | ~0.8s | 3% | 0.6479 |
| **param k=256** | **25.8s** | **81.3K** | **5.1×** | **~0.4s** | **1.6%** | 0.4908 |

\* grad 模式 loss 是冻结参数大 batch 口径，与 baseline 不可直接比。

**param k=256 × 30 iter 质量长训**：loss 0.8303 → **0.0528**，25.75s/iter 全程稳定，
consistency PASS。与 baseline 对比 iter3 差 +7.9% 后持续收敛——loss 代理口径质量损耗单位数%，
**严格值需下游 win-rate eval**（loss 不是终局指标）。注意：Local SGD 漂移代价随机器数稀释
（N 台平均方差 ↓N），2 台 A/B 是 K 的最坏情形，20 台只会更好。

## 7. 生产配置决策（2026-08-19 用户拍板）

- **transformer：embed=392, depth=4, patch=4, heads=4, ff_factor=4 → 7,461,336 参数**（7.5M 目标）。
  论文 depth=6，取更稳妥的 4（时间/金钱成本权衡）。
- **单机单卡实测 21.8K sps（48.1s/iter）**——比 depth=2/embed=512（39.4K sps）慢 1.8×：FLOPs
  只多 17%，但 4 层顺序 transformer 在 DCU 上小 batch 前向是延迟主导，层数翻倍开销超线性。
  用户预估 28-30K 未达标，如实记录。
- 20 卡（10 机×2 卡）预计 ~436K sps（每卡 envs 1638 → 迭代 38.5s + 通信 ~1s）→
  **260B ≈ 6.9 天**（vs 单机 8 卡 depth-2 9.9 天；vs depth=2/embed=512 20 卡 ~3.9 天）。
  深度 4 的稳健性代价真实存在，是否换回 depth 2 由用户定夺。
- 20 卡通信：param k=256，每节点 ~92MB/同步 × 4 ≈ 370MB/迭代 ≈ 1.5s → 效率损耗 ~7%
  （fp32）；`--lsgd-bf16` 减半到 ~3.5%。48s 计算量级下通信占比更小。
- 推荐启动：`--lsgd-k 256 --lsgd-mode param`（无内存限制、效率损耗小）。

## 8. 10 机 × 2 卡部署清单

- 部署包：`dcu_deploy_10node.tar.gz`（jaxbomb.tgz + optax wheels + setup_notebook.sh + README）。
- `setup_notebook.sh`：解代码到 `/root/private_data/qqt-gpu-sim/`，`pip install --no-index
  --find-links=wheels --no-deps optax chex dm-tree toolz wrapt etils typing_extensions
  absl-py attrs`（**不碰 numpy/jax，防破坏平台 jax 0.6.0**），DTK env + LD_PRELOAD + 自检。
- 启动（第 N 台，N=0..9）：
  ```
  export WORLD_SIZE=10 RANK=<N> MASTER_ADDR=<rank0-ip:172.31.x> MASTER_PORT=29500
  export LSGD_K=256 LSGD_MODE=param CKPT_DIR=ckpt CKPT_EVERY=30
  python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 \
    --heads 4 --ff-factor 4 --num-envs 32768 --num-steps 256 --minibatch 32768 \
    --epochs 2 --iters 2000 --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE
  ```
- **同步启动**：rank0（coordinator）先起，其余 9 台在 rendezvous 超时（600s）内启动即可；
  建议编排脚本 30s 内全部拉起。rank 从 workerN/hostname 推导的兜底在 10 台场景不可靠，
  **必须显式传 RANK/WORLD_SIZE/MASTER_ADDR**。
- **落盘策略**：每台按 `ckpt_<iter>_r<rank>.pkl` 存本地（断点续训，rank 各自的状态/keys）；
  **rank0 额外每 30 分钟存 params 轻量快照**（`ckpt_local/params_it*.pkl`，~25MB pickle），
  供 `pull_ckpt_local.sh` 拉回本地/评估——参数小，不拖速度（`--ckpt-local-dir` /
  `--ckpt-local-every`，环境变量 CKPT_LOCAL_DIR / CKPT_LOCAL_EVERY 可调）。
- **编排**：`dcu_deploy_10node.tar.gz` 内含 `launch_10nodes.sh`（部署+取 IP+同步启动）、
  `watch_10nodes.sh`（监控）、`pull_ckpt_local.sh`（拉快照）；nodes.txt 每行
  `<端口> <主机> <密码>`，第 1 行 = rank0。
- 校验：每迭代 `consistency PASS`；启动打印 `LSGD: k=… → N 次同步/迭代 ≈ XMB`。

## 9. 环境坑速查（notebook / 平台容器）

- JAX 0.6.0 需要 `source /opt/dtk/env.sh`，否则 `libMIOpen-recommend.so` 缺失。
- `hipfftMp` 缺 NEEDED libmpi：`export LD_PRELOAD=/usr/mpi/gcc/openmpi-4.1.7a1/lib/libmpi.so`
  （否则 `undefined symbol ompi_mpi_int`）。
- 平台容器默认代理会让 JAX rendezvous 超时：`unset HTTP_PROXY HTTPS_PROXY http_proxy
  https_proxy ALL_PROXY all_proxy`。
- `jax.distributed.initialize` 必须在任何 JAX 后端调用之前（jax_bomb 模块 import 会初始化
  后端，所以 import 放在 initialize 之后）。
- optax 0.2.8 需从 wheels 离线装（notebook 无外网），`--no-deps` 防 numpy 升级。
- RCCL 调优不要碰（见 §3）；`--ff-factor` 显式传参注意 float shape bug（已修）。
