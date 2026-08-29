# 多卡 DP 训练：SCNet「模型训练」平台路径

## 为什么走平台服务而不是 HPC sbatch

实测（2026-08）：账号 `actts28ojm` 的配额是**同时最多 1 张 DCU**
（`AssocGrpGRES`，与节点无关）。在 zzeshell HPC 集群内提交 2 个
1 卡作业、或 1 个 2 卡作业，第二个 DCU 都会被拒。因此双卡验证
改用超算互联网的「模型训练」服务 —— 它是托管式多卡训练，创建任务时
直接选卡数/实例数，不受该配额限制，跨实例走平台预置的 RDMA
（InfiniBand/RoCE，NCCL_IB_* 环境变量由平台注入，RCCL 兼容读取）。

参考：https://www.scnet.cn/help/docs/mainsite/ai/model-training/practice/

## 平台注入的环境变量

创建任务后，每个实例（容器）自动注入：

| 变量 | 含义 |
|---|---|
| `WORLD_SIZE` | 实例数 |
| `RANK` | 本实例序号 0..WORLD_SIZE-1 |
| `MASTER_ADDR` | process 0 的 IP（coordinator 由 process 0 自动拉起） |
| `MASTER_PORT` | rendezvous 端口 |

## 两种验证形态（同一份脚本）

| 形态 | 每实例卡数 | 实例数 | 总卡数 | 通信 |
|---|---|---|---|---|
| A. 单容器多卡（推荐先跑） | 2 | 1 | 2 | 卡间 RCCL（无网络依赖） |
| B. 跨节点 RDMA | 1 | 2 | 2 | 卡间 RCCL over RDMA fabric |

两份跑同一 `multicard_train.py`，DP 语义相同：每个副本独立 rollout
（零通信），梯度对全局 replica 轴 pmean allreduce，全部副本参数逐位一致。

## 文件

- `jax_bomb/multicard_train.py` —— 平台多卡入口（`python3 -m jax_bomb.multicard_train`）
- `scripts/scnet_model_train.sh` —— 控制台「启动脚本」模板（粘贴全文）
- DP 内核与 `jax_bomb/jax_train.py` 的 `build_dp_one_iter` 逐位一致；
  `--check-consistency` 每轮 all_gather 参数摘要，跨副本逐位校验
  （pmean/RCCL 正确性的直接证据，失败即退出）

## 前置：镜像准备（一次性）

平台「模型训练」用「我的镜像」。按最佳实践文档的 Notebook 流程：

1. 控制台 → Notebook → 创建（区域选昆山、加速卡 AI-64GB、基础镜像）
2. 终端里装依赖并保存镜像：

```bash
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jax==0.6.0 jax-rocm60-plugin optax \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -c "import jax; print(jax.__version__, jax.devices())"
```

（DCU2 实测环境即 JAX 0.6.0 + jax-rocm60-plugin + optax，DTK 25.04。）

3. 把仓库上传到 `/root/private_data/<你的目录>/qqt-gpu-sim`
   （`private_data` 是用户个人目录，任务内可见）。

## 控制台创建任务

1. 控制台 → 人工智能 → 模型训练 → 创建任务
2. 选昆山区域「异构加速卡 AI-64GB」，**每实例加速卡数量=2，实例数=1**
   （跨节点验证时改为：每实例 1、实例数=2）
3. 镜像选上一步保存的
4. 「启动脚本」粘贴 `scripts/scnet_model_train.sh` 内容
   （脚本第一行 `cd /root/private_data/qqt-gpu-sim` 按实际路径改）

## 预期日志（DCU2 单卡实测，E512/d2/p4，4096 envs×256 steps）

```
[0] arch=transformer embed=512 depth=2 patch=4 params=6,383,112 envs/replica=4096 mb_local=4096
[0] 44.015 s/iter × 2 → 47,647 sps (4096 envs × 256 steps)
[0] consistency PASS: 1 个 replica 参数逐位一致 (digest=...)
```

单卡 pmap 路径参考值 **≈47.6K sps**；平台 2 卡（实例数×每实例卡数=2）预期
≈2 倍（95K 量级，扣少量 allreduce 开销）。一致性 FAIL 说明 RCCL
pmean/all_gather 异常，脚本默认以非零退出。

## 已验证的分布式语义（DCU2，JAX 0.6.0）

- `jax.distributed.initialize` 由 process 0 自动拉起 coordinator，各实例
  传入相同 `coordinator_address` 即可；必须在任何 jax 后端调用之前执行
  （jax_bomb 的模块级 `jnp.array` 会触发后端，故脚本把 jax_bomb 的
  import 放到 initialize 之后）。
- 多进程 pmap 输入轴 = `jax.local_device_count()`（每进程本地设备数），
  **pmap 轴本身跨进程**（axis_size = 实例数×每实例卡数）：
  用 2 进程 CPU 实测，rank0 全 1（sum=4）与 rank1 全 2（sum=8）的
  pmean = 6.0 两进程一致；all_gather 摘要跨进程逐位相同（518.2865）。
- 因此 `ppo_update` 的 `pmean(grads, axis_name="dev")` 就是全局梯度
  allreduce，参数在全部副本逐位一致 —— 脚本的 consistency 校验即
  RCCL 正确性的直接证据。

## 注意事项

- `--num-envs` 和 `--minibatch` 必须能被总卡数整除（脚本会校验）。
- 容器默认 HTTP(S)_PROXY 会让 JAX rendezvous 超时 —— 启动脚本已 unset。
- 部分 DTK 镜像 hipfftMp 缺 NEEDED libmpi，需 LD_PRELOAD —— 启动脚本
  已做自动探测；若换镜像后 import 报 undefined symbol，手动指定路径。
- 不要 unset 平台的 `NCCL_IB_*`/`NCCL_SOCKET_IFNAME`（RCCL 沿用）。
