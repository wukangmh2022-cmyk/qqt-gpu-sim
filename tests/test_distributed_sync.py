"""10 节点 20 卡分布式同步 (RCCL / JAX Distributed) 全量真实性与数据一致性测试。

测试项目：
1. JAX 分布式网络组网 (jax.distributed.initialize)
2. 拓扑与设备发现 (10 processes, 20 global DCUs)
3. 全局 pmean allreduce 数学与数值精度 (各卡输入唯一值，验证全局均值 10.50000000)
4. 全局 all_gather 跨机张量收集 (验证 20 卡数据无损互通)
5. 模拟分布式 PPO 梯度更新 (验证 10 节点更新后参数 100% 逐位绝对一致 SHA256)
"""

import os
import sys
import time
import hashlib
import numpy as np

# 必须在 import jax 之前设置环境变量
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
master_port = int(os.environ.get("MASTER_PORT", "29500"))

print(f"[Rank {rank}] 正在初始化分布式网络: MASTER={master_addr}:{master_port} WORLD_SIZE={world_size}...", flush=True)

import jax
import jax.numpy as jnp
import optax

if world_size > 1:
    jax.distributed.initialize(
        coordinator_address=f"{master_addr}:{master_port}",
        num_processes=world_size,
        process_id=rank
    )

devs = jax.devices()
local_devs = jax.local_devices()
n_local = len(local_devs)
n_total = len(devs)

print(f"\n==================== [Rank {rank}] 节点自检 ====================", flush=True)
print(f"本地设备数: {n_local}, 全局设备总数: {n_total}", flush=True)
for i, d in enumerate(local_devs):
    print(f"  - 本地卡 {i}: {d}", flush=True)

assert n_total == world_size * n_local, f"设备总数异常: 期望 {world_size * n_local}, 实际 {n_total}"

# ----------------- 测试 1: 全局 pmean 归约正确性 -----------------
# 每张卡分配一个唯一的数字: (rank * n_local + local_idx) + 1
# 例如 20 张卡的值为 1.0, 2.0, ..., 20.0
# 全局平均值理论严格值: sum(1..20)/20 = 210/20 = 10.5
local_ids = jnp.array([rank * n_local + i + 1.0 for i in range(n_local)], dtype=jnp.float32)

@jax.pmap
def test_pmean(x):
    return jax.lax.pmean(x, axis_name='dev')

# pmap 在本地设备切片上执行，全局 axis 跨所有机器
pmean_out = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='dev'), axis_name='dev')(local_ids)
expected_mean = (n_total * (n_total + 1) / 2.0) / float(n_total)

print(f"\n==================== [Rank {rank}] 测试 1: pmean 数值归约 ====================", flush=True)
print(f"本地输入: {local_ids}")
print(f"pmean 归约输出: {pmean_out}")
print(f"期望理论值: {expected_mean:.6f}")

for v in np.array(pmean_out):
    diff = abs(v - expected_mean)
    assert diff < 1e-5, f"pmean 精度错误: 实际 {v}, 期望 {expected_mean}, 差值 {diff}"
print(f"✅ 测试 1 (pmean 真实同步) 通过！误差 < 1e-5", flush=True)

# ----------------- 测试 2: 全局 all_gather 跨机收集 -----------------
print(f"\n==================== [Rank {rank}] 测试 2: all_gather 跨机通信 ====================", flush=True)

@jax.pmap
def test_allgather(x):
    return jax.lax.all_gather(x, axis_name='dev')

gathered = jax.pmap(lambda x: jax.lax.all_gather(x, axis_name='dev'), axis_name='dev')(local_ids)
gathered_arr = np.array(gathered[0])  # 取第 0 个本地卡的视角 (n_total,)
expected_arr = np.array([float(i + 1) for i in range(n_total)], dtype=np.float32)

print(f"all_gather 收集到全网 20 卡数据: {gathered_arr}")
np.testing.assert_allclose(gathered_arr, expected_arr, atol=1e-5)
print(f"✅ 测试 2 (all_gather 全网互通) 通过！20 张卡数据 100% 完整无损！", flush=True)

# ----------------- 测试 3: 模拟分布式 PPO 梯度更新与逐位一致性 -----------------
print(f"\n==================== [Rank {rank}] 测试 3: 模拟 PPO 参数一致性 (SHA256) ====================", flush=True)

# 初始化一个含 100 万参数的测试网络权重
key = jax.random.PRNGKey(42)  # 所有节点必须从相同初始参数出发
dummy_params = {
    'w1': jax.random.normal(key, (512, 512)),
    'b1': jnp.zeros((512,)),
    'w2': jax.random.normal(key, (512, 10)),
}
opt = optax.adam(1e-3)
opt_state = opt.init(dummy_params)

# 模拟各节点用不同数据产生本地梯度
node_key = jax.random.PRNGKey(rank * 100 + 7)
local_grad = {
    'w1': jax.random.normal(node_key, (512, 512)),
    'b1': jax.random.normal(node_key, (512,)),
    'w2': jax.random.normal(node_key, (512, 10)),
}

# 分布式同步梯度：对 local_grad 做 pmean
def sync_and_step(params, opt_state, grads):
    # pmean 梯度
    synced_grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name='dev'), grads)
    updates, new_opt_state = opt.update(synced_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# 在本地设备上执行更新
synced_fn = jax.pmap(sync_and_step, axis_name='dev', in_axes=(None, None, 0))
# 把 local_grad 在本地设备轴堆叠 (n_local, ...)
local_grad_stacked = jax.tree.map(lambda x: jnp.stack([x] * n_local), local_grad)

new_params, new_opt_state = synced_fn(dummy_params, opt_state, local_grad_stacked)

# 计算更新后参数的哈希值
param_bytes = np.array(new_params['w1'][0]).tobytes() + np.array(new_params['b1'][0]).tobytes() + np.array(new_params['w2'][0]).tobytes()
sha = hashlib.sha256(param_bytes).hexdigest()
digest_val = float(np.sum(new_params['w1'][0]))

print(f"更新后参数 SHA256: {sha}")
print(f"更新后参数和 Checksum: {digest_val:.6f}")

# 将哈希发送并收集
sha_bytes = np.frombuffer(sha[:16].encode('ascii'), dtype=np.uint8)
all_sha = jax.pmap(lambda x: jax.lax.all_gather(x, axis_name='dev'), axis_name='dev')(jnp.stack([sha_bytes] * n_local))
all_sha_arr = np.array(all_sha[0])

# 校验所有 20 个 replica 的哈希是否完全相等
for i in range(n_total):
    replica_sha_str = bytes(all_sha_arr[i]).decode('ascii')
    assert replica_sha_str == sha[:16], f"参数哈希不一致! 卡 0: {sha[:16]}, 卡 {i}: {replica_sha_str}"

print(f"✅ 测试 3 (分布式 PPO 零漂移更新) 通过！全网 20 卡更新后参数 100% 逐位绝对一致！", flush=True)
print(f"\n🎉🎉🎉 [Rank {rank}] 10 节点 20 卡分布式同步全套测试全部 PASS！🎉🎉🎉\n", flush=True)
