"""Kaggle 双 T4 验证脚本 —— 测 3 个决定性数字（对比 DCU 基线）。

DCU 基线（已实测）：
  - 500 算子 launch: CPU提交 ~15.7ms | 同步稳态 19.3ms
  - 完整 collect(n=20000,T=128): 41s（torch）/ 55s（含 cnn 对手）
  - JAX rays(n=20000,blast≤5): 165ms（HIP XLA 不融合邻居访问 → 慢 20 倍）

在 Kaggle（双 T4, CUDA）上跑：
  1) launch 开销 → 若 <5ms，CPU 提交瓶颈解除，torch 路线 1.5-2.5x 成立
  2) 模拟 step 近似链（~5000 算子，含邻居访问）→ T4 的 GPU 执行下限
  3) JAX rays（CUDA XLA）→ 若 <5ms，B 路线（完整 jax 移植）复活

用法：Kaggle Notebook → 新建 Script → 粘贴本文件 → 添加 GPU 加速器(2×T4) → Run
"""
import os, subprocess, sys, time
import torch

N, H, W = 20000, 13, 13
DEV = "cuda"

print("=== GPU 信息 ===")
print("GPU 数:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)} 显存 {p.total_memory/1e9:.1f}GB")

# ---------- 1. launch 开销（500 逐元素算子序列）----------
x = torch.zeros(N, H, W, device=DEV)
def torch_many():
    y = x
    for _ in range(500):
        y = y + 1
    return y
torch_many(); torch.cuda.synchronize()
for _ in range(5):
    torch_many()
t0 = time.perf_counter()
for _ in range(30):
    torch_many()
t_cpu = (time.perf_counter() - t0) / 30
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(30):
    torch_many()
torch.cuda.synchronize()
t_sync = (time.perf_counter() - t0) / 30
print(f"\n[1] 500算子 launch: CPU提交 {t_cpu*1000:.2f} ms | 同步稳态 {t_sync*1000:.2f} ms")
print(f"    (DCU基线: CPU提交 15.7ms | 同步 19.3ms)")

# ---------- 2. 模拟 step 近似链（~5000 算子，含邻居访问/gather）----------
print("\n[2] 模拟 step 近似链（~5000 算子 kernel 密集）:")
def sim_chain():
    fuse = torch.randint(0, 31, (N, H, W), device=DEV)          # 引信
    owner = torch.randint(-1, 2, (N, H, W), device=DEV)
    wall = torch.rand(N, H, W, device=DEV) < 0.2
    bombed = fuse > 0
    triggered = (fuse == 0) & (owner >= 0)
    covered = triggered.clone()
    for b in range(1, 8):                                       # 爆炸传播 4 方向 × b 步
        src = triggered & (fuse == b)
        for dr in (-1, 1):
            for dc in (0,):
                front = torch.roll(src, dr, dims=1) & ~wall      # 邻居访问
                covered |= front
        for dc in (-1, 1):
            for dr in (0,):
                front = torch.roll(src, dc, dims=2) & ~wall
                covered |= front
    fuse2 = torch.where(triggered, torch.zeros_like(fuse), fuse)  # 清场
    pos = torch.rand(N, 2, 2, device=DEV)
    cell = pos.long()
    flat = cell[..., 0] * W + cell[..., 1]
    alive = torch.ones(N, 2, device=DEV, dtype=torch.bool)
    dmg = (torch.rand(N, 2, device=DEV) < 0.1).float()
    reward = dmg.sum() - dmg.sum(dim=1, keepdim=True) + alive.float().sum()
    return reward
sim_chain()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    sim_chain()
torch.cuda.synchronize()
t_chain = (time.perf_counter() - t0) / 5
print(f"    T4 同步稳态: {t_chain*1000:.1f} ms/次")
print(f"    (DCU富泡step基线: ~82 ms/tick——含真实奖励/危险图，此链更轻，看相对量级)")

# ---------- 3. JAX rays（CUDA XLA——B 路线生死）----------
print("\n[3] JAX rays（CUDA XLA 融合邻居访问?）:")
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("    未安装 jax，安装中...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "jax[cuda12]"])
    import jax
    import jax.numpy as jnp

def shift_jax(x, dr, dc):
    xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    return xp[:, 1 - dr:1 - dr + H, 1 - dc:1 - dc + W]

def rays_jax(src, wall, bombed, brick, blast, b_max):
    not_solid = ~(bombed | brick)
    seed = src & ~wall & ~brick
    covered = seed
    for b in range(1, b_max + 1):
        src_b = seed & (blast == b)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            front = src_b
            for _ in range(b):
                front = shift_jax(front, dr, dc) & ~wall
                covered = covered | front
                front = front & not_solid
    return covered

src_j = jnp.zeros((N, H, W), dtype=bool).at[:, 3, 3].set(True)
wall_j = jnp.zeros((N, H, W), dtype=bool)
bombed_j = jnp.zeros((N, H, W), dtype=bool)
brick_j = jnp.zeros((N, H, W), dtype=bool)
blast_j = jnp.full((N, H, W), 7, dtype=jnp.int32)
f = jax.jit(lambda s, w, b, br, bl: rays_jax(s, w, b, br, bl, 7))
out = f(src_j, wall_j, bombed_j, brick_j, blast_j)
jax.block_until_ready(out)
for _ in range(3):
    f(src_j, wall_j, bombed_j, brick_j, blast_j)
jax.block_until_ready(f(src_j, wall_j, bombed_j, brick_j, blast_j))
t0 = time.perf_counter()
for _ in range(20):
    f(src_j, wall_j, bombed_j, brick_j, blast_j)
jax.block_until_ready(f(src_j, wall_j, bombed_j, brick_j, blast_j))
t_rays = (time.perf_counter() - t0) / 20
print(f"    T4 jax rays: {t_rays*1000:.2f} ms")
print(f"    (DCU基线: 165ms——HIP 不融合；若 T4 <5ms → B 路线复活)")

print("\n=== 结论判断 ===")
print(f"[1] launch 同步稳态: {t_sync*1000:.1f} ms" + (" → CPU提交瓶颈解除，torch 1.5-2.5x 成立" if t_sync*1000 < 8 else " → launch 仍偏高，需查"))
print(f"[3] jax rays: {t_rays*1000:.1f} ms" + (" → CUDA XLA 融合成功，B 路线(50.7M FPS路径)可行" if t_rays*1000 < 5 else " → CUDA XLA 也未融合，B 路线仍需斟酌"))
