"""验证 triton 危险图（阶段 A 连锁修正 + 阶段 B 扩散）bitwise vs torch danger_map。

覆盖：
  1. max_chain=1 回归（阶段 B only——原路径行为不变）
  2. max_chain=16（阶段 A+B，torch danger_map 全语义）
  3. 富泡多 tick 场景（25 tick 连续随机动作，含连锁组）+ 稀疏场景
  4. 速度对比
用法：.venv/bin/python verify_triton_dangerA.py
"""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/root")

from sim.triton_sim import _HAS_TRITON, danger_triton
if not _HAS_TRITON:
    print("triton 不可用"); sys.exit(1)
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.blast import danger_map
from sim.dev import pick_device

dev = pick_device()
N, H, W = 4096, 13, 13
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10)
torch.manual_seed(0)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()


def compare(tag, fuse, wall, brick, blast_map, mc):
    ref = danger_map(fuse, wall, blast_map, cfg.fuse, brick, max_chain=mc)
    got = danger_triton(fuse, wall, fuse > 0, brick, blast_map, cfg.fuse,
                        max_chain=mc)
    sync()
    diff = (ref - got).abs().max().item()
    pos = int(((ref > 0) != (got > 0)).sum())
    ok = (diff == 0.0) and pos == 0
    print(f"  [{tag}] max_chain={mc}: maxdiff {diff:.3e} 非零位置差 {pos}"
          f" {'✓' if ok else '✗'}")
    return ok


def build_rich_states(ticks=25):
    """连续随机动作制造富泡/连锁状态，收集若干 snapshot。"""
    sim = BatchedSim(cfg, N, device=dev, seed=0)
    sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
    sim.reset_all()
    states = []
    for i in range(ticks):
        mv = torch.randint(0, 5, (N, 2), device=dev)
        bmb = (torch.rand(N, 2, device=dev) < 0.45).long()
        acts = torch.stack([mv, bmb], dim=-1)
        sim.step(acts, auto_reset=False)
        if i % 3 == 0:   # 采样：不同引爆阶段的状态
            states.append((sim.fuse.clone(), sim.wall.clone(),
                           sim.brick.clone(), sim._blast_map().clone()))
    sync()
    return states


ok_all = True
print(f"=== 1. 回归：max_chain=1（阶段 B 原路径）===")
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(15):
    sim.step(acts, auto_reset=False)
sync()
ok_all &= compare("富泡", sim.fuse, sim.wall, sim.brick, sim._blast_map(), 1)

print(f"=== 2. 阶段 A+B（max_chain=16）：富泡多 tick 快照 ===")
states = build_rich_states(25)
for i, (f, wl, br, bl) in enumerate(states):
    ok_all &= compare(f"snap{i}", f, wl, br, bl, 16)

print(f"=== 3. 稀疏场景（无连锁，早退路径）===")
fuse_s = torch.zeros(N, H, W, device=dev, dtype=torch.uint8)
fuse_s[:, 3, 3] = 8; fuse_s[:, 5, 9] = 5; fuse_s[:, 9, 7] = 2   # 三颗孤立泡
wall_s = torch.zeros(N, H, W, device=dev, dtype=torch.bool)
brick_s = torch.zeros(N, H, W, device=dev, dtype=torch.bool)
blast_s = torch.zeros(N, H, W, device=dev, dtype=torch.int32) + 3
ok_all &= compare("孤立泡", fuse_s, wall_s, brick_s, blast_s, 16)

print(f"=== 4. 空场短路 ===")
fuse_e = torch.zeros(N, H, W, device=dev, dtype=torch.uint8)
ok_all &= compare("无泡", fuse_e, wall_s, brick_s, blast_s, 16)

print(f"=== 5. 速度（N={N}，富泡 max_chain=16）===")
f, wl, br, bl = states[-1]
for _ in range(3):
    danger_map(f, wl, bl, cfg.fuse, br, max_chain=16)
    danger_triton(f, wl, f > 0, br, bl, cfg.fuse, max_chain=16)
sync()
t0 = time.perf_counter()
for _ in range(10):
    danger_map(f, wl, bl, cfg.fuse, br, max_chain=16)
sync()
t_t = (time.perf_counter() - t0) / 10 * 1e3
t0 = time.perf_counter()
for _ in range(10):
    danger_triton(f, wl, f > 0, br, bl, cfg.fuse, max_chain=16)
sync()
t_k = (time.perf_counter() - t0) / 10 * 1e3
print(f"  torch danger_map: {t_t:.1f} ms | triton: {t_k:.2f} ms | x{t_t/t_k:.1f}")

print(f"\n{'全部通过 ✓' if ok_all else '有差异 ✗'}")
sys.exit(0 if ok_all else 1)
