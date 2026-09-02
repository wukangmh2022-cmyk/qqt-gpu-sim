"""910B 同步 vs 计算权衡：danger/resolve 的 early_exit on/off 对比。

若 fixed（无同步）比 early_exit（有同步）快 → 同步比省的计算贵，删早退。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.blast import resolve_explosions, danger_map
from sim.dev import pick_device

dev = pick_device()
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
acts = torch.stack([
    torch.randint(0, 5, (N, 2), device=dev),
    (torch.rand(N, 2, device=dev) < 0.4).long()], dim=-1)
for _ in range(30):
    sim.step(acts)
sync()
bmap = sim._blast_map()

def bench(name, fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    dt = (time.perf_counter() - t0) / it * 1e3
    print(f"  {name:<42}: {dt:8.3f} ms")
    return dt

print(f"N={N} 泡数 {int((sim.fuse>0).sum().item())}")
t_ee = bench("resolve early_exit=True（有同步）", lambda: resolve_explosions(
    sim.fuse, sim.owner, sim.wall, bmap, cfg.max_chain, sim.brick,
    early_exit=True))
t_fx = bench("resolve early_exit=False（固定轮无同步）", lambda: resolve_explosions(
    sim.fuse, sim.owner, sim.wall, bmap, cfg.max_chain, sim.brick,
    early_exit=False))
print(f"  resolve: fixed/ee = {t_fx/t_ee:.2f}x")

t_ee2 = bench("danger early_exit=True", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=cfg.max_chain,
    early_exit=True))
t_fx2 = bench("danger early_exit=False", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=cfg.max_chain,
    early_exit=False))
print(f"  danger: fixed/ee = {t_fx2/t_ee2:.2f}x")
