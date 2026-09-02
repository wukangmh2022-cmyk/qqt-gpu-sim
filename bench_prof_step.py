"""Profile corridor+open crate/growth path kernels on DCU (device time top)."""

from __future__ import annotations

import argparse
import time

import torch
torch.compile = lambda fn, **kw: fn

import sim.torch_sim as _ts
_ts._HAS_TRITON = False
_ts._move_triton = None

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.dev import resolve_device


p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=16384)
p.add_argument("--map-mode", default="corridor",
               choices=["corridor", "open"])
p.add_argument("--open-fraction", type=float, default=1.0)
p.add_argument("--open-blast", type=int, default=None, help="覆盖 open_growth_blast")
args = p.parse_args()

dev = resolve_device("cuda")
if args.map_mode == "open":
    cfg = SimConfig()
else:
    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=args.open_fraction, ring_fraction=0.0,
                    hazard_fraction=0.0, crate_speed_only=False,
                    timeout_draw=False, combo_reward=0.10, combo_gap_factor=0.9)
    if args.open_blast is not None:
        cfg.open_growth_blast = args.open_blast
        cfg.open_growth_bombs = args.open_blast
N = args.num_envs
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
acts = torch.stack([
    torch.randint(0, 5, (N, 2), device=dev),
    (torch.rand(N, 2, device=dev) < 0.3).long(),
], dim=-1)
for _ in range(3):
    sim.step(acts)
torch.cuda.synchronize()
print(f"[prof] map={args.map_mode} open_f={args.open_fraction} N={N} "
      f"cfg.map_mode={cfg.map_mode}")

import torch.profiler as prof
acts_list = [prof.ProfilerActivity.CPU]
if hasattr(prof.ProfilerActivity, "CUDA"):
    acts_list.append(prof.ProfilerActivity.CUDA)
with prof.profile(activities=acts_list, record_shapes=False) as pr:
    for _ in range(5):
        sim.step(acts)
    torch.cuda.synchronize()

kavg = pr.key_averages()
use_dev = hasattr(prof.ProfilerActivity, "CUDA")
attr = "self_device_time_total" if use_dev else "self_cpu_time_total"
rows = sorted(kavg, key=lambda e: -getattr(e, attr))[:25]
print(f"=== top25 by {attr} (ms) ===")
for e in rows:
    v = getattr(e, attr)
    print(f"{v/1e6:9.2f} ms x{e.count:6d}  {e.key}")
tot = sum(getattr(e, attr) for e in kavg)
print(f"total {attr}: {tot/1e6:.2f} ms over {len(kavg)} ops")

# ---- wall-clock 分段（observe / legal / step），同步后计时 ----
print("\n=== wall per-tick (synchronized) ===")
def wall(fn, reps=20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0

to = wall(lambda: sim.observe())
tl = wall(lambda: sim.legal_mask())
ts = wall(lambda: sim.step(acts, auto_reset=False))
print(f"observe  {to:7.2f} ms/tick")
print(f"legal    {tl:7.2f} ms/tick")
print(f"step     {ts:7.2f} ms/tick")
print(f"total    {to+tl+ts:7.2f} ms/tick")

