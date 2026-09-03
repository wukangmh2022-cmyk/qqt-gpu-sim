"""910B SPS 基准：triton_step_full 单 tick 稳态计时 × N 扫描。

SPS = N × ticks_per_sec。N 取训练典型批次（8192/16384）。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_full, triton_step_core
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

for N in (4096, 8192, 16384):
    sim_t = BatchedSim(cfg, N, device=dev, seed=0)
    sim_k = BatchedSim(cfg, N, device=dev, seed=0)
    sim_t.reset_all(); sim_k.reset_all()
    mv = torch.randint(0, 5, (N, 2), device=dev)
    acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
    # 预热
    for _ in range(3):
        sim_t.step(acts)
        triton_step_full(sim_k, acts)
    sync()
    def bt(fn, it=10):
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        sync()
        return (time.perf_counter() - t0) / it * 1000
    t_t = bt(lambda: sim_t.step(acts))
    t_k = bt(lambda: triton_step_full(sim_k, acts))
    print(f"N={N:>6}: torch {t_t:8.2f} ms/tick ({N/t_t*1e3:8.0f} SPS) | "
          f"triton_full {t_k:8.2f} ms/tick ({N/t_k*1e3:8.0f} SPS) | x{t_t/t_k:.2f}")
