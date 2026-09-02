"""N 扫描：单卡 910B 的 SPS 峰值在哪？N ∈ {8k..256k}。"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def bench(sim, acts, it=6):
    for _ in range(3):
        sim.step(acts)
    torch.npu.synchronize()
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        for _ in range(it):
            sim.step(acts)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / it * 1000)
    return min(ts)

for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
    try:
        sim = BatchedSim(cfg, N, device=dev, seed=0)
        sim.reset_all()
        mv = torch.randint(0, 5, (N, 2), device=dev)
        acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
        ms = bench(sim, acts)
        sps = N / ms * 1e3
        print(f"N={N:>7}: {ms:8.2f} ms/tick  {sps:10.0f} SPS  ({sps/1e4:.2f}万)  "
              f"每 env {ms*1e3/N:.3f} µs")
        del sim, mv, acts
        torch.npu.synchronize()
    except Exception as ex:
        print(f"N={N}: FAIL {type(ex).__name__}: {ex}")
