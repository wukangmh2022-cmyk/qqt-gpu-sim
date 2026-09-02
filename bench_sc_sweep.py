"""_sc 标量 op 替换后的 910B 稳态 SPS：N=16384 min-of-3 + item 计数。

与 208e4cf 同口径（corridor speed=3.0），对比 91.1ms/17.9万 SPS。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
N = 16384

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

# 预热
for _ in range(5):
    sim.step(acts)
sync()

def one_batch(it=10):
    t0 = time.perf_counter()
    for _ in range(it):
        sim.step(acts)
    sync()
    return (time.perf_counter() - t0) / it * 1000

times = sorted(one_batch() for _ in range(3))
best = times[0]
print(f"N={N}: min={best:7.2f} ms/tick  {N/best*1e3:8.0f} SPS  ({N/best*1e3/1e4:.2f}万) | runs={['%.2f'%t for t in times]}")

# item() 同步计数（torch_npu 内部 _local_scalar_dense 类 dispatch）
try:
    from torch_npu.contrib import profiling as _  # noqa: 触达可用性
except Exception:
    pass
import torch.profiler as prof
if dev.startswith("npu"):
    with prof.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        sim.step(acts)
        torch.npu.synchronize()
    keys = [e.key for e in p.key_averages() if 'item' in e.key.lower() or 'local_scalar' in e.key.lower()]
    n_items = sum(e.self_device_time_total for e in p.key_averages() if 'item' in e.key.lower() or 'local_scalar' in e.key.lower())
    print(f"item/local_scalar 事件数: {len(keys)}  self_device 总时间: {n_items/1e3:.2f} ms")
