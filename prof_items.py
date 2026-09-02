"""定位剩余 2 个 item/local_scalar 事件 + step 各段耗时分布。"""
import sys, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
N = 16384

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(3):
    sim.step(acts)
torch.npu.synchronize()

import torch.profiler as prof
from torch.profiler import ProfilerActivity
with prof.profile(activities=[ProfilerActivity.CPU],
                  with_stack=True, record_shapes=True) as p:
    sim.step(acts)
    torch.npu.synchronize()

print("=== item/local_scalar 事件（含调用栈） ===")
for e in p.key_averages():
    if 'item' in e.key.lower() or 'local_scalar' in e.key.lower() or 'scalar_tensor' in e.key.lower():
        print(f"  {e.key}: count={e.count} cpu={e.self_cpu_time_total/1e3:.3f}ms")
        for st in e.stack[:6]:
            print(f"      at {st}")
        print()

print("=== CPU 总耗时 Top 20 ===")
rows = sorted(p.key_averages(), key=lambda e: -e.self_cpu_time_total)[:20]
for e in rows:
    print(f"  {e.self_cpu_time_total/1e3:9.3f}ms x{e.count:5d}  {e.key}")
print(f"  总 CPU self: {sum(e.self_cpu_time_total for e in p.key_averages())/1e3:.1f} ms")
