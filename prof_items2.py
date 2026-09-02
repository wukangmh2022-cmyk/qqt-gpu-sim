"""定位 item/_local_scalar_dense 的确切调用栈（用 events 而非 key_averages）。"""
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

evs = [e for e in p.events() if e.name in ("aten::item", "aten::_local_scalar_dense", "aten::nonzero")]
print(f"=== {len(evs)} 事件 ===")
for e in evs:
    print(f"\n--- {e.name} (cpu={e.self_cpu_time_total/1e3:.3f}ms) ---")
    for fr in e.stack[:10]:
        print(f"    {fr}")
