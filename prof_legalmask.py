"""legal_mask 剖析：16.4ms/step 花在哪？"""
import sys, torch, torch_npu, torch.profiler as prof
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.obs import legal_mask
from sim.dev import pick_device

dev = pick_device()
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(3):
    sim.step(acts)
    legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
               sim.brick, sim.bombs_cap)
torch.npu.synchronize()

with prof.profile(activities=[prof.ProfilerActivity.CPU]) as p:
    legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
               sim.brick, sim.bombs_cap)
    torch.npu.synchronize()

kavg = p.key_averages()
tot = sum(e.self_cpu_time_total for e in kavg)
n_ev = sum(e.count for e in kavg)
print(f"legal_mask 总 CPU self: {tot/1e3:.2f} ms, 事件数: {n_ev}")
rows = sorted(kavg, key=lambda e: -e.self_cpu_time_total)[:20]
for e in rows:
    print(f"  {e.self_cpu_time_total/1e3:9.2f} ms x{e.count:5d}  {e.key}")
