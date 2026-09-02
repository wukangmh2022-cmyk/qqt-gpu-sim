"""NPU 设备侧剖析：step 的 device kernel 时间 + op 数分段。"""
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

# NPU activity 是否可用
try:
    from torch.profiler import ProfilerActivity
    acts_list = [a for a in [ProfilerActivity.CPU] + ([ProfilerActivity.NPU] if hasattr(ProfilerActivity, 'NPU') else [])]
    print("activities:", acts_list)
    from torch.profiler import profile
    with profile(activities=acts_list, record_shapes=False) as p:
        sim.step(acts)
        torch.npu.synchronize()
    kavg = p.key_averages()
    # device 总时间
    tot_dev = sum(e.self_device_time_total for e in kavg)
    tot_cpu = sum(e.self_cpu_time_total for e in kavg)
    n_ops = len(kavg)
    print(f"设备时间合计: {tot_dev/1e6:.2f} ms, CPU self 合计: {tot_cpu/1e3:.1f} ms, 唯一算子: {n_ops}")
    rows = sorted(kavg, key=lambda e: -e.self_device_time_total)[:25]
    print("=== device Top25 ===")
    for e in rows:
        print(f"  {e.self_device_time_total/1e3:9.2f} ms x{e.count:5d}  {e.key}")
except Exception as ex:
    print("NPU profile failed:", type(ex).__name__, ex)
    # fallback: 纯 CPU activity
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as p:
        sim.step(acts)
        torch.npu.synchronize()
    kavg = p.key_averages()
    tot_cpu = sum(e.self_cpu_time_total for e in kavg)
    rows = sorted(kavg, key=lambda e: -e.self_cpu_time_total)[:25]
    print(f"CPU self 合计: {tot_cpu/1e3:.1f} ms")
    for e in rows:
        print(f"  {e.self_cpu_time_total/1e3:9.2f} ms x{e.count:5d}  {e.key}")
