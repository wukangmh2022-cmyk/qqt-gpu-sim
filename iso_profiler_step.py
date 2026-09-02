"""910B 用 torch.profiler 剖析整步，列出最耗时的前 20 个算子。"""
import sys, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
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

# 预热（编译/缓存）
for _ in range(3):
    sim.step(acts)
sync()

try:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        for _ in range(5):
            sim.step(acts)
        sync()
    print("=== 最耗时算子 Top 20（CPU dispatch 时间）===")
    evs = prof.key_averages()
    rows = [(e.key, e.self_cpu_time_total / 1000, e.count)
            for e in evs if e.self_cpu_time_total > 0]
    rows.sort(key=lambda r: -r[1])
    tot = sum(r[1] for r in rows)
    for k, t, c in rows[:20]:
        print(f"  {t:8.1f} ms ({t/tot*100:5.1f}%) x{c:<4} {k[:60]}")
    print(f"  合计 {tot:.0f} ms")
    # 导出 stack 并按函数名分组统计 _local_scalar_dense 的调用来源
    try:
        prof.export_stacks("/tmp/stacks.txt", metric="self_cpu_time_total")
        print("stacks 已导出 /tmp/stacks.txt")
    except Exception as e:
        print("export_stacks 失败:", str(e)[:120])
except Exception as e:
    print("profiler 不可用:", str(e)[:200])
