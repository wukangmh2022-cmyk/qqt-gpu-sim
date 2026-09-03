"""真实 SelfPlayRunner.collect 计时（open 13x13, N=2048, MLP learner + bot 对手）。

验证真实训练 sps=17k 与 bench collect 5.26万 的 3 倍差距来源。
"""
import sys, time, torch, torch.profiler as prof
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
from sim.torch_sim import BatchedSim
from sim.bots import make_bot
from train.model import ActorCritic
from train.ppo import SelfPlayRunner, PPOConfig

dev = pick_device()
assert dev.startswith("npu")
N = 2048
cfg = SimConfig(map_mode="open", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
learner = ActorCritic(cfg.obs_shape, arch="mlp").to(dev).eval()
opponent = make_bot(sim, "astar", mode=True)
runner = SelfPlayRunner(sim, learner, [opponent], PPOConfig(), handicap=1.0)

def sync():
    torch.npu.synchronize()

# 预热一个 collect
buf, last_val = runner.collect()
sync()
print("collect 预热完成")

# 计时
t0 = time.perf_counter()
buf, last_val = runner.collect()
sync()
t_collect = (time.perf_counter() - t0) * 1000
n_tick = PPOConfig().rollout_steps
per_tick = t_collect / n_tick
print(f"真实 collect 一次: {t_collect:8.1f} ms ({n_tick} tick, 每 tick {per_tick:.2f} ms)")
print(f"真实 collect SPS: {N*n_tick/t_collect*1e3/1e4:.2f}万")

# profile collect 热点（CPU）
with prof.profile(activities=[prof.ProfilerActivity.CPU]) as p:
    runner.collect()
    sync()
kavg = p.key_averages()
tot = sum(e.self_cpu_time_total for e in kavg)
print(f"\ncollect CPU self 合计: {tot/1e3:.0f} ms, 事件数: {sum(e.count for e in kavg)}")
rows = sorted(kavg, key=lambda e: -e.self_cpu_time_total)[:18]
for e in rows:
    print(f"  {e.self_cpu_time_total/1e3:8.1f} ms x{e.count:6d}  {e.key}")
