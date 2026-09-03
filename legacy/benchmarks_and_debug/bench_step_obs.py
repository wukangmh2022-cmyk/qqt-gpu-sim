"""训练侧整体 SPS：step + observe + legal_mask（PPO 每 tick 的 env 交互部分）。

对比：纯 step vs step+obs+legal_mask。N=16384 corridor。
"""
import sys, time, torch
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

def sync():
    if dev.startswith("npu"):
        torch.npu.synchronize()
    else:
        torch.mps.synchronize()

torch.manual_seed(0)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

def bench(fn, it=8):
    for _ in range(4):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    return (time.perf_counter() - t0) / it * 1000

# 纯 step
t_s = bench(lambda: sim.step(acts))
print(f"纯 step        : {t_s:7.2f} ms/tick  {N/t_s*1e3/1e4:.2f}万 SPS")

# step + obs（_dng_cache 复用）
def step_obs():
    sim.step(acts)
    sim.observe()
t_so = bench(step_obs)
print(f"step+obs       : {t_so:7.2f} ms/tick  {N/t_so*1e3/1e4:.2f}万 SPS  (obs 占比 {(t_so-t_s)/t_so*100:.0f}%)")

# step + obs + legal_mask
def step_obs_mask():
    sim.step(acts)
    sim.observe()
    legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
               sim.brick, sim.bombs_cap)
t_all = bench(step_obs_mask)
print(f"step+obs+mask  : {t_all:7.2f} ms/tick  {N/t_all*1e3/1e4:.2f}万 SPS  (mask 占比 {(t_all-t_so)/t_all*100:.0f}%)")

# obs 单独成本（复用缓存后）
def obs_only():
    sim.observe()
t_o = bench(obs_only)
print(f"obs 单独       : {t_o:7.2f} ms/tick")

# legal_mask 单独
def mask_only():
    legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
               sim.brick, sim.bombs_cap)
t_m = bench(mask_only)
print(f"legal_mask 单独 : {t_m:7.2f} ms/tick")
