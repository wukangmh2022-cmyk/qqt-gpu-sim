"""Short benchmark for CPU Simulator + accelerator learner boundaries.

Run inside the target Slurm environment, for example:
  python bench_hybrid.py --train-device cuda --sim-device cpu \
      --num-envs 256 --rollout 16 --iters 2
"""

from __future__ import annotations

import argparse
import time

import torch
# DCU 上无 backend='npu' 编译后端；benchmark 关心吞吐，不关心编译加速。
torch.compile = lambda fn, **kw: fn

import sim.torch_sim as _ts
_ts._HAS_TRITON = False
_ts._move_triton = None

from sim.dev import resolve_device
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic
from train.ppo import PPOConfig, SelfPlayRunner, ppo_update
from train.model_pool import clone_frozen


p = argparse.ArgumentParser()
p.add_argument("--train-device", default=None)
p.add_argument("--sim-device", default="cpu")
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--rollout", type=int, default=16)
p.add_argument("--iters", type=int, default=2)
p.add_argument("--map-mode", default="open", choices=["open", "corridor"],
               help="corridor = 与 train.py 一致的走廊混合地图（更重的 sim）")
args = p.parse_args()

train_device = resolve_device(args.train_device)
sim_device = resolve_device(args.sim_device)
if train_device == sim_device:
    print("warning: train and sim devices are identical; this is the baseline path")

if args.map_mode == "corridor":
    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0,
                    crate_speed_only=False, timeout_draw=False,
                    combo_reward=0.10, combo_gap_factor=0.9)
else:
    cfg = SimConfig()
sim = BatchedSim(cfg, args.num_envs, device=sim_device, seed=0)
learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(train_device)
opp = clone_frozen(learner)
pcfg = PPOConfig(rollout_steps=args.rollout, epochs=1, minibatches=2)
runner = SelfPlayRunner(sim, learner, [opp], pcfg, measure_timing=True)
opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

for _ in range(1):
    runner.collect()

start = time.perf_counter()
for i in range(args.iters):
    tick = time.perf_counter()
    buf, last_val = runner.collect()
    collect_s = time.perf_counter() - tick
    update_s = time.perf_counter()
    ppo_update(learner, opt, buf, last_val, pcfg, 0.1)
    update_s = time.perf_counter() - update_s
    steps = args.num_envs * args.rollout
    print(f"iter={i} collect={collect_s:.3f}s update={update_s:.3f}s "
          f"sps={steps/(collect_s+update_s):.1f} "
          f"timing={runner.last_timing}", flush=True)
print(f"total_sps={args.num_envs*args.rollout*args.iters/(time.perf_counter()-start):.1f}")
