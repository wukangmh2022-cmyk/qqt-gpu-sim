"""Rollout 分段瓶颈测试：钉死混合模式的四个关键数字。

每个分量单独计时（毫秒/tick），并给出 overlap 后的 SPS 上下界：
  SPS = N / max(T_sim_CPU, T_transfer + T_policy)

用法（在目标 Slurm 环境）：
  python bench_breakdown.py --train-device cuda --sim-device cpu \
      --num-envs 5632 --rollout 32 --iters 2
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

from sim.dev import resolve_device, synchronize
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic
from train.model_pool import clone_frozen


p = argparse.ArgumentParser()
p.add_argument("--train-device", default=None)
p.add_argument("--sim-device", default="cpu")
p.add_argument("--num-envs", type=int, default=5632)
p.add_argument("--rollout", type=int, default=32)
p.add_argument("--iters", type=int, default=2)
args = p.parse_args()

train_device = resolve_device(args.train_device)
sim_device = resolve_device(args.sim_device)
cfg = SimConfig()
N, T = args.num_envs, args.rollout


def bench(name, fn, sync) -> float:
    torch.manual_seed(0)
    for _ in range(1):
        fn()
    if sync:
        synchronize(train_device)
    start = time.perf_counter()
    for _ in range(args.iters):
        fn()
    if sync:
        synchronize(train_device)
    total = time.perf_counter() - start
    per = total / args.iters / T * 1000.0
    print(f"{name:28s} {per:9.2f} ms/tick")
    return per


# 1) CPU Simulator：obs + legal + step（不含网络）
sim = BatchedSim(cfg, N, device=sim_device, seed=0)
sim.reset_all()
acts = torch.zeros((N, 2, 2), dtype=torch.long, device=sim_device)

def sim_only():
    for _ in range(T):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a = torch.randint(0, 5, (N, 2), device=sim_device)
        b = (torch.rand(N, 2, device=sim_device) < 0.3).long()
        acts[:, :, 0] = a
        acts[:, :, 1] = b
        sim.step(acts)
        if _ % 4 == 3:
            sim.flush_recycle()
    sim.flush_recycle()

t_sim = bench("sim_only (CPU/GPU)", sim_only, False)

# 2) H2D 传输：obs fp16 + 掩码 → train device
obs_dev = sim.observe()
mm, bm = sim.legal_mask()
train_obs = torch.empty_like(obs_dev, device=train_device)
train_mm = torch.empty_like(mm, device=train_device)
train_bm = torch.empty_like(bm, device=train_device)

def transfer_only():
    for _ in range(T):
        train_obs.copy_(obs_dev, non_blocking=True)
        train_mm.copy_(mm, non_blocking=True)
        train_bm.copy_(bm, non_blocking=True)
    synchronize(train_device)

t_transfer = bench("transfer obs->train", transfer_only, True)

# 3) DCU policy：5 次 MLP 前向（learner + 4 opp 最坏情况）
learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(train_device)
opps = [clone_frozen(learner) for _ in range(4)]
ob, mm0, bm0 = train_obs, train_mm[:, 0], train_bm[:, 0]

def policy_only():
    with torch.no_grad():
        for _ in range(T):
            learner.act(ob, mm0, bm0, 0)
            for net in opps:
                net.act(ob, mm0, bm0, 1)
    synchronize(train_device)

t_policy = bench("policy fwd x5 (DCU)", policy_only, True)

# 4) DCU ppo_update：单次完整更新
from train.ppo import PPOConfig, SelfPlayRunner, ppo_update

runner = SelfPlayRunner(sim, learner, opps[:1], PPOConfig(rollout_steps=T))
buf, last_val = runner.collect()
opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

def update_only():
    ppo_update(learner, opt, buf, last_val, PPOConfig(rollout_steps=T), 0.1)

t_update = bench("ppo_update (DCU)", update_only, True)

# 5) 汇总
tick_serial = t_sim + t_transfer + t_policy
tick_overlap = max(t_sim, t_transfer + t_policy)
print(f"\nserial   SPS = {N / tick_serial * 1000:,.0f}")
print(f"overlap  SPS = {N / tick_overlap * 1000:,.0f}")
print(f"update/iter = {t_update / 1000 * T:.2f}s (bottleneck only if > tick)")
