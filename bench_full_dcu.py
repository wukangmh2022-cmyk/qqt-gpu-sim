"""DCU 端到端训练 SPS：eager danger vs inductor 级联。

结构与 Blackwell 日志同口径：SelfPlayRunner.collect() + ppo_update()，
报告 collect/update/sps，并单独测 sim.step 均耗时给出分段占比。
"""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.model import ActorCritic
from train.ppo import SelfPlayRunner, PPOConfig, ppo_update
from sim.config import SimConfig
from sim.torch_sim import BatchedSim

p = argparse.ArgumentParser()
p.add_argument('--num-envs', type=int, default=2048)
p.add_argument('--arch', default='cnn')
p.add_argument('--rollout', type=int, default=128)
p.add_argument('--iters', type=int, default=3)
p.add_argument('--mode', default='cascade', choices=['cascade', 'eager'])
p.add_argument('--map-mode', default='corridor')
p.add_argument('--open-fraction', type=float, default=1.0)
args = p.parse_args()

torch.manual_seed(20260816)
import sim.torch_sim as ts
if args.mode == 'eager':
    ts.BatchedSim._triton_ok = staticmethod(lambda: False)

device = 'cuda'
cfg = SimConfig(map_mode=args.map_mode, speed=3.0, max_steps=1800,
                open_fraction=args.open_fraction, ring_fraction=0.0,
                hazard_fraction=0.0, crate_speed_only=False, timeout_draw=False,
                combo_reward=0.10, combo_gap_factor=0.9)
N = args.num_envs
sim = BatchedSim(cfg, N, device)
sim.reset_all()
net = ActorCritic(sim.cfg.obs_shape, arch=args.arch, n_players=2).to(device)
print(f'[{args.mode}] N={N} arch={args.arch} rollout={args.rollout} '
      f'map={args.map_mode} open_frac={args.open_fraction} '
      f'params={net.n_params():,}', flush=True)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
pcfg = PPOConfig()
pcfg.rollout_steps = args.rollout
runner = SelfPlayRunner(sim, net, [net], pcfg)

# 预热（编译档位 + TBE/算子缓存）
print(f'[{args.mode}] warmup', end='', flush=True)
for _ in range(3):
    buf, last_val = runner.collect()
    print('.', end='', flush=True)
print(' done')

# 分段：sim.step 单独计时（同口径）
g = torch.Generator(device=device).manual_seed(99)
def sim_acts():
    mmask, bmask = sim.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)
for _ in range(10):
    sim.step(sim_acts())
torch.cuda.synchronize()
t0 = time.time(); T = 60
for _ in range(T):
    sim.step(sim_acts())
torch.cuda.synchronize()
dt = time.time() - t0
print(f'[{args.mode}] sim.step avg={dt/T*1000:.2f}ms tiers={len(sim._dng_tier)}', flush=True)

t0 = time.time()
for it in range(args.iters):
    t1 = time.time()
    buf, last_val = runner.collect()
    t_collect = time.time() - t1
    t2 = time.time()
    stats = ppo_update(net, opt, buf, last_val, pcfg, 0.1)
    t_update = time.time() - t2
    steps = N * args.rollout
    sps = steps / (time.time() - t1)
    lt = runner.last_timing
    print(f'[{args.mode} iter{it}] collect={t_collect:.1f}s '
          f'update={t_update:.1f}s sps={sps/1e3:.1f}k | '
          f'sim={lt["sim_ms"]:.0f}ms transfer={lt["transfer_ms"]:.0f}ms '
          f'policy={lt["policy_ms"]:.0f}ms of total={lt["total_ms"]:.0f}ms',
          flush=True)
tot = N * args.rollout * args.iters / (time.time() - t0)
print(f'[{args.mode}] final sps = {tot/1e3:.1f}k env-steps/s', flush=True)
