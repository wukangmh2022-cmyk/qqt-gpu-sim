# 与 DCU 同口径的完整训练 SPS 测量（MLP 345k + n=20000 + collect+ppo）
# DCU 参考: HANDOFF L208 'sps 35k @n=20000' / L269 '38k'
import os, sys, time, argparse
import torch
import torch_npu  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.model import ActorCritic
from train.ppo import SelfPlayRunner, PPOConfig, ppo_update
from sim.config import SimConfig
from sim.torch_sim import BatchedSim

p = argparse.ArgumentParser()
p.add_argument('--num-envs', type=int, default=20000)
p.add_argument('--arch', default='mlp')
p.add_argument('--rollout', type=int, default=128)
p.add_argument('--iters', type=int, default=2)
args = p.parse_args()

torch.npu.set_device(0)
device = 'npu:0'
print(f'[parity] N={args.num_envs} arch={args.arch} rollout={args.rollout} iters={args.iters}')

cfg = SimConfig()
sim = BatchedSim(cfg, args.num_envs, device)
sim.reset_all()
net = ActorCritic(sim.cfg.obs_shape, arch=args.arch, n_players=2).to(device)
print(f'[model] params={net.n_params():,} obs_shape={sim.cfg.obs_shape}')
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
pcfg = PPOConfig()
pcfg.rollout_steps = args.rollout
runner = SelfPlayRunner(sim, net, [net], pcfg, device if False else None) if False else None
# 修正: opponents 是 P-1 个, n_players=2 -> 1 个对手
runner = SelfPlayRunner(sim, net, [net], pcfg)

# 预热一 tick（TBE 编译缓存）
print('[warmup]', end='', flush=True)
for _ in range(3):
    buf, last_val = runner.collect()
    print('.', end='', flush=True)
print(' done')

t0 = time.time()
for it in range(args.iters):
    t1 = time.time()
    buf, last_val = runner.collect()
    t_collect = time.time() - t1
    t2 = time.time()
    stats = ppo_update(net, opt, buf, last_val, pcfg, 0.1)
    t_update = time.time() - t2
    steps = args.num_envs * args.rollout
    sps = steps / (time.time() - t1)
    print(f'[iter {it}] collect={t_collect:.1f}s update={t_update:.1f}s '
          f'total={time.time()-t1:.1f}s sps={sps/1e3:.1f}k', flush=True)
print(f'[parity] final sps = {args.num_envs*args.rollout*args.iters/(time.time()-t0)/1e3:.1f}k')
