"""LSTM 架构（局部 7×7 视野 + 相对坐标 + 全局状态 + LSTM）从零训练速度测试。

与 bench_dcu_parity.py 同口径：完整训练（collect 128 tick + ppo_update BPTT），
SPS = N × rollout / 迭代秒。跑真实模拟器 + 局部特征生成 + LSTM 前向 + BPTT。

用法：
    python3 train_lstm_speed.py --num-envs 2048 --rollout 128 --iters 2
"""
import os, sys, time, argparse
import torch

import torch_npu  # noqa: F401  (910B)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.model import ActorCritic            # noqa: E402
from train.ppo import SelfPlayRunner, PPOConfig, ppo_update  # noqa: E402
from sim.config import SimConfig               # noqa: E402
from sim.torch_sim import BatchedSim           # noqa: E402
from sim.bots import BotWrapper, make_bot      # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.npu.set_device(0)
    device = "npu:0"
    print(f"[lstm] N={args.num_envs} rollout={args.rollout} "
          f"iters={args.iters} epochs={args.epochs} mb={args.minibatches}")

    cfg = SimConfig()
    sim = BatchedSim(cfg, args.num_envs, device)
    sim.reset_all()
    net = ActorCritic(sim.cfg.obs_shape, arch="lstm", n_players=2).to(device)
    print(f"[model] arch=lstm params={net.n_params():,} "
          f"obs_shape={sim.cfg.obs_shape} device={device}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    pcfg = PPOConfig()
    pcfg.rollout_steps = args.rollout
    pcfg.epochs = args.epochs
    pcfg.minibatches = args.minibatches

    # 纯规则 bot 对手（greedy）：无状态、不需要神经网络对手
    bot = make_bot(sim, "greedy")
    runner = SelfPlayRunner(sim, net, [bot], pcfg)

    # 预热（TBE 编译缓存）
    print("[warmup]", end="", flush=True)
    for _ in range(3):
        buf, last_val = runner.collect()
        ppo_update(net, opt, buf, last_val, pcfg, 0.1)
        print(".", end="", flush=True)
    print(" done")

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
        print(f"[iter {it}] collect={t_collect:.1f}s update={t_update:.1f}s "
              f"total={time.time()-t1:.1f}s sps={sps/1e3:.1f}k "
              f"pg={stats['pg']:.3f} ent={stats['ent']:.3f}", flush=True)
    print(f"[lstm] final sps = "
          f"{args.num_envs*args.rollout*args.iters/(time.time()-t0)/1e3:.1f}k")


if __name__ == "__main__":
    main()
