#!/usr/bin/env python3
"""独立 env 性能验证（本地 MPS/CPU，不碰服务器训练）。

复现 DCU 真实训练负载的**结构**（corridor 满成长 + 10 泡 + combo + astar 对手）：
  observe+legal_mask / learner.act / bot.act / sim.step 分段计时 + torch.profiler
  kernel 级分布 —— 找 A（网络前向 38%）/ B（模拟器 44%）里的热点。

用法：
  python -m scripts.prof_env --n 1024 --ticks 30 --profile  # 默认走 mps/cpu
"""
import argparse
import time

import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic
from sim.bots import make_bot


def build_env(n: int, device: str, opp: str = "astar"):
    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.5, ring_fraction=0, hazard_fraction=0,
                    timeout_draw=True, combo_reward=0.10, combo_gap_factor=0.9)
    sim = BatchedSim(cfg, n, device=device)
    learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(device).eval()
    for p in learner.parameters():
        p.requires_grad_(False)
    bot = make_bot(sim, opp) if opp != "none" else None
    return cfg, sim, learner, bot


def tick(sim, learner, bot, actions):
    obs = sim.observe()
    mmask, bmask = sim.legal_mask()
    with torch.no_grad():
        a0, _, _ = learner.act(obs, mmask[:, 0], bmask[:, 0], 0)
    actions[:, 0] = a0
    if bot is not None:
        actions[:, 1] = bot.act(obs, mmask[:, 1], bmask[:, 1], 1)
    sim.step(actions)


def sync(device):
    dev = torch.device(device)
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--ticks", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--opp", default="astar",
                    help="astar/greedy/random/none（none=无 bot，纯网络对手）")
    args = ap.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[prof_env] device={device} n={args.n} ticks={args.ticks} opp={args.opp}")
    cfg, sim, learner, bot = build_env(args.n, device, args.opp)
    actions = torch.zeros((args.n, 2, 2), dtype=torch.long, device=device)

    # warmup（建图/编译/kernel 预热）
    for _ in range(5):
        tick(sim, learner, bot, actions)
    sync(device)

    # ---- 分段计时 ----
    t_obs = t_act = t_bot = t_step = 0.0
    for _ in range(args.ticks):
        t0 = time.perf_counter()
        obs = sim.observe()
        mmask, bmask = sim.legal_mask()
        t1 = time.perf_counter()
        with torch.no_grad():
            a0, _, _ = learner.act(obs, mmask[:, 0], bmask[:, 0], 0)
        t2 = time.perf_counter()
        actions[:, 0] = a0
        if bot is not None:
            actions[:, 1] = bot.act(obs, mmask[:, 1], bmask[:, 1], 1)
        t3 = time.perf_counter()
        sim.step(actions)
        t4 = time.perf_counter()
        t_obs += t1 - t0
        t_act += t2 - t1
        t_bot += t3 - t2
        t_step += t4 - t3
    sync(device)
    tot = t_obs + t_act + t_bot + t_step
    print(f"\n=== 分段（{args.ticks} ticks × {args.n} env）===")
    for name, t in (("observe+legal_mask", t_obs), ("learner.act", t_act),
                    ("bot.act (astar)", t_bot), ("sim.step", t_step)):
        print(f"  {name:<18}: {t/args.ticks*1000:7.2f} ms/tick ({t/tot*100:5.1f}%)")
    print(f"  总计: {tot/args.ticks*1000:.2f} ms/tick | "
          f"{args.n*args.ticks/tot/1e3:.1f}k env-step/s")

    if not args.profile:
        return

    # ---- kernel 级分布（20 tick 采样）----
    import torch.profiler as prof
    act_cuda = prof.ProfilerActivity.CUDA if torch.cuda.is_available() else None
    activities = [prof.ProfilerActivity.CPU]
    if act_cuda is not None:
        activities.append(act_cuda)
    with prof.profile(activities=activities, record_shapes=True,
                      with_stack=True) as p:
        for _ in range(20):
            tick(sim, learner, bot, actions)
        sync(device)
    print("\n=== kernel 级 top-45（self CPU）===")
    print(p.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=45))
    if act_cuda is not None:
        print("\n=== kernel 级 top-25（CUDA）===")
        print(p.key_averages().table(sort_by="self_device_time_total",
                                     row_limit=25))
    # 同步点（_local_scalar_dense）的调用栈分布 —— 定位 137 个/tick 的同步来源
    print("\n=== 同步点（_local_scalar_dense）top 调用栈 ===")
    syncs = p.key_averages(group_by_input_shape=False)
    for ev in p.events():
        if ev.key.startswith("aten::_local_scalar_dense"):
            for st in ev.stack:
                print(f"  {st}")
            print("  ----")
            break  # 只显示第一个同步点的栈（同类）
    # 统计总调用数
    total_scalar = sum(1 for ev in p.events()
                       if ev.key.startswith("aten::_local_scalar_dense"))
    print(f"_local_scalar_dense 总调用: {total_scalar}（/20 tick = {total_scalar/20:.0f}/tick）")


if __name__ == "__main__":
    main()
