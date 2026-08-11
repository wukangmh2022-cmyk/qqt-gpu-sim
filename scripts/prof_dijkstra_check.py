#!/usr/bin/env python3
"""dijkstra CHECK_EVERY 在 DCU 的最优值扫描。

收敛判定（bool(torch.equal)）是 GPU→CPU 同步；DCU 上 sync 贵，但更粗的
CHECK_EVERY 让"提前停"延迟多跑空转 kernel。扫 4/8/16/32/169 找最优。
用法：python -m scripts.prof_dijkstra_check --n 2048 --ticks 15
"""
import argparse
import time

import torch

import sim.bots as B
from sim.config import SimConfig
from sim.torch_sim import BatchedSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=15)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prof_dijkstra] device={dev} n={args.n}")

    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.5, timeout_draw=True, combo_reward=0.10)
    sim = BatchedSim(cfg, args.n, device=dev, seed=0)
    bot = B.make_bot(sim, "astar")
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    actions = torch.zeros(args.n, 2, 2, dtype=torch.long, device=dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    orig_check = B.CHECK_EVERY
    results = []
    for ce in (4, 8, 16, 32, 169):
        B.CHECK_EVERY = ce
        for _ in range(3):
            bot.act(obs, mm[:, 1], bm[:, 1], 1)
            sim.step(actions, auto_reset=False)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.ticks):
            bot.act(obs, mm[:, 1], bm[:, 1], 1)
            sim.step(actions, auto_reset=False)
        sync()
        t1 = time.perf_counter()
        dt = (t1 - t0) / args.ticks
        results.append((ce, dt))
        print(f"CHECK_EVERY={ce:4d}: {dt*1000:7.2f} ms/tick")
    B.CHECK_EVERY = orig_check
    best = min(results, key=lambda r: r[1])
    print(f"\n最优: CHECK_EVERY={best[0]} ({best[1]*1000:.2f} ms/tick)")


if __name__ == "__main__":
    main()
