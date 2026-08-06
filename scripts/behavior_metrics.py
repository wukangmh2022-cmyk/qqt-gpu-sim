"""行为指标验收（"留炮/节奏"专项）：量化 AI 是不是"一股脑全丢"。

配合放炮奖励调整（chain_blast 自连爆归属修复 + passivity 收紧）验收：
目标 = 放炮间隔分布拉长（连丢占比下降、憋炮占比上升），学"留炮/async 节奏"。

指标（对指定模型 vs 指定对手，corridor 70%）：
- 放炮间隔均值/中位（tick）：30 引信下 <10 说明弹幕化
- 连丢占比（间隔 ≤3 tick）：实测旧版 73% —— 目标是显著下降
- 憋炮占比（间隔 ≥20 tick）：留炮能力，目标是上升
- 在场泡数均值 / 上限：常驻半场 = 全丢，稀疏 = 有留

用法：
    python scripts/behavior_metrics.py --ckpt ckpt/duel_course_*.pt --ticks 800
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.factory import make_sim
from train.train import load_fixed_checkpoint

CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.3, ring_fraction=0.0, hazard_fraction=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--opponent", default="astar", choices=["astar", "greedy"])
    args = ap.parse_args()
    device = args.device
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")

    sim = make_sim(CFG, 256, backend="torch", device=device, seed=0)
    net = load_fixed_checkpoint(args.ckpt, CFG.obs_shape, device)
    bot = make_bot(sim, args.opponent)

    intervals: list[int] = []
    live_hist: list[int] = []
    since = torch.zeros(256, dtype=torch.long, device=device)
    for _ in range(args.ticks):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        with torch.no_grad():
            a0 = net.act(obs, mm[:, 0], bm[:, 0], 0)[0]
        a1 = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        placed = a0[:, 1] == 1
        intervals.extend((since[placed] + 1).cpu().tolist())
        since[placed] = 0
        since[~placed] += 1
        sim.step(torch.stack([a0, a1], dim=1))
        live = ((sim.owner == 0) & (sim.fuse > 0)).flatten(1).sum(dim=1)
        live_hist.extend(live[sim.alive[:, 0]].cpu().tolist())

    iv = np.array(intervals)
    lh = np.array(live_hist)
    cap = int(sim.bombs_cap[0, 0])
    print(f"[{args.ckpt}] vs {args.opponent} | {args.ticks} tick × 256 env")
    print(f"放炮间隔 tick: 均值 {iv.mean():.1f} | 中位 {np.median(iv):.0f}")
    print(f"连丢占比(≤3tick): {(iv <= 3).mean() * 100:.0f}%   "
          f"(旧版实测 73%，目标显著下降)")
    print(f"急放占比(≤5tick): {(iv <= 5).mean() * 100:.0f}%")
    print(f"憋炮占比(≥20tick): {(iv >= 20).mean() * 100:.0f}%  (留炮能力，目标上升)")
    print(f"在场泡数: 均值 {lh.mean():.2f} / 上限 {cap}   "
          f"(≈上限=全丢，稀疏=有留)")


if __name__ == "__main__":
    main()
