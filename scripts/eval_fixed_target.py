"""固定靶验证：P0 完全不动（每 tick IDLE + 不放泡），各对手打它，看胜率。

目的：检验 teacher/规则 bot/陪练在本地评估环境（distill 地图）是否"崩坏"。
若连固定靶都打不赢（胜率低），说明该对手在评估环境下策略失效；
若对固定靶胜率高，说明对手正常，student 输赢是策略层面的真实差距。

用法：
    python scripts/eval_fixed_target.py --device mps --episodes 48
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from deploy.collect_distill import make_cfg, load_net, _swap_player_channels
from sim.bots import make_bot
from sim.factory import make_sim


def duel(sim, pol0, pol1, episodes: int) -> tuple[float, float, float]:
    """pol0/1 = callable(obs, mm_p, bm_p) -> (N,2)。返回 (win0, draw, win1)。"""
    n = sim.num_envs
    dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    rounds = 0
    t0 = time.time()
    while rounds < episodes:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = pol0(obs, mm[:, 0], bm[:, 0])
        a1 = pol1(obs, mm[:, 1], bm[:, 1])
        _, d, info = sim.step(torch.stack([a0, a1], dim=1))
        just = d & ~done
        w0 += int((just & info["winner"][:, 0]).sum())
        w1 += int((just & info["winner"][:, 1]).sum())
        dr += int((just & ~info["winner"][:, 0] & ~info["winner"][:, 1]).sum())
        done |= d
        rounds += int(just.sum())
        if bool(done.all()):
            sim.reset_all()
            done.zero_()
    tot = max(w0 + w1 + dr, 1)
    return w0 / tot, dr / tot, w1 / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--episodes", type=int, default=48)
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")
    torch.manual_seed(0)

    sim = make_sim(make_cfg(), 256, backend="torch", device=device, seed=0)

    @torch.no_grad()
    def fixed(o, m, b):                      # 固定靶：IDLE + 不放泡
        return torch.zeros((o.shape[0], 2), dtype=torch.long, device=o.device)

    opps = {}
    for nm in ("duel_nobc_11b_live", "duel_nobc_8b_live", "duel_cnn", "duel_5x3"):
        fpath = f"ckpt/{nm}.pt"
        if os.path.exists(fpath):
            net = load_net(fpath, device)
            opps[nm] = (lambda t=net: lambda o, m, b: t.act(
                _swap_player_channels(o), m, b, 1)[0])()
    for kind in ("astar", "greedy", "random"):
        opps[kind] = (lambda k=kind: lambda o, m, b: make_bot(sim, k).act(o, m, b, 1))()

    print("=== 固定靶（P0 不动）vs 对手（P1）===")
    for i, (name, pol) in enumerate(opps.items()):
        sim.gen.manual_seed(1000 + i)      # 各对手独立随机序列，防互相污染
        sim.reset_all()
        w0, dr, w1 = duel(sim, fixed, pol, args.episodes)
        print(f"固定靶 vs {name:<20}: 靶胜 {w0:.1%} / 平 {dr:.1%} / 对手胜 {w1:.1%}  "
              f"({args.episodes}局)")


if __name__ == "__main__":
    main()
