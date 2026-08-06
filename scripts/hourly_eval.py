"""每小时评估：最新 ckpt 对历史对手 + 寻路 bot 的胜率趋势。

验证"逐步有效提升"：每次训练快照，对**固定历史对手**（500M / 600M / cnn /
rw8 / astar / hunter）打一批局，胜率逐次记录 → eval_trend.csv 时间序列。
胜率单调爬升 = 训练在进步（对手固定，无 self-play 水涨船高干扰）。

用法：
    python scripts/hourly_eval.py --ckpt ckpt/course_xxx.pt \
        --trend eval_trend.csv --episodes 96
对手默认：500m=ckpt/course_501m.pt 600m=ckpt/course_597m.pt
          cnn=ckpt/duel_cnn.pt rw8=ckpt/duel_rw8.pt astar(规则) hunter(规则)
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.train import load_fixed_checkpoint

DEV = "cpu"
CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0,
                open_crate_cross=True, hit_attr_penalty=2)


def swap_channels(obs: torch.Tensor, p: int = 2) -> torch.Tensor:
    """P1 侧网络模型视角：per-player 通道 0↔1 互换 + pid=0（与 duel/launcher
    真实对打一致）。训练模型只优化了 pid=0 视角，P1 位直接看原始 obs 会把
    对手当自己（per-player 通道错位）→ 疯狂自爆 → 胜率虚高/虚低（评测 bug）。"""
    base = 2 * p + 3
    idx = list(range(obs.shape[1]))
    for seg in (range(0, p), range(p, 2 * p),
                range(base + 1, base + 1 + p),
                range(base + 1 + p, base + 1 + 2 * p),
                range(base + 1 + 2 * p, base + 1 + 3 * p)):
        seg = list(seg)
        idx[seg[0]], idx[seg[1]] = idx[seg[1]], idx[seg[0]]
    return obs[:, idx]

DEFAULT_OPPS = "500m=ckpt/course_501m.pt,600m=ckpt/course_597m.pt," \
               "cnn=ckpt/duel_cnn.pt,rw8=ckpt/duel_rw8.pt,astar=bot,hunter=bot"


def duel(sim, pol0, pol1, episodes: int, p1_is_net: bool = False) -> dict:
    """pol0 = 被测模型（pid=0 视角）；pol1 = 对手。p1_is_net=True 时 pol1 内部
    需已做 swap（网络对手 P1 视角）；规则 bot 直接 act。返回 (win0, draw, win1,
    suicide, killed) —— suicide = 被测模型死亡时自己名下有在场泡（自爆）。"""
    n = sim.num_envs
    dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = rounds = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    suicide = torch.zeros(n, dtype=torch.long, device=dev)
    killed = torch.zeros(n, dtype=torch.long, device=dev)
    while rounds < episodes:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = pol0(obs, mm[:, 0], bm[:, 0])
        a1 = pol1(obs, mm[:, 1], bm[:, 1])
        owner_snap = sim.owner.clone()
        fuse_snap = sim.fuse.clone()
        _, d, info = sim.step(torch.stack([a0, a1], dim=1))
        died0 = info["died"][:, 0]
        own_cnt = ((owner_snap == 0) & (fuse_snap > 0)).flatten(1).sum(dim=1)
        suicide += (died0 & (own_cnt > 0)).long()
        killed += (died0 & (own_cnt == 0)).long()
        just = d & ~done
        win0 = just & info["winner"][:, 0]
        win1 = just & info["winner"][:, 1]
        w0 += int(win0.sum())
        w1 += int(win1.sum())
        dr += int((just & ~win0 & ~win1).sum())
        done |= d
        rounds += int(just.sum())
        if bool(done.all()):
            sim.reset_all()
            done.zero_()
    tot = max(1, rounds)
    return {"win": w0 / tot, "draw": dr / tot, "loss": w1 / tot,
            "suicide": int(suicide.sum()), "killed": int(killed.sum()),
            "rounds": rounds}


def parse_opps(spec: str) -> list[tuple[str, object]]:
    out = []
    for item in spec.split(","):
        name, _, path = item.strip().partition("=")
        if path == "bot":
            out.append((name, "bot:" + name))   # 标记规则 bot（按名建）
        else:
            out.append((name, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trend", default="eval_trend.csv")
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--opponents", default=DEFAULT_OPPS)
    ap.add_argument("--sim-envs", type=int, default=64)
    args = ap.parse_args()

    sim = BatchedSim(CFG, args.sim_envs, device=DEV, seed=0)
    net = load_fixed_checkpoint(args.ckpt, CFG.obs_shape, DEV)
    net.eval()
    try:
        step = int(torch.load(args.ckpt, map_location="cpu",
                              weights_only=False)["global_step"])
        elo = float(torch.load(args.ckpt, map_location="cpu",
                               weights_only=False)["elo"])
    except Exception:
        step, elo = 0, 0.0

    @torch.no_grad()
    def pol0(o, m, b):
        return net.act(o, m, b, 0)[0]

    opps = parse_opps(args.opponents)
    row = {"date": time.strftime("%m-%d %H:%M"), "step": step, "elo": round(elo, 1)}
    print(f"[eval] {args.ckpt} step={step}M elo={elo:.0f} — {len(opps)} 对手\n")
    for name, src in opps:
        if isinstance(src, str) and src.startswith("bot:"):
            kind = src.split(":", 1)[1]
            b = make_bot(sim, kind)
            pol1 = (lambda f=b: lambda o, m, bm: f.act(o, m, bm, 1))()
            is_net = False
        else:
            f = load_fixed_checkpoint(src, CFG.obs_shape, DEV)
            f.eval()
            # 网络对手 P1 位：swap 视角 + pid=0（否则 per-player 通道错位 → 自爆）
            pol1 = (lambda n=f: lambda o, m, b: n.act(swap_channels(o), m, b, 0)[0])()
            is_net = True
        t0 = time.time()
        r = duel(sim, pol0, pol1, args.episodes)
        row[name] = round(r["win"], 3)
        row[name + "_su"] = round(r["suicide"] / max(1, r["rounds"]), 3)
        print(f"  vs {name:<8}: win {r['win']:.1%} / draw {r['draw']:.1%} "
              f"/ loss {r['loss']:.1%}   自爆 {r['suicide']}/{r['killed']} "
              f"({time.time()-t0:.0f}s)")

    # 追加趋势 csv
    new = not os.path.exists(args.trend)
    with open(args.trend, "a", newline="") as f:
        wtr = csv.writer(f)
        if new:
            wtr.writerow(list(row.keys()))
        wtr.writerow(list(row.values()))
    print(f"\n[tren] {args.trend}")


if __name__ == "__main__":
    main()
