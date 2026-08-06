"""课程训练阶段性检验：course ckpt vs 规则 bot + 固定陪练全维度对战。

跑在 corridor 70%（open 0.3 / ring 0 / hazard 0），与课程训练主线一致。
每对 256 局，输出胜/平/负率。

用法：
    python scripts/duel_arena.py --ckpt ckpt/duel_course_208M.pt --device mps
"""

from __future__ import annotations

import argparse
import time

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.factory import make_sim
from train.train import load_fixed_checkpoint

CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.3, ring_fraction=0.0, hazard_fraction=0.0)


def make_cfg(map_mode: str) -> SimConfig:
    """按地图模式给配置（与启动器/训练同款：open 走 open_fraction=1.0，
    ring 走 ring_fraction=1.0，corridor 走 70% 混合）。"""
    if map_mode == "open":
        return SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                         open_fraction=1.0, ring_fraction=0.0, hazard_fraction=0.0)
    if map_mode == "ring":
        return SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                         open_fraction=0.0, ring_fraction=1.0, hazard_fraction=0.0)
    return CFG


def duel(sim, pol0, pol1, episodes: int) -> tuple[float, float, float]:
    """pol0/1 = callable(obs, mm_p, bm_p) -> (N,2)。返回 (win0, draw, win1)。"""
    n = sim.num_envs
    dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    rounds = 0
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
    tot = w0 + w1 + dr
    return w0 / tot, dr / tot, w1 / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/duel_course_208M.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--map-mode", default="corridor",
                    choices=["open", "corridor", "ring"],
                    help="对局地图：open 纯空场 / corridor 70% 混合 / ring 环岛")
    args = ap.parse_args()
    device = args.device
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")
    torch.manual_seed(0)

    cfg = make_cfg(args.map_mode)
    sim = make_sim(cfg, 256, backend="torch", device=device, seed=0)
    net = load_fixed_checkpoint(args.ckpt, cfg.obs_shape, device)

    @torch.no_grad()
    def course(o, m, b):
        return net.act(o, m, b, 0)[0]

    bots = {}
    for kind in ("astar", "greedy", "random"):
        bots[kind] = (lambda k=kind: lambda o, m, b: make_bot(sim, k).act(o, m, b, 1))()
    fixed = {}
    for nm in ("5x2", "5x3", "rw8", "cnn"):
        f = load_fixed_checkpoint(f"ckpt/duel_{nm}.pt", cfg.obs_shape, device)
        fixed[nm] = (lambda n=f: lambda o, m, b: n.act(o, m, b, 1)[0])()

    print(f"[{args.ckpt}] obs={net.obs_shape} arch={net.arch} 地图={args.map_mode}")
    print("=== course vs 规则 bot（课程敌人）===")
    for name, pol in bots.items():
        t0 = time.time()
        w, d, l = duel(sim, course, pol, args.episodes)
        print(f"course vs {name:<8}: win {w:.1%} / draw {d:.1%} / loss {l:.1%}  "
              f"({args.episodes}局, {time.time()-t0:.0f}s)")
    print("=== course vs 固定陪练（绝对锚点）===")
    for name, pol in fixed.items():
        t0 = time.time()
        w, d, l = duel(sim, course, pol, args.episodes)
        print(f"course vs {name:<8}: win {w:.1%} / draw {d:.1%} / loss {l:.1%}  "
              f"({args.episodes}局, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
