"""jax student（蒸馏产物）对战胜率评估：torch 复现 jax 2层768 MLP，本地直接跑。

student.pt = jax params pickle（纯 numpy dict）：w1(1183,768)/b1/w2/b2 +
wm/bm(5) wb/bb(2) wv/bv(1)。推理用 torch fp32 复现（权重原样拷贝，
bf16 训练精度差异对胜率评估无影响）。输入 = collect_distill.obs7_batch
（7 通道双视角，与蒸馏/PPO 数据同源）。

用法：
    python scripts/eval_student.py --student /tmp/student.pt --device cpu
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from deploy.collect_distill import make_cfg, load_net, obs7_batch, _swap_player_channels
from sim.bots import make_bot
from sim.factory import make_sim


def load_student(path: str, device) -> dict:
    """jax params pickle → torch tensors（dict，同 jax_net.init_mlp 布局）。"""
    with open(path, "rb") as f:
        p = pickle.load(f)
    return {k: torch.from_numpy(np.asarray(v)).to(device) for k, v in p.items()}


@torch.no_grad()
def student_forward(p: dict, o7: torch.Tensor, pid: int):
    """o7 (N,2,7,13,13) → 取 pid 视角 → 2层768 MLP → (mv,bv)。与 jax mlp_forward 同构。"""
    x = o7[:, pid].reshape(o7.shape[0], -1).float()  # sim obs 可能半精度，权重 fp32
    x = torch.relu(x @ p["w1"] + p["b1"])
    x = torch.relu(x @ p["w2"] + p["b2"])
    return x @ p["wm"] + p["bm"], x @ p["wb"] + p["bb"]


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
    ap.add_argument("--student", default="/tmp/student.pt")
    ap.add_argument("--teacher", default="ckpt/duel_nobc_11b_live.pt")
    ap.add_argument("--device", default=None,
                    help="mps/cpu；不传自动（MPS 可用则 MPS，否则 CPU）")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--map-mode", default="distill",
                    choices=["distill", "open", "corridor"],
                    help="distill=收集同款（纯50%+变换50%）；open 纯空场；corridor 70% 混合")
    ap.add_argument("--swap-sides", action="store_true",
                    help="student 当 player1、对手当 player0：双方都用自己最熟的 "
                         "P0 视角姿势（对手 pid=0 官方姿势，student 用 obs7 的 P1 视角）")
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")
    torch.manual_seed(0)

    if args.map_mode == "distill":
        cfg = make_cfg()
    elif args.map_mode == "open":
        cfg = make_cfg(); cfg.open_fraction = 1.0; cfg.pure_open_fraction = 0.0
    else:
        cfg = make_cfg(); cfg.open_fraction = 0.3; cfg.ring_fraction = 0.0
        cfg.pure_open_fraction = 0.0

    n = 256
    sim = make_sim(cfg, n, backend="torch", device=device, seed=0)
    p = load_student(args.student, device)
    print(f"student {args.student} params={sum(v.numel() for v in p.values()):,} "
          f"device={device} 地图={args.map_mode}")

    def make_student(pid: int):
        @torch.no_grad()
        def pol(obs, m, b):
            o7 = obs7_batch(sim, obs)                 # (N,2,7,13,13)
            mv, bv = student_forward(p, o7, pid)
            am = torch.where(m, mv, torch.full_like(mv, float("-inf"))).argmax(-1)
            ab = torch.where(b, bv, torch.full_like(bv, float("-inf"))).argmax(-1)
            return torch.stack([am, ab], dim=-1)
        return pol

    student = make_student(1 if args.swap_sides else 0)

    # 对手（默认 player1）。torch 网络当 P1 时用 collect 同款姿势：通道互换 + pid=1
    # （实测 4 种姿势：swap+pid1=95.8% 正确，裸 pid1=43.8% 错位 —— 2026-08-17）。
    # --swap-sides 时对手当 player0：用官方 pid=0 姿势（不 swap）。
    opps = {}
    if args.teacher and os.path.exists(args.teacher):
        teacher = load_net(args.teacher, device)
        tname = os.path.splitext(os.path.basename(args.teacher))[0]
        if args.swap_sides:
            opps[f"teacher({tname})"] = (
                lambda t=teacher: lambda o, m, b: t.act(o, m, b, 0)[0])()
        else:
            opps[f"teacher({tname})"] = (
                lambda t=teacher: lambda o, m, b: t.act(
                    _swap_player_channels(o), m, b, 1)[0])()
    for kind in ("astar", "greedy", "random"):
        if args.swap_sides:
            opps[kind] = (lambda k=kind: lambda o, m, b: make_bot(sim, k).act(o, m, b, 0))()
        else:
            opps[kind] = (lambda k=kind: lambda o, m, b: make_bot(sim, k).act(o, m, b, 1))()
    for nm in ("duel_cnn", "duel_5x3"):
        fpath = f"ckpt/{nm}.pt"
        if os.path.exists(fpath):
            net = load_net(fpath, device)
            if args.swap_sides:
                opps[nm] = (lambda t=net: lambda o, m, b: t.act(o, m, b, 0)[0])()
            else:
                opps[nm] = (
                    lambda t=net: lambda o, m, b: t.act(
                        _swap_player_channels(o), m, b, 1)[0])()

    print("=== student(蒸馏) vs 对手 "
          + ("[swap-sides: student=P1, 对手=P0 官方姿势]" if args.swap_sides
             else "[student=P0, 对手=P1]")
          + " ===")
    for i, (name, pol) in enumerate(opps.items()):
        sim.gen.manual_seed(1000 + i)      # 各对手独立随机序列，防互相污染
        sim.reset_all()
        t0 = time.time()
        if args.swap_sides:
            w, d, l = duel(sim, pol, student, args.episodes)   # pol0=对手, pol1=student
        else:
            w, d, l = duel(sim, student, pol, args.episodes)   # pol0=student, pol1=对手
        print(f"student vs {name:<14}: s_win {w:.1%} / draw {d:.1%} / opp_win {l:.1%}  "
              f"({args.episodes}局, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
