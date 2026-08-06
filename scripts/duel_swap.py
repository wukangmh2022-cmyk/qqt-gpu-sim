"""同模型 P0/P1 位置对称性验证 + 本地实力测评。

- 位置对称：net vs net_swap（同一个 net，P1 侧用 _swap_player_channels 同款
  通道重排），A/B 两侧交换各打一批局 —— 若视角修复生效，两侧胜率应接近
  （随机差异 ~±5%）。
- 对手锚点：astar / greedy / random（规则）+ 5x2 / 5x3 / rw8 / cnn（固定陪练）。

用法：
    python scripts/duel_swap.py ckpt/course_501m.pt
"""

from __future__ import annotations

import sys
import time

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.factory import make_sim
from sim.torch_sim import BatchedSim
from train.train import load_fixed_checkpoint

DEV = "cpu"
CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0,
                open_crate_cross=True, hit_attr_penalty=2)


def swap_channels(obs: torch.Tensor, p: int = 2) -> torch.Tensor:
    """与 play/duel.py 的 _swap_player_channels 同款：物理 P1 侧模型的
    per-player 通道 0↔1 互换，让它"自己 = 通道 0、pid=0 视角"。"""
    base = 2 * p + 3
    idx = list(range(obs.shape[1]))
    for seg in (range(0, p), range(p, 2 * p),
                range(base + 1, base + 1 + p),
                range(base + 1 + p, base + 1 + 2 * p),
                range(base + 1 + 2 * p, base + 1 + 3 * p)):
        seg = list(seg)
        idx[seg[0]], idx[seg[1]] = idx[seg[1]], idx[seg[0]]
    return obs[:, idx]


def duel_policies(sim, pol0, pol1, episodes: int) -> tuple[float, float, float]:
    """polX(obs, mm_p, bm_p) -> (N,2)。终局计数与 sim 的 winner/loser 一致：
    n_alive==1 活着胜/死者输、双亡=平、超时血多胜。返回 (win0, draw, win1)。"""
    n = sim.num_envs
    dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = rounds = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    while rounds < episodes:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = pol0(obs, mm[:, 0], bm[:, 0])
        a1 = pol1(obs, mm[:, 1], bm[:, 1])
        _, d, info = sim.step(torch.stack([a0, a1], dim=1))
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
    return w0 / max(1, rounds), dr / max(1, rounds), w1 / max(1, rounds)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/course_501m.pt"
    episodes = 256
    sim = BatchedSim(CFG, 128, device=DEV, seed=0)
    net = load_fixed_checkpoint(path, CFG.obs_shape, DEV)
    net.eval()
    print(f"[{path}] obs={net.obs_shape} arch={net.arch} 地图=corridor50% "
          f"hit_attr_penalty={CFG.hit_attr_penalty} episodes={episodes}")

    @torch.no_grad()
    def pol0(o, m, b):
        return net.act(o, m, b, 0)[0]

    @torch.no_grad()
    def pol1_swap(o, m, b):
        return net.act(swap_channels(o), m, b, 0)[0]

    print("\n=== 位置对称性：同模型，A/B 两侧互换 ===")
    w, d, l = duel_policies(sim, pol0, pol1_swap, episodes)
    print(f"P0(直看) vs P1(swap) : win {w:.1%} / draw {d:.1%} / loss {l:.1%}")
    w, d, l = duel_policies(sim, pol1_swap, pol0, episodes)
    print(f"P1(swap) vs P0(直看) : win {w:.1%} / draw {d:.1%} / loss {l:.1%}")

    bots = {}
    for kind in ("astar", "greedy", "random"):
        b = make_bot(sim, kind)
        bots[kind] = (lambda f=b: lambda o, m, bm: f.act(o, m, bm, 1))()
    fixed = {}
    for nm in ("5x2", "5x3", "rw8", "cnn"):
        f = load_fixed_checkpoint(f"ckpt/duel_{nm}.pt", CFG.obs_shape, DEV)
        f.eval()
        fixed[nm] = (lambda n=f: lambda o, m, b: n.act(o, m, b, 1)[0])()

    print("\n=== course vs 规则 bot ===")
    for name, pol in bots.items():
        t0 = time.time()
        w, d, l = duel_policies(sim, pol0, pol, episodes)
        print(f"vs {name:<8}: win {w:.1%} / draw {d:.1%} / loss {l:.1%}  "
              f"({time.time()-t0:.0f}s)")
    print("=== course vs 固定陪练 ===")
    for name, pol in fixed.items():
        t0 = time.time()
        w, d, l = duel_policies(sim, pol0, pol, episodes)
        print(f"vs {name:<8}: win {w:.1%} / draw {d:.1%} / loss {l:.1%}  "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
