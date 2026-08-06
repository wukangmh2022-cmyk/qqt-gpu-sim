"""课程强度排序 sanity：Greedy > Random，已训模型 > Greedy。

验证课程敌人的强度梯度是对的 —— 训练从"打得过的对手"（random）升级到
"势均力敌/略强的对手"（greedy），再到固定陪练（5x2），每级都得真的
更强，课程才有意义。纯对局计数，跑在 corridor（训练主线地图）上。

用法：
    python scripts/sanity_curriculum.py [--device mps|cpu] [--episodes 200]
"""

from __future__ import annotations

import argparse
import time

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.factory import make_sim
from train.train import load_fixed_checkpoint


def duel(sim, pol0, pol1, episodes: int, max_ticks: int = 1800) -> tuple[int, int, int]:
    """pol0/1 = callable(obs, mm_p, bm_p) -> (N,2)。返回 (win0, win1, draw)。"""
    n = sim.num_envs
    w0 = w1 = draw = 0
    done = torch.zeros(n, dtype=torch.bool, device=sim.device)
    for _ in range(episodes // n):
        while not bool(done.all()):
            obs = sim.observe()
            mm, bm = sim.legal_mask()
            a0 = pol0(obs, mm[:, 0], bm[:, 0])
            a1 = pol1(obs, mm[:, 1], bm[:, 1])
            acts = torch.stack([a0, a1], dim=1)
            _, d, info = sim.step(acts)
            just = d & ~done
            w0 += int((just & info["winner"][:, 0]).sum())
            w1 += int((just & info["winner"][:, 1]).sum())
            draw += int((just & ~info["winner"][:, 0] & ~info["winner"][:, 1]).sum())
            done |= d
        sim.reset_all()
        done.zero_()
    return w0, w1, draw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--episodes", type=int, default=256)
    args = ap.parse_args()
    device = args.device
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")
    torch.manual_seed(0)

    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.3, ring_fraction=0.0, hazard_fraction=0.0)
    n_envs = 256
    sim = make_sim(cfg, n_envs, backend="torch", device=device, seed=0)

    random_bot = make_bot(sim, "random")
    greedy_bot = make_bot(sim, "greedy")

    def run(name0, pol0, name1, pol1) -> None:
        t0 = time.time()
        w0, w1, dr = duel(sim, pol0, pol1, args.episodes)
        tot = w0 + w1 + dr
        print(f"{name0} vs {name1}: win {w0/tot:.1%} / draw {dr/tot:.1%} / "
              f"loss {w1/tot:.1%}  ({tot} 局, {time.time()-t0:.0f}s)")

    print(f"[corridor] {args.episodes} 局 × {n_envs} envs")
    run("GreedyBot", lambda o, m, b: greedy_bot.act(o, m, b, 1),
        "RandomBot", lambda o, m, b: random_bot.act(o, m, b, 1))

    # 5x2 冻结网络 vs GreedyBot：5x2 应占优（课程最高一级）
    fivex2 = load_fixed_checkpoint("ckpt/duel_5x2.pt", sim.cfg.obs_shape, device)
    print(f"[5x2] obs_shape={fivex2.obs_shape} arch={fivex2.arch} n_players={fivex2.n_players}")

    @torch.no_grad()
    def fivex2_act(obs, mm, bm):
        return fivex2.act(obs, mm, bm, 0)[0]

    run("5x2(冻结)", fivex2_act, "GreedyBot", lambda o, m, b: greedy_bot.act(o, m, b, 1))


if __name__ == "__main__":
    main()
