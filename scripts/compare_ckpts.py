"""本地双模型对打：A vs B 双向（消除出生点/视角偏差），击杀口径。

- P1 侧模型用 swap_channels 把 per-player 通道 0↔1 互换 → "自己=P0 视角"
  （与 play/duel._swap_player_channels 同款，也和 hourly_eval 的 P1 swap 一致）。
- 终局口径与训练一致（timeout_draw=True）：只有死亡终局(n_alive==1)记胜负，
  超时/同归于尽=平局 —— 胜率统计是纯"击杀"口径，不靠血判生。
- 双向：A在P0 vs B在P1(swap)，再 B在P0 vs A在P1(swap)，各自半局。

用法：
    python scripts/compare_ckpts.py ckpt/A.pt ckpt/B.pt [episodes_per_side]
"""

from __future__ import annotations

import sys
import time

import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from scripts.duel_swap import swap_channels
from train.train import load_fixed_checkpoint

DEV = "cpu"

CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0,
                open_crate_cross=True, hit_attr_penalty=2,
                timeout_draw=True)          # 与训练一致：超时=平局，击杀才赢


def duel_policies(sim, pol0, pol1, episodes: int) -> tuple[float, float, float]:
    """pol(obs, mm_p, bm_p) -> (N,2)。终局计数用 sim.info["winner"]：
    winner 只在死亡终局(n_alive==1)置位 → 击杀=胜；超时/双亡=平局 0。"""
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


def load_net(path: str, obs_shape, dev):
    net = load_fixed_checkpoint(path, obs_shape, dev)
    net.obs_shape = obs_shape          # act() 内部用到
    return net


def main() -> None:
    pa, pb = sys.argv[1], sys.argv[2]
    eps = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    sim = BatchedSim(CFG, 128, device=DEV, seed=0)
    na = load_net(pa, CFG.obs_shape, DEV)
    nb = load_net(pb, CFG.obs_shape, DEV)
    na.eval(); nb.eval()
    print(f"[{pa}] arch={na.arch} obs={na.obs_shape}")
    print(f"[{pb}] arch={nb.arch} obs={nb.obs_shape}")
    print(f"地图 corridor50% · 超时=平局(击杀口径) · 每侧 {eps} 局")

    @torch.no_grad()
    def pol(o, m, b, net):
        return net.act(o, m, b, 0)[0]

    # 方向 1：A 在 P0，B 在 P1(swap)
    t0 = time.time()
    w, dr, l = duel_policies(
        sim,
        lambda o, m, b: pol(o, m, b, na),
        lambda o, m, b: pol(swap_channels(o), m, b, nb),
        eps)
    print(f"  A在P0: A胜 {w:.1%} / 平 {dr:.1%} / A败 {l:.1%}  ({time.time()-t0:.0f}s)")
    # 方向 2：B 在 P0，A 在 P1(swap)
    t0 = time.time()
    w, dr, l = duel_policies(
        sim,
        lambda o, m, b: pol(o, m, b, nb),
        lambda o, m, b: pol(swap_channels(o), m, b, na),
        eps)
    print(f"  B@P0: A胜 {l:.4%} / 平 {dr:.1%} / A败 {w:.4%}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()