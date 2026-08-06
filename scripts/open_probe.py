"""纯 open 地图专项测评：模型 vs astar 杀得死吗 + 前四行去不去。

背景：用户反馈 (1) open 空图模型从不上前四行（疑 corridor 顶部 4 行永久墙
记忆污染）；(2) 模型杀不死新版寻路 astar，尤其空图。

测两组初始属性：
  - 40% 起点 3/3/0.84（训练同款 open 关）
  - 80% 起点 6/6/1.68（duel 启动器默认，用户体感）
各 vs astar / greedy 打 128 局；同时统计双方角色行分布（前 4 行占比）。

用法：
    python scripts/open_probe.py ckpt/course_547m.pt
"""

from __future__ import annotations

import sys

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.train import load_fixed_checkpoint

DEV = "cpu"


def make_open_cfg(pct: float) -> SimConfig:
    return SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                     open_fraction=1.0, ring_fraction=0.0, hazard_fraction=0.0,
                     open_crate_cross=True, hit_attr_penalty=2,
                     open_growth_bombs=max(1, round(7 * pct)),
                     open_growth_blast=max(1, round(7 * pct)),
                     open_growth_speed=round(2.1 * pct, 2))


def duel_and_track(sim, pol0, pol1, episodes: int) -> tuple[dict, dict]:
    """打 episodes 局，返回 (胜率统计, 双方行分布计数)。
    行分布：记录每 tick 各玩家中心格所在行 → 前 4 行占比（corridor 顶部永久墙
    区域；open 关该区域是空地）。"""
    n = sim.num_envs
    dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = rounds = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    row_cnt = {0: [0] * sim.cfg.height, 1: [0] * sim.cfg.height}
    alive_cnt = {0: 0, 1: 0}
    from sim.move import center_cell
    while rounds < episodes:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = pol0(obs, mm[:, 0], bm[:, 0])
        a1 = pol1(obs, mm[:, 1], bm[:, 1])
        _, d, info = sim.step(torch.stack([a0, a1], dim=1))
        # 行分布：本 tick 每个存活玩家的中心格行
        cell = center_cell(sim.pos)
        for pl in (0, 1):
            alive_pl = sim.alive[:, pl]
            rows = cell[:, pl, 0].long()
            for e in range(n):
                if bool(alive_pl[e]):
                    row_cnt[pl][int(rows[e])] += 1
                    alive_cnt[pl] += 1
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
    stats = {"win": w0 / tot, "draw": dr / tot, "loss": w1 / tot}
    dist = {}
    for pl in (0, 1):
        a = max(1, alive_cnt[pl])
        top4 = sum(row_cnt[pl][:4]) / a
        bottom4 = sum(row_cnt[pl][sim.cfg.height - 4:]) / a
        dist[pl] = {"top4": top4, "bottom4": bottom4,
                    "rows": [c / a for c in row_cnt[pl]]}
    return stats, dist


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/course_547m.pt"
    episodes = 128
    for pct, tag in ((0.4, "40%起点(训练同款)"), (0.8, "80%起点(duel默认)")):
        cfg = make_open_cfg(pct)
        sim = BatchedSim(cfg, 128, device=DEV, seed=0)
        net = load_fixed_checkpoint(path, cfg.obs_shape, DEV)
        net.eval()

        @torch.no_grad()
        def pol0(o, m, b):
            return net.act(o, m, b, 0)[0]

        bots = {}
        for kind in ("astar", "greedy"):
            b = make_bot(sim, kind)
            bots[kind] = (lambda f=b: lambda o, m, bm: f.act(o, m, bm, 1))()

        print(f"\n=== {path} 纯open {tag}（开血{cfg.open_growth_bombs}/"
              f"{cfg.open_growth_blast}/{cfg.open_growth_speed}）===")
        for name, pol in bots.items():
            s, dist = duel_and_track(sim, pol0, pol, episodes)
            p0, p1 = dist[0], dist[1]
            print(f"vs {name:<8}: win {s['win']:.1%} / draw {s['draw']:.1%} "
                  f"/ loss {s['loss']:.1%}   "
                  f"模型前4行占比 {p0['top4']:.1%} / 底4行 {p0['bottom4']:.1%} | "
                  f"{name}前4行 {p1['top4']:.1%} / 底4行 {p1['bottom4']:.1%}")


if __name__ == "__main__":
    main()
