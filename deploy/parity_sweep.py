#!/usr/bin/env python3
"""随机状态扫描：生成 60 个随机对局状态（torch 参考实现），供 web/parity.js
以 --sweep 模式批量对拍，覆盖随机炮位/引信/威力/玩家位置/属性等边界情况。

用法：
    .venv/bin/python deploy/parity_sweep.py > deploy/sweep_states.json
    node web/parity.js --sweep deploy/sweep_states.json
"""

from __future__ import annotations

import json
import os
import sys

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from sim.blast import danger_map, resolve_explosions  # noqa: E402
from sim.config import SimConfig  # noqa: E402
from sim.obs import encode_obs, legal_mask  # noqa: E402


class LCG:
    """与 JS 端 mulberry32 无关的简易 LCG，两侧各自重放同一序列即可复现状态。"""
    def __init__(self, seed: int = 12345) -> None:
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s / 0xFFFFFFFF


def main() -> None:
    cfg = SimConfig(height=13, width=13, n_players=2, map_mode="corridor",
                    speed=3.0, max_steps=1800,
                    obs_extra_enabled=True, obs_fp16=False)
    n, h, w = 1, cfg.height, cfg.width
    rng = LCG()

    # 与 JS 侧一致的 corridor 布局（顶墙 + 侧砖）
    wall = torch.zeros((n, h, w), dtype=torch.bool)
    wall[:, : cfg.top_wall_rows, :] = True
    c0 = (w - cfg.corridor_width) // 2
    c1 = c0 + cfg.corridor_width
    brick = torch.zeros((n, h, w), dtype=torch.bool)
    for r in range(cfg.top_wall_rows, h):
        for c in range(w):
            if c < c0 or c >= c1:
                brick[:, r, c] = True
    base_wall = wall[0].clone()
    base_brick = brick[0].clone()
    passable = ~(base_wall | base_brick)
    # 注意：2D 张量 nonzero(as_tuple=True) 返回 (行索引, 列索引)，[0] 是**行**，
    # 必须拍平后取平铺索引（否则炮会落到 row0 墙格，产生不可能状态）。
    cells = passable.view(-1).nonzero(as_tuple=True)[0]

    states_out = []
    n_states = 60
    for k in range(n_states):
        fuse = torch.zeros((n, h, w), dtype=torch.int16)
        owner = torch.full((n, h, w), -1, dtype=torch.int8)
        bomb_blast = torch.zeros((n, h, w), dtype=torch.int16)
        # 0..6 颗炮，落在可通行格
        nb = int(rng.next() * 7)
        cells = passable.nonzero(as_tuple=True)[0]
        for _ in range(nb):
            i = int(cells[int(rng.next() * cells.numel())])
            fuse[:, i // w, i % w] = int(1 + rng.next() * 30)   # fuse 1..30
            owner[:, i // w, i % w] = int(rng.next() * 2)       # 0/1
            bomb_blast[:, i // w, i % w] = int(2 + rng.next() * 6)  # 威 2..7
        # 玩家位置：可通行格内连续坐标
        pos = torch.zeros((n, 2, 2), dtype=torch.float32)
        for p in range(2):
            rr = int(cells[int(rng.next() * cells.numel())])
            pos[:, p, 0] = rr // w + 0.3 + rng.next() * 0.4
            pos[:, p, 1] = rr % w + 0.3 + rng.next() * 0.4
        alive = torch.tensor([[rng.next() < 0.9, rng.next() < 0.9]])
        t = torch.tensor([int(rng.next() * 1800)])
        crate = torch.zeros((n, h, w), dtype=torch.bool)
        for _ in range(int(rng.next() * 4)):
            i = int(cells[int(rng.next() * cells.numel())])
            crate[:, i // w, i % w] = True
        invuln = torch.tensor([[int(rng.next() * 40), int(rng.next() * 40)]],
                              dtype=torch.long)
        bombs_p = torch.tensor([[1 + int(rng.next() * 10), 1 + int(rng.next() * 10)]],
                               dtype=torch.long)
        blast_p = torch.tensor([[1 + int(rng.next() * 7), 1 + int(rng.next() * 7)]],
                               dtype=torch.long)

        blast_map = torch.where(bomb_blast > 0, bomb_blast.long(), cfg.blast)
        dng = danger_map(fuse, wall, blast_map, cfg.fuse, brick, cfg.max_chain)
        obs = encode_obs(cfg, wall, fuse, owner, pos, alive, t, brick,
                         bomb_blast, crate, invuln, bombs_p,
                         danger_precomputed=dng)
        mm, bm = legal_mask(cfg, wall, fuse, owner, pos, alive, brick, bombs_p)
        covered, triggered = resolve_explosions(fuse, owner, wall, blast_map,
                                                cfg.max_chain, brick)

        states_out.append({
            "state": {
                "wall": [int(v) for v in wall[0].reshape(-1).tolist()],
                "brick": [int(v) for v in brick[0].reshape(-1).tolist()],
                "fuse": [int(v) for v in fuse[0].reshape(-1).tolist()],
                "owner": [int(v) for v in owner[0].reshape(-1).tolist()],
                "bomb_blast": [int(v) for v in bomb_blast[0].reshape(-1).tolist()],
                "pos": [float(v) for v in pos[0].reshape(-1).tolist()],
                "alive": [bool(v) for v in alive[0].tolist()],
                "t": int(t[0]),
                "crate": [int(v) for v in crate[0].reshape(-1).tolist()],
                "invuln": [int(v) for v in invuln[0].tolist()],
                "bombs_cap": [int(v) for v in bombs_p[0].tolist()],
                "blast_cap": [int(v) for v in blast_p[0].tolist()],
            },
            "danger": [float(v) for v in dng[0].reshape(-1).tolist()],
            "obs": [float(v) for v in obs[0].reshape(-1).tolist()],
            "mm": [[int(v) for v in row] for row in mm[0].tolist()],
            "bm": [[int(v) for v in row] for row in bm[0].tolist()],
            "covered": [int(v) for v in covered[0].reshape(-1).tolist()],
            "triggered": [int(v) for v in triggered[0].reshape(-1).tolist()],
        })
    print(json.dumps({"cfg": {"h": h, "w": w, "channels": cfg.n_channels,
                              "max_chain": cfg.max_chain},
                      "states": states_out}, separators=(",", ":")))
    print(f"// {n_states} states generated", file=sys.stderr)


if __name__ == "__main__":
    main()
