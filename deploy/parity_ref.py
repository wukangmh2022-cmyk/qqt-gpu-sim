#!/usr/bin/env python3
"""生成一个固定对局状态的参考输出（torch 参考实现），供 web/parity.js 对拍。

构造 corridor 13×13 状态：顶墙/侧砖 + 固定位置的多颗炮（不同 owner/威力/引信
剩余，覆盖连锁与非连锁情况）+ 固定玩家位置/属性/无敌期/宝箱，然后输出
    danger_map / encode_obs(14ch) / legal_mask / resolve_explosions
四个结果到 JSON。JS 侧用同样状态重算并逐元素比较。

用法：
    .venv/bin/python deploy/parity_ref.py > deploy/ref_state.json
    node web/parity.js deploy/ref_state.json
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


def main() -> None:
    torch.manual_seed(0)
    cfg = SimConfig(height=13, width=13, n_players=2, map_mode="corridor",
                    speed=3.0, max_steps=1800,
                    obs_extra_enabled=True, obs_fp16=False)
    n, h, w = 1, cfg.height, cfg.width

    # ---- 固定状态（直接写张量，不跑 reset，保证 JS 侧可复现）----
    wall = torch.zeros((n, h, w), dtype=torch.bool)
    wall[:, : cfg.top_wall_rows, :] = True                       # 顶 4 行永久墙
    c0 = (w - cfg.corridor_width) // 2
    c1 = c0 + cfg.corridor_width
    brick = torch.zeros((n, h, w), dtype=torch.bool)
    for r in range(cfg.top_wall_rows, h):
        for c in range(w):
            if c < c0 or c >= c1:
                brick[:, r, c] = True
    fuse = torch.zeros((n, h, w), dtype=torch.int16)
    owner = torch.full((n, h, w), -1, dtype=torch.int8)
    bomb_blast = torch.zeros((n, h, w), dtype=torch.int16)
    # 三颗炮：(5,5) 引信0(即将爆,owner0,威2) 连锁到 (5,7)；(8,9) 独立炮(引信10,owner1,威4)
    for (r, c, f, o, b) in ((5, 5, 0, 0, 2), (5, 7, 10, 1, 4), (8, 9, 10, 0, 3)):
        fuse[:, r, c] = f
        owner[:, r, c] = o
        bomb_blast[:, r, c] = b
    pos = torch.tensor([[[8.5, 5.5], [8.5, 8.5]]], dtype=torch.float32)
    alive = torch.tensor([[True, True]])
    t = torch.tensor([100])
    crate = torch.zeros((n, h, w), dtype=torch.bool)
    crate[:, 9, 5] = True
    invuln = torch.tensor([[5, 0]], dtype=torch.long)
    bombs_p = torch.tensor([[8, 6]], dtype=torch.long)
    blast_p = torch.tensor([[6, 5]], dtype=torch.long)

    blast_map = torch.where(bomb_blast > 0, bomb_blast.long(), cfg.blast)

    dng = danger_map(fuse, wall, blast_map, cfg.fuse, brick, cfg.max_chain)
    obs = encode_obs(cfg, wall, fuse, owner, pos, alive, t, brick,
                     bomb_blast, crate, invuln, bombs_p,
                     danger_precomputed=dng)
    mm, bm = legal_mask(cfg, wall, fuse, owner, pos, alive, brick, bombs_p)
    covered, triggered = resolve_explosions(fuse, owner, wall, blast_map,
                                            cfg.max_chain, brick)

    out = {
        "cfg": {"h": h, "w": w, "channels": cfg.n_channels,
                "top_wall_rows": cfg.top_wall_rows,
                "corridor_width": cfg.corridor_width,
                "fuse": cfg.fuse, "blast": cfg.blast, "max_chain": cfg.max_chain,
                "growth_bombs_max": cfg.growth_bombs_max},
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
    }
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
