"""关卡生成器测试：健壮性 = 无永久墙死区（连通性）+ 出生点空旷。

"永远进不去"只可能来自永久墙（brick 可炸、理论全通），所以唯一硬约束是
忽略 brick 后从出生点 BFS 能到达所有非永久墙格。生成器每放一柱永久墙都
校验，这里钉住这个不变量。
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from sim.config import SimConfig
from sim.levelgen import connectivity_ok, generate_level


def _cfg() -> SimConfig:
    return SimConfig(height=13, width=13, n_players=2,
                     map_mode="corridor", corridor_width=5, top_wall_rows=4,
                     speed=3.0, max_steps=1800)


def test_connectivity_detects_dead_zone():
    """永久墙围死一个格 → 连通性必须 False（"永远进不去"的检测）。"""
    wall = np.zeros((5, 5), dtype=bool)
    # (2,2) 被四邻永久墙围死，无出生点可达
    wall[1, 2] = wall[3, 2] = wall[2, 1] = wall[2, 3] = True
    assert not connectivity_ok(wall, [(0, 0)]), "围死格应判不可达"
    # 去掉一面墙 → 可达
    wall[2, 1] = False
    assert connectivity_ok(wall, [(0, 0)]), "开口后应可达"


def test_generated_level_is_robust():
    """生成的关卡：连通（无死区）+ 出生点空旷（不被墙闷死）+ 有随机性。"""
    cfg = _cfg()
    rng = random.Random(7)
    lv = generate_level(cfg, rng)
    spawns = [(int(r), int(c)) for r, c in cfg.spawn_pos()]
    assert connectivity_ok(lv.wall, spawns), "生成关卡必须连通"
    for r, c in spawns:
        assert not lv.wall[r, c] and not lv.brick[r, c], "出生点必须空旷"
    # 顶墙存在、左右 brick 存在
    assert bool(lv.wall[:4].all()), "顶部 4 行永久墙"
    assert int(lv.brick[:, :4].sum()) > 0 and int(lv.brick[:, 9:].sum()) > 0


def test_generated_levels_vary():
    """不同 seed 生成不同布局（随机性有效，不是同一张图）。"""
    cfg = _cfg()
    a = generate_level(cfg, random.Random(1))
    b = generate_level(cfg, random.Random(2))
    assert not np.array_equal(a.brick, b.brick) or not np.array_equal(a.wall, b.wall)


@pytest.mark.skipif(not os.path.isdir("levels"), reason="levels/ 未生成")
def test_saved_levels_all_connected():
    """levels/ 里预生成的全部关卡：逐关连通 + 出生空旷。"""
    cfg = _cfg()
    spawns = [(int(r), int(c)) for r, c in cfg.spawn_pos()]
    import torch
    files = sorted(f for f in os.listdir("levels") if f.endswith(".pt"))
    assert len(files) >= 64, "至少 64 关（泛化训练的数据量）"
    for fn in files:
        d = torch.load(os.path.join("levels", fn), weights_only=False)
        w, b = d["wall"].numpy(), d["brick"].numpy()
        assert connectivity_ok(w, spawns), f"{fn} 有永久墙死区"
        for r, c in spawns:
            assert not w[r, c] and not b[r, c], f"{fn} 出生点被墙堵"
