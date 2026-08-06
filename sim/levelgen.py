"""随机关卡生成器 —— 为"地图泛化"训练预生成一批健壮关卡。

为什么不能纯随机：
- **永久墙（wall）会堵死区域**：被永久墙完全包围的格永远进不去 → 浪费地图空间。
- **可炸墙（brick）不会**（炸开就行），但要保证初始空旷区连通（出生点可达）。
- 所以唯一硬约束是：**忽略 brick（可炸）后，从出生点出发能到达所有非永久墙格**。
  生成时每放一柱永久墙都用 BFS 校验，破坏连通就换位置/重试。

生成流程（corridor 骨架 + 中间随机，和现有训练地图兼容）：
1. 骨架：顶部 top_wall_rows 行永久墙 + 左右 brick 列 + 中间空旷区（corridor 同款）。
2. 空旷区随机撒 brick（密度 brick_density）—— 可炸墙随机排布。
3. 空旷区随机放少量永久墙柱（密度 pillar_density）—— 每柱 BFS 校验连通。
4. 全部通过 → 存盘 levels/level_XXXX.pt（wall + brick + 生成 meta）。

用法：
    python -m sim.levelgen --count 96 --out levels
    python -m sim.levelgen --count 96 --out levels --seed 42 --brick-density 0.25
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import DIRS, SimConfig

# 关卡的"出生点"固定用 corridor 空旷区中心（与训练出生一致），
# 但泛化时也可传自定义出生点（多敌人/换出生点场景）。


def connectivity_ok(wall: np.ndarray, spawns: list[tuple[int, int]]) -> bool:
    """忽略 brick（可炸），从出生点 BFS，所有非永久墙格是否可达。

    wall: (H, W) bool，True = 永久墙。brick 不传进来（可炸 = 不堵死）。
    返回 True ⇔ 每个格"是永久墙 或 被 BFS 访问到" —— 没有永久墙死区。
    """
    h, w = wall.shape
    visited = np.zeros_like(wall, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in spawns:
        if 0 <= r < h and 0 <= c < w and not wall[r, c] and not visited[r, c]:
            visited[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + int(dr), c + int(dc)
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] \
                    and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return bool((visited | wall).all())


@dataclass
class Level:
    wall: np.ndarray        # (H, W) bool 永久墙
    brick: np.ndarray       # (H, W) bool 可炸墙
    seed: int
    meta: dict              # 生成参数（density 等）


def _skeleton(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """corridor 骨架：顶部永久墙 + 左右 brick 列 + 中间空旷区。"""
    h, w = cfg.height, cfg.width
    c0 = (w - cfg.corridor_width) // 2
    c1 = c0 + cfg.corridor_width
    wall = np.zeros((h, w), dtype=bool)
    wall[: cfg.top_wall_rows, :] = True                  # 顶部永久墙
    cols = np.arange(w)
    side = (cols < c0) | (cols >= c1)
    rows = np.arange(h)[:, None]
    brick = side[None, :] & ~(rows < cfg.top_wall_rows)  # 左右 brick（顶部行是永久墙）
    return wall, brick


def generate_level(cfg: SimConfig, rng: random.Random,
                   brick_density: float = 0.25,
                   pillar_density: float = 0.06,
                   max_tries: int = 64) -> Level:
    """生成一关：corridor 骨架 + 中间随机 brick + 带连通校验的永久墙柱。

    返回 Level；任何一步校验不过会重试（max_tries 后抛错，调用方换 seed）。
    """
    h, w = cfg.height, cfg.width
    spawns = [(int(r), int(c)) for r, c in cfg.spawn_pos()]

    for attempt in range(max_tries):
        wall, brick = _skeleton(cfg)
        # 空旷区 = 非骨架墙 且 非出生点四邻
        open_mask = ~wall & ~brick
        open_cells = [(r, c) for r in range(h) for c in range(w)
                      if open_mask[r, c]]
        # 清掉出生点及四邻（保证开局不闷死，和 mapgen 一致）
        clear = set(spawns)
        for r, c in spawns:
            for dr, dc in DIRS:
                clear.add((r + int(dr), c + int(dc)))
        open_cells = [p for p in open_cells if p not in clear]

        # 2. 空旷区随机撒 brick
        for r, c in open_cells:
            if rng.random() < brick_density:
                brick[r, c] = True

        # 3. 随机放永久墙柱（少，且每柱校验连通）
        #    柱只放"初始空旷"格（不是 brick 区/顶墙），避免和骨架冲突
        pillar_cells = [p for p in open_cells if not brick[p]]
        rng.shuffle(pillar_cells)
        placed_pillars = 0
        target = int(pillar_density * len(open_cells))
        for (r, c) in pillar_cells:
            if placed_pillars >= target:
                break
            wall[r, c] = True
            if not connectivity_ok(wall, spawns):
                wall[r, c] = False           # 堵死了 → 撤掉
            else:
                placed_pillars += 1

        # 4. 最终校验：无永久墙死区 + 出生点空旷
        if connectivity_ok(wall, spawns) and all(not wall[p] and not brick[p]
                                                 for p in spawns):
            return Level(wall=wall, brick=brick, seed=0, meta={
                "brick_density": brick_density,
                "pillar_density": pillar_density,
            })
    raise RuntimeError("generate_level 重试耗尽：调小密度或检查骨架")


def save_levels(path: str, levels: list[Level], cfg: SimConfig) -> None:
    """存到文件夹：level_XXXX.pt（wall, brick, meta），外加 LEVELS.md 说明。"""
    os.makedirs(path, exist_ok=True)
    for i, lv in enumerate(levels):
        torch.save({
            "wall": torch.from_numpy(lv.wall),
            "brick": torch.from_numpy(lv.brick),
            "meta": lv.meta,
            "cfg": {k: getattr(cfg, k) for k in
                    ("height", "width", "n_players", "corridor_width",
                     "top_wall_rows")},
        }, os.path.join(path, f"level_{i:04d}.pt"))
    with open(os.path.join(path, "LEVELS.md"), "w") as f:
        f.write(f"# 预生成关卡（{len(levels)} 关）\n\n"
                f"骨架：corridor（顶 {cfg.top_wall_rows} 行永久墙、左右 "
                f"{cfg.corridor_width} 列可炸墙、中间空旷区）。\n"
                f"空旷区随机可炸墙 + 少量永久墙柱（**每柱 BFS 校验，无死区**）。\n"
                f"用法：加载 level_XXXX.pt 的 wall/brick 替换 make_walls/make_bricks。\n")


def load_levels(path: str = "levels") -> list[dict]:
    """加载预生成关卡池（地图泛化训练用）。

    返回 [{wall, brick, meta, cfg}]，按编号排序。训练侧可在 reset 时
    每局随机抽一关替换 make_walls/make_bricks —— 生成器保证全部无死区。
    """
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    out = []
    for fn in files:
        d = torch.load(os.path.join(path, fn), weights_only=False)
        out.append(d)
    return out


def level_at(i: int, path: str = "levels") -> dict:
    """取第 i 关（越界取模循环，方便训练里滚动使用）。"""
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    fn = files[i % len(files)]
    return torch.load(os.path.join(path, fn), weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成健壮随机关卡（地图泛化用）")
    ap.add_argument("--count", type=int, default=96)
    ap.add_argument("--out", default="levels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brick-density", type=float, default=0.25,
                    help="空旷区可炸墙密度（0~1）")
    ap.add_argument("--pillar-density", type=float, default=0.06,
                    help="空旷区永久墙柱密度（低，防死区）")
    args = ap.parse_args()

    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    speed=3.0, max_steps=1800)
    rng = random.Random(args.seed)
    levels = []
    for i in range(args.count):
        lv = generate_level(cfg, random.Random(rng.randrange(1 << 30)),
                            args.brick_density, args.pillar_density)
        lv.seed = i
        levels.append(lv)
    save_levels(args.out, levels, cfg)
    print(f"生成 {len(levels)} 关 → {args.out}/（全部通过连通校验）")


if __name__ == "__main__":
    main()
