"""场地生成 —— 两个 backend 共用，保证 parity 测试里地图完全一致。

地图生成放在 host 侧（torch 算子）而不是 kernel 里：它每局只跑一次
（约每 400 tick 一次），不在热路径上，放到 kernel 里只会引入一套
需要和 Python RNG 对齐的随机数发生器，纯属自找麻烦。
"""

from __future__ import annotations

import torch

from .config import DIRS, SimConfig


def make_walls(
    cfg: SimConfig, count: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """返回 (count, H, W) bool。wall_density=0 时是纯空场。

    corridor 模式：**顶部 top_wall_rows 行全部永久墙**（不可炸），与
    wall_density 无关（即使 density=0 也有顶墙）；wall_density>0 时空旷区
    额外按经典炸弹人图案随机立柱（障碍物渐进课程用，密度克制递增）。
    非零 density 时按经典炸弹人的"奇数行奇数列立柱"图案随机保留柱子，
    并强制清空出生点及其四邻，避免开局就被闷死。
    """
    h, w = cfg.height, cfg.width
    if cfg.map_mode == "corridor":
        # 顶部 top_wall_rows 行全部永久墙 —— 出生点在空旷区下方，
        # 顶墙把场地顶住，火也不会烧到地图外。与 density 无关。
        wall = torch.zeros((count, h, w), dtype=torch.bool, device=device)
        wall[:, : cfg.top_wall_rows, :] = True
        # wall_density>0：空旷区也随机立柱（每局图案不同，防地图过拟合）
        if cfg.wall_density > 0:
            rows = torch.arange(h).view(-1, 1)
            cols = torch.arange(w).view(1, -1)
            pillars = (rows % 2 == 1) & (cols % 2 == 1)
            keep = torch.rand((count, h, w), generator=gen) < cfg.wall_density
            wall |= pillars.unsqueeze(0) & keep
    elif cfg.wall_density <= 0:
        return torch.zeros((count, h, w), dtype=torch.bool, device=device)
    else:
        rows = torch.arange(h).view(-1, 1)
        cols = torch.arange(w).view(1, -1)
        pillars = (rows % 2 == 1) & (cols % 2 == 1)
        keep = torch.rand((count, h, w), generator=gen) < cfg.wall_density
        wall = pillars.unsqueeze(0) & keep
    for row, col in cfg.spawn_cells():
        wall[:, row, col] = False
        for drow, dcol in DIRS:
            nr, nc = row + int(drow), col + int(dcol)
            if 0 <= nr < h and 0 <= nc < w:
                wall[:, nr, nc] = False
    return wall.to(device)


def make_bricks(
    cfg: SimConfig, count: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """返回 (count, H, W) bool 的**可炸墙**（brick）。与 make_walls 分开。

    - open 模式：无 brick（纯空场）。
    - corridor 模式：中间 corridor_width 列可通行，左右两侧**整列** brick。
      开局出生点及四邻清空，保证不被墙闷死。
    可炸墙被火焰覆盖即摧毁（见 blast.py / torch_sim），挡火但会被烧掉。
    """
    h, w = cfg.height, cfg.width
    if cfg.map_mode != "corridor":
        return torch.zeros((count, h, w), dtype=torch.bool, device=device)

    c0 = (w - cfg.corridor_width) // 2          # 可通行区左边界列
    c1 = c0 + cfg.corridor_width                # 可通行区右边界列（开区间）
    rows = torch.arange(h).view(-1, 1)
    cols = torch.arange(w).view(1, -1)
    side = (cols < c0) | (cols >= c1)           # 左右两侧 brick
    # 顶部 top_wall_rows 行已经是永久墙（make_walls），不再铺 brick
    in_top = rows < cfg.top_wall_rows
    brick = side & ~in_top
    out = brick.expand(count, h, w).clone()
    # **不**清出生点四邻：corridor 里出生点在可通行区（脚下非 brick），
    # 四邻可能正好是 brick 墙（如 (8.5,8.5) 的东邻 (8,9) 是右墙第一列）——
    # 清空会把墙炸出一个开局缺口。可通行区内部本来就通，不会闷死。
    return out.to(device)


def make_ring_bricks(
    cfg: SimConfig, count: int, gen: torch.Generator, device: torch.device
) -> torch.Tensor:
    """环岛（ring）模式的可炸墙：中间是**永久墙山体**（torch_sim 单独铺 wall），
    山体外围一圈按 ring_brick_density 的**稀疏密度**铺 brick —— 不是全部充满，
    否则开局没有立锥之地。出生点及四邻强制清空，保证不被 brick 闷死。

    环形区域：**山体外圈 1 格宽的环带**（矩形 [r0-1, r1] × [c0-1, c1] 减去
    山体本身，即山体外侧紧邻的一圈 13×13 - 7×7 = 120 格）。brick 只在环带内
    随机铺（稀疏 density），山体内部（永久墙）不铺 —— 中间整块区域不可行走，
    玩家围着山体绕圈。出生点在场地四角（ring_spawns），清空其四邻后脚下必定
    安全。
    """
    h, w = cfg.height, cfg.width
    r0 = (h - cfg.ring_center_h) // 2
    c0 = (w - cfg.ring_center_w) // 2
    r1 = r0 + cfg.ring_center_h
    c1 = c0 + cfg.ring_center_w
    band = torch.zeros((count, h, w), dtype=torch.bool, device=device)
    band[:, r0 - 1:r1 + 1, c0 - 1:c1 + 1] = True   # 山体 + 外圈 1 格
    band[:, r0:r1, c0:c1] = False                  # 减去山体本身 → 纯环带
    rnd = torch.rand((count, h, w), generator=gen).to(device)  # gen 是 CPU 生成器 → 先随机再搬到目标设备
    brick = band & (rnd < cfg.ring_brick_density)
    # 出生点四角安全：清掉每个出生点及四邻的 brick（角落互不相邻，无重叠冲突）
    for row, col in cfg.ring_spawn_cells():
        brick[:, row, col] = False
        for drow, dcol in DIRS:
            nr, nc = row + int(drow), col + int(dcol)
            if 0 <= nr < h and 0 <= nc < w:
                brick[:, nr, nc] = False
    return brick.to(device)


def make_ring_walls(cfg: SimConfig, count: int, device: torch.device) -> torch.Tensor:
    """环岛（ring）模式中间山体的**永久墙**：不可行走、不可炸。

    玩家被隔在山体外的环带里绕圈 —— 这就是"环岛"：中央大障碍物 + 周边
    稀疏可炸墙 + 四角出生点。brick 只铺在环带（make_ring_bricks），
    二者在空间上互斥。
    """
    h, w = cfg.height, cfg.width
    r0 = (h - cfg.ring_center_h) // 2
    c0 = (w - cfg.ring_center_w) // 2
    wall = torch.zeros((count, h, w), dtype=torch.bool, device=device)
    wall[:, r0:r0 + cfg.ring_center_h, c0:c0 + cfg.ring_center_w] = True
    return wall
