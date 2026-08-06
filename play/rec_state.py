"""录像回放共享：从 obs 通道构造渲染状态（FakeSim）→ 复用 duel 的 RES 真实渲染。

录像 obs[t] = (C, H, W) 14 通道共享观测（step 前快照，人类视角=player0）：
  0/1 位置、2/3 引信（分 owner）、4 墙|砖、5 危险、6 进度、7 宝箱、8.. 扩展。
FakeSim 只读字段喂给 build_static/draw_grid（精灵/砖贴图/Z 排序/危险红区/
血条/无敌罩全用真实素材）。replay.py（交互回放）与 scripts/render_gif.py
（GIF/webm 渲染）共用。

时序语义（重要）：obs[t] 是 step **前**快照，放炮/吃箱/爆炸发生在 tick t 内、
到 obs[t+1] 才反映。逐 tick 回放（replay）直接显示 obs[t] 原始状态 → 完全同步。
插值渲染（render_gif）的位置插值 + 泡/宝箱用 cur（step 前）→ 动作与状态同步。
"""

from __future__ import annotations

import torch


class FakeSim:
    """从录像 obs 提取的状态容器，喂给 build_static/draw_grid（只读字段）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.wall = torch.zeros(1, cfg.height, cfg.width, dtype=torch.bool)
        self.brick = torch.zeros(1, cfg.height, cfg.width, dtype=torch.bool)
        self.fuse = torch.zeros(1, cfg.height, cfg.width, dtype=torch.int16)
        self.owner = torch.full((1, cfg.height, cfg.width), -1, dtype=torch.int16)
        self.pos = torch.zeros(1, 2, 2, dtype=torch.float32)
        self.alive = torch.ones(1, 2, dtype=torch.bool)
        self.hp = torch.full((1, 2), cfg.max_hp, dtype=torch.uint8)
        self.invuln = torch.zeros(1, 2, dtype=torch.long)
        self.crate = torch.zeros(1, cfg.height, cfg.width, dtype=torch.bool)
        self.t = torch.zeros(1, dtype=torch.long)

    def set_frame(self, o, cfg) -> None:
        """从一帧 obs (C,H,W) 填状态（o 已 /255 归一化，numpy 或 torch 均可）。"""
        if not torch.is_tensor(o):
            o = torch.from_numpy(o).float()
        h, w = cfg.height, cfg.width
        self.wall[0] = False
        self.brick[0] = o[4] > 0.5                      # 墙|砖 合并通道 → 砖块
        f0, f1 = o[2] * cfg.fuse, o[3] * cfg.fuse
        self.fuse[0] = 0
        self.fuse[0] = torch.where(f0 > 0.5, f0.long(), torch.zeros_like(f0)).short()
        self.fuse[0] = torch.where(f1 > 0.5, f1.long(),
                                   self.fuse[0]).short()
        self.owner[0] = -1
        self.owner[0] = torch.where(f0 > 0.5, torch.zeros_like(f0).long(),
                                    self.owner[0]).short()
        self.owner[0] = torch.where(f1 > 0.5, torch.ones_like(f1).long(),
                                    self.owner[0]).short()
        for pid in range(2):
            p = self._player_center(o[pid])
            if p is None:
                self.alive[0, pid] = False
            else:
                self.alive[0, pid] = True
                self.pos[0, pid, 0] = p[0]
                self.pos[0, pid, 1] = p[1]
        self.crate[0] = o[7] > 0.5 if o.shape[0] > 7 else False
        self.t[0] = int(o[6, 0, 0] * cfg.max_steps)   # 进度通道是常量平面

    @staticmethod
    def _player_center(plane: torch.Tensor) -> tuple[float, float] | None:
        """从双线性 splat 平面（格下标坐标）反推角色亚格位置（格中心坐标）。"""
        tot = float(plane.sum())
        if tot < 1e-6:
            return None
        h, w = plane.shape
        ys = torch.arange(h, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, -1)
        cy = float((plane * ys).sum()) / tot + 0.5
        cx = float((plane * xs).sum()) / tot + 0.5
        return cy, cx


def explosion_between(cur: FakeSim, nxt: FakeSim, cfg):
    """cur 有泡、nxt 无泡的格子 = 本 tick 爆炸格 → (blast_mask, trig_mask) bool。

    录像 obs 只有 10Hz 快照没有爆炸帧；泡泡消失 = 爆炸。用相邻两帧引信通道差
    推断爆炸格，向 4 方向扩展 blast（挡墙），中间帧画爆炸闪光。"""
    cb = (cur.fuse[0] > 0).numpy()
    nb = (nxt.fuse[0] > 0).numpy()
    trig = cb & ~nb
    if not trig.any():
        return None
    wall = (cur.wall[0] | cur.brick[0]).numpy()
    h, w = cfg.height, cfg.width
    blast = trig.copy()
    for y, x in zip(*trig.nonzero()):
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            for k in range(1, int(cfg.blast) + 1):
                ny, nx = y + dr * k, x + dc * k
                if not (0 <= ny < h and 0 <= nx < w) or wall[ny, nx]:
                    break
                blast[ny, nx] = True
    return (torch.from_numpy(blast).bool(), torch.from_numpy(trig).bool())
