"""观测编码与合法动作掩码 —— 参考实现。

通道定义见 `sim/config.py` 文件头的 OBS_LAYOUT 注释和 RULES.md。
所有函数都对 batch 向量化，只在角色维度上做小循环（P <= 4，展开成 Python
循环比 gather 更清楚，也和 CUDA 侧一一对应）。

`encode_obs` 输出的是**一个 env 一份的共享张量** `(N, 2P+3, H, W)`，
不是每个角色一份。角色视角靠 `cfg.view_perm(me)` 这个纯置换表达，
在 `train/model.py` 里被吸收进第一层卷积的权重索引，零数据搬运。

位置通道用**双线性铺开**而不是 one-hot：连续坐标下 one-hot 会把 0.2 格的
位移直接量化掉，网络看到的是"卡在格中间不动"。铺开之后亚格偏移完整保留，
"还差几帧能拐过这个角"这件事才有可能被学到。
"""

from __future__ import annotations

import torch

from .blast import danger_map
from .config import DIRS, N_BOMB, N_MOVES, SimConfig
from .move import _EPS, _resolve_axis, center_cell


def _splat(pos_me: torch.Tensor, gate: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """把连续坐标双线性铺开成 (N, H, W) 平面，总质量为 1（gate 为 False 时全 0）。"""
    n = pos_me.shape[0]
    dev = pos_me.device
    # 格中心 i 对应 fy = i，所以先减去半格
    fy = (pos_me[:, 0] - 0.5).clamp(0.0, h - 1.0)
    fx = (pos_me[:, 1] - 0.5).clamp(0.0, w - 1.0)
    y0 = fy.floor().long().clamp(0, h - 1)
    x0 = fx.floor().long().clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    wy = (fy - y0.to(fy.dtype)).clamp(0.0, 1.0)
    wx = (fx - x0.to(fx.dtype)).clamp(0.0, 1.0)
    g = gate.to(fy.dtype)

    out = torch.zeros((n, h * w), dtype=fy.dtype, device=dev)
    for yy, ww_y in ((y0, 1.0 - wy), (y1, wy)):
        for xx, ww_x in ((x0, 1.0 - wx), (x1, wx)):
            out.scatter_add_(1, (yy * w + xx).unsqueeze(1),
                             (ww_y * ww_x * g).unsqueeze(1))
    return out.view(n, h, w)


def encode_obs(
    cfg: SimConfig,
    wall: torch.Tensor,
    fuse: torch.Tensor,
    owner: torch.Tensor,
    pos: torch.Tensor,
    alive: torch.Tensor,
    t: torch.Tensor,
    brick: torch.Tensor | None = None,
    bomb_blast: torch.Tensor | None = None,
    crate: torch.Tensor | None = None,
    invuln: torch.Tensor | None = None,
    bombs_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 (N, 2P+3+obs_extra(P), H, W)：**整个 env 一份**的共享观测。

    通道顺序见 `sim/config.py` 文件头。基础 2P+3 个通道与"我是谁"无关，
    角色视角是这份张量的一个置换（`cfg.view_perm(me)`），不需要额外内存；
    尾部扩展通道（宝箱/无敌/可用泡数/泡数上限）是**世界信息**，原样保留。

    中间量一律 fp32 计算，只在最后一次性 cast 成存储 dtype ——
    这样精度行为可预测，CUDA 侧照抄同样的顺序即可对齐。
    """
    n, p = alive.shape
    h, w = cfg.height, cfg.width
    fuse_norm = fuse.float() / float(cfg.fuse)
    zeros = torch.zeros_like(fuse_norm)

    obs = torch.zeros((n, cfg.n_channels, h, w), device=wall.device,
                      dtype=torch.float32)
    for i in range(p):
        obs[:, i] = _splat(pos[:, i], alive[:, i], h, w)
        obs[:, p + i] = torch.where((owner == i) & (fuse > 0), fuse_norm, zeros)
    # 墙通道 = 永久墙 | 可炸墙（都不可通行）。corridor 里格内墙全 brick，
    # "有墙 = 可炸"对网络是隐式可学的；状态变化（变 0）即"炸开了"。
    obs[:, 2 * p] = (wall | (brick if brick is not None
                             else torch.zeros_like(wall))).float()
    # 危险图用**每颗泡自己的威力**（成长系统 bomb_blast），否则 UI/网络看到
    # 的危险范围恒为 cfg.blast 而实际爆炸随成长变大 —— "危险区不更新"的根因。
    # 0（手工种泡/未设置）回退 cfg.blast，与 torch_sim._blast_map 一致。
    if bomb_blast is not None:
        blast_map = torch.where(bomb_blast > 0, bomb_blast.long(), cfg.blast)
    else:
        blast_map = cfg.blast
    obs[:, 2 * p + 1] = danger_map(fuse, wall, blast_map, cfg.fuse, brick)
    obs[:, 2 * p + 2] = (t.float() / float(cfg.max_steps)).view(n, 1, 1)

    # ---------------- 扩展通道（世界信息，尾部原样保留） ----------------
    # 由 cfg.obs_extra_enabled 开关：关掉时保持旧 7 通道布局（兼容旧 ckpt 评估）。
    if not cfg.obs_extra_enabled:
        return obs.half() if cfg.obs_fp16 else obs
    # 2P+3: 宝箱位置（0/1）
    if crate is not None:
        obs[:, 2 * p + 3] = crate.float()
    # 2P+4..+P: 玩家 i 无敌标记（位置格 1，其余 0）
    if invuln is not None:
        cell = center_cell(pos)
        flat = (cell[..., 0] * w + cell[..., 1])
        for i in range(p):
            ch = 2 * p + 4 + i
            mask = (invuln[:, i] > 0) & alive[:, i]
            obs[:, ch].view(n, -1).scatter_(
                1, flat[:, i].unsqueeze(1), mask.float().unsqueeze(1))
    # 后两组 P 通道：玩家 i 可用泡泡数（位置格 = 可用/上限档）与泡泡上限
    # （位置格 = 上限/上限档）。归一化到 (0,1]，越接近 1 越接近上限。
    if bombs_p is not None:
        cap = bombs_p.float().clamp(min=1)
        live = torch.stack([(owner == i) & (fuse > 0) for i in range(p)],
                           dim=1).flatten(2).sum(dim=2)      # (n,p) 在场泡数
        avail = (bombs_p - live).float().clamp(min=0)
        cell = center_cell(pos)
        flat = (cell[..., 0] * w + cell[..., 1])
        for i in range(p):
            base = 2 * p + 4 + p    # 越过 crate + invuln 两段
            obs[:, base + i].view(n, -1).scatter_(
                1, flat[:, i].unsqueeze(1),
                (avail[:, i] / cap[:, i]).unsqueeze(1) * alive[:, i].float().unsqueeze(1))
            obs[:, base + p + i].view(n, -1).scatter_(
                1, flat[:, i].unsqueeze(1),
                (cap[:, i] / cfg.growth_bombs_max).unsqueeze(1) * alive[:, i].float().unsqueeze(1))
    return obs.half() if cfg.obs_fp16 else obs


def can_place(
    cfg: SimConfig,
    fuse: torch.Tensor,
    owner: torch.Tensor,
    pos: torch.Tensor,
    brick: torch.Tensor | None = None,
    bombs_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """(N, P) bool：这个 tick 放泡能否成立（中心格没泡/没墙 + 在场数没满）。

    bombs_p (N,P)：**每个玩家**当前的泡数上限（corridor 逐人成长）。
    """
    n, p, _ = pos.shape
    w = cfg.width
    cell = center_cell(pos)
    flat = (cell[..., 0] * w + cell[..., 1])
    fuse_flat = fuse.view(n, -1)
    ok = torch.zeros((n, p), dtype=torch.bool, device=pos.device)
    for me in range(p):
        under = fuse_flat.gather(1, flat[:, me].unsqueeze(1)).squeeze(1) > 0
        live = ((owner == me) & (fuse > 0)).flatten(1).sum(1)
        # 脚下是墙/brick 也不能放泡（中心格必须可通行）
        blocked_flat = (fuse > 0) | (brick if brick is not None
                                     else torch.zeros_like(fuse, dtype=torch.bool))
        blocked_flat = blocked_flat.view(n, -1)
        under_brick = blocked_flat.gather(1, flat[:, me].unsqueeze(1)).squeeze(1)
        cap = bombs_p[:, me] if bombs_p is not None else cfg.max_bombs   # (N,)
        ok[:, me] = ~under & ~under_brick & (live < cap)
    return ok


def legal_mask(
    cfg: SimConfig,
    wall: torch.Tensor,
    fuse: torch.Tensor,
    owner: torch.Tensor,
    pos: torch.Tensor,
    alive: torch.Tensor,
    brick: torch.Tensor | None = None,
    bombs_p: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (move_mask (N,P,5), bomb_mask (N,P,2))。

    方向掩码只屏蔽"按了也一格都动不了"的方向（已经贴住墙或泡泡）。
    MOVE_IDLE 永远合法，所以方向掩码不可能全 False —— 格子版里那个
    "全掩码 → softmax 出 NaN"的兜底分支在这套动作空间下自然消失了。

    放炮头的 bomb=0（不放）也永远合法，同理不需要兜底。
    """
    n, p, _ = pos.shape
    h, w, rad = cfg.height, cfg.width, cfg.radius
    blocked_flat = (wall | (fuse > 0)
                    | (brick if brick is not None
                       else torch.zeros_like(wall))).view(n, -1)

    move_mask = torch.ones((n, p, N_MOVES), dtype=torch.bool, device=pos.device)
    for me in range(p):
        y, x = pos[:, me, 0], pos[:, me, 1]
        for k, (ky, kx) in enumerate(DIRS):
            dy = torch.full_like(y, ky * cfg.step_len)
            dx = torch.full_like(x, kx * cfg.step_len)
            if ky:
                ny = _resolve_axis(y + dy, dy, x, y, x, blocked_flat, rad, h, w, True)
                moved = (ny - y).abs() > _EPS * 2
            else:
                nx = _resolve_axis(x + dx, dx, y, y, x, blocked_flat, rad, h, w, False)
                moved = (nx - x).abs() > _EPS * 2
            move_mask[:, me, k] = moved
        # 死亡角色的动作不会被执行，整行放开，省掉调用方的特殊分支
    move_mask = (move_mask & alive.unsqueeze(-1)) | (~alive).unsqueeze(-1)

    place = can_place(cfg, fuse, owner, pos, brick, bombs_p) & alive
    bomb_mask = torch.ones((n, p, N_BOMB), dtype=torch.bool, device=pos.device)
    # 死亡角色两个头都整行放开（动作反正不会被执行），调用方不需要特殊分支
    bomb_mask[..., 1] = place | ~alive
    return move_mask, bomb_mask
