"""连续坐标的移动与碰撞 —— 参考实现。

这是"炸弹人玩法"和"一 tick 跳一格"的分界线。角色按 `speed` 格/秒匀速走，
一个 tick 位移 `speed / tick_hz`（默认 3/15 = 0.2 格），要走满一格需要 5 个 tick。
方向键按下即生效、松手（MOVE_IDLE）即停，没有惯性也没有加速度。

碰撞用轴对齐盒（半宽 `radius`）。因为约束了 `radius < 0.5`，碰撞盒最多同时
压在**两行两列**上，所以沿单轴移动时只需要检查前进方向上的 **2 个格子**，
不需要遍历邻域。单 tick 位移默认远小于格宽（3.6/10 = 0.36 格），不穿模；
成长系统把速度推到 1.5×（0.45 格/tick）时，靠"前缘区间枚举"一次消解：
前缘最多跨过 2 个格边界，枚举旧前缘格..新前缘格，任一障碍就贴着停 ——
高速也永不穿模，不需要 substep。

一条容易漏的规则：**刚在脚下放的泡泡必须能走出去**。做法不是记"哪颗泡该放行"，
而是判断"碰撞盒当前是否已经压在这一格上" —— 压着就放行，走出去之后自然变成实体。
无状态，两个 backend 都好实现。
"""

from __future__ import annotations

import torch

from .config import DIRS, MOVE_IDLE, N_MOVES, SimConfig

# torch.compile 时排除 move_players：其 gather 索引（_resolve_axis_batch 的
# blocked_flat gather + scatter）在 HIP triton 融合下语义出错（2026-08-10 实测
# 0.045 格位置分叉）。graph break 让它在 compile step 里保持 eager 执行。
try:
    from torch.compiler import disable as _dynamo_disable
except ImportError:  # torch < 2.7
    from torch._dynamo import disable as _dynamo_disable

_EPS = 1e-4

# 方向动作 → (dy, dx) 位移表缓存（key=(device,dtype,step)；step 由 cfg 固定）
_STEP_TABLE_CACHE: dict = {}


def _step_table(device, dtype, step: float) -> torch.Tensor:
    """(N_MOVES, 2) float 表：动作 k → (dy, dx)。IDLE 恒为 0。"""
    key = (device, dtype, step)
    t = _STEP_TABLE_CACHE.get(key)
    if t is None:
        t = torch.tensor(
            [[ky * step, kx * step] for ky, kx in DIRS] + [[0.0, 0.0]],
            device=device, dtype=dtype)
        _STEP_TABLE_CACHE[key] = t
    return t


def _impassable(
    blocked_flat: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    rad: float,
    h: int,
    w: int,
) -> torch.Tensor:
    """(N,) bool：格 (row, col) 对这个角色是否不可通行。越界算不可通行。"""
    oob = (row < 0) | (row >= h) | (col < 0) | (col >= w)
    idx = (row.clamp(0, h - 1) * w + col.clamp(0, w - 1)).unsqueeze(1)
    solid = blocked_flat.gather(1, idx).squeeze(1)
    # 碰撞盒当前已经覆盖这一格 → 放行（脚下自己刚放的泡）
    # 与 JS sim.js impassable 一致：下界 floor、上界 **ceil** + 严格小于
    # （左闭右开）。盒右/下缘恰贴格边界（y+R=整数）时不覆盖下一格 → 防穿炮。
    r0, r1 = (y - rad).floor().long(), (y + rad).ceil().long()
    c0, c1 = (x - rad).floor().long(), (x + rad).ceil().long()
    inside = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
    return oob | (solid & ~inside)


def _impassable_pair(
    blocked_flat: torch.Tensor,
    row0: torch.Tensor, col0: torch.Tensor,
    row1: torch.Tensor, col1: torch.Tensor,
    y: torch.Tensor, x: torch.Tensor,
    rad: float, h: int, w: int,
) -> torch.Tensor:
    """(N,) bool：两格 (row0,col0)/(row1,col1) **任一**不可通行（合并 gather）。

    语义与两次 _impassable 的 OR 逐位一致，但只做 1 次 gather（DCU 上
    小 gather 的 launch 开销是大头，合并后掩码的 gather 数减半）。
    """
    oob = ((row0 < 0) | (row0 >= h) | (col0 < 0) | (col0 >= w)
           | (row1 < 0) | (row1 >= h) | (col1 < 0) | (col1 >= w))
    idx = torch.stack([
        row0.clamp(0, h - 1) * w + col0.clamp(0, w - 1),
        row1.clamp(0, h - 1) * w + col1.clamp(0, w - 1),
    ], dim=-1)                                  # (N, 2)
    solid = blocked_flat.gather(1, idx)         # 一次 gather 两格
    r0, r1 = (y - rad).floor().long(), (y + rad).ceil().long()
    c0, c1 = (x - rad).floor().long(), (x + rad).ceil().long()
    in0 = (row0 >= r0) & (row0 < r1) & (col0 >= c0) & (col0 < c1)
    in1 = (row1 >= r0) & (row1 < r1) & (col1 >= c0) & (col1 < c1)
    return oob | (solid[:, 0] & ~in0) | (solid[:, 1] & ~in1)


def _resolve_axis(
    coord: torch.Tensor,      # 移动轴上的新坐标（未消解碰撞）
    delta: torch.Tensor,      # 该轴位移，符号决定前进方向
    other: torch.Tensor,      # 另一轴坐标（本 tick 不变）
    y: torch.Tensor,          # 当前 y（判"脚下放行"用）
    x: torch.Tensor,          # 当前 x
    blocked_flat: torch.Tensor,
    rad: float,
    h: int,
    w: int,
    vertical: bool,
) -> torch.Tensor:
    """沿单轴消解碰撞：撞上就贴着障碍物停下（滑动），而不是整步作废。

    **不需要 substep**：位移 < 2 格时，碰撞盒前缘最多跨过 2 个格边界，所以
    只需枚举"旧前缘格 → 新前缘格"（最多 2 格），任一格是障碍就贴着它停。
    大位移（成长 1.5× → 0.45 格/tick）同样一次算完，不会穿模 —— 比如前缘
    从 3.7 挪到 4.1，若格 3 是障碍，检查 old_lead=3 即命中，停在 3 - rad 处，
    不会穿过它。下一 tick 还想往该方向走就被判定为无法前进。
    """
    sgn = torch.sign(delta)
    # 前缘 = 碰撞盒朝移动方向的那一边；扫过的格 = 旧前缘格..新前缘格
    old_lead = (coord - delta + sgn * rad).floor().long()
    new_lead = (coord + sgn * rad).floor().long()
    lo = torch.minimum(old_lead, new_lead)
    hi = torch.maximum(old_lead, new_lead)
    span0 = (other - rad).floor().long()               # 横跨的另一轴范围（最多 2）
    span1 = (other + rad).floor().long()

    def hit_at(lead: torch.Tensor) -> torch.Tensor:
        # span0/span1 两格合并成一次 gather（_impassable_pair）
        if vertical:
            return _impassable_pair(blocked_flat, lead, span0, lead, span1,
                                    y, x, rad, h, w)
        return _impassable_pair(blocked_flat, span0, lead, span1, lead,
                                y, x, rad, h, w)

    hit_lo = hit_at(lo)
    hit_hi = hit_at(hi)
    # 沿移动方向取**最近**的障碍格：先查新前缘格（角色想去的方向），
    # 撞上它就贴着停。新前缘 = hi（sgn>0）或 lo（sgn<0）。
    # 不能用哨兵值标记"无碰撞"（越界格索引可以是 -1，会冲突），
    # 用独立的 has 掩码 + 先新后旧两次检查。
    first_lead = torch.where(sgn > 0, hi, lo)
    second_lead = torch.where(sgn > 0, lo, hi)
    first_hit = torch.where(sgn > 0, hit_hi, hit_lo)
    second_hit = torch.where(sgn > 0, hit_lo, hit_hi)
    has = first_hit | second_hit
    first = torch.where(first_hit, first_lead,
                        torch.where(second_hit, second_lead, torch.zeros_like(lo)))
    stop_pos = torch.where(sgn > 0,
                           first.to(coord.dtype) - rad - _EPS,
                           first.to(coord.dtype) + 1.0 + rad + _EPS)
    return torch.where(has, stop_pos, coord)


def _impassable_pair_batch(
    r0: torch.Tensor, c0: torch.Tensor,
    r1: torch.Tensor, c1: torch.Tensor,
    yb: torch.Tensor, xb: torch.Tensor,
    blocked_flat: torch.Tensor, rad: float, h: int, w: int,
) -> torch.Tensor:
    """(N,K) 批量版 _impassable_pair：两格任一不可通行（一次 gather 两格）。

    与 _impassable_pair 逐位一致，但输入是 (N,K) 批量（r0/c0/r1/c1 是每探针
    的两格坐标，yb/xb 是 (N,K) **每探针自己的当前坐标**）。DCU 上小 kernel
    launch 是大头，legal_mask 的 10 次独立 _resolve_axis 合并成 2 次批量调用
    → kernel 数 ÷5。
    """
    n, k = r0.shape
    oob = ((r0 < 0) | (r0 >= h) | (c0 < 0) | (c0 >= w)
           | (r1 < 0) | (r1 >= h) | (c1 < 0) | (c1 >= w))            # (N,K)
    idx = torch.stack([
        r0.clamp(0, h - 1) * w + c0.clamp(0, w - 1),
        r1.clamp(0, h - 1) * w + c1.clamp(0, w - 1),
    ], dim=-1)                                                      # (N,K,2)
    solid = blocked_flat.gather(1, idx.reshape(n, -1)).reshape(n, k, 2)
    r0c = (yb - rad).floor().long()
    r1c = (yb + rad).ceil().long()
    c0c = (xb - rad).floor().long()
    c1c = (xb + rad).ceil().long()
    in0 = (r0 >= r0c) & (r0 < r1c) & (c0 >= c0c) & (c0 < c1c)
    in1 = (r1 >= r0c) & (r1 < r1c) & (c1 >= c0c) & (c1 < c1c)
    return oob | (solid[..., 0] & ~in0) | (solid[..., 1] & ~in1)


def _resolve_axis_batch(
    coord: torch.Tensor,      # (N,K) 移动轴上的新坐标（未消解）
    delta: torch.Tensor,      # (N,K) 该轴位移，符号决定前进方向
    other: torch.Tensor,      # (N,K) 另一轴坐标（本 tick 不变）
    y: torch.Tensor,          # (N,1) 当前 y（判"脚下放行"）
    x: torch.Tensor,          # (N,1) 当前 x
    blocked_flat: torch.Tensor,
    rad: float,
    h: int,
    w: int,
    vertical: bool,
) -> torch.Tensor:
    """(N,K) 批量版 _resolve_axis —— 与单点版逐位一致，kernel 数 ÷K。

    用于 legal_mask 的 5 方向探针（K=2P 垂直 + 2P 水平）和 move_players 的
    2P 次消解。内部数学与 _resolve_axis 完全相同，只是把 (N,) 变成 (N,K)，
    gather 批量化成一次 (N,K*2)。y/x 传 (N,K)（**每探针自己的当前坐标**，
    不能广播——不同玩家的坐标不同）。
    """
    sgn = torch.sign(delta)
    old_lead = (coord - delta + sgn * rad).floor().long()
    new_lead = (coord + sgn * rad).floor().long()
    lo = torch.minimum(old_lead, new_lead)
    hi = torch.maximum(old_lead, new_lead)
    span0 = (other - rad).floor().long()
    span1 = (other + rad).floor().long()
    if vertical:
        hit_lo = _impassable_pair_batch(lo, span0, lo, span1, y, x,
                                        blocked_flat, rad, h, w)
        hit_hi = _impassable_pair_batch(hi, span0, hi, span1, y, x,
                                        blocked_flat, rad, h, w)
    else:
        hit_lo = _impassable_pair_batch(span0, lo, span1, lo, y, x,
                                        blocked_flat, rad, h, w)
        hit_hi = _impassable_pair_batch(span0, hi, span1, hi, y, x,
                                        blocked_flat, rad, h, w)
    first_lead = torch.where(sgn > 0, hi, lo)
    second_lead = torch.where(sgn > 0, lo, hi)
    first_hit = torch.where(sgn > 0, hit_hi, hit_lo)
    second_hit = torch.where(sgn > 0, hit_lo, hit_hi)
    has = first_hit | second_hit
    first = torch.where(first_hit, first_lead,
                        torch.where(second_hit, second_lead,
                                    torch.zeros_like(lo)))
    stop_pos = torch.where(sgn > 0,
                           first.to(coord.dtype) - rad - _EPS,
                           first.to(coord.dtype) + 1.0 + rad + _EPS)
    return torch.where(has, stop_pos, coord)


@_dynamo_disable
def move_players(
    cfg: SimConfig,
    pos: torch.Tensor,        # (N, P, 2) float，格坐标（角色中心）
    move: torch.Tensor,       # (N, P) long，方向头动作
    alive: torch.Tensor,      # (N, P) bool
    blocked: torch.Tensor,    # (N, H, W) bool，墙 | 泡泡
    speed_mult: torch.Tensor | None = None,   # (N, P) float 或 None；None = 全 1.0
) -> torch.Tensor:
    """返回新的 pos。角色之间**不碰撞**（可穿过），只和墙/泡泡碰撞。

    角色互不碰撞是原作的行为，也顺手消掉了格子版里那段 O(P²) 的换位/同格消解。
    `speed_mult` 是**按玩家独立**的速度倍率（对打窗口里给玩家侧 +30% 用，
    AI 保持 1.0；训练不传它，行为与旧版逐位一致）。
    """
    n, p, _ = pos.shape
    h, w, rad, step = cfg.height, cfg.width, cfg.radius, cfg.step_len
    if speed_mult is None:
        speed_mult = torch.ones((n, p), dtype=pos.dtype, device=pos.device)
    blocked_flat = blocked.view(n, -1)
    out = pos.clone()
    tbl = _step_table(device=pos.device, dtype=pos.dtype, step=step)

    # 批量消解：2P 次 _resolve_axis（每玩家 y/x 各一）→ 垂直/水平各一次 (N,P)
    # 批量调用（_resolve_axis_batch，数学逐位一致）—— DCU 小 kernel launch
    # 是大头，kernel 数 ÷P。
    delta = tbl[move.clamp(0, N_MOVES - 1)]                 # (N,P,2)
    delta = delta * speed_mult.unsqueeze(-1)
    moving = alive & (move != MOVE_IDLE)
    delta = torch.where(moving.unsqueeze(-1), delta, torch.zeros_like(delta))
    dy, dx = delta[..., 0], delta[..., 1]
    y, x = pos[..., 0], pos[..., 1]                          # (N,P)
    ny = _resolve_axis_batch(y + dy, dy, x, y, x,
                             blocked_flat, rad, h, w, True)
    nx = _resolve_axis_batch(x + dx, dx, y, y, x,
                             blocked_flat, rad, h, w, False)
    out[..., 0] = torch.where(dy != 0, ny, y)
    out[..., 1] = torch.where(dx != 0, nx, x)
    # 防御性边界夹紧：坐标保持在 [rad, h-rad]×[rad, w-rad] —— 碰撞盒最贴边
    # 但不出界。不能用格中心 [0.5, h-0.5]：贴边站 0.3 是合法姿势（碰撞盒
    # [0.0, 0.6]），钳到 0.5 会让掩码说"能动"、实际却动不了。
    # 这条防线真正挡的是 `_resolve_axis` 的 stop_pos 在边界格滑动时算出
    # `lead ± 1 + rad + EPS` 溢出地图（用户实测看到角色穿出界面）。
    out[..., 0] = out[..., 0].clamp(rad, h - rad)
    out[..., 1] = out[..., 1].clamp(rad, w - rad)
    return out


def center_cell(pos: torch.Tensor) -> torch.Tensor:
    """(N, P, 2) float → (N, P, 2) long：角色中心所在格。

    放泡位置和命中判定都用中心格：**命中盒故意比碰撞盒小**，
    否则贴着墙走的时候会被隔着墙的火焰蹭死，手感上说不通。
    """
    return pos.floor().long()
