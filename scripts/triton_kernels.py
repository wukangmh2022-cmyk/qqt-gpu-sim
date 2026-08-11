#!/usr/bin/env python3
"""手写 triton 融合 kernel（DCU 100k 目标的核心工程）。

架构：**每 env 一个 triton program**（5632 programs 并行），program 内做该
env 的完整计算 —— 消除 step 每 tick ~1000 个小 kernel 的 launch 开销
（DCU 实测：sim.step GPU 执行仅 3.85ms/tick，CPU launch 50.7ms，92% 是
launch）。当前实现 move_players（碰撞消解），逐位对比 torch 版验证。

注意：triton jit 内**不支持嵌套函数**，碰撞检查必须内联展开。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _move_players_kernel(
    pos_ptr, move_ptr, alive_ptr, blocked_ptr, out_ptr,
    speed_ptr,
    N,
    P: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    RAD: tl.constexpr, STEP: tl.constexpr, EPS: tl.constexpr,
):
    """每 program 一个 (env, player)。与 sim/move.py::move_players 逐位一致。

    blocked (N,H,W) bool → 展开 (N, H*W)。越界格不可通行。
    碰撞：前缘扫过格（lo..hi）任一不可通行 → 贴着停（sgn>0: lead-rad-EPS,
    sgn<0: lead+1+rad+EPS）；脚下放行（碰撞盒已覆盖的格不算障碍）。
    speed_ptr：每玩家速度倍率 (N,P) float（训练 = spd_g 成长速度，非 None）。
    """
    pid = tl.program_id(0)
    env = pid // P
    me = pid % P
    base = env * P * 2 + me * 2
    y = tl.load(pos_ptr + base)
    x = tl.load(pos_ptr + base + 1)
    act = tl.load(move_ptr + env * P + me)
    al = tl.load(alive_ptr + env * P + me)
    sm = tl.load(speed_ptr + env * P + me)

    # 方向→位移查表（与 _step_table 一致）：0上 1下 2左 3右 4 idle，×速度倍率
    S = STEP
    dy = tl.where(act == 0, -S, tl.where(act == 1, S, 0.0)) * sm
    dx = tl.where(act == 2, -S, tl.where(act == 3, S, 0.0)) * sm
    moving = al & (act != 4)
    dy = tl.where(moving, dy, 0.0)
    dx = tl.where(moving, dx, 0.0)

    # 脚下放行的碰撞盒范围（y/x 当前坐标，两轴共用）
    r0c = tl.floor(y - RAD).to(tl.int32)
    r1c = tl.floor(y + RAD).to(tl.int32)
    c0c = tl.floor(x - RAD).to(tl.int32)
    c1c = tl.floor(x + RAD).to(tl.int32)

    # ---- Y 轴消解（vertical）----
    coord_y = y + dy
    sgn_y = tl.where(dy > 0, 1.0, tl.where(dy < 0, -1.0, 0.0))
    old_lead = tl.floor(coord_y - dy + sgn_y * RAD).to(tl.int32)
    new_lead = tl.floor(coord_y + sgn_y * RAD).to(tl.int32)
    lo_y = tl.minimum(old_lead, new_lead)
    hi_y = tl.maximum(old_lead, new_lead)
    span0 = tl.floor(x - RAD).to(tl.int32)
    span1 = tl.floor(x + RAD).to(tl.int32)

    # ---- hit_at(lo_y)：两格 (lo_y, span0)/(lo_y, span1) ----
    idx0 = lo_y * W + span0
    idx1 = lo_y * W + span1
    oob = (lo_y < 0) | (lo_y >= H) | (span0 < 0) | (span0 >= W) \
        | (span1 < 0) | (span1 >= W)
    b0 = tl.load(blocked_ptr + env * H * W + idx0,
                 mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
    b1 = tl.load(blocked_ptr + env * H * W + idx1,
                 mask=(idx1 >= 0) & (idx1 < H * W), other=0.0)
    in_row = (lo_y >= r0c) & (lo_y <= r1c)
    in0 = in_row & (span0 >= c0c) & (span0 <= c1c)
    in1 = in_row & (span1 >= c0c) & (span1 <= c1c)
    h0 = tl.where(in0, 0.0, b0)
    h1 = tl.where(in1, 0.0, b1)
    hit_lo_y = tl.where(oob, 1.0, tl.where((h0 > 0.5) | (h1 > 0.5), 1.0, 0.0))
    # ---- hit_at(hi_y) ----
    idx0 = hi_y * W + span0
    idx1 = hi_y * W + span1
    oob = (hi_y < 0) | (hi_y >= H) | (span0 < 0) | (span0 >= W) \
        | (span1 < 0) | (span1 >= W)
    b0 = tl.load(blocked_ptr + env * H * W + idx0,
                 mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
    b1 = tl.load(blocked_ptr + env * H * W + idx1,
                 mask=(idx1 >= 0) & (idx1 < H * W), other=0.0)
    in_row = (hi_y >= r0c) & (hi_y <= r1c)
    in0 = in_row & (span0 >= c0c) & (span0 <= c1c)
    in1 = in_row & (span1 >= c0c) & (span1 <= c1c)
    h0 = tl.where(in0, 0.0, b0)
    h1 = tl.where(in1, 0.0, b1)
    hit_hi_y = tl.where(oob, 1.0, tl.where((h0 > 0.5) | (h1 > 0.5), 1.0, 0.0))

    first_lead_y = tl.where(sgn_y > 0, hi_y, lo_y)
    second_lead_y = tl.where(sgn_y > 0, lo_y, hi_y)
    first_hit_y = tl.where(sgn_y > 0, hit_hi_y, hit_lo_y)
    second_hit_y = tl.where(sgn_y > 0, hit_lo_y, hit_hi_y)
    has_y = (first_hit_y > 0.5) | (second_hit_y > 0.5)
    first_y = tl.where(first_hit_y > 0.5, first_lead_y,
                       tl.where(second_hit_y > 0.5, second_lead_y, 0))
    stop_y = tl.where(sgn_y > 0, first_y.to(tl.float32) - RAD - EPS,
                      first_y.to(tl.float32) + 1.0 + RAD + EPS)
    ny = tl.where(has_y, stop_y, coord_y)
    ny = tl.where(dy != 0.0, ny, y)

    # ---- X 轴消解（horizontal，对称）----
    coord_x = x + dx
    sgn_x = tl.where(dx > 0, 1.0, tl.where(dx < 0, -1.0, 0.0))
    old_lead_x = tl.floor(coord_x - dx + sgn_x * RAD).to(tl.int32)
    new_lead_x = tl.floor(coord_x + sgn_x * RAD).to(tl.int32)
    lo_x = tl.minimum(old_lead_x, new_lead_x)
    hi_x = tl.maximum(old_lead_x, new_lead_x)
    span0x = tl.floor(y - RAD).to(tl.int32)
    span1x = tl.floor(y + RAD).to(tl.int32)

    # ---- hit_at(lo_x)：两格 (span0x, lo_x)/(span1x, lo_x) ----
    idx0 = span0x * W + lo_x
    idx1 = span1x * W + lo_x
    oob = (lo_x < 0) | (lo_x >= W) | (span0x < 0) | (span0x >= H) \
        | (span1x < 0) | (span1x >= H)
    b0 = tl.load(blocked_ptr + env * H * W + idx0,
                 mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
    b1 = tl.load(blocked_ptr + env * H * W + idx1,
                 mask=(idx1 >= 0) & (idx1 < H * W), other=0.0)
    in_col = (lo_x >= c0c) & (lo_x <= c1c)
    in0 = in_col & (span0x >= r0c) & (span0x <= r1c)
    in1 = in_col & (span1x >= r0c) & (span1x <= r1c)
    h0 = tl.where(in0, 0.0, b0)
    h1 = tl.where(in1, 0.0, b1)
    hit_lo_x = tl.where(oob, 1.0, tl.where((h0 > 0.5) | (h1 > 0.5), 1.0, 0.0))
    # ---- hit_at(hi_x) ----
    idx0 = span0x * W + hi_x
    idx1 = span1x * W + hi_x
    oob = (hi_x < 0) | (hi_x >= W) | (span0x < 0) | (span0x >= H) \
        | (span1x < 0) | (span1x >= H)
    b0 = tl.load(blocked_ptr + env * H * W + idx0,
                 mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
    b1 = tl.load(blocked_ptr + env * H * W + idx1,
                 mask=(idx1 >= 0) & (idx1 < H * W), other=0.0)
    in_col = (hi_x >= c0c) & (hi_x <= c1c)
    in0 = in_col & (span0x >= r0c) & (span0x <= r1c)
    in1 = in_col & (span1x >= r0c) & (span1x <= r1c)
    h0 = tl.where(in0, 0.0, b0)
    h1 = tl.where(in1, 0.0, b1)
    hit_hi_x = tl.where(oob, 1.0, tl.where((h0 > 0.5) | (h1 > 0.5), 1.0, 0.0))

    first_lead_x = tl.where(sgn_x > 0, hi_x, lo_x)
    second_lead_x = tl.where(sgn_x > 0, lo_x, hi_x)
    first_hit_x = tl.where(sgn_x > 0, hit_hi_x, hit_lo_x)
    second_hit_x = tl.where(sgn_x > 0, hit_lo_x, hit_hi_x)
    has_x = (first_hit_x > 0.5) | (second_hit_x > 0.5)
    first_x = tl.where(first_hit_x > 0.5, first_lead_x,
                       tl.where(second_hit_x > 0.5, second_lead_x, 0))
    stop_x = tl.where(sgn_x > 0, first_x.to(tl.float32) - RAD - EPS,
                      first_x.to(tl.float32) + 1.0 + RAD + EPS)
    nx = tl.where(has_x, stop_x, coord_x)
    nx = tl.where(dx != 0.0, nx, x)

    # 防御性边界夹紧（与 torch 版一致）
    ny = tl.minimum(tl.maximum(ny, RAD), H - RAD)
    nx = tl.minimum(tl.maximum(nx, RAD), W - RAD)
    tl.store(out_ptr + base, ny)
    tl.store(out_ptr + base + 1, nx)


def move_players_triton(cfg, pos, move, alive, blocked, speed_mult=None):
    """triton 版 move_players（与 sim/move.py 逐位一致）。

    speed_mult 支持 None（全 1.0）或 (N,P) 倍率（训练传 spd_g）。返回新 pos。
    **必须 contiguous**：triton 的指针算术假设连续布局，非连续 view（如
    acts[..., 0]）会读错内存（2026-08-10 实测 P1 act 读成 0）。
    """
    n, p, _ = pos.shape
    h, w = cfg.height, cfg.width
    rad = cfg.radius
    step = cfg.step_len
    move = move.contiguous()
    alive = alive.contiguous()
    blocked = blocked.contiguous()
    pos_c = pos.contiguous()
    if speed_mult is None:
        speed_mult = torch.ones((n, p), dtype=pos.dtype, device=pos.device)
    else:
        speed_mult = speed_mult.contiguous()
    out = torch.empty_like(pos_c)
    grid = (n * p,)
    _move_players_kernel[grid](
        pos_c, move, alive, blocked, out, speed_mult, n,
        P=p, H=h, W=w, RAD=rad, STEP=step, EPS=1e-4,
    )
    return out


@triton.jit
def _danger_b_kernel(
    weight_ptr, blast_ptr, bombed_ptr, passable_ptr, not_solid_ptr, danger_ptr,
    N,
    H: tl.constexpr, W: tl.constexpr, MAXB: tl.constexpr, BLOCK: tl.constexpr,
):
    """danger 阶段 B：每 env 一个 program，BLOCK×BLOCK tile（13×13 padding）。

    语义（与 blast.py 阶段 B 逐位一致）：danger[r,c] = max over 4 方向、
    距离 s ≤ blast[src] 的炮格 src（bombed 且 blast ≥ s），路径（src 与
    (r,c) 之间的 s-1 格）全部 non-solid（泡/砖挡火）。炮格自身权重入 danger。
    "每格检查法"（无迭代/无 barrier）：每格独立看 4 方向 × MAXB 距离。
    BLOCK 必须 ≥ max(H,W) 且 2 的幂（tl.arange 限制）——13×13 用 16。
    """
    env = tl.program_id(0)
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    rmask = rows < H
    cmask = cols < W
    valid = rmask & cmask
    offs = env * H * W + rows * W + cols
    w = tl.load(weight_ptr + offs, mask=valid, other=0.0)     # 炮格权重（修正后）
    bl = tl.load(blast_ptr + offs, mask=valid, other=0.0)     # 每格 blast 档
    bm = tl.load(bombed_ptr + offs, mask=valid, other=0.0)    # 炮格 mask
    pa = tl.load(passable_ptr + offs, mask=valid, other=0.0)
    ns = tl.load(not_solid_ptr + offs, mask=valid, other=0.0)  # 1 = 非泡非砖
    danger = w * pa                           # 炮格自身（×passable 同生产）

    # 4 方向展开（上/下/左/右），每方向 s=1..MAXB
    for d in range(4):
        if d == 0:
            dy, dx = -1, 0
        elif d == 1:
            dy, dx = 1, 0
        elif d == 2:
            dy, dx = 0, -1
        else:
            dy, dx = 0, 1
        # 路径干净性（中间格 non-solid），随 s 累积
        path_ok = (w * 0 + 1).to(tl.int32)   # 全 1（triton 无 ones_like）
        for s in range(1, MAXB + 1):
            # 更新路径：第 s-1 格（(r,c)+dir*(s-1)）需 non-solid
            if s > 1:
                mr = rows + dy * (s - 1)
                mc = cols + dx * (s - 1)
                minb = (mr >= 0) & (mr < H) & (mc >= 0) & (mc < W)
                mr_c = tl.minimum(tl.maximum(mr, 0), H - 1)
                mc_c = tl.minimum(tl.maximum(mc, 0), W - 1)
                ns_mid = tl.load(not_solid_ptr + env * H * W + mr_c * W + mc_c,
                                 mask=minb & valid, other=0.0)
                path_ok = path_ok & (ns_mid > 0.5).to(tl.int32)
            # src = (r,c) + dir*s
            r_s = rows + dy * s
            c_s = cols + dx * s
            inb = (r_s >= 0) & (r_s < H) & (c_s >= 0) & (c_s < W)
            r_sc = tl.minimum(tl.maximum(r_s, 0), H - 1)
            c_sc = tl.minimum(tl.maximum(c_s, 0), W - 1)
            src_offs = env * H * W + r_sc * W + c_sc
            w_src = tl.load(weight_ptr + src_offs, mask=inb & valid, other=0.0)
            bl_src = tl.load(blast_ptr + src_offs, mask=inb & valid, other=0.0)
            bm_src = tl.load(bombed_ptr + src_offs, mask=inb & valid, other=0.0)
            # src 是炮（bm>0）且 blast ≥ s 且路径干净 → 覆盖
            ok = (bm_src > 0.5) & (bl_src >= s) & (path_ok > 0) & inb
            cand = tl.where(ok, w_src, 0.0)
            danger = tl.maximum(danger, cand)

    tl.store(danger_ptr + offs, danger, mask=valid)


def danger_b_triton(weight, blast, bombed, passable, not_solid, max_b):
    """triton 版 danger 阶段 B（输入 = 修正后权重等，输出 (N,H,W) danger）。"""
    n, h, w = weight.shape
    danger = torch.empty_like(weight)
    grid = (n,)
    block = 1
    while block < max(h, w):
        block *= 2
    _danger_b_kernel[grid](
        weight.contiguous(), blast.contiguous(), bombed.contiguous(),
        passable.contiguous(), not_solid.contiguous(), danger, n,
        H=h, W=w, MAXB=max_b, BLOCK=block,
    )
    return danger


@triton.jit
def _dijkstra_kernel(
    src_ptr, danger_ptr, blocked_ptr, cost_ptr, dist_ptr, tmp_ptr,
    N,
    H: tl.constexpr, W: tl.constexpr, LAM: tl.constexpr,
    MAXP: tl.constexpr, BLOCK: tl.constexpr,
):
    """多源 Dijkstra（Bellman-Ford）：每 env 一个 program，BLOCK tile + barrier。

    语义与 _dijkstra_field 一致：V(c) = 到最近 source 的最短代价，进入格 c
    的代价 = 1 + LAM×danger[c]，blocked（泡/墙）不可通行。
    趟间同步用全局 buffer ping-pong + tl.debug_barrier（block 内 169 线程）。
    MAXP = 网格直径上界（13×13 → 24 趟收敛）。
    """
    env = tl.program_id(0)
    rows = tl.arange(0, BLOCK)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    rmask = rows < H
    cmask = cols < W
    valid = rmask & cmask
    offs = env * H * W + rows * W + cols
    # 初始 dist：source 格 = 0，其余 inf；cost = 1+lam*danger，blocked → inf
    s = tl.load(src_ptr + offs, mask=valid, other=0.0)
    dn = tl.load(danger_ptr + offs, mask=valid, other=0.0)
    bk = tl.load(blocked_ptr + offs, mask=valid, other=0.0)
    c = 1.0 + LAM * dn
    c = tl.where(bk > 0.5, 1e9, c)
    d0 = tl.where(s > 0.5, 0.0, 1e9)
    d0 = tl.where(valid, d0, 1e9)
    tl.store(dist_ptr + offs, d0)
    tl.store(cost_ptr + offs, c, mask=valid)
    tl.debug_barrier()
    d2 = d0
    for _ in range(MAXP):
        d = tl.load(dist_ptr + offs, mask=valid, other=1e9, volatile=True)
        # 4 邻松弛：d <= min over neighbors of d[nb] + cost[本格]（进入本格代价，
        # 与 _dijkstra_field 的 `nd + cost` 一致 —— 用本格 cost 不用邻格）
        for ddir in range(4):
            if ddir == 0:
                dy, dx = -1, 0
            elif ddir == 1:
                dy, dx = 1, 0
            elif ddir == 2:
                dy, dx = 0, -1
            else:
                dy, dx = 0, 1
            nr = rows + dy
            nc = cols + dx
            nb_ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W) & valid
            nr_c = tl.minimum(tl.maximum(nr, 0), H - 1)
            nc_c = tl.minimum(tl.maximum(nc, 0), W - 1)
            nb_offs = env * H * W + nr_c * W + nc_c
            d_nb = tl.load(dist_ptr + nb_offs, mask=nb_ok, other=1e9, volatile=True)
            cand = d_nb + c
            cand = tl.where(nb_ok, cand, 1e9)
            d = tl.minimum(d, cand)
        tl.store(tmp_ptr + offs, d, mask=valid)
        tl.debug_barrier()
        d2 = tl.load(tmp_ptr + offs, mask=valid, other=1e9)
        tl.store(dist_ptr + offs, d2)
        tl.debug_barrier()
    tl.store(dist_ptr + offs, d2, mask=valid)


def dijkstra_triton(sim, sources, danger, lam=2.0, max_passes=None,
                    blocked=None):
    """triton 版多源 Dijkstra（与 _dijkstra_field 一致）。sources (N,H,W)。"""
    n, h, w = danger.shape
    if blocked is None:
        blocked = sim.wall | sim.brick
    dist = torch.empty(n, h, w, dtype=torch.float32, device=danger.device)
    tmp = torch.empty_like(dist)
    cost = torch.empty_like(dist)
    grid = (n,)
    block = 1
    while block < max(h, w):
        block *= 2
    maxp = max_passes or (h * w)
    _dijkstra_kernel[grid](
        sources.contiguous(), danger.contiguous(), blocked.contiguous(),
        cost, dist, tmp, n,
        H=h, W=w, LAM=lam, MAXP=maxp, BLOCK=block,
    )
    # 不可达格（1e9 哨兵）→ inf（与 _dijkstra_field 一致，astar 的 isfinite 依赖）
    dist = torch.where(dist >= 1e8, torch.full_like(dist, float("inf")), dist)
    return dist.reshape(n, -1)
