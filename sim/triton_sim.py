"""triton 化模拟核心 kernel —— 完全 GPU 端（每 tick 少数 launch，无 CPU 参与）。

硬件无关（MPS / HIP / Ascend / CUDA 同一份代码），本地 MPS 验证逻辑 →
Ascend 910B（triton-ascend）直接跑。每 kernel 与 sim/torch_sim.py /
sim/blast.py / sim/move.py 的参考实现逐位一致（各自有 verify 脚本）。

当前包含：
  - move_players_triton   移动（AABB 滑动碰撞，56-90x，DCU 已验证）
  - explode_triton        爆炸传播（per-cell 4 方向扫描，bitwise 0）
  - place_bombs_triton    放泡（脚下格 scatter，设计见 TODO）
  - danger_stageB_triton  危险图阶段 B（per-cell 权重传播，TODO）

状态传递用指针（in-place / out buffer），无 Python 循环。
"""
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # 本地 MPS 编译好之前 / 无 triton 环境
    _HAS_TRITON = False


# ---------------- 移动（AABB 滑动碰撞）----------------

if _HAS_TRITON:
    @triton.jit
    def _move_players_kernel(
        pos_ptr, move_ptr, alive_ptr, blocked_ptr, out_ptr,
        speed_ptr, N,
        P: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
        RAD: tl.constexpr, STEP: tl.constexpr, EPS: tl.constexpr,
    ):
        """每 program 一个 (env, player)。与 sim/move.py::move_players 逐位一致。"""
        pid = tl.program_id(0)
        env = pid // P
        me = pid % P
        base = env * P * 2 + me * 2
        y = tl.load(pos_ptr + base)
        x = tl.load(pos_ptr + base + 1)
        act = tl.load(move_ptr + env * P + me)
        al = tl.load(alive_ptr + env * P + me)
        sm = tl.load(speed_ptr + env * P + me)

        S = STEP
        dy = tl.where(act == 0, -S, tl.where(act == 1, S, 0.0)) * sm
        dx = tl.where(act == 2, -S, tl.where(act == 3, S, 0.0)) * sm
        moving = al & (act != 4)
        dy = tl.where(moving, dy, 0.0)
        dx = tl.where(moving, dx, 0.0)

        r0c = tl.floor(y - RAD).to(tl.int32)
        r1c = tl.floor(y + RAD).to(tl.int32)
        c0c = tl.floor(x - RAD).to(tl.int32)
        c1c = tl.floor(x + RAD).to(tl.int32)

        # ---- Y 轴消解 ----
        coord_y = y + dy
        sgn_y = tl.where(dy > 0, 1.0, tl.where(dy < 0, -1.0, 0.0))
        old_lead = tl.floor(coord_y - dy + sgn_y * RAD).to(tl.int32)
        new_lead = tl.floor(coord_y + sgn_y * RAD).to(tl.int32)
        lo_y = tl.minimum(old_lead, new_lead)
        hi_y = tl.maximum(old_lead, new_lead)
        span0 = tl.floor(x - RAD).to(tl.int32)
        span1 = tl.floor(x + RAD).to(tl.int32)

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

        # ---- X 轴消解（对称）----
        coord_x = x + dx
        sgn_x = tl.where(dx > 0, 1.0, tl.where(dx < 0, -1.0, 0.0))
        old_lead_x = tl.floor(coord_x - dx + sgn_x * RAD).to(tl.int32)
        new_lead_x = tl.floor(coord_x + sgn_x * RAD).to(tl.int32)
        lo_x = tl.minimum(old_lead_x, new_lead_x)
        hi_x = tl.maximum(old_lead_x, new_lead_x)
        span0x = tl.floor(y - RAD).to(tl.int32)
        span1x = tl.floor(y + RAD).to(tl.int32)

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

        ny = tl.minimum(tl.maximum(ny, RAD), H - RAD)
        nx = tl.minimum(tl.maximum(nx, RAD), W - RAD)
        tl.store(out_ptr + base, ny)
        tl.store(out_ptr + base + 1, nx)

    @triton.jit
    def _explode_kernel(src, wall, bombed, brick, blst, out,
                        HW, NENV, H: tl.constexpr, W: tl.constexpr,
                        BMAX: tl.constexpr, BLOCK: tl.constexpr):
        """per-cell 4 方向扫描：1 kernel 替代 rays 的 ~4×Σb 次 shift。

        与 blast.rays 逐位一致（已 bitwise 验证）。wall 格不可覆盖；炮/brick
        覆盖但不穿透（更远的格被挡）。BMAX 固定（growth_blast_max=7）。
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = HW * NENV
        m = offs < total
        nenv = offs // HW
        cidx = offs % HW
        h = cidx // W
        w = cidx % W
        base = nenv * HW + cidx
        covered = tl.load(src + base, mask=m, other=False)
        wself = tl.load(wall + base, mask=m, other=True)
        brk_self = tl.load(brick + base, mask=m, other=False)
        covered = covered & ~wself & ~brk_self   # 源过滤：墙/砖不可作爆源（与 eager seed 一致）
        for d in tl.static_range(4):
            dr = -1 if d == 0 else (1 if d == 1 else 0)
            dc = -1 if d == 2 else (1 if d == 3 else 0)
            cum = tl.zeros((BLOCK,), tl.int1)
            for step in tl.static_range(1, BMAX + 1):
                gh = h + dr * step
                gw = w + dc * step
                ok = (gh >= 0) & (gh < H) & (gw >= 0) & (gw < W)
                gidx = nenv * HW + gh * W + gw
                gm = m & ok
                wv = tl.load(wall + gidx, mask=gm, other=True)
                sv = tl.load(src + gidx, mask=gm, other=False)
                bv = tl.load(blst + gidx, mask=gm, other=0)
                bmb = tl.load(bombed + gidx, mask=gm, other=False)
                brk = tl.load(brick + gidx, mask=gm, other=False)
                hit = ((sv & ~wv & ~brk) & (bv >= step)) & (cum == 0)
                covered = covered | hit
                blk = wv | bmb | brk
                cum = cum | blk
        tl.store(out + base, tl.where(wself, False, covered).to(tl.int8), mask=m)

    @triton.jit
    def _count_bombs_kernel(owner, fuse, out, HW, NENV,
                            P: tl.constexpr, BLOCK: tl.constexpr):
        """每 env 一个 program：统计每玩家在场泡数（owner==me & fuse>0）。

        输出 (NENV, P) int32。BLOCK 覆盖 H*W（pad 到 2 的幂）。
        """
        pid = tl.program_id(0)
        offs = pid * HW + tl.arange(0, BLOCK)
        m = offs < (pid + 1) * HW
        o = tl.load(owner + offs, mask=m, other=-1)
        f = tl.load(fuse + offs, mask=m, other=0)
        for me in tl.static_range(P):
            cnt = tl.sum((o == me) & (f > 0))
            tl.store(out + pid * P + me, cnt)

    @triton.jit
    def _place_bombs_kernel(fuse, owner, bomb_blast, pos, alive, bomb,
                            bombs_cap, blast_cap, brick, wall, live_count,
                            placed, N,
                            P: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                            FUSE: tl.constexpr):
        """放泡：每 program 一个 (env, player)。脚下格 fuse==0 & 非墙/砖 &
        在场泡数（_count_bombs_kernel 预计算）< 上限 → 写 fuse/owner/blast。
        """
        pid = tl.program_id(0)
        env = pid // P
        me = pid % P
        y = tl.load(pos + env * P * 2 + me * 2)
        x = tl.load(pos + env * P * 2 + me * 2 + 1)
        r = tl.floor(y).to(tl.int32)
        c = tl.floor(x).to(tl.int32)
        idx = r * W + c
        cur_fuse = tl.load(fuse + env * H * W + idx)
        cur_owner = tl.load(owner + env * H * W + idx)
        cur_blast = tl.load(bomb_blast + env * H * W + idx)
        on_brick = tl.load(brick + env * H * W + idx)
        on_wall = tl.load(wall + env * H * W + idx)
        al = tl.load(alive + env * P + me)
        bm = tl.load(bomb + env * P + me)
        cap = tl.load(bombs_cap + env * P + me)
        bl_cap = tl.load(blast_cap + env * P + me)
        own = tl.load(live_count + env * P + me)      # 在场泡数（预计算）
        ok = al & bm & (cur_fuse <= 0) & (on_brick == 0) & (on_wall == 0) \
            & (own < cap)
        tl.store(placed + env * P + me, ok.to(tl.int8))
        tl.store(fuse + env * H * W + idx,
                 tl.where(ok, FUSE, cur_fuse))
        tl.store(owner + env * H * W + idx,
                 tl.where(ok, me, cur_owner))
        tl.store(bomb_blast + env * H * W + idx,
                 tl.where(ok, bl_cap, cur_blast))

    @triton.jit
    def _danger_kernel(fuse, wall, bombed, brick, blst, out,
                       HW, NENV, H: tl.constexpr, W: tl.constexpr,
                       BMAX: tl.constexpr, BLOCK: tl.constexpr,
                       FUSE_MAX: tl.constexpr, EXP: tl.constexpr):
        """危险图阶段 B（per-cell 权重传播，与 danger_map(max_chain=1) 一致）。

        每 cell 4 方向扫描：源 s（fuse>0 的泡）的权重 weight[s] 传播 blast[s]
        格，途中无墙/泡/brick 挡 → danger[c] = max(weight[s])。权重在 kernel 内
        算（w_raw=(1-(fuse-1)/FUSE)^EXP，与 torch danger_map 同）。墙格恒 0。
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = HW * NENV
        m = offs < total
        nenv = offs // HW
        cidx = offs % HW
        h = cidx // W
        w = cidx % W
        base = nenv * HW + cidx
        fuse_v = tl.load(fuse + base, mask=m, other=0)
        wself = tl.load(wall + base, mask=m, other=True)
        # 权重（仅泡格）：w_raw=(1-(f-1)/FUSE).clamp(0)^EXP
        w_raw = 1.0 - (fuse_v.to(tl.float32) - 1.0) / FUSE_MAX
        w_c = tl.maximum(w_raw, 0.0)
        if EXP == 2.0:                       # 训练默认 exp=2（乘法最稳）
            p_w = w_c * w_c
        else:
            p_w = tl.exp(EXP * tl.log(w_c))
        weight = tl.where(fuse_v > 0, p_w, 0.0)
        dng = tl.where(wself, 0.0, weight)          # 墙格恒 0；自身权重为 seed
        for d in tl.static_range(4):
            dr = -1 if d == 0 else (1 if d == 1 else 0)
            dc = -1 if d == 2 else (1 if d == 3 else 0)
            cum = tl.zeros((BLOCK,), tl.int1)
            for step in tl.static_range(1, BMAX + 1):
                gh = h + dr * step
                gw = w + dc * step
                ok = (gh >= 0) & (gh < H) & (gw >= 0) & (gw < W)
                gidx = nenv * HW + gh * W + gw
                gm = m & ok
                wv = tl.load(wall + gidx, mask=gm, other=True)
                sv = tl.load(fuse + gidx, mask=gm, other=0)
                bv = tl.load(blst + gidx, mask=gm, other=0)
                bmb = tl.load(bombed + gidx, mask=gm, other=False)
                brk = tl.load(brick + gidx, mask=gm, other=False)
                # 源的权重（同款公式）
                s_c = tl.maximum(1.0 - (sv.to(tl.float32) - 1.0) / FUSE_MAX, 0.0)
                if EXP == 2.0:
                    p_s = s_c * s_c
                else:
                    p_s = tl.exp(EXP * tl.log(s_c))
                sw = tl.where(sv > 0, p_s, 0.0)
                cand = tl.where((sv > 0) & (bv >= step) & (cum == 0), sw, 0.0)
                dng = tl.maximum(dng, cand)
                cum = cum | (wv | bmb | brk)
        tl.store(out + base, tl.where(wself, 0.0, dng), mask=m)


# ---------------- wrappers ----------------

def move_players_triton(cfg, pos, move, alive, blocked, speed_mult=None):
    """triton 版 move_players（与 sim/move.py 逐位一致）。"""
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
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


def explode_triton(src, wall, bombed, brick, blast, b_max=7):
    """爆炸传播（与 blast.rays 逐位一致）。返回 bool covered。

    src/wall/bombed/brick: (N,H,W) bool；blast: (N,H,W) int（每格档位）。
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
    n, h, w = src.shape
    hw = h * w
    src_c = src.contiguous()
    wall_c = wall.contiguous()
    bombed_c = bombed.contiguous()
    brick_c = brick.contiguous()
    blst_c = blast.to(torch.int32).contiguous()
    out = torch.empty((n, h, w), dtype=torch.int8, device=src.device)
    BLOCK = 1024
    grid = ((hw * n + BLOCK - 1) // BLOCK,)
    _explode_kernel[grid](
        src_c, wall_c, bombed_c, brick_c, blst_c, out,
        hw, n, h, w, BMAX=max(1, int(b_max)), BLOCK=BLOCK,
    )
    return out.bool()


def count_bombs_triton(owner, fuse):
    """每 env 每玩家在场泡数（owner==me & fuse>0）。返回 (N,P) int32。"""
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
    n, h, w = fuse.shape
    hw = h * w
    p = owner.shape[1] if owner.dim() == 2 else 2
    out = torch.empty((n, p), dtype=torch.int32, device=fuse.device)
    BLOCK = 1 << (hw - 1).bit_length()
    _count_bombs_kernel[(n,)](owner.contiguous(), fuse.contiguous(), out,
                              hw, n, P=p, BLOCK=BLOCK)
    return out


def place_bombs_triton(cfg, fuse, owner, bomb_blast, pos, alive, bomb,
                       bombs_cap, blast_cap, brick, wall):
    """放泡（计数 kernel + 放泡 kernel，2 个 launch）。与 torch _place_bombs 一致。

    返回 placed (N,P) bool（放置成功掩码，since_bomb 清零用）。
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
    n = fuse.shape[0]
    p = cfg.n_players
    h, w = cfg.height, cfg.width
    hw = h * w
    live_count = torch.empty((n, p), dtype=torch.int32,
                             device=fuse.device)
    BLOCK = 1 << (hw - 1).bit_length()
    _count_bombs_kernel[(n,)](
        owner.contiguous(), fuse.contiguous(), live_count, hw, n,
        P=p, BLOCK=BLOCK)
    placed = torch.zeros((n, p), dtype=torch.int8, device=fuse.device)
    _place_bombs_kernel[(n * p,)](
        fuse.contiguous(), owner.contiguous(), bomb_blast.contiguous(),
        pos.contiguous(), alive.contiguous(), bomb.contiguous(),
        bombs_cap.contiguous(), blast_cap.contiguous(),
        brick.contiguous(), wall.contiguous(), live_count, placed, n,
        P=p, H=h, W=w, FUSE=cfg.fuse)
    return placed.bool()


def resolve_triton(fuse, owner, wall, bomb_blast, brick, max_chain=16,
                  b_max=7):
    """爆炸与连锁（triton explode kernel 链）。返回 (covered, triggered)。

    连锁轮固定（不早退——XLA/triton 图内无同步），与固定轮 torch 逐位一致。
    """
    import torch
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    covered = explode_triton(triggered, wall, live, brick, bomb_blast, b_max)
    for _ in range(max_chain - 1):
        newly = live & covered & ~triggered
        covered = covered | explode_triton(newly, wall, live, brick,
                                           bomb_blast, b_max)
        triggered = triggered | newly
    return covered, triggered


def danger_triton(fuse, wall, bombed, brick, blast, fuse_max,
                  b_max=7, exp=2.0):
    """危险图阶段 B（per-cell 权重传播，与 danger_map(max_chain=1) 一致）。

    返回 float32 (N,H,W) 危险图。阶段 A（连锁修正）待补。
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
    n, h, w = fuse.shape
    hw = h * w
    out = torch.empty((n, h, w), dtype=torch.float32, device=fuse.device)
    BLOCK = 1024
    grid = ((hw * n + BLOCK - 1) // BLOCK,)
    _danger_kernel[grid](
        fuse.contiguous(), wall.contiguous(), bombed.contiguous(),
        brick.contiguous(), blast.to(torch.int32).contiguous(), out,
        hw, n, h, w, BMAX=max(1, int(b_max)), BLOCK=BLOCK,
        FUSE_MAX=int(fuse_max), EXP=float(exp))
    return out
