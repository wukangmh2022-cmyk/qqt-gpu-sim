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
        # 2D grid (p, n)：program_id(1)=env、program_id(0)=player。
        # 1D grid (n*p,) 在 N≥32768（grid≥65536）触发 triton-ascend 上限
        # （"grid should be less than 65536"，2026-08-11 实测）——拆二维后
        # 每维 < 65536，N 无上限，结果逐 program 独立位级不变。
        pid = tl.program_id(1) * P + tl.program_id(0)
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
        b0 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx0, 0), H * W - 1),
                     mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
        b1 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx1, 0), H * W - 1),
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
        b0 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx0, 0), H * W - 1),
                     mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
        b1 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx1, 0), H * W - 1),
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
        b0 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx0, 0), H * W - 1),
                     mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
        b1 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx1, 0), H * W - 1),
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
        b0 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx0, 0), H * W - 1),
                     mask=(idx0 >= 0) & (idx0 < H * W), other=0.0)
        b1 = tl.load(blocked_ptr + env * H * W + tl.minimum(tl.maximum(idx1, 0), H * W - 1),
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
                gidx = tl.minimum(tl.maximum(gidx, 0), HW * NENV - 1)  # 防御 OOB
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
        offs = tl.minimum(offs, NENV * HW - 1)   # 防御 OOB（mask 仍按原 offs 判）
        m = offs < (pid + 1) * HW
        o = tl.load(owner + offs, mask=m, other=-1)
        f = tl.load(fuse + offs, mask=m, other=0)
        for me in tl.static_range(P):
            # triton-ascend 3.2.0: tl.sum(bool) 结果错（恒 1）→ 先转 int32 再 sum
            cnt = tl.sum(tl.where((o == me) & (f > 0), 1, 0))
            tl.store(out + pid * P + me, cnt)

    @triton.jit
    def _place_bombs_kernel(fuse, owner, bomb_blast, pos, alive, bomb,
                            bombs_cap, blast_cap, brick, wall, live_count,
                            placed, N,
                            P: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                            FUSE: tl.constexpr):
        """放泡：每 program 一个 **env**，内层按玩家顺序处理（grid=(N,)）。

        **为什么不是每 program 一个 (env, player)**：pl 0 / pl 1 可能同格
        （角色不碰撞），两人同 tick 都按放泡键时，torch 顺序循环里 pl 1 能看到
        pl 0 刚写的 fuse=30 → 拒绝（脚下已有泡不能叠放）。若 pl 0/1 分属两个
        并发 program，pl 1 的 `cur_fuse<=0` 读可能赶在 pl 0 写之前 → 同格叠放
        （910B 实测 placed 差：env 397 双人同格 (8,7)，triton 叠放、torch 拒绝）。
        内层 static_range(P) 顺序处理：程序内 store→load 同址可见，逐位复刻
        torch `_place_bombs` 的 for 循环语义。
        """
        env = tl.program_id(0)
        for me in tl.static_range(P):
            y = tl.load(pos + env * P * 2 + me * 2)
            x = tl.load(pos + env * P * 2 + me * 2 + 1)
            # triton-ascend 3.2.0：标量 tl.floor 在存储地址链上编译失败
            # （unresolved materialization → tensor<1xf32>）。pos 恒 >= 0，
            # 截断 == floor（verify 对拍固化）。
            r = y.to(tl.int32)
            c = x.to(tl.int32)
            idx = r * W + c
            g = tl.minimum(tl.maximum(env * H * W + idx, 0), N * H * W - 1)  # 防御 OOB
            cur_fuse = tl.load(fuse + g)
            on_brick = tl.load(brick + g)
            on_wall = tl.load(wall + g)
            al = tl.load(alive + env * P + me)
            bm = tl.load(bomb + env * P + me) > 0   # int64 -> bool（triton-ascend 不能 bool&int）
            cap = tl.load(bombs_cap + env * P + me)
            bl_cap = tl.load(blast_cap + env * P + me)
            own = tl.load(live_count + env * P + me)      # 在场泡数（预计算）
            # mask 参数要求严格 i1：~on_brick 替代 on_brick==0（bool==int 会升型）
            ok = (al & bm & (cur_fuse <= 0) & (~on_brick) & (~on_wall)
                  & (own < cap)).to(tl.int1)
            tl.store(placed + env * P + me, ok.to(tl.int8))
            # 条件写用 mask 参数（in-place 更新：ok 处写新值，其余保持原值）。
            # tl.store(ptr, tl.where(ok, a, b)) 在 triton-ascend 是编译雷区。
            tl.store(fuse + g, FUSE, mask=ok)          # FUSE 是 constexpr，triton 自动 cast 到 uint8
            tl.store(owner + g, me, mask=ok)           # me 是 static_range 的 constexpr，自动 cast
            tl.store(bomb_blast + g, bl_cap, mask=ok)

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
                gidx = tl.minimum(tl.maximum(gidx, 0), HW * NENV - 1)  # 防御 OOB
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

    @triton.jit
    def _danger_stageA_round_kernel(front, fdist, w, bombed, wall, brick, blast_f,
                                    fnew, fdnew, HW, NENV,
                                    H: tl.constexpr, W: tl.constexpr,
                                    BMAX: tl.constexpr, BLOCK: tl.constexpr):
        """危险图阶段 A 单轮（torch danger_map max_chain>1 一轮的逐位等价）。

        每 cell gather：从 4 方向距离 ≤BMAX 的波前源（front>0，剩余距离 fdist）
        接收最大到达权重。规则与 torch 阶段 A 一致（先记录后挡）：
          - 源 = 本轮波前（front>0 的炮格），剩余距离 fdist[源]（>= 到达距离才记录）；
          - 途中墙/泡/brick 挡（cum 累积，同 resolve/rays 规则）；
          - 到达值 = 源权重（路径全通；挡/墙都算到不了——passable∈{0,1} 乘法等价门控）；
          - **只有炮格接收**（spread×bombed —— v1 非炮格混入 w 的 bug 防御）。
        单轮内顺带完成整轮状态更新（全部逐 cell）：
          newly = (spread>w)&bombed；w = max(w,spread)；fnew/fdnew = where(newly, spread, blast_f)。
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = HW * NENV
        m = offs < total
        nenv = offs // HW
        cidx = offs % HW
        h = cidx // W
        wc = cidx % W
        base = nenv * HW + cidx
        wself = tl.load(wall + base, mask=m, other=True)
        bmb_c = tl.load(bombed + base, mask=m, other=False)
        spread = tl.zeros((BLOCK,), tl.float32)
        for d in tl.static_range(4):
            dr = -1 if d == 0 else (1 if d == 1 else 0)
            dc = -1 if d == 2 else (1 if d == 3 else 0)
            cum = tl.zeros((BLOCK,), tl.int1)
            for step in tl.static_range(1, BMAX + 1):
                gh = h + dr * step
                gw = wc + dc * step
                ok = (gh >= 0) & (gh < H) & (gw >= 0) & (gw < W)
                gidx = nenv * HW + gh * W + gw
                gidx = tl.minimum(tl.maximum(gidx, 0), HW * NENV - 1)  # 防御 OOB
                gm = m & ok
                wv = tl.load(wall + gidx, mask=gm, other=True)
                bmb = tl.load(bombed + gidx, mask=gm, other=False)
                brk = tl.load(brick + gidx, mask=gm, other=False)
                fv = tl.load(front + gidx, mask=gm, other=0.0)
                fdv = tl.load(fdist + gidx, mask=gm, other=0.0)
                cand = tl.where((fv > 0.0) & (fdv >= step) & (cum == 0) & (~wv),
                                fv, 0.0)
                spread = tl.maximum(spread, cand)
                cum = cum | (wv | bmb | brk)
        spread = tl.where(wself, 0.0, spread)      # 墙格无接收
        spread = tl.where(bmb_c, spread, 0.0)      # 只有炮格接收（v1 bug 防御）
        w_c = tl.load(w + base, mask=m, other=0.0)
        newly = (spread > w_c) & bmb_c
        tl.store(w + base, tl.maximum(w_c, spread), mask=m)
        tl.store(fnew + base, tl.where(newly, spread, 0.0), mask=m)
        tl.store(fdnew + base,
                 tl.where(newly,
                          tl.load(blast_f + base, mask=m, other=0.0), 0.0),
                 mask=m)

    @triton.jit
    def _danger_diffuse_kernel(weight, wall, bombed, brick, blst, out,
                               HW, NENV, H: tl.constexpr, W: tl.constexpr,
                               BMAX: tl.constexpr, BLOCK: tl.constexpr):
        """危险图阶段 B：从**已修正权重**（阶段 A 后）扩散（_danger_kernel 同款扫描）。

        与 danger_map(max_chain=1) 的 _danger_kernel 唯一差别：源权重直接读
        传入的 weight 格（阶段 A 修正过），不再从 fuse 重算。逐位等价由
        verify_triton_dangerA.py 固化（torch danger_map max_chain=16 对拍）。
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = HW * NENV
        m = offs < total
        nenv = offs // HW
        cidx = offs % HW
        h = cidx // W
        wc = cidx % W
        base = nenv * HW + cidx
        wself = tl.load(wall + base, mask=m, other=True)
        w_self = tl.load(weight + base, mask=m, other=0.0)
        dng = tl.where(wself, 0.0, w_self)          # seed = 自身权重（墙格恒 0）
        for d in tl.static_range(4):
            dr = -1 if d == 0 else (1 if d == 1 else 0)
            dc = -1 if d == 2 else (1 if d == 3 else 0)
            cum = tl.zeros((BLOCK,), tl.int1)
            for step in tl.static_range(1, BMAX + 1):
                gh = h + dr * step
                gw = wc + dc * step
                ok = (gh >= 0) & (gh < H) & (gw >= 0) & (gw < W)
                gidx = nenv * HW + gh * W + gw
                gidx = tl.minimum(tl.maximum(gidx, 0), HW * NENV - 1)  # 防御 OOB
                gm = m & ok
                wv = tl.load(wall + gidx, mask=gm, other=True)
                bmb = tl.load(bombed + gidx, mask=gm, other=False)
                brk = tl.load(brick + gidx, mask=gm, other=False)
                bv = tl.load(blst + gidx, mask=gm, other=0)
                sw = tl.load(weight + gidx, mask=gm, other=0.0)
                cand = tl.where((sw > 0.0) & (bv >= step) & (cum == 0), sw, 0.0)
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
    grid = (p, n)
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
    _place_bombs_kernel[(n,)](                              # 每 env 一 program（同格竞争见 kernel 注释）
        fuse.contiguous(), owner.contiguous(), bomb_blast.contiguous(),
        pos.contiguous(), alive.contiguous(), bomb.contiguous(),
        bombs_cap.contiguous(), blast_cap.contiguous(),
        brick.contiguous(), wall.contiguous(), live_count, placed, n,
        P=p, H=h, W=w, FUSE=cfg.fuse)
    return placed.bool()


def resolve_triton(fuse, owner, wall, bomb_blast, brick, max_chain=16,
                  b_max=7, early_exit=True):
    """爆炸与连锁（triton explode kernel 链）。返回 (covered, triggered)。

    语义与 torch resolve_explosions 逐位一致（verify_triton_boom 固化）。
    `early_exit`：None = 自动（全设备早退）；True 强制；False 强制关
    （CUDA graph 捕获段必须 False——回放要求固定 kernel 序列，早退的
    host break 会把轮数钉死在捕获值）。**910B 实测：无爆炸 tick 若跑满
    max_chain 轮 explode kernel（~24ms/轮）要 384ms+；早退短路到 ~1ms。**
    """
    import torch
    _CHECK_EVERY = 2    # 与 sim/blast.py CHECK_EVERY 一致（早退检查每 2 轮一次）
    should_ee = True if early_exit is None else early_exit
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    # 无爆炸短路（同 torch resolve_explosions）：大多数 tick 只是倒计时，
    # 没有引信走完的泡 → 直接返回空覆盖（1 次 sync 换掉 16 轮 explode kernel）。
    if should_ee and not bool(triggered.any()):
        return torch.zeros_like(fuse, dtype=torch.bool), triggered
    covered = explode_triton(triggered, wall, live, brick, bomb_blast, b_max)
    for i in range(max_chain - 1):
        newly = live & covered & ~triggered
        if should_ee and i % _CHECK_EVERY == 0 and not bool((newly).any()):
            break
        covered = covered | explode_triton(newly, wall, live, brick,
                                           bomb_blast, b_max)
        triggered = triggered | newly
    return covered, triggered


def danger_triton(fuse, wall, bombed, brick, blast, fuse_max,
                  b_max=7, exp=2.0, max_chain=1, early_exit=True):
    """危险图（阶段 A 连锁修正 + 阶段 B 扩散），与 torch danger_map 逐位等价。

    - max_chain=1：阶段 B only（走 _danger_kernel 原路径，行为不变）。
    - max_chain>1：阶段 A 波前连锁（每轮 _danger_stageA_round_kernel）修正
      炮格权重 → 阶段 B 用修正权重扩散（_danger_diffuse_kernel）。
      阶段 A 空场短路 / 早退语义与 torch danger_map 相同（DANGER_CHECK_EVERY=2）。
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton 不可用")
    import torch
    n, h, w = fuse.shape
    hw = h * w
    dev = fuse.device
    BLOCK = 1024
    grid = ((hw * n + BLOCK - 1) // BLOCK,)
    BMAX = max(1, int(b_max))
    if max_chain <= 1:
        # 阶段 B only：原路径（kernel 内从 fuse 算权重，bitwise 已验证）
        out = torch.empty((n, h, w), dtype=torch.float32, device=dev)
        _danger_kernel[grid](
            fuse.contiguous(), wall.contiguous(), bombed.contiguous(),
            brick.contiguous(), blast.to(torch.int32).contiguous(), out,
            hw, n, h, w, BMAX=BMAX, BLOCK=BLOCK,
            FUSE_MAX=int(fuse_max), EXP=float(exp))
        return out
    # ---- 阶段 A：权重 + 连锁修正（语义同 torch danger_map max_chain>1）----
    w_raw = 1.0 - (fuse.float() - 1.0) / float(fuse_max)
    weight = torch.where(bombed, w_raw.clamp_min(0.0).pow(exp),
                         torch.zeros_like(fuse, dtype=torch.float32))
    if early_exit and not bool(bombed.any()):
        return torch.zeros_like(fuse, dtype=torch.float32)   # 空场短路
    blast_f = torch.where(bombed, blast.float(), torch.zeros_like(weight))
    front_a = torch.where(bombed, weight, torch.zeros_like(weight))
    fdist_a = blast_f.clone()
    front_b = torch.empty_like(front_a)
    fdist_b = torch.empty_like(front_a)
    w_c = weight
    w_b = bombed.contiguous()
    wal_c = wall.contiguous()
    brk_c = brick.contiguous()
    bf_c = blast_f.contiguous()
    _DANGER_CHECK_EVERY = 2    # 与 sim/blast.py DANGER_CHECK_EVERY 一致
    for i in range(max_chain):
        # 乒乓缓冲：输入/输出必须异体（同体会竞态——kernel 读 front 时另一
        # block 正在写它）
        if i % 2 == 0:
            _danger_stageA_round_kernel[grid](
                front_a.contiguous(), fdist_a.contiguous(), w_c, w_b,
                wal_c, brk_c, bf_c, front_b, fdist_b,
                hw, n, h, w, BMAX=BMAX, BLOCK=BLOCK)
            newly_cur = front_b
        else:
            _danger_stageA_round_kernel[grid](
                front_b.contiguous(), fdist_b.contiguous(), w_c, w_b,
                wal_c, brk_c, bf_c, front_a, fdist_a,
                hw, n, h, w, BMAX=BMAX, BLOCK=BLOCK)
            newly_cur = front_a
        if early_exit and i % _DANGER_CHECK_EVERY == 0 \
                and not bool((newly_cur > 0).any()):
            break
    # ---- 阶段 B：用修正权重扩散 ----
    out = torch.empty((n, h, w), dtype=torch.float32, device=dev)
    _danger_diffuse_kernel[grid](
        w_c, wal_c, w_b, brk_c, blast.to(torch.int32).contiguous(), out,
        hw, n, h, w, BMAX=BMAX, BLOCK=BLOCK)
    return out
