"""JAX bomberman —— 对齐正式版空场景规则（sim/config.py + sim/move.py + sim/blast.py）。

与 sim/torch_sim.py 的 step 逐段对齐（除道具/成长/奖励外）：
  - 连续坐标移动：speed=7.56 格/秒 → 每 tick 0.756 格，AABB 滑动碰撞
    （radius=0.3，_resolve_axis 逻辑），角色互不碰撞，脚下刚放的泡放行
  - 放泡：移动前落在起始中心格，脚下无泡 + 在场泡数 < max_bombs(10)，
    威力按放泡时刻快照（BLAST=7）存进 bomb_blast
  - 引信 FUSE=30，爆炸与连锁 max_chain=16（泡挡火、每颗泡自己的威力）
  - 伤害：中心格着火扣 1 血（HP=5），无敌期 INVULN=30 tick，血归 0 死亡
  - 终局：n_alive <= 1 或 t >= MAX_STEPS(1800)，done 后就地重置（auto_reset）
观测 = 正式版 2P+3=7 通道视角（view_perm）：自己位置/自己泡剩余时间/
对手位置/对手泡剩余时间/墙(空场景 0)/泡威力(blast)/进度。**不预计算危险图**
（化繁为简：大网络从炸弹基础信息自己学危险推理，省掉每 tick 的危险图
连锁传播+扩散——那曾是环境每 tick 最贵的部分）。

纯函数式 jax，vmap over batch；爆炸连锁用 lax.while_loop 动态链长
（jax 无 host 同步，等价 torch 的动态早退）。
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------- 正式版数值（sim/config.py 对齐） ----------------
H = W = 13
TICK_HZ = 10
SPEED = 7.56            # 格/秒（3.6 × growth_speed_max 2.1）→ 每 tick 0.756 格
STEP = SPEED / TICK_HZ  # 0.756 格/tick
RADIUS = 0.3            # 碰撞盒半宽（< 0.5）
EPS = 1e-4
FUSE = 30               # 引信（3 秒）
BLAST = 7               # 十字威力（不含中心格）
MAX_BOMBS = 10          # 单角色在场泡数上限
MAX_HP = 5
INVULN = 30             # 被炸伤后无敌 tick
MAX_CHAIN = 16          # 连锁最多迭代轮数
MAX_STEPS = 1800        # 局长 180 秒
N_MOVES, N_BOMB = 5, 2
_MOVE_DELTA = jnp.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                         [1.0, 0.0], [-1.0, 0.0]], jnp.float32)   # 停/右/左/下/上
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# 观测通道：0 我位置, 1 我泡, 2 对手位置, 3 对手泡, 4 墙(0), 5 危险图, 6 进度
N_OBS_CH = 7


class BombState(NamedTuple):
    pos: jnp.ndarray       # (2, 2) float32 连续坐标（角色中心，格单位）
    fuse: jnp.ndarray      # (H, W) int32 引信倒计时，0 = 无泡
    owner: jnp.ndarray     # (H, W) int32 -1 无，0/1 玩家
    bomb_blast: jnp.ndarray  # (H, W) int32 每颗泡自己的威力（放泡时快照）
    wall: jnp.ndarray      # (H, W) bool 永久墙（不可通行不可炸）
    alive: jnp.ndarray     # (2,) bool
    hp: jnp.ndarray        # (2,) int32
    invuln: jnp.ndarray    # (2,) int32 剩余无敌 tick
    t: jnp.ndarray         # () int32


def _fresh() -> BombState:
    """新局状态（对角出生点，同正式版 open 关）。"""
    pos = jnp.array([[1.0, 1.0], [H - 2.0, W - 2.0]], jnp.float32)
    fuse = jnp.zeros((H, W), jnp.int32)
    owner = -jnp.ones((H, W), jnp.int32)
    bomb_blast = jnp.zeros((H, W), jnp.int32)
    wall = jnp.zeros((H, W), jnp.bool_)
    alive = jnp.ones((2,), jnp.bool_)
    hp = jnp.full((2,), MAX_HP, jnp.int32)
    invuln = jnp.zeros((2,), jnp.int32)
    t = jnp.zeros((), jnp.int32)
    return BombState(pos, fuse, owner, bomb_blast, wall, alive, hp, invuln, t)


def init_batch(key, n: int) -> BombState:
    keys = jax.random.split(key, n)
    return jax.vmap(lambda _k: _fresh())(keys)


def _shift(x, drow, dcol):
    """result[i,j] = x[i-drow, j-dcol]，越界补 0（同 blast.py 语义）。"""
    if drow == 1:
        x = jnp.concatenate([jnp.zeros((1, x.shape[1]), x.dtype), x[:-1]], axis=0)
    elif drow == -1:
        x = jnp.concatenate([x[1:], jnp.zeros((1, x.shape[1]), x.dtype)], axis=0)
    if dcol == 1:
        x = jnp.concatenate([jnp.zeros((x.shape[0], 1), x.dtype), x[:, :-1]], axis=1)
    elif dcol == -1:
        x = jnp.concatenate([x[:, 1:], jnp.zeros((x.shape[0], 1), x.dtype)], axis=1)
    return x


# ---------------- 爆炸传播（同 sim/blast.py 的距离缓冲 v2） ----------------

def _rays(seed, bombed, blast_map, wall):
    """从 seed 出发的十字覆盖。泡/brick 挡火但被覆盖；wall 永久墙完全挡火；blast 每格自己的威力。

    固定 max_b = BLAST 轮（多跑空步结果不变），与 blast.py 的
    distance-buffer 传播逐位一致。
    """
    covered = seed
    fd = seed.astype(jnp.int8) * jnp.clip(blast_map, 0, 127).astype(jnp.int8)
    one = jnp.ones_like(fd)
    not_solid = ~bombed & ~wall
    for drow, dcol in _DIRS:
        fd_p = fd
        for _ in range(BLAST):
            fd1 = _shift(fd_p, drow, dcol)
            covered = covered | (fd1 > 0)
            fd1 = fd1 - one
            fd1 = fd1 * not_solid.astype(jnp.int8)     # 泡/墙挡火：记录后不穿透
            fd_p = fd1
    return covered


def _resolve_explosions(fuse, owner, bomb_blast, wall):
    """返回 (covered, triggered)，连锁最多 MAX_CHAIN 轮（动态早退）。

    triggered = 本 tick 爆炸的泡；被 covered 覆盖的活泡连锁点燃。
    链长用 lax.while_loop 动态（jax 无 host 同步；多跑空轮结果不变）。
    """
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    blast_map = jnp.where(bomb_blast > 0, bomb_blast, BLAST)
    covered = _rays(triggered, live, blast_map, wall)

    def cond(c):
        newly = c[0]
        return jnp.any(newly)

    def body(c):
        newly, covered, triggered = c
        covered2 = covered | _rays(newly, live, blast_map, wall)
        triggered2 = triggered | newly
        newly2 = live & covered2 & ~triggered2
        return newly2, covered2, triggered2

    _, covered, triggered = jax.lax.while_loop(
        cond, body, (live & covered & ~triggered, covered, triggered))
    return covered, triggered


# ---------------- 移动（同 sim/move.py _resolve_axis） ----------------

def _impassable_pair(blocked_flat, r0, c0, r1, c1, y, x):
    """两格任一不可通行（越界算不可通行；碰撞盒覆盖中的格放行）。"""
    h, w = H, W
    oob = ((r0 < 0) | (r0 >= h) | (c0 < 0) | (c0 >= w)
           | (r1 < 0) | (r1 >= h) | (c1 < 0) | (c1 >= w))
    idx = jnp.stack([
        jnp.clip(r0, 0, h - 1) * w + jnp.clip(c0, 0, w - 1),
        jnp.clip(r1, 0, h - 1) * w + jnp.clip(c1, 0, w - 1)], axis=-1)
    solid = blocked_flat[idx]
    r0c = (y - RADIUS).astype(jnp.int32)
    r1c = (y + RADIUS).astype(jnp.int32)
    c0c = (x - RADIUS).astype(jnp.int32)
    c1c = (x + RADIUS).astype(jnp.int32)
    in0 = (r0 >= r0c) & (r0 <= r1c) & (c0 >= c0c) & (c0 <= c1c)
    in1 = (r1 >= r0c) & (r1 <= r1c) & (c1 >= c0c) & (c1 <= c1c)
    return oob | (solid[0] & ~in0) | (solid[1] & ~in1)


def _resolve_axis(coord, delta, other, y, x, blocked_flat, vertical):
    """沿单轴消解碰撞：撞上贴着障碍停下（滑动）。同 move.py _resolve_axis。"""
    sgn = jnp.sign(delta)
    old_lead = (coord - delta + sgn * RADIUS).astype(jnp.int32)
    new_lead = (coord + sgn * RADIUS).astype(jnp.int32)
    lo = jnp.minimum(old_lead, new_lead)
    hi = jnp.maximum(old_lead, new_lead)
    span0 = (other - RADIUS).astype(jnp.int32)
    span1 = (other + RADIUS).astype(jnp.int32)

    def hit_at(lead):
        if vertical:
            return _impassable_pair(blocked_flat, lead, span0, lead, span1, y, x)
        return _impassable_pair(blocked_flat, span0, lead, span1, lead, y, x)

    hit_lo = hit_at(lo)
    hit_hi = hit_at(hi)
    first_lead = jnp.where(sgn > 0, hi, lo)
    second_lead = jnp.where(sgn > 0, lo, hi)
    first_hit = jnp.where(sgn > 0, hit_hi, hit_lo)
    second_hit = jnp.where(sgn > 0, hit_lo, hit_hi)
    has = first_hit | second_hit
    first = jnp.where(first_hit, first_lead,
                      jnp.where(second_hit, second_lead, jnp.zeros_like(lo)))
    stop_pos = jnp.where(sgn > 0,
                         first.astype(jnp.float32) - RADIUS - EPS,
                         first.astype(jnp.float32) + 1.0 + RADIUS + EPS)
    return jnp.where(has, stop_pos, coord)


def _move_player(pos_me, move, alive_me, blocked):
    """单玩家移动：连续坐标 + AABB 滑动碰撞（角色互不碰撞）。"""
    delta = _MOVE_DELTA[move] * STEP
    moving = alive_me & (move != 4)              # MOVE_IDLE=4
    delta = jnp.where(moving, delta, jnp.zeros_like(delta))
    dy, dx = delta[0], delta[1]
    y, x = pos_me[0], pos_me[1]
    blocked_flat = blocked.reshape(-1)
    ny = _resolve_axis(y + dy, dy, x, y, x, blocked_flat, True)
    nx = _resolve_axis(x + dx, dx, y, y, x, blocked_flat, False)
    out_y = jnp.where(dy != 0, ny, y)
    out_x = jnp.where(dx != 0, nx, x)
    out_y = jnp.clip(out_y, RADIUS, H - RADIUS)
    out_x = jnp.clip(out_x, RADIUS, W - RADIUS)
    return jnp.stack([out_y, out_x])


# ---------------- step（对齐 torch_sim.step 的顺序） ----------------

def step(state: BombState, actions: jnp.ndarray) -> tuple[BombState, jnp.ndarray]:
    """actions: (2, 2) int32 = [dir(0-4), bomb(0/1)]，每玩家。返回 (state, done)。"""
    pos, fuse, owner, bomb_blast, wall, alive, hp, invuln, t = state
    dirs, bombs = actions[:, 0], actions[:, 1]
    alive0 = alive

    # 1. 引信递减
    fuse = jnp.where(fuse > 0, fuse - 1, fuse)

    # 2. 放泡（移动前，落在起始中心格；威力按此刻档位快照；墙格不可放）
    cell = pos.astype(jnp.int32)                 # center_cell = floor
    for me in range(2):
        y, x = cell[me, 0], cell[me, 1]
        idx = jnp.clip(y, 0, H - 1) * W + jnp.clip(x, 0, W - 1)
        cur_f = fuse.reshape(-1)[idx]
        live = ((owner == me) & (fuse > 0)).sum()
        wall_cell = wall.reshape(-1)[idx]
        ok = (alive0[me] & (bombs[me] == 1) & (cur_f <= 0)
              & (live < MAX_BOMBS) & ~wall_cell)
        fuse = fuse.at[y, x].set(jnp.where(ok, FUSE, cur_f))
        owner = owner.at[y, x].set(jnp.where(ok, me, owner[y, x]))
        bomb_blast = bomb_blast.at[y, x].set(jnp.where(ok, BLAST, bomb_blast[y, x]))

    # 3. 移动（blocked = 泡 | 墙）
    blocked = (fuse > 0) | wall
    p0 = _move_player(pos[0], dirs[0], alive0[0], blocked)
    p1 = _move_player(pos[1], dirs[1], alive0[1], blocked)
    newpos = jnp.stack([p0, p1])

    # 4. 爆炸与连锁（墙挡火）
    covered, triggered = _resolve_explosions(fuse, owner, bomb_blast, wall)

    # 5. 伤害：中心格着火扣 1 血，无敌期不掉血，血归 0 死
    ccell = newpos.astype(jnp.int32)
    hit = alive0 & covered[ccell[:, 0], ccell[:, 1]]
    invuln_ok = invuln <= 0
    hit_eff = hit & invuln_ok
    hp_new = jnp.clip(hp - hit_eff.astype(jnp.int32), 0, None)
    died = hit_eff & (hp_new == 0)
    alive_new = alive0 & ~died
    invuln = jnp.clip(invuln - 1, 0, None)
    invuln = jnp.where(hit_eff, INVULN, invuln)

    # 6. 清场（爆炸泡 fuse→0、owner→-1、威力→0）
    fuse = jnp.where(triggered, 0, fuse)
    owner = jnp.where(triggered, -1, owner)
    bomb_blast = jnp.where(triggered, 0, bomb_blast)

    # 7. 计步与终局（done 后就地重置 = 正式版 auto_reset）
    t = t + 1
    n_alive = alive_new.sum()
    done = (n_alive <= 1) | (t >= MAX_STEPS)
    new_state = BombState(newpos, fuse, owner, bomb_blast, wall, alive_new,
                          hp_new, invuln, t)
    return jax.lax.cond(done, lambda _: _fresh(), lambda _: new_state, None), done


# ---------------- 合法动作掩码（同 sim/obs.py::legal_mask 语义） ----------------

def legal_mask(state: BombState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """返回 (move_mask (2,5) bool, bomb_mask (2,2) bool)。

    方向掩码只屏蔽"按了也一格都动不了"的方向（贴住墙/泡）——用探针位移
    判定（IDLE 永远合法）。放泡掩码：存活 & 不在墙格 & 脚下无泡 & 未超上限
    （bomb=0 不放永远合法）。与 torch sim/obs.py::legal_mask 语义一致。
    """
    pos, fuse, owner, _bb, wall, alive, _hp, _invuln, _t = state
    blocked = (fuse > 0) | wall
    # 四个方向各探一次位移：能移动 = 合法（IDLE=4 恒合法）
    move = jnp.zeros((2, 5), jnp.bool_)
    for d in range(4):
        np_ = _move_player(pos[0], d, alive[0], blocked)
        moved0 = (np_ != pos[0]).any()
        np1 = _move_player(pos[1], d, alive[1], blocked)
        moved1 = (np1 != pos[1]).any()
        move = move.at[0, d].set(moved0 & alive[0])
        move = move.at[1, d].set(moved1 & alive[1])
    move = move.at[:, 4].set(True)                      # IDLE 恒合法
    # 放泡：存活 & 不在墙格 & 脚下无泡 & 在场泡数 < 上限
    cell = pos.astype(jnp.int32)
    cell = jnp.stack([jnp.clip(cell[:, 0], 0, H - 1),
                      jnp.clip(cell[:, 1], 0, W - 1)], axis=-1)
    idx = cell[:, 0] * W + cell[:, 1]
    cur_f = fuse.reshape(-1)[idx]
    on_wall = wall.reshape(-1)[idx]
    live = jnp.stack([
        ((owner == 0) & (fuse > 0)).sum(),
        ((owner == 1) & (fuse > 0)).sum(),
    ])
    can_bomb = alive & (cur_f <= 0) & ~on_wall & (live < MAX_BOMBS)
    bomb = jnp.stack([~alive, can_bomb], axis=-1)       # bomb=0 恒合法
    return move, bomb


# ---------------- 观测（正式版 2P+3=7 通道视角） ----------------

def _splat(pos_me, gate, h, w):
    """把连续坐标双线性铺开成 (H, W) 平面，总质量 1（同 obs.py _splat）。"""
    fy = jnp.clip(pos_me[0] - 0.5, 0.0, h - 1.0)
    fx = jnp.clip(pos_me[1] - 0.5, 0.0, w - 1.0)
    y0 = jnp.clip(fy.astype(jnp.int32), 0, h - 1)
    x0 = jnp.clip(fx.astype(jnp.int32), 0, w - 1)
    y1 = jnp.clip(y0 + 1, 0, h - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)
    wy = jnp.clip(fy - y0.astype(jnp.float32), 0.0, 1.0)
    wx = jnp.clip(fx - x0.astype(jnp.float32), 0.0, 1.0)
    g = gate.astype(jnp.float32)
    out = jnp.zeros((h, w), jnp.float32)
    out = out.at[y0, x0].add((1 - wy) * (1 - wx) * g)
    out = out.at[y0, x1].add((1 - wy) * wx * g)
    out = out.at[y1, x0].add(wy * (1 - wx) * g)
    out = out.at[y1, x1].add(wy * wx * g)
    return out


def make_obs(state: BombState, pid: int) -> jnp.ndarray:
    """(7, H, W) float32，玩家 pid 视角。

    通道 = 炸弹**基础信息**（不预计算危险图——大网络自己学危险推理）：
      ch0 我位置（双线性 splat）   ch1 我的泡剩余时间 fuse/FUSE
      ch2 对手位置（splat）        ch3 对手泡剩余时间 fuse/FUSE
      ch4 墙（wall 二值）          ch5 泡威力 blast/BLAST（在场泡格 = 范围）
      ch6 局内进度 t/MAX_STEPS
    """
    pos, fuse, owner, bomb_blast, wall, alive, _hp, _invuln, t = state
    me, opp = pid, 1 - pid
    fuse_norm = fuse.astype(jnp.float32) / float(FUSE)
    bombed = fuse > 0
    blast_norm = jnp.where(
        bombed,
        jnp.where(bomb_blast > 0, bomb_blast, BLAST).astype(jnp.float32) / float(BLAST),
        jnp.zeros_like(fuse, jnp.float32))
    obs = jnp.stack([
        _splat(pos[me], alive[me], H, W),
        jnp.where(owner == me, fuse_norm, jnp.zeros_like(fuse_norm)),
        _splat(pos[opp], alive[opp], H, W),
        jnp.where(owner == opp, fuse_norm, jnp.zeros_like(fuse_norm)),
        wall.astype(jnp.float32),                                # 墙（二值）
        blast_norm,                                              # 泡威力（范围）
        jnp.full((H, W), t.astype(jnp.float32) / float(MAX_STEPS), jnp.float32),
    ])
    return obs
