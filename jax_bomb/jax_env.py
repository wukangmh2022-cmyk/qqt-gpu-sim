"""JAX bomberman —— 对齐正式版空场景规则（sim/config.py + sim/move.py + sim/blast.py）。

与 sim/torch_sim.py 的 step 逐段对齐（除道具/成长/奖励外）：
  - 连续坐标移动：speed=3.0 格/秒 → 每 tick 0.3 格（对齐 Web sim.js stepLen），
    AABB 滑动碰撞（radius=0.36，_resolve_axis 逻辑），角色互不碰撞，脚下刚放的泡放行
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

import os

import jax
import jax.numpy as jnp

from . import levels   # 标准化关卡（set_active 后 _fresh 走关卡采样）

# ---------------- 正式版数值（sim/config.py 对齐） ----------------
H = 13                            # 高（行数）
W = 15                            # 宽（列数；宽度 +2：13→15。ViT patch 数不变，见 jax_net）
TICK_HZ = 10
SPEED = 3.0             # 格/秒（对齐 Web sim.js CFG.speed=3.0 / stepLen=0.3）→ 每 tick 0.3 格
STEP = SPEED / TICK_HZ  # 0.3 格/tick（spd_g=1.0 时；实际位移 = STEP × spd_g，满速 0.63）
RADIUS = 0.45 * 0.8     # 训练双方碰撞盒半宽 0.36 格（自对弈不能按 P0/P1 区分）
EPS = 1e-4
MAX_SWEEP = 3           # _resolve_axis 跨格扫描上限：位移≤0.63 格 → 前缘跨≤2 格（3 是余量）
FUSE = 30               # 引信（3 秒）
BLAST = 7               # 十字威力（不含中心格）
MAX_BOMBS = 10          # 泡数上限（成长属性 growth_bombs_max）
MAX_HP = 5
INVULN = 30             # 被炸伤后无敌 tick
MAX_CHAIN = 16          # 连锁最多迭代轮数
MAX_STEPS = 1800        # 局长 180 秒
PUSH_TIME = 0.3         # 推箱子：持续推 ≥0.3s（3 tick）箱子移一格（对齐 Web）
N_MOVES, N_BOMB = 5, 2
# 方向编码与 torch（sim/config.py）一致：0=上 1=下 2=左 3=右 4=停(IDLE)。
# 之前用 [停/右/左/下/上] 与 torch 错位 —— 对拍/蒸馏动作语义全乱，2026-08-17 修。
_MOVE_DELTA = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
                         [0.0, 0.0]], jnp.float32)   # 上/下/左/右/停
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# 贪婪转向垂直方向对：dir 0(上)/1(下) → 左右(2,3)；dir 2(左)/3(右) → 上下(0,1)
_PERP = jnp.array([[2, 3], [2, 3], [0, 1], [0, 1]], jnp.int32)

# ---------------- 混合地图（sim/config.py + sim/mapgen.py 语义对齐） ----------------
# 每局按概率掷三类：纯空场 50% / open 带障碍 25% / corridor 砖墙 25%
# （collect_distill make_cfg 同款：pure_open_fraction / open_fraction / 余量）。
# QQT_ENV_MIX 环境变量切换训练分布（在 import 本模块前设置）：
#   mixed（默认）  = 纯空 50% / open 带障碍 25% / corridor 砖墙 25%（课程/混合训练）
#   open_obstacle  = 纯空 0% / open 带障碍 100% —— 论文式：每局 0-5 个随机
#                    小数量障碍（Pommerman 论文同款设定，真实训练用）
#   open           = 纯空 40% / open 带障碍 60%（去掉 corridor 砖墙）
_QQT_ENV_MIX = os.environ.get("QQT_ENV_MIX", "mixed")
if _QQT_ENV_MIX == "open_obstacle":
    PURE_OPEN_FRACTION = 0.0
    OPEN_FRACTION = 1.0
elif _QQT_ENV_MIX == "open":
    PURE_OPEN_FRACTION = 0.4
    OPEN_FRACTION = 0.6
else:
    PURE_OPEN_FRACTION = 0.5
    OPEN_FRACTION = 0.25
RING_FRACTION = 0.0
OPEN_OBSTACLE_MAX = 5        # open 关随机单障碍（永久墙）上限
TOP_WALL_ROWS = 4            # corridor 顶/底墙行总数（每局随机分上下）
CORRIDOR_WIDTH = 5           # corridor 可通行区宽（左右整列 brick）
WALL_DENSITY = 0.45          # corridor 边缘连续 brick 段概率

# 成长属性（sim/config.py 默认值）
GROWTH_BOMBS_START, GROWTH_BLAST_START = 2, 2   # corridor 初始
GROWTH_SPEED_START = 1.0
GROWTH_BOMBS_MAX, GROWTH_BLAST_MAX = MAX_BOMBS, BLAST   # 上限
GROWTH_SPEED_MAX = 2.1
GROWTH_SPEED_STEP = 0.15
OPEN_GROWTH_BOMBS, OPEN_GROWTH_BLAST, OPEN_GROWTH_SPEED = 3, 3, 0.84  # open 初始
CRATE_PROB = 0.5             # corridor 炸砖宝箱爆率（open 关恒 1.0 必升）
HIT_ATTR_PENALTY = 2         # 掉血扣泡/威/速各几层（clamp 回模式起点）
OPEN_CRATE_CROSS = True      # open 关开局中心十字宝箱池（≈46 格，100% 有东西）
MAX_RECYCLE = 2 * 3 * HIT_ATTR_PENALTY   # 单 tick 掉血回收箱上限（2 玩家 × 3 属性 × 层数）

# 观测通道：0 我位置, 1 我泡, 2 对手位置, 3 对手泡, 4 墙, 5 危险图, 6 进度,
#            7 宝箱（共享地图通道，torch obs14 ch7 同源）
N_OBS_CH = 14


# corridor / open 出生点（对齐 config.spawn_pos / torch_sim._open_spawns）：
#   corridor：空旷区行列中心（13×15、top_wall_rows=4、corridor_width=5 → (8.5,6.5)/(8.5,9.5)）
#   open：整宽中线均分（P=2 → (6.5,5.2)/(6.5,9.8)）
CORRIDOR_SPAWN = jnp.array([
    [(TOP_WALL_ROWS + (H - 1)) / 2.0 + 0.5, (W - CORRIDOR_WIDTH) / 2.0 + 0.5 + 1.0],
    [(TOP_WALL_ROWS + (H - 1)) / 2.0 + 0.5,
     (W - CORRIDOR_WIDTH) / 2.0 + 0.5 + 1.0 + (CORRIDOR_WIDTH - 2)],  # P=2: step=3
], jnp.float32)
OPEN_SPAWN = jnp.array([
    [(H - 1) / 2.0 + 0.5, (W - 1) / 3.0 + 0.5],
    [(H - 1) / 2.0 + 0.5, (W - 1) * 2.0 / 3.0 + 0.5],
], jnp.float32)

# 出生点所在格（Python int 常量）：清四邻 / 回收排除用 —— 在 vmap trace 里
# jnp 常量索引也会产生 traced 值（Python if 崩），必须用静态 Python int。
# 13×15 取值 = int(torch config.spawn_pos)：(13-5)/2+1.5=6.5→6、+3→9.5→9；
# open (15-1)/3+0.5=5.17→5、(15-1)*2/3+0.5=9.83→9。
_CORRIDOR_CELLS = ((8, 6), (8, 9))    # corridor 空旷区中心两格
_OPEN_CELLS = ((6, 5), (6, 9))        # open 中线均分两格


class BombState(NamedTuple):
    pos: jnp.ndarray       # (2, 2) float32 连续坐标（角色中心，格单位）
    fuse: jnp.ndarray      # (H, W) int32 引信倒计时，0 = 无泡
    owner: jnp.ndarray     # (H, W) int32 -1 无，0/1 玩家
    bomb_blast: jnp.ndarray  # (H, W) int32 每颗泡自己的威力（放泡时快照）
    wall: jnp.ndarray      # (H, W) bool 永久墙（不可通行不可炸）
    brick: jnp.ndarray     # (H, W) bool 可炸墙（挡火、被覆盖即摧毁→宝箱）
    pushable: jnp.ndarray  # (H, W) bool 可推箱（推箱子关；必 ⊆ brick，被炸整箱消失）
    push_t: jnp.ndarray    # (H, W) float32 每格可推箱的推动计时（≥PUSH_TIME 移一格）
    bush: jnp.ndarray      # (H, W) bool 灌木：可通行 + 可炸毁（被覆盖即摧毁→概率宝箱）
    crate: jnp.ndarray     # (H, W) int8 道具编码（>0 有道具，踩到必升，与 Web 一致）：
                           #   0=无 1=泡泡+1 2=威力+1 3=速度+1 4=超级泡泡+4
                           #   5=超级威力+4 6=超级速度+4 7=问号随机（预置宝箱/回收箱）
    rec_crate: jnp.ndarray  # (H, W) bool 回收宝箱标记（踩到必升，不掷爆率）
    alive: jnp.ndarray     # (2,) bool
    hp: jnp.ndarray        # (2,) int32
    invuln: jnp.ndarray    # (2,) int32 剩余无敌 tick
    bombs_cap: jnp.ndarray  # (2,) float32 每玩家泡数上限（成长属性）
    blast_cap: jnp.ndarray  # (2,) float32 每玩家威力上限（成长属性）
    spd_g: jnp.ndarray     # (2,) float32 每玩家速度倍率（位移 = STEP × spd_g）
    # ---- 预留位（2026-08-20，后训练增强用；当前全 0 不参与玩法） ----
    buffs: jnp.ndarray     # (2,) int8 变身 buff（3 bit：0=无，1-7 熊猫/螃蟹等）
    debuffs: jnp.ndarray   # (2,) int8 减速等 debuff（2 bit：0=无，1-3，如慢慢胶）
    items: jnp.ndarray     # (2, 4) int8 道具栏（4 槽 × int6 道具 ID：0=空，1-63；
                           #   夺宝=宝石数量、飞镖等按竞技类型解释）
    gametype: jnp.ndarray  # () int8 竞技类型（int4：0=普通对抗，1=夺宝，2=...）
    is_open: jnp.ndarray   # () bool 本局 open 关（掉血惩罚起点、爆率；关卡模式=无砖）
    t: jnp.ndarray         # () int32
    level_id: jnp.ndarray = jnp.int32(-1)  # () int32 关卡 id（-1 = 过程式生成）


def _recycle_excl() -> jnp.ndarray:
    """掉血回收的排除格（(H,W) bool）：三类出生点四邻 + open 十字带。

    对齐 torch torch_sim._recycle_excl：回收宝箱只落在无墙无砖可通行格，
    绝不叠出生点脚下/开局十字池（corridor 无 ring 关，仍按 torch 排除）。
    """
    excl = jnp.zeros((H, W), jnp.bool_)
    for rr, cc in _CORRIDOR_CELLS + _OPEN_CELLS:
        for dr, dc in ((0, 0),) + _DIRS:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                excl = excl.at[nr, nc].set(True)
    excl = excl | _cross_crates()          # open 十字带（开局池专属）
    return excl


def _cross_crates() -> jnp.ndarray:
    """open 关开局中心十字宝箱池：(H,W) bool（确定性，不依赖 RNG）。

    横竖各 2 排 —— 行带 {cy-1, cy} 全宽 ∪ 列带 {cx-1, cx} 全高（13×15 →
    2×13+2×15−4 = 52 格），排除 open 出生点四邻（对齐 torch _open_geometry）。
    """
    cy, cx = (H - 1) // 2, (W - 1) // 2
    cross = jnp.zeros((H, W), jnp.bool_)
    cross = cross.at[cy - 1, :].set(True)
    cross = cross.at[cy, :].set(True)
    cross = cross.at[:, cx - 1].set(True)
    cross = cross.at[:, cx].set(True)
    for rr, cc in _OPEN_CELLS:
        for dr, dc in ((0, 0),) + _DIRS:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                cross = cross.at[nr, nc].set(False)
    return cross


def _make_map(key) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """混合地图生成，语义对齐 sim/mapgen.py（RNG 算法不同，不逐位一致）。

    每局掷一类：
      - 纯空场（50%）：无墙无砖，open 属性/十字宝箱；
      - open（25%）：open_obstacle_max 个随机单障碍永久墙（不放回）；
      - corridor（25%）：顶/底共 top_wall_rows 行永久墙（顶行数每局随机）
        + 左右整列 brick + 可通行区边缘连续 brick 段（wall_density 概率）。
    返回 (wall, brick, is_open, spawn(2,2))；spawn 已按 torch 规则约一半概率
    交换 P0/P1 出生点（消除"恒打物理左侧"偏置）。
    """
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    r = jax.random.uniform(k1, ())
    is_pure = r < PURE_OPEN_FRACTION
    # open 关 = 纯空场 ∪ open 障碍（torch 的 open_type = is_pure | is_open）
    is_open = r < (PURE_OPEN_FRACTION + OPEN_FRACTION)
    is_corr = ~is_open

    row_ar = jnp.arange(H)[:, None]                      # (H,1)
    col_ar = jnp.arange(W)[None, :]                      # (1,W)

    # ---- corridor：顶/底墙 + 左右 brick + 边缘连续段 ----
    top = jax.random.randint(k2, (), 0, TOP_WALL_ROWS + 1)
    bot = TOP_WALL_ROWS - top
    wall_c = (row_ar < top) | (row_ar >= H - bot)
    wall_c = jnp.broadcast_to(wall_c, (H, W))
    c0 = (W - CORRIDOR_WIDTH) // 2
    c1 = c0 + CORRIDOR_WIDTH
    side = (col_ar < c0) | (col_ar >= c1)
    in_top = row_ar < top
    brick_c = side & ~in_top
    brick_c = jnp.broadcast_to(brick_c, (H, W))
    if WALL_DENSITY > 0:
        # 垂直连续段：可通行区边缘列（贴左右 brick 内侧），2-4 格高。
        # corridor_width=5 固定 → 4 个边列 [c0, c0+1, c1-2, c1-1] 恒在
        # [c0, c1) 内（torch mapgen 的动态过滤在固定宽度下结果相同）。
        edge_cols = jnp.array([c0, c0 + 1, c1 - 2, c1 - 1], jnp.int32)
        ncols = edge_cols.shape[0]
        k3a, k3b, k3c = jax.random.split(k3, 3)
        span = (H - bot - 4 - top).astype(jnp.float32)
        span = jnp.maximum(span, 1.0)
        starts = (jax.random.uniform(k3a, (ncols,)) * span
                  + top.astype(jnp.float32)).astype(jnp.int32)
        lens = 2 + (jax.random.uniform(k3b, (ncols,)) * 3).astype(jnp.int32)
        act = jax.random.uniform(k3c, (ncols,)) < WALL_DENSITY
        rows_v = jnp.arange(H)
        for ci in range(ncols):
            seg = act[ci] & (rows_v >= starts[ci]) & (rows_v < starts[ci] + lens[ci])
            ec = edge_cols[ci]
            brick_c = brick_c.at[:, ec].set(brick_c[:, ec] | seg)
        # 水平连续段：贴顶墙下方 2 行，2-4 格宽（可通行区宽 > 4 才放）
        if (c1 - c0) > 4:
            k4a, k4b, k4c = jax.random.split(k4, 3)
            hstarts = (jax.random.uniform(k4a, ()) * (c1 - c0 - 4)
                       + c0).astype(jnp.int32)
            hlens = 2 + (jax.random.uniform(k4b, ()) * 3).astype(jnp.int32)
            hact = jax.random.uniform(k4c, ()) < WALL_DENSITY
            cols_h = jnp.arange(c0, c1)
            hseg = hact & (cols_h >= hstarts) & (cols_h < hstarts + hlens)
            hr = top + jnp.arange(2)
            for rr in range(2):
                rrow = hr[rr]
                brick_c = brick_c.at[rrow, c0:c1].set(
                    brick_c[rrow, c0:c1] | hseg)

    # ---- open：随机单障碍（不放回 = permutation 取前 n 个）----
    n_obs = jax.random.randint(k5, (), 0, OPEN_OBSTACLE_MAX + 1)
    perm = jax.random.permutation(k5, H * W)
    obs_idx = perm[:OPEN_OBSTACLE_MAX]
    obs_ok = jnp.arange(OPEN_OBSTACLE_MAX) < n_obs
    wall_o = jnp.zeros((H, W), jnp.bool_)
    wall_o = wall_o.at[obs_idx // W, obs_idx % W].set(obs_ok)

    wall = jnp.where(is_corr, wall_c, wall_o)
    wall = jnp.where(is_pure, jnp.zeros((H, W), jnp.bool_), wall)
    brick = jnp.where(is_corr, brick_c, jnp.zeros((H, W), jnp.bool_))

    # 出生点：corridor 空旷区中心 / open 中线均分（模块常量，非 traced）
    spawn = jnp.where(is_corr, CORRIDOR_SPAWN, OPEN_SPAWN)

    # 清出生点四邻（corridor + open 两套都清：cell 是 Python int 常量，
    # 静态索引；比 torch mapgen 只清 cfg.spawn_cells() 更严 —— open 关也
    # 保证脚下/四邻无墙无砖）
    for rr, cc in _CORRIDOR_CELLS + _OPEN_CELLS:
        for dr, dc in ((0, 0),) + _DIRS:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                wall = wall.at[nr, nc].set(False)
                brick = brick.at[nr, nc].set(False)

    # 位置对称化：约一半 env 交换 P0/P1 出生点（属性按 pid 绑定，与出生侧无关）
    sw = jax.random.uniform(k6, ()) < 0.5
    spawn = jnp.where(sw, spawn[::-1], spawn)
    return wall, brick, is_open, spawn


def _fresh(key) -> BombState:
    """新局：随机地图 + 按类型初始属性 + 宝箱池。

    levels 激活时（jax_bomb.levels.set_active）：从 241 张 QQ堂标准化关卡
    加权采样（wall/brick/出生点/初始属性/预置宝箱全来自关卡数据）。
    未激活：过程式生成（_make_map：纯空/open/corridor 混合 + 十字宝箱池）。
    """
    if levels.active() is not None:
        s = levels.active().sample(key)
        fuse = jnp.zeros((H, W), jnp.int32)
        owner = -jnp.ones((H, W), jnp.int32)
        bomb_blast = jnp.zeros((H, W), jnp.int32)
        alive = jnp.ones((2,), jnp.bool_)
        hp = jnp.full((2,), MAX_HP, jnp.int32)
        invuln = jnp.zeros((2,), jnp.int32)
        bombs_cap = jnp.full((2,), s.lo[0], jnp.float32)
        blast_cap = jnp.full((2,), s.lo[1], jnp.float32)
        spd_g = jnp.full((2,), s.lo[2], jnp.float32)
        buffs = jnp.zeros((2,), jnp.int8)          # 预留位（见 BombState 注释）
        debuffs = jnp.zeros((2,), jnp.int8)
        items = jnp.zeros((2, 4), jnp.int8)
        gametype = jnp.zeros((), jnp.int8)
        t = jnp.zeros((), jnp.int32)
        return BombState(s.pos, fuse, owner, bomb_blast, s.wall, s.brick,
                         s.pushable, jnp.zeros((H, W), jnp.float32), s.bush,
                         jnp.where(s.crate, 7, 0).astype(jnp.int8), s.rec,
                         alive, hp, invuln, bombs_cap, blast_cap, spd_g, buffs,
                         debuffs, items, gametype, s.is_open, t, s.level_id)
    wall, brick, is_open, pos = _make_map(key)
    fuse = jnp.zeros((H, W), jnp.int32)
    owner = -jnp.ones((H, W), jnp.int32)
    bomb_blast = jnp.zeros((H, W), jnp.int32)
    bush = jnp.zeros((H, W), jnp.bool_)   # 过程式生成无灌木
    crate = jnp.where(is_open & OPEN_CRATE_CROSS, _cross_crates(),
                      jnp.zeros((H, W), jnp.bool_)).astype(jnp.int8) * 7
    rec_crate = jnp.zeros((H, W), jnp.bool_)
    alive = jnp.ones((2,), jnp.bool_)
    hp = jnp.full((2,), MAX_HP, jnp.int32)
    invuln = jnp.zeros((2,), jnp.int32)
    b0 = jnp.where(is_open, OPEN_GROWTH_BOMBS, GROWTH_BOMBS_START)
    z0 = jnp.where(is_open, OPEN_GROWTH_BLAST, GROWTH_BLAST_START)
    s0 = jnp.where(is_open, OPEN_GROWTH_SPEED, GROWTH_SPEED_START)
    bombs_cap = jnp.full((2,), b0, jnp.float32)
    blast_cap = jnp.full((2,), z0, jnp.float32)
    spd_g = jnp.full((2,), s0, jnp.float32)
    buffs = jnp.zeros((2,), jnp.int8)          # 预留：变身 buff（3 bit）
    debuffs = jnp.zeros((2,), jnp.int8)        # 预留：debuff（2 bit）
    items = jnp.zeros((2, 4), jnp.int8)        # 预留：道具栏（4 槽 × int6）
    gametype = jnp.zeros((), jnp.int8)         # 预留：竞技类型（int4，0=普通对抗）
    t = jnp.zeros((), jnp.int32)
    pushable = jnp.zeros((H, W), jnp.bool_)  # 过程式无可推墙
    push_t = jnp.zeros((H, W), jnp.float32)  # 推箱计时
    return BombState(pos, fuse, owner, bomb_blast, wall, brick, pushable, push_t,
                     bush, crate, rec_crate, alive, hp, invuln, bombs_cap,
                     blast_cap, spd_g, buffs, debuffs, items, gametype,
                     is_open, t, jnp.int32(-1))


def init_batch(key, n: int) -> BombState:
    keys = jax.random.split(key, n)
    return jax.vmap(_fresh)(keys)


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


def _shift4(x4):
    """x4: (4, H, W)，四方向（UP/DOWN/LEFT/RIGHT）各自移位（越界补 0）。

    语义与 _shift 一致（result[i,j] = x[i-drow, j-dcol]），4 方向合并成
    一次 stack —— danger_map 每 tick 的热点（原 4×BLAST 次独立 concat，
    batch 后 concat 数 ÷4，方向并行）。
    """
    zr = jnp.zeros((1, W), x4.dtype)
    zc = jnp.zeros((H, 1), x4.dtype)
    return jnp.stack([
        jnp.concatenate([x4[0, 1:], zr], axis=0),       # UP   (drow=-1)
        jnp.concatenate([zr, x4[1, :-1]], axis=0),      # DOWN (drow=1)
        jnp.concatenate([x4[2, :, 1:], zc], axis=1),    # LEFT (dcol=-1)
        jnp.concatenate([zc, x4[3, :, :-1]], axis=1),   # RIGHT(dcol=1)
    ])


def _spread_all(fw, fd, passable, not_solid):
    """从 (H,W) 波前 fw/剩余距离 fd 出发的 4 方向传播，返回传播覆盖 (H,W)。

    danger_map 阶段 A/B 共用（torch 的 4×BLAST 固定步展开）：
    - 4 方向 batch（_shift4）每轮各推进 1 步（concat 数 ÷4、方向并行）；
    - 固定 BLAST 轮（XLA 对固定循环完全展开融合，实测比 while_loop 按
      `any(fd>0)` 早退快 25% —— 动态 trip count 的 while 每轮带归约且
      编译优化差；多跑的空步 keep=False 零贡献，结果逐位一致）；
    - 墙置 0（passable）、泡/brick 记录后不穿透（not_solid）。
    """
    one = jnp.ones_like(fd)
    fw4 = jnp.stack([fw] * 4)
    fd4 = jnp.stack([fd] * 4)
    p4 = jnp.broadcast_to(passable[None], (4, H, W))
    ns4 = jnp.broadcast_to(not_solid[None], (4, H, W))
    spread = jnp.zeros_like(fw4)
    for _ in range(BLAST):
        fw1 = _shift4(fw4) * p4
        fd1 = _shift4(fd4) * p4 - one
        keep = fd1 >= 0
        fw1 = fw1 * keep
        spread = jnp.maximum(spread, fw1)
        fw1 = fw1 * ns4                      # 泡/brick 记录后不穿透
        fd1 = fd1 * ns4
        fw4, fd4 = fw1, fd1
    return jnp.max(spread, axis=0)


# ---------------- 爆炸传播（同 sim/blast.py 的距离缓冲 v2） ----------------

def _rays(seed, bombed, blast_map, wall, brick=None):
    """从 seed 出发的十字覆盖。泡/brick 挡火但被覆盖；wall 永久墙挡火**且不被
    覆盖**；blast 每格自己的威力。

    对齐 torch sim/blast.py::rays：`fd1 = _shift * ~wall` 在 covered 之前
    置 0 —— 墙格既不覆盖也不穿透（旧版先 covered 后置 0 会让墙格进覆盖图，
    对拍 3 格差异）；brick 与泡同语义（solid 吸收火焰，覆盖先于挡火）。
    固定 max_b = BLAST 轮（多跑空步结果不变）。
    """
    seed = seed & ~wall & ~(brick if brick is not None
                            else jnp.zeros_like(wall))  # 源不在墙/砖上（防御）
    covered = seed
    fd = seed.astype(jnp.int8) * jnp.clip(blast_map, 0, 127).astype(jnp.int8)
    one = jnp.ones_like(fd)
    not_wall = (~wall).astype(jnp.int8)
    solid = bombed | (brick if brick is not None else jnp.zeros_like(wall))
    not_solid = ~solid                   # 泡/brick 挡火（覆盖后不穿透）；墙靠前置置 0
    for drow, dcol in _DIRS:
        fd_p = fd
        for _ in range(BLAST):
            fd1 = _shift(fd_p, drow, dcol) * not_wall   # 墙格置 0（不覆盖不穿透）
            covered = covered | (fd1 > 0)
            fd1 = fd1 - one
            fd1 = fd1 * not_solid.astype(jnp.int8)      # 泡/brick 记录后不穿透
            fd_p = fd1
    return covered


def _reach_dp(solid):
    """solid: (H,W) bool（墙|砖|泡，都挡火）。返回四方向每格"沿该方向到
    第一个 solid 的连续格数"（不含自身；边界算 solid）。

    用户方案的 compute_reach 移植：正向/反向扫描一次算完，无嵌套循环，
    XLA 固定展开 —— 是阶段 A 矩阵版挡火检查的基础。
    """
    up = jnp.zeros((H, W), jnp.int32)
    up = up.at[0, :].set(0)
    for r in range(1, H):
        up = up.at[r, :].set(jnp.where(solid[r - 1], 0, up[r - 1] + 1))
    down = jnp.zeros((H, W), jnp.int32)
    down = down.at[H - 1, :].set(0)
    for r in range(H - 2, -1, -1):
        down = down.at[r, :].set(jnp.where(solid[r + 1], 0, down[r + 1] + 1))
    left = jnp.zeros((H, W), jnp.int32)
    left = left.at[:, 0].set(0)
    for c in range(1, W):
        left = left.at[:, c].set(jnp.where(solid[:, c - 1], 0, left[:, c - 1] + 1))
    right = jnp.zeros((H, W), jnp.int32)
    right = right.at[:, W - 1].set(0)
    for c in range(W - 2, -1, -1):
        right = right.at[:, c].set(jnp.where(solid[:, c + 1], 0, right[:, c + 1] + 1))
    return up, down, left, right


def _stage_a_matrix(weight, blast_f, bombed, wall, brick, max_chain):
    """阶段 A 矩阵版（替代 while 波前接力）：有向 max-plus 传播。

    torch 阶段 A 语义：被更早爆炸的泡的 blast 覆盖的泡 → 爆炸时刻提前到
    组内最早（权重域取最大）。等价于有向图传递闭包：
      - 泡 j 直接覆盖泡 i ⇔ i 在 j 的 blast 十字内（dist ≤ blast_j）且 j→i
        开区间无 solid（wall|brick|bomb 都挡火；用 _reach_dp 距离 ≥ dist-1
        判定 —— 相邻泡开区间 0 格恒真）；
      - 每轮 wv_i = max(wv_i, max_j reach[j,i]·wv_j)：同步广播 = torch 的
        newly 波前接力（每轮全体源用最新权重传播，max 幂等，多跑空轮不变）；
      - 固定 max_chain 轮（= torch while 的 cap；链深 ≤ 泡数 ≤ 20）。
    权重域精确 max、无累积误差 → 与 while 波前版**逐位一致**，但 O(K²)
    矩阵传播替代每轮全图 4×7 传播（K=20），XLA 友好。
    """
    K = MAX_BOMBS * 2
    bomb_flat = bombed.reshape(-1)
    # 每 env 至多 K 颗泡（top_k 取最大 key：True=1 > False=0 → 泡索引在前）
    _, order = jax.lax.top_k(bomb_flat.astype(jnp.int32), K)
    mask = bomb_flat[order]
    cr, cc = order // W, order % W
    wv = jnp.where(mask, weight.reshape(-1)[order], 0.0)
    bv = jnp.where(mask, blast_f.reshape(-1)[order], 0.0)

    up, down, left, right = _reach_dp(wall | brick | bombed)
    # 每泡 i 沿四方向到第一个 solid 的连续格数（不含 i 自身）
    u_i, d_i = up[cr, cc], down[cr, cc]
    l_i, r_i = left[cr, cc], right[cr, cc]

    dist = jnp.abs(cr[:, None] - cr[None, :]) + jnp.abs(cc[:, None] - cc[None, :])
    aligned = (cr[:, None] == cr[None, :]) | (cc[:, None] == cc[None, :])
    in_blast = (dist > 0) & (dist <= bv[:, None])          # 源 j 的 blast
    up_ok = (cr[:, None] < cr[None, :]) & (u_i[None, :] >= dist - 1)
    down_ok = (cr[:, None] > cr[None, :]) & (d_i[None, :] >= dist - 1)
    left_ok = (cc[:, None] < cc[None, :]) & (l_i[None, :] >= dist - 1)
    right_ok = (cc[:, None] > cc[None, :]) & (r_i[None, :] >= dist - 1)
    reach = aligned & in_blast & (up_ok | down_ok | left_ok | right_ok)
    reach = reach & mask[:, None] & mask[None, :]

    # 固定 max_chain 轮有向 max-plus 传播（scan：body 只展开一次，避免
    # Python for 展开 16 份 body 使 HLO 膨胀、ROCm hsaco 编译产物超限；
    # 每轮 (K,K) 小矩阵运算，比 while 波前的全图 4×7 传播便宜得多）
    def prop_body(wv, _):
        recv = (reach.transpose(1, 0) * wv[None, :]).max(axis=-1)
        return jnp.maximum(wv, recv), None

    wv, _ = jax.lax.scan(prop_body, wv, None, length=max_chain)

    # 回填炮格（order 唯一；无效槽写原值）
    weight = weight.at[cr, cc].set(jnp.where(mask, wv, weight[cr, cc]))
    return weight


def _danger_map(fuse, wall, bomb_blast, brick=None, fuse_max=FUSE,
                max_chain=MAX_CHAIN):
    """torch sim/blast.py::danger_map 的 jax 移植（ch5 危险图，逐位对齐）。

    在场所有泡泡的"时空影响范围"，越接近爆炸值越大，落在 (0, 1]：
    - 权重 = 指数压缩的引信（(1-(fuse-1)/FUSE)^2，刚放的泡几乎无色）；
    - 阶段 A（max_chain>1）：炮格间连锁修正（矩阵版有向 max-plus 传播，
      与 torch 波前接力逐位一致，XLA 友好 —— 见 _stage_a_matrix）；
    - 阶段 B：每颗泡沿 4 方向扩散自己的 blast 档距离（固定 BLAST 步，
      多跑空步 keep=False 零贡献），墙不穿不覆盖、泡/brick 记录后不穿透。
    值域天然 0-1，直接作 obs 通道（torch 同款，无额外归一化）。
    """
    bombed = fuse > 0
    brick_t = brick if brick is not None else jnp.zeros_like(wall)
    w_raw = 1.0 - (fuse.astype(jnp.float32) - 1.0) / float(fuse_max)
    weight = jnp.where(bombed, jnp.clip(w_raw, 0.0, None) ** 2,
                       jnp.zeros_like(w_raw))
    blast_f = jnp.where(bomb_blast > 0, bomb_blast, BLAST).astype(jnp.float32)
    not_solid = (~bombed & ~brick_t).astype(jnp.float32)  # 泡/brick 挡火
    passable = (~wall).astype(jnp.float32)

    if max_chain > 1:
        weight = _stage_a_matrix(weight, blast_f, bombed, wall, brick,
                                 max_chain)

    seed = weight * passable                       # 阶段 B：每方向从 seed 出发
    fd_seed = jnp.where(bombed, blast_f, jnp.zeros_like(seed))
    danger = jnp.maximum(seed, _spread_all(seed, fd_seed, passable, not_solid))
    return danger


def _resolve_explosions(fuse, owner, bomb_blast, wall, brick=None,
                        chain_cap=8):
    """返回 (covered, triggered)，连锁最多 chain_cap 轮。

    **与 torch 生产对齐**（torch_sim._resolve_c 的 chain_cap = chain_cap_rounds
    =8，见 2026-08-17 对拍修复：原 resolve cap=4 在满级 blast=7 的密集泡阵
    链深 >4 时漏爆）。while_loop 动态早退：实际轮数 = 链深，cap 只是上限，
    调大不增加计算（jax 无 host 同步）。链深 >cap 时与 torch 固定轮截断
    方式一致 → 覆盖/伤害逐位对齐。brick 挡火但被覆盖（同 torch _resolve_c）。
    """
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    blast_map = jnp.where(bomb_blast > 0, bomb_blast, BLAST)
    covered = _rays(triggered, live, blast_map, wall, brick)

    def cond(c):
        i, newly, _cov, _trig = c
        return (i < chain_cap) & jnp.any(newly)

    def body(c):
        i, newly, covered, triggered = c
        covered2 = covered | _rays(newly, live, blast_map, wall, brick)
        triggered2 = triggered | newly
        newly2 = live & covered2 & ~triggered2
        return i + 1, newly2, covered2, triggered2

    _, _, covered, triggered = jax.lax.while_loop(
        cond, body, (0, live & covered & ~triggered, covered, triggered))
    return covered, triggered


def _resolve_explosions_matrix(fuse, owner, bomb_blast, wall, brick=None,
                               chain_cap=8):
    """矩阵版爆炸连锁：布尔传递闭包 + 一次 _rays，替代 while 波前接力。

    torch 连锁语义（_resolve_c，chain_cap=8）：初始 triggered（fuse==0）泡
    传播 blast → 被覆盖的 live 泡变 newly 下轮爆 → 直到无新增或到 cap。
    每轮 `_rays(newly, live, ...)` 的挡火集合**恒定**（live 泡 + brick + wall，
    triggered 泡已爆不挡火），`_rays` 对源集合线性（∪_seeds 传播 = 各源
    传播之并）→ 最终 covered = 从**所有爆泡**一次 _rays 的覆盖并集。
    因此只需求出全部爆泡集合：
      - 有向边 j→i：泡 j 的 blast 十字覆盖 live 泡 i，j→i 开区间无
        solid（wall|brick|live 泡；triggered 泡不挡火 → 不在 solid 里）；
      - 初始爆集 = triggered；每轮爆集 |= 被覆盖的 live 泡（布尔 OR 传播，
        固定 chain_cap 轮 = torch 截断语义，多跑空轮零变化）；
      - covered = _rays(爆集, live, blast_map, wall, brick) 一次。
    返回 (covered, 爆集)，爆集 = 原 while 的 triggered 累积（含连锁新增），
    step 清场直接用。与 while 波前版逐位一致（对拍验证），但每次连锁只
    一次全图传播（4 方向 × BLAST 步）替代最多 chain_cap 轮，XLA 友好。
    """
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    blast_map = jnp.where(bomb_blast > 0, bomb_blast, BLAST)

    K = MAX_BOMBS * 2
    bomb_flat = (live | triggered).reshape(-1)
    _, order = jax.lax.top_k(bomb_flat.astype(jnp.int32), K)
    mask = bomb_flat[order]
    cr, cc = order // W, order % W
    burst = jnp.where(mask, triggered.reshape(-1)[order], False)
    live_node = jnp.where(mask, live.reshape(-1)[order], False)
    bv = jnp.where(mask, blast_map.reshape(-1)[order], 0)

    up, down, left, right = _reach_dp(wall | brick | live)
    u_i, d_i = up[cr, cc], down[cr, cc]
    l_i, r_i = left[cr, cc], right[cr, cc]

    dist = jnp.abs(cr[:, None] - cr[None, :]) + jnp.abs(cc[:, None] - cc[None, :])
    aligned = (cr[:, None] == cr[None, :]) | (cc[:, None] == cc[None, :])
    in_blast = (dist > 0) & (dist <= bv[:, None])
    up_ok = (cr[:, None] < cr[None, :]) & (u_i[None, :] >= dist - 1)
    down_ok = (cr[:, None] > cr[None, :]) & (d_i[None, :] >= dist - 1)
    left_ok = (cc[:, None] < cc[None, :]) & (l_i[None, :] >= dist - 1)
    right_ok = (cc[:, None] > cc[None, :]) & (r_i[None, :] >= dist - 1)
    reach = aligned & in_blast & (up_ok | down_ok | left_ok | right_ok)
    reach = reach & mask[:, None] & mask[None, :] & live_node[None, :]

    def prop_body(burst, _):
        recv = (reach.transpose(1, 0) & burst[None, :]).any(axis=-1)
        return burst | recv, None

    burst, _ = jax.lax.scan(prop_body, burst, None, length=chain_cap)

    burst_map = jnp.zeros((H, W), jnp.bool_)
    burst_map = burst_map.at[cr, cc].set(jnp.where(mask, burst, False))
    covered = _rays(burst_map, live, blast_map, wall, brick)
    return covered, burst_map


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
    # 盒覆盖豁免：与 JS sim.js impassable 一致 —— 下界 floor、上界 **ceil** +
    # 严格小于（左闭右开）。盒右/下缘恰贴格边界（y+R=整数）时不覆盖下一格，
    # 泡照常挡 → 防穿炮；闭区间会误判"压着"放行（旧实现，已对齐 JS）。
    r0c = jnp.floor(y - RADIUS).astype(jnp.int32)
    r1c = jnp.ceil(y + RADIUS).astype(jnp.int32)
    c0c = jnp.floor(x - RADIUS).astype(jnp.int32)
    c1c = jnp.ceil(x + RADIUS).astype(jnp.int32)
    in0 = (r0 >= r0c) & (r0 < r1c) & (c0 >= c0c) & (c0 < c1c)
    in1 = (r1 >= r0c) & (r1 < r1c) & (c1 >= c0c) & (c1 < c1c)
    return oob | (solid[0] & ~in0) | (solid[1] & ~in1)


def _resolve_axis(coord, delta, other, y, x, blocked_flat, vertical):
    """沿单轴消解碰撞：撞上贴着障碍停下（滑动）。同 move.py _resolve_axis。

    跨格保护：JAX 单 tick 位移 = STEP×spd_g 可达 0.63 格（spd_g 2.1），前缘扫过
    lo..hi 跨 ≤2 格。仍逐格扫描全部 lead（上限 MAX_SWEEP=3 余量），**终点侧优先**
    （对齐 JS/torch firstLead 语义：两 lead 都堵时停在终点侧前，贴地图边 span
    越界场景才出现两 lead 同堵）；中心路径硬约束（_move_player）兜底防穿。
    """
    sgn = jnp.sign(delta)
    old_lead = jnp.floor(coord - delta + sgn * RADIUS).astype(jnp.int32)
    new_lead = jnp.floor(coord + sgn * RADIUS).astype(jnp.int32)
    lo = jnp.minimum(old_lead, new_lead)
    hi = jnp.maximum(old_lead, new_lead)
    span0 = jnp.floor(other - RADIUS).astype(jnp.int32)
    span1 = jnp.floor(other + RADIUS).astype(jnp.int32)

    def hit_at(lead):
        if vertical:
            return _impassable_pair(blocked_flat, lead, span0, lead, span1, y, x)
        return _impassable_pair(blocked_flat, span0, lead, span1, lead, y, x)

    sweep = hi - lo + 1                       # 扫过的格数（动态，≤ MAX_SWEEP）

    def body(i, carry):
        # 终点侧优先：向下（sgn>0）hi→lo，向上 lo→hi，取**第一个**障碍贴停
        # （与 JS/torch 的 firstLead 一致；中间格也扫，防止跨格跳过砖）
        lead = jnp.where(sgn > 0, hi - i, lo + i)
        hit = hit_at(lead) & (i < sweep)
        found, first = carry
        first = jnp.where(hit & ~found, lead, first)
        return found | hit, first

    found, first = jax.lax.fori_loop(0, MAX_SWEEP, body,
                                     (jnp.zeros((), jnp.bool_),
                                      jnp.zeros_like(lo)))
    stop_pos = jnp.where(sgn > 0,
                         first.astype(jnp.float32) - RADIUS - EPS,
                         first.astype(jnp.float32) + 1.0 + RADIUS + EPS)
    return jnp.where(found, stop_pos, coord)


def _move_player(pos_me, move, alive_me, blocked, spd=1.0):
    """单玩家移动：连续坐标 + AABB 滑动碰撞（角色互不碰撞）。

    位移 = STEP × spd_g（速度成长倍率，torch spd_p = spd_g × step_len 同义）。
    方向编码同 torch：0=上 1=下 2=左 3=右 4=停(IDLE)。
    """
    delta = _MOVE_DELTA[move] * STEP * spd
    moving = alive_me & (move != 4)              # MOVE_IDLE=4（torch 编码）
    delta = jnp.where(moving, delta, jnp.zeros_like(delta))
    dy, dx = delta[0], delta[1]
    y, x = pos_me[0], pos_me[1]
    blocked_flat = blocked.reshape(-1)
    ny = _resolve_axis(y + dy, dy, x, y, x, blocked_flat, True)
    # 中心路径硬约束（防穿炮/穿墙）：中心点扫过的每一格都必须可通行（起点格
    # 脚下豁免）。_impassable_pair 的"盒覆盖格豁免"允许碰撞盒擦边探入，
    # 但中心不能进入泡/墙/砖格——布置炸弹后可以贴着泡走，但不能踩回泡格
    # 中心（更不能穿炮）。逐轴检查：y 段（列=起点列）、x 段（行=ny 行）。
    start_r = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, H - 1)
    start_c = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, W - 1)
    y_lo = jnp.clip(jnp.floor(jnp.minimum(y, ny)).astype(jnp.int32), 0, H - 1)
    y_hi = jnp.clip(jnp.floor(jnp.maximum(y, ny)).astype(jnp.int32), 0, H - 1)
    # 动态 slice（vmap 下切片索引必须静态）：固定窗口 MAX_SWEEP 行，掩码排除
    seg_y = jax.lax.dynamic_slice(blocked, (y_lo, start_c), (MAX_SWEEP, 1))[:, 0]
    rows = jnp.arange(MAX_SWEEP) + y_lo
    seg_y = jnp.where((rows <= y_hi) & (rows != start_r), seg_y, False)
    ok_y = ~jnp.any(seg_y)
    ny = jnp.where(ok_y, ny, y)                  # 中心会进泡/墙/砖 → 该轴回弹
    nx = _resolve_axis(x + dx, dx, y, ny, x, blocked_flat, False)
    x_lo = jnp.clip(jnp.floor(jnp.minimum(x, nx)).astype(jnp.int32), 0, W - 1)
    x_hi = jnp.clip(jnp.floor(jnp.maximum(x, nx)).astype(jnp.int32), 0, W - 1)
    cy0 = jnp.clip(jnp.floor(ny).astype(jnp.int32), 0, H - 1)
    seg_x = jax.lax.dynamic_slice(blocked, (cy0, x_lo), (1, MAX_SWEEP))[0]
    cols = jnp.arange(MAX_SWEEP) + x_lo
    seg_x = jnp.where((cols <= x_hi)
                      & ~((cols == start_c) & (cy0 == start_r)), seg_x, False)
    ok_x = ~jnp.any(seg_x)
    nx = jnp.where(ok_x, nx, x)
    out_y = jnp.where(dy != 0, ny, y)
    out_x = jnp.where(dx != 0, nx, x)
    out_y = jnp.clip(out_y, RADIUS, H - RADIUS)
    out_x = jnp.clip(out_x, RADIUS, W - RADIUS)
    return jnp.stack([out_y, out_x])


def _steer(pos_me, target_dir, alive_me, blocked, spd=1.0, pushable=None):
    """贪婪转向适配器：模型输出=目标相邻格，转向器选第一个能动的方向。

    优先级：直走(target_dir) > 垂直偏转1 > 垂直偏转2。垂直回退方向按目标
    行/列的**斜对角开闭**排序：楔死时朝开口侧滑——朝墙侧滑永远进不了目标
    行/列，会背向目标绕路（点上方开口却左滑绕墙）；两边同开/同堵时朝
    当前格中心线归中，避免固定偏左/偏上把玩家带向错误一侧。每 tick无状态
    决策，底层 _move_player 不变。
    """
    # IDLE(4) 不参与转向：_PERP 无第 5 行，JAX 越界索引会静默 clamp 成
    # [0,1]（上/下）→ IDLE 玩家被转向器推走
    idle = target_dir == 4
    p = _move_player(pos_me, target_dir, alive_me, blocked, spd)
    moved_dist = jnp.abs(p - pos_me).sum()
    moved = moved_dist > 2 * EPS
    full_step = moved_dist >= (STEP * spd) * 0.95
    y, x = pos_me[0], pos_me[1]
    r0 = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, H - 1)
    c0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, W - 1)
    if pushable is None:
        pushable = jnp.zeros_like(blocked)
    dr = jnp.where(target_dir == 0, -1, jnp.where(target_dir == 1, 1, 0))
    dc = jnp.where(target_dir == 2, -1, jnp.where(target_dir == 3, 1, 0))
    ptr = jnp.clip(r0 + dr, 0, H - 1)
    ptc = jnp.clip(c0 + dc, 0, W - 1)
    push_target = (target_dir < 4) & pushable[ptr, ptc]
    vert = target_dir < 2
    tr = jnp.clip(r0 + jnp.where(target_dir == 0, -1, 1), 0, H - 1)   # 目标行
    tc = jnp.clip(c0 + jnp.where(target_dir == 2, -1, 1), 0, W - 1)   # 目标列
    diag_l = (c0 - 1 < 0) | blocked[tr, jnp.clip(c0 - 1, 0, W - 1)]
    diag_r = (c0 + 1 >= W) | blocked[tr, jnp.clip(c0 + 1, 0, W - 1)]
    diag_u = (r0 - 1 < 0) | blocked[jnp.clip(r0 - 1, 0, H - 1), tc]
    diag_d = (r0 + 1 >= H) | blocked[jnp.clip(r0 + 1, 0, H - 1), tc]
    # _PERP 的默认顺序是 左→右 / 上→下。仅一侧开放时优先开放侧；两侧
    # 状态相同时，按连续坐标朝当前格中心线归中（与 Web Sim._steer 对齐）。
    same_lr = diag_l == diag_r
    same_ud = diag_u == diag_d
    swap_vert = (diag_l & ~diag_r) | (same_lr & (x < c0.astype(jnp.float32) + 0.5))
    swap_horz = (diag_u & ~diag_d) | (same_ud & (y < r0.astype(jnp.float32) + 0.5))
    swap = jnp.where(vert, swap_vert, swap_horz)
    perp = _PERP[target_dir]                     # (2,)——IDLE 已在出口拦截
    p1 = _move_player(pos_me, perp[0], alive_me, blocked, spd)
    p2 = _move_player(pos_me, perp[1], alive_me, blocked, spd)
    # 一格宽通道内，侧移只钳制到当前格中心线，防止越过中心后下一 tick
    # 反向修正形成振荡。仅一侧开放的真正绕障碍场景不做钳制。
    both_blocked = jnp.where(vert, diag_l & diag_r, diag_u & diag_d)
    center_x = c0.astype(jnp.float32) + 0.5
    center_y = r0.astype(jnp.float32) + 0.5
    p1_vert_x = jnp.where(perp[0] == 2, jnp.maximum(p1[1], center_x),
                          jnp.minimum(p1[1], center_x))
    p2_vert_x = jnp.where(perp[1] == 2, jnp.maximum(p2[1], center_x),
                          jnp.minimum(p2[1], center_x))
    p1_horz_y = jnp.where(perp[0] == 0, jnp.maximum(p1[0], center_y),
                          jnp.minimum(p1[0], center_y))
    p2_horz_y = jnp.where(perp[1] == 0, jnp.maximum(p2[0], center_y),
                          jnp.minimum(p2[0], center_y))
    p1 = p1.at[1].set(jnp.where(both_blocked & vert, p1_vert_x, p1[1]))
    p2 = p2.at[1].set(jnp.where(both_blocked & vert, p2_vert_x, p2[1]))
    p1 = p1.at[0].set(jnp.where(both_blocked & ~vert, p1_horz_y, p1[0]))
    p2 = p2.at[0].set(jnp.where(both_blocked & ~vert, p2_horz_y, p2[0]))
    m1 = jnp.abs(p1 - pos_me).sum() > 2 * EPS
    m2 = jnp.abs(p2 - pos_me).sum() > 2 * EPS
    fa = jnp.where(swap, p2, p1)
    ma = jnp.where(swap, m2, m1)
    fb = jnp.where(swap, p1, p2)
    mb = jnp.where(swap, m1, m2)
    # 顶着可推箱时保持直走结果（通常是原地，接近箱子时可能是半步），让
    # step 的 push_t 连续累计到 0.3s；不能进入垂直兜底。
    out = jnp.where(push_target, p,
                    jnp.where(full_step, p,
                              jnp.where(ma, fa, jnp.where(mb, fb,
                                        jnp.where(moved, p, pos_me)))))
    return jnp.where(idle, pos_me, out)


# ---------------- step（对齐 torch_sim.step 的顺序） ----------------

def _scatter_recycle(key, wall, brick, bush, crate, rec_crate, lost):
    """掉血回收：每玩家被扣的层数 → 纯地面格随机撒宝箱。

    对齐 torch _scatter_recycle（语义）：排除出生点四邻 + open 十字带
    （_recycle_excl）；**只落纯粹地面**——墙/砖/灌木都不可落（灌木可通行
    但非地面，回收箱不允许停在灌木上）；不放回抽样（permutation 全图取前
    N 个可用候选），掉 N 层必现 N 个新箱（严格守恒）；回收箱标记 rec_crate
    （踩到必升）。lost 是 (2,) int32（掉血玩家被扣层数，非掉血者恒 0）；
    total=0 时 at 空集无操作（RNG 白取几个数，无副作用）。
    """
    total = lost.sum()
    avail = (~wall & ~brick & ~bush) & ~_recycle_excl()
    perm = jax.random.permutation(key, H * W)
    sel = perm[:MAX_RECYCLE]
    good = avail.reshape(-1)[sel]                    # 候选是否可落
    order = jnp.cumsum(good) - 1                     # 可用候选的序号
    valid = good & (order >= 0) & (order < total)    # 取前 total 个**可用**候选
    # jax 禁用非具体 bool 索引 → 用 where 掩码写回（无效候选写原值）
    y = jnp.clip(sel // W, 0, H - 1)
    x = jnp.clip(sel % W, 0, W - 1)
    crate = crate.at[y, x].set(jnp.where(valid, 7, crate[y, x]))   # 7=问号随机
    rec_crate = rec_crate.at[y, x].set(
        jnp.where(valid, True, rec_crate[y, x]))
    return crate, rec_crate


def step(state: BombState, actions: jnp.ndarray, key, auto_reset: bool = True,
         return_info: bool = False):
    """actions: (2, 2) int32 = [dir(0-4), bomb(0/1)]，每玩家。

    return_info=False：返回 (state, done)（默认，现有调用方不变）。
    return_info=True：返回 (state, done, info)，info = {"alive": alive_new,
    "hp": hp_new, "dmg": hit_eff.astype(int32), "cell": ccell} —— 终局
    结算后、auto_reset 重置**前**的存活/血量/掉血/落点快照，供 collect_rollout
    计算稠密奖励
    （对齐 torch step 返回的 reward 所需信号：掉血 dmg、终局 alive/hp）。
    auto_reset=True 时 info 取**结算后**值（不是重置后），否则奖励会算到
    新局头上。Python 静态分支，jit 折叠，无运行时开销。

    key：地图生成（auto_reset）/ 掉血回收 / 宝箱拾取的 RNG。语义对齐
    torch_sim.step + torch_sim.hit_attr_penalty 块 + 宝箱拾取块：
      放泡/威力按各自成长上限（bombs_cap/blast_cap）；速度 × spd_g；
      brick 挡火但被覆盖，炸掉的砖 → 宝箱；掉血扣 泡/威/速 各 2 层并回收
      成随机宝箱；站在宝箱上掷爆率开箱成长（open 必升、回收箱必升）。
    auto_reset=True（训练/收集默认）：终局就地重置为新局（随机新地图）。
    False（对拍用）：终局保留原状态。Python 静态分支，jit 折叠，无运行时开销。
    """
    (pos, fuse, owner, bomb_blast, wall, brick, pushable, push_t, bush, crate,
     rec_crate, alive, hp, invuln, bombs_cap, blast_cap, spd_g, _buffs,
     _debuffs, _items, _gametype, is_open, t, level_id) = state
    dirs, bombs = actions[:, 0], actions[:, 1]
    alive0 = alive

    # 1. 引信递减
    fuse = jnp.where(fuse > 0, fuse - 1, fuse)

    # 2. 放泡（移动前，落在起始中心格；威力按此刻档位快照；墙/砖格不可放）
    cell = pos.astype(jnp.int32)                 # center_cell = floor
    for me in range(2):
        y, x = cell[me, 0], cell[me, 1]
        idx = jnp.clip(y, 0, H - 1) * W + jnp.clip(x, 0, W - 1)
        cur_f = fuse.reshape(-1)[idx]
        live = ((owner == me) & (fuse > 0)).sum()
        solid_cell = (wall | brick).reshape(-1)[idx]
        ok = (alive0[me] & (bombs[me] == 1) & (cur_f <= 0)
              & (live < bombs_cap[me]) & ~solid_cell)
        fuse = fuse.at[y, x].set(jnp.where(ok, FUSE, cur_f))
        owner = owner.at[y, x].set(jnp.where(ok, me, owner[y, x]))
        bomb_blast = bomb_blast.at[y, x].set(
            jnp.where(ok, blast_cap[me].astype(jnp.int32),
                      bomb_blast[y, x]))

    # 2.5 推箱子（对齐 Web sim.js:373-410）：玩家**前缘**顶着可推箱时，
    #     持续推 ≥PUSH_TIME(0.3s = 3 tick) 箱子移一格；目标格有墙/砖/泡/
    #     道具/其他箱 → 推不动（计时清零）。箱子格 = brick（blocked 挡玩家，
    #     本 tick 玩家顶箱不动，箱子移走后下 tick 跟进）。逐格模型：数据
    #     里 push_boxes 全是 1×1 单格箱，每格独立计时推动。
    for me in range(2):
        y, x = pos[me, 0], pos[me, 1]
        dy = jnp.where(dirs[me] == 0, -1, jnp.where(dirs[me] == 1, 1, 0))
        dx = jnp.where(dirs[me] == 2, -1, jnp.where(dirs[me] == 3, 1, 0))
        # 前缘格（Web: dy>0 → floor(y+radius+EPS*8)；dy<0 → ceil(y-radius)-1）
        pr = jnp.where(dy > 0, jnp.floor(y + RADIUS + EPS * 8),
                       jnp.where(dy < 0, jnp.ceil(y - RADIUS) - 1,
                                 jnp.floor(y)))
        pc = jnp.where(dx > 0, jnp.floor(x + RADIUS + EPS * 8),
                       jnp.where(dx < 0, jnp.ceil(x - RADIUS) - 1,
                                 jnp.floor(x)))
        in_f = (pr >= 0) & (pr < H) & (pc >= 0) & (pc < W)  # Web: 前缘出界 → 无箱可推
        pr = jnp.clip(pr, 0, H - 1).astype(jnp.int32)
        pc = jnp.clip(pc, 0, W - 1).astype(jnp.int32)
        pi = pr * W + pc
        fl = pushable.reshape(-1)[pi]               # 前缘格是可推箱
        ti = pi + dy * W + dx
        in_b = (ti >= 0) & (ti < H * W)
        ti_c = jnp.clip(ti, 0, H * W - 1)
        ok = (in_b & ~wall.reshape(-1)[ti_c] & ~brick.reshape(-1)[ti_c]
              & (fuse.reshape(-1)[ti_c] <= 0)
              & (crate.reshape(-1)[ti_c] == 0)
              & ~pushable.reshape(-1)[ti_c])
        push_me = alive0[me] & (dirs[me] != 4) & fl & in_f
        pt = push_t.reshape(-1)[pi]
        # 写入必须用 2D 索引（flat 索引在 (H,W) 数组上会被静默 clamp 到末行）
        tr, tc = ti_c // W, ti_c % W                # 目标格 2D（~in_b 时 do_move=False 写入无效）
        push_t = push_t.at[pr, pc].set(
            jnp.where(push_me & ok, pt + 0.1,
                      jnp.where(push_me, 0.0, pt)))   # 推不动清零
        do_move = push_me & ok & ((pt + 0.1) >= PUSH_TIME)
        brick = brick.at[pr, pc].set(jnp.where(do_move, False, brick[pr, pc]))
        brick = brick.at[tr, tc].set(jnp.where(do_move, True, brick[tr, tc]))
        pushable = pushable.at[pr, pc].set(
            jnp.where(do_move, False, pushable[pr, pc]))
        pushable = pushable.at[tr, tc].set(
            jnp.where(do_move, True, pushable[tr, tc]))
        # 箱子移走 → 原格计时清零；新格计时清零（Web: pushT[box.o]=0，箱随人走计时归零）
        push_t = push_t.at[pr, pc].set(jnp.where(do_move, 0.0, push_t[pr, pc]))
        push_t = push_t.at[tr, tc].set(jnp.where(do_move, 0.0, push_t[tr, tc]))

    # 3. 移动（blocked = 泡 | 墙 | 砖；位移 × spd_g）—— _steer 贪婪转向：
    #    模型输出=目标相邻格，直走被挡自动试垂直方向（对齐 Web Sim._steer）
    blocked = (fuse > 0) | wall | brick
    p0 = _steer(pos[0], dirs[0], alive0[0], blocked, spd_g[0], pushable)
    p1 = _steer(pos[1], dirs[1], alive0[1], blocked, spd_g[1], pushable)
    newpos = jnp.stack([p0, p1])

    # 4. 爆炸与连锁（墙挡火不覆盖；brick 挡火但被覆盖）
    covered, triggered = _resolve_explosions_matrix(fuse, owner, bomb_blast,
                                                    wall, brick)
    # 4b. 炸掉的砖/灌木 → 宝箱（JS 语义：被覆盖瞬间按本关 crate_rate 掷爆率，
    #      bush 与 brick 互斥且同规则；open 无砖无灌木恒无操作；踩到必升见 6b）
    if levels.active() is not None:
        rate_b = levels.active().rate[jnp.maximum(level_id, 0)]
        super_f = levels.active().super_f[jnp.maximum(level_id, 0)]
    else:
        rate_b = jnp.where(is_open, 1.0, CRATE_PROB)
        super_f = jnp.zeros((), jnp.float32)        # 过程式无超级道具
    destroy = (brick | bush) & covered
    # 与 Web 一致（sim.js:349-359）：crate_rate 判定掉落 → superFraction 判定
    # 超级（+4档）→ floor(rng*3) 定种类（0=泡 1=威 2=速）。编码：
    # 1/2/3 = 泡/威/速 +1；4/5/6 = 超级泡/威/速 +4
    k0, k1, k2 = jax.random.split(key, 3)
    drop = destroy & (jax.random.uniform(k0, (H, W)) < rate_b)
    is_super = jax.random.uniform(k1, (H, W)) < super_f
    kind = (jax.random.uniform(k2, (H, W)) * 3).astype(jnp.int32)
    crate = jnp.where(drop, (1 + kind + is_super.astype(jnp.int32) * 3)
                           .astype(jnp.int8), crate)
    brick = brick & ~destroy
    bush = bush & ~destroy
    pushable = pushable & ~destroy   # 可推箱被炸 → 箱子消失（brick 同步清）
    push_t = jnp.where(destroy, 0.0, push_t)  # 箱子没了 → 该格计时一并清零

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

    # 5b. 掉血惩罚 + 宝箱回收（对齐 torch hit_attr_penalty 块）：被炸掉血的
    #     玩家泡/威/速各扣 2 层（clamp 回模式起点），扣掉的以宝箱随机回收
    #     （每扣 1 层回收 1 箱，总量守恒；起点时扣 0 → 不掉层也不生箱）。
    if HIT_ATTR_PENALTY > 0:
        pen = HIT_ATTR_PENALTY
        if levels.active() is not None:
            # 关卡模式：掉血 clamp 回本关初始属性（level_id < 0 时防御性回退）
            lo_all = levels.active().lo[jnp.maximum(level_id, 0)]
            lo_bombs, lo_blast, lo_spd = lo_all[0], lo_all[1], lo_all[2]
        else:
            lo_bombs = jnp.where(is_open, OPEN_GROWTH_BOMBS,
                                 GROWTH_BOMBS_START).astype(jnp.float32)
            lo_blast = jnp.where(is_open, OPEN_GROWTH_BLAST,
                                 GROWTH_BLAST_START).astype(jnp.float32)
            lo_spd = jnp.where(is_open, OPEN_GROWTH_SPEED,
                               GROWTH_SPEED_START).astype(jnp.float32)
        nb = jnp.clip(bombs_cap - pen, lo_bombs, None)
        nz = jnp.clip(blast_cap - pen, lo_blast, None)
        ns = jnp.maximum(spd_g - pen * GROWTH_SPEED_STEP, lo_spd)
        lost_all = ((bombs_cap - nb) + (blast_cap - nz)
                    + jnp.round((spd_g - ns) / GROWTH_SPEED_STEP)
                    ).astype(jnp.int32)
        hit_any = hit_eff                       # 实际掉血（无敌期不掉血不扣属性）
        bombs_cap = jnp.where(hit_any, nb, bombs_cap)
        blast_cap = jnp.where(hit_any, nz, blast_cap)
        spd_g = jnp.where(hit_any, ns, spd_g)
        lost_eff = lost_all * hit_any.astype(jnp.int32)
        # 没掉血（绝大多数 tick）跳过全图 permutation + scatter（性能）
        crate, rec_crate = jax.lax.cond(
            lost_eff.sum() > 0,
            lambda k: _scatter_recycle(k, wall, brick, bush, crate, rec_crate,
                                       lost_eff),
            lambda _: (crate, rec_crate), key)

    # 6. 清场（爆炸泡 fuse→0、owner→-1、威力→0）
    fuse = jnp.where(triggered, 0, fuse)
    owner = jnp.where(triggered, -1, owner)
    bomb_blast = jnp.where(triggered, 0, bomb_blast)

    # 6b. 道具拾取（与 Web sim.js:410-448 一致）：爆率已在炸开时判定、踩到必升；
    #     编码解释：1-3=泡/威/速+1，4-6=超级+4，7=问号随机（预置宝箱/回收箱）。
    #     成长 clamp 到**每关上限**（levels.json bombs_max/blast_max/speed_max）。
    key2, key3 = jax.random.split(key, 2)
    ccell2 = jnp.stack([jnp.clip(ccell[:, 0], 0, H - 1),
                        jnp.clip(ccell[:, 1], 0, W - 1)], axis=-1)
    cy2, cx2 = ccell2[:, 0], ccell2[:, 1]
    flat = cy2 * W + cx2
    stood = (crate > 0).reshape(-1)[flat]        # (2,) 脚下有道具
    kind_code = crate.reshape(-1)[flat].astype(jnp.int32)   # 1-7
    rb_grow = jax.random.uniform(key3, (2,))
    hits = stood & alive0                        # torch 用 alive0（死人不开箱）
    # A crate is shared map state. Resolve a simultaneous same-cell pickup once,
    # with P0 matching the deterministic turn order used elsewhere in the sim.
    same_crate_cell = (cy2[0] == cy2[1]) & (cx2[0] == cx2[1])
    hits = hits.at[1].set(hits[1] & ~(same_crate_cell & hits[0]))
    fixed = (kind_code - 1) % 3                  # 1/2/3→0/1/2；4/5/6→0/1/2（超级同种类）
    attr = jnp.where(kind_code == 7, (rb_grow * 3).astype(jnp.int32), fixed)
    is_super = (kind_code >= 4) & (kind_code <= 6)
    add = jnp.where(is_super, 4, 1).astype(jnp.int32) * hits.astype(jnp.int32)
    add_b = ((attr == 0).astype(jnp.int32)) * add
    add_z = ((attr == 1).astype(jnp.int32)) * add
    add_s = ((attr == 2).astype(jnp.float32)) * add.astype(jnp.float32)
    if levels.active() is not None:
        caps = levels.active().caps[jnp.maximum(level_id, 0)]
    else:
        caps = jnp.array([GROWTH_BOMBS_MAX, GROWTH_BLAST_MAX, GROWTH_SPEED_MAX],
                         jnp.float32)
    # Capped pickups are still consumed, but only actual attribute growth is a
    # reward event. Snapshots are after this tick's hit penalty, so a pickup that
    # restores a just-lost attribute remains meaningful.
    prev_bombs_cap = bombs_cap
    prev_blast_cap = blast_cap
    prev_spd_g = spd_g
    bombs_cap = jnp.clip(bombs_cap + add_b, None, caps[0])
    blast_cap = jnp.clip(blast_cap + add_z, None, caps[1])
    spd_g = jnp.minimum(spd_g + add_s * GROWTH_SPEED_STEP, caps[2])
    grew = hits & ((bombs_cap > prev_bombs_cap)
                   | (blast_cap > prev_blast_cap)
                   | (spd_g > prev_spd_g))
    crate = crate.at[cy2, cx2].set(0)
    rec_crate = rec_crate.at[cy2, cx2].set(False)

    # 7. 计步与终局（done 后就地重置 = 正式版 auto_reset）
    t = t + 1
    n_alive = alive_new.sum()
    done = (n_alive <= 1) | (t >= MAX_STEPS)
    new_state = BombState(newpos, fuse, owner, bomb_blast, wall, brick,
                          pushable, push_t, bush, crate, rec_crate, alive_new,
                          hp_new, invuln, bombs_cap, blast_cap, spd_g, _buffs,
                          _debuffs, _items, _gametype, is_open, t, level_id)
    if auto_reset:
        out = jax.lax.cond(done, lambda _: _fresh(key), lambda _: new_state,
                           None)
        if return_info:
            return out, done, {"alive": alive_new, "hp": hp_new,
                               "dmg": hit_eff.astype(jnp.int32),
                               "cell": ccell2,
                               "crate": grew,
                               "walls": destroy.sum().astype(jnp.int32)}
        return out, done
    if return_info:
        return new_state, done, {"alive": alive_new, "hp": hp_new,
                                 "dmg": hit_eff.astype(jnp.int32),
                                 "cell": ccell2,
                                 "crate": grew,
                                 "walls": destroy.sum().astype(jnp.int32)}
    return new_state, done


# ---------------- 合法动作掩码（同 sim/obs.py::legal_mask 语义） ----------------

def legal_mask(state: BombState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """返回 (move_mask (2,5) bool, bomb_mask (2,2) bool)。

    方向编码同 torch：0=上 1=下 2=左 3=右 4=停(IDLE)。动作语义=目标相邻格：
    掩码只查目标格是否为障碍（O(1) 格查表，不探测碰撞物理）——模型看障碍
    图即可直接读出答案。目标格是可推箱 → 合法（推箱豁免）；目标格是泡/墙/
    砖 → 非法。IDLE(4) 恒合法；死亡角色整行放开（动作不会被执行，调用方免
    特殊分支）。放泡掩码：存活 & 不在墙/砖格 & 脚下无泡 & 未超成长上限；
    bomb=0（不放）恒合法，死亡角色 bomb=1 也放开。
    """
    (pos, fuse, owner, _bb, wall, brick, pushable, _pt, _bush, _crate, _rc, alive,
     _hp, _invuln, bombs_cap, _blast_cap, spd_g, _buff, _debuff, _item, _gtype,
     _is_open, _t, _level_id) = state
    # 推箱格豁免：mask 把朝可推箱方向标记为合法（模型才会选这个方向去推），
    # 但 step() 的移动仍用含 brick 的 blocked 挡住玩家（推箱期间贴箱不动，
    # 3 tick 后箱子移走玩家跟进）。
    blocked = (fuse > 0) | wall | (brick & ~pushable)
    # 目标格查询：cell + DIRS 的 4 个相邻格，blocked 查表（不调 _move_player）
    cell = jnp.clip(pos.astype(jnp.int32), 0, jnp.array([H - 1, W - 1]))
    targets = cell[:, None, :] + jnp.array(_DIRS, jnp.int32)     # (2,4,2)
    targets = jnp.clip(targets, 0, jnp.array([H - 1, W - 1]))[None, :, :, :]
    target_idx = targets[..., 0] * W + targets[..., 1]           # (1,2,4)
    target_blocked = blocked.reshape(-1)[target_idx]             # (1,2,4)
    move = jnp.zeros((2, 5), jnp.bool_)
    move = move.at[:, :4].set(~target_blocked[0])
    move = move.at[:, 4].set(True)                      # IDLE 恒合法
    move = (move & alive[:, None]) | ~alive[:, None]    # 死亡整行放开
    # 放泡：存活 & 不在墙/砖格 & 脚下无泡 & 在场泡数 < 成长上限
    cell = pos.astype(jnp.int32)
    cell = jnp.stack([jnp.clip(cell[:, 0], 0, H - 1),
                      jnp.clip(cell[:, 1], 0, W - 1)], axis=-1)
    idx = cell[:, 0] * W + cell[:, 1]
    cur_f = fuse.reshape(-1)[idx]
    on_solid = (wall | brick).reshape(-1)[idx]
    live = jnp.stack([
        ((owner == 0) & (fuse > 0)).sum(),
        ((owner == 1) & (fuse > 0)).sum(),
    ])
    can_bomb = alive & (cur_f <= 0) & ~on_solid & (live < bombs_cap)
    bomb = jnp.stack([jnp.ones_like(alive),        # bomb=0（不放）恒合法
                      can_bomb | ~alive], axis=-1)  # 死亡 bomb=1 放开
    return move, bomb


# ---------------- 观测（正式版 2P+3=7 通道视角） ----------------

def _splat(pos_me, gate, h, w):
    """把连续坐标双线性铺开成 (H, W) 平面，总质量 1（同 obs.py _splat）。

    4 个角点合并成一次 batch scatter_add（重复目标格 add 可交换，与原
    4 次 `.at.add` 逐位一致；少 3 次 scatter kernel launch）。
    """
    fy = jnp.clip(pos_me[0] - 0.5, 0.0, h - 1.0)
    fx = jnp.clip(pos_me[1] - 0.5, 0.0, w - 1.0)
    y0 = jnp.clip(fy.astype(jnp.int32), 0, h - 1)
    x0 = jnp.clip(fx.astype(jnp.int32), 0, w - 1)
    y1 = jnp.clip(y0 + 1, 0, h - 1)
    x1 = jnp.clip(x0 + 1, 0, w - 1)
    wy = jnp.clip(fy - y0.astype(jnp.float32), 0.0, 1.0)
    wx = jnp.clip(fx - x0.astype(jnp.float32), 0.0, 1.0)
    g = gate.astype(jnp.float32)
    ys = jnp.stack([y0, y0, y1, y1])
    xs = jnp.stack([x0, x1, x0, x1])
    ws = jnp.stack([(1 - wy) * (1 - wx) * g, (1 - wy) * wx * g,
                    wy * (1 - wx) * g, wy * wx * g])
    return jnp.zeros((h, w), jnp.float32).at[ys, xs].add(ws)


def global_vec(state: BombState, pid: int) -> jnp.ndarray:
    """玩家 pid 视角的全局状态向量 (G=24,)，全部归一化到 [0,1]。

    基础 11 维：[局内进度 t/MAX, 我HP, 敌HP, 我泡数/威/速, 敌泡数/威/速,
    我存活, 敌存活]。HP 直接进向量 → 价值头能学到"残血≈低胜率"。
    预留 13 维（2026-08-20，后训练增强：变身/道具/竞技类型，当前全 0）：
    [我buff(3bit), 敌buff, 我debuff(2bit), 敌debuff, 我道具栏4槽(int6),
    敌道具栏4槽, 竞技类型(int4)]。

    与 make_obs 的格子通道互补：血量/成长属性/存活/道具是"时间序列标量"，
    论文式双序列输入的第二路（state token）。
    """
    hp = state.hp.astype(jnp.float32) / float(MAX_HP)
    b = state.bombs_cap / float(GROWTH_BOMBS_MAX)
    z = state.blast_cap / float(GROWTH_BLAST_MAX)
    sp = state.spd_g / float(GROWTH_SPEED_MAX)
    alive = state.alive.astype(jnp.float32)
    t = state.t.astype(jnp.float32) / float(MAX_STEPS)
    buff = state.buffs.astype(jnp.float32) / 7.0
    debuff = state.debuffs.astype(jnp.float32) / 3.0
    items = state.items.astype(jnp.float32) / 63.0
    gtype = state.gametype.astype(jnp.float32) / 15.0
    return jnp.stack([
        t, hp[pid], hp[1 - pid],
        b[pid], z[pid], sp[pid],
        b[1 - pid], z[1 - pid], sp[1 - pid],
        alive[pid], alive[1 - pid],
        buff[pid], buff[1 - pid],
        debuff[pid], debuff[1 - pid],
        items[pid, 0], items[pid, 1], items[pid, 2], items[pid, 3],
        items[1 - pid, 0], items[1 - pid, 1], items[1 - pid, 2], items[1 - pid, 3],
        gtype,
    ])


def make_obs(state: BombState, pid: int, danger=None) -> jnp.ndarray:
    """(N_OBS_CH, H, W) float32，玩家 pid 视角。

    通道 = 炸弹**基础信息 + 危险图**（ch5 = torch 同款 danger_map，网络直接
    读"火会烧到哪"，不需要自己从泡坐标反推威胁方向）：
      ch0 我位置（双线性 splat）   ch1 我的泡剩余时间 fuse/FUSE
      ch2 对手位置（splat）        ch3 对手泡剩余时间 fuse/FUSE
      ch4 墙|砖（都不可通行，obs.py ch4 同源）
      ch5 危险图 danger_map（0-1，越接近爆炸越大）
      ch6 局内进度 t/MAX_STEPS     ch7 道具存在性（crate>0，共享地图通道）
      ch8 灌木 bush（二值：可通行 + 可炸，炸毁按本关爆率出道具）
      ch9 泡道具（1/4）  ch10 威力道具（2/5）  ch11 速度道具（3/6）
      ch12 超级道具（4/5/6，+4 档；问号宝箱=7 时 ch9-12 全 0，只有 ch7）
      ch13 可推箱 pushable（二值；外观是砖但可被持续推动，AI 靠它区分
      "炸"还是"推"——没有此通道推箱玩法对策略不可见）

    `danger` 可传入预计算的危险图（两个视角共享同一份，省一半计算；
    both_perspectives 用）；None 时内部现算（单视角调用/对拍用）。
    血量和成长属性等时间序列标量不进格子通道，走 global_vec（state token）。
    """
    (pos, fuse, owner, bomb_blast, wall, brick, pushable, _pt, bush, crate, _rc,
     alive, _hp, _invuln, _bombs_cap, _blast_cap, _spd_g, _buff, _debuff,
     _item, _gtype, _is_open, t, _level_id) = state
    me, opp = pid, 1 - pid
    fuse_norm = fuse.astype(jnp.float32) / float(FUSE)
    bombed = fuse > 0
    if danger is None:
        danger = _danger_map(fuse, wall, bomb_blast, brick=brick)
    obs = jnp.stack([
        _splat(pos[me], alive[me], H, W),
        jnp.where(owner == me, fuse_norm, jnp.zeros_like(fuse_norm)),
        _splat(pos[opp], alive[opp], H, W),
        jnp.where(owner == opp, fuse_norm, jnp.zeros_like(fuse_norm)),
        (wall | brick).astype(jnp.float32),                  # 墙|砖（都不可通行）
        danger,                                              # 危险图（共享）
        jnp.full((H, W), t.astype(jnp.float32) / float(MAX_STEPS), jnp.float32),
        (crate > 0).astype(jnp.float32),                     # 道具存在性（二值）
        bush.astype(jnp.float32),                            # 灌木（可通行可炸）
        ((crate == 1) | (crate == 4)).astype(jnp.float32),   # 泡道具
        ((crate == 2) | (crate == 5)).astype(jnp.float32),   # 威力道具
        ((crate == 3) | (crate == 6)).astype(jnp.float32),   # 速度道具
        ((crate >= 4) & (crate <= 6)).astype(jnp.float32),   # 超级道具（+4 档）
        pushable.astype(jnp.float32),                        # 可推箱（推箱子玩法）
    ])
    return obs
