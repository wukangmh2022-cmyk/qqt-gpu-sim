"""jax 宝箱/成长系统 smoke 测试（本地/远端 jax 环境均可跑）。

覆盖 RNG 部分（对拍只覆盖确定性共享逻辑）：
  1. 地图生成分布：纯空场/open/corridor 比例 ≈ 50/25/25，spawn 合法性
     （落在可通行格）、corridor 有 brick、open 有十字宝箱池
  2. 炸砖变箱：corridor 关 brick 被爆炸覆盖 → brick 清除 + crate 出现
  3. 拾取成长：站箱上多次 → 属性单调增长、不超上限（open 必升）
  4. 掉血扣属性 + 回收守恒：掉血 tick 后 泡/威/速 扣 2 层、回收箱数 = 扣层数
  5. 掉血在起点时不扣不产箱（守恒边界）

注意：**必须从仓库根运行**（jax_bomb/platform.py 会劫持标准库 platform，
脚本放 jax_bomb/ 下 import jax 会崩）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (BLAST, CRATE_PROB, FUSE, GROWTH_BLAST_MAX,
                              GROWTH_BLAST_START, GROWTH_BOMBS_MAX,
                              GROWTH_BOMBS_START, GROWTH_SPEED_MAX,
                              GROWTH_SPEED_START, GROWTH_SPEED_STEP, H,
                              HIT_ATTR_PENALTY, MAX_HP, MAX_RECYCLE, N_OBS_CH,
                              OPEN_CRATE_CROSS, OPEN_GROWTH_BLAST,
                              OPEN_GROWTH_BOMBS, OPEN_GROWTH_SPEED, BombState,
                              _fresh, _make_map, init_batch, legal_mask,
                              make_obs, step)

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"  ✔ {name}")
    else:
        FAILS += 1
        print(f"  ✘ {name} {extra}")


def test_map_distribution():
    print("1. 地图生成分布")
    key = jax.random.PRNGKey(0)
    n = 4000
    keys = jax.random.split(key, n)
    wall, brick, is_open, spawn = jax.vmap(_make_map)(keys)
    wall, brick, is_open, spawn = (jax.device_get(x) for x in
                                   (wall, brick, is_open, spawn))
    is_open = is_open.astype(bool)
    # open 类型 = is_open（纯空场 + open 障碍）；corridor = ~is_open
    frac_open = is_open.mean()
    print(f"  open 关比例 {frac_open:.3f}（期望 ~0.75；纯空场+open 障碍）")
    check("open≈75%", abs(frac_open - 0.75) < 0.03)
    # corridor 必有 brick，open 无 brick
    corr = ~is_open
    n_corr = corr.sum()
    n_open = is_open.sum()
    check("corridor 有 brick", n_corr > 0 and brick[corr].any())
    check("open 无 brick", n_open > 0 and not brick[is_open].any())
    # spawn 不在墙/砖上（清四邻后）
    cell = spawn.astype(jnp.int32)
    on_wall = jax.vmap(
        lambda w, b, p: (w | b)[p[0, 0], p[0, 1]] | (w | b)[p[1, 0], p[1, 1]]
    )(wall, brick, cell)
    on_wall = jax.device_get(on_wall)
    check("spawn 不在墙/砖上", not bool(on_wall.any()))
    # open 关有十字宝箱池（构造 _fresh 看 crate 数量）
    st = _fresh(keys[0])
    crate0 = jax.device_get(st.crate)
    is_open0 = bool(jax.device_get(st.is_open))
    if is_open0 and OPEN_CRATE_CROSS:
        n_crate = int(crate0.sum())
        check(f"open 关开局十字池 {n_crate} 格 > 30", n_crate > 30)
    # obs 通道数
    obs = make_obs(st, 0)
    check(f"N_OBS_CH=8（实际 {obs.shape[0]}）", obs.shape[0] == N_OBS_CH == 8)
    # 属性按类型
    bc = jax.device_get(st.bombs_cap)
    zc = jax.device_get(st.blast_cap)
    sg = jax.device_get(st.spd_g)
    exp_b = OPEN_GROWTH_BOMBS if is_open0 else GROWTH_BOMBS_START
    exp_z = OPEN_GROWTH_BLAST if is_open0 else GROWTH_BLAST_START
    exp_s = OPEN_GROWTH_SPEED if is_open0 else GROWTH_SPEED_START
    check("初始属性按地图类型", bool((bc == exp_b).all()) and bool((zc == exp_z).all())
          and bool(jnp.allclose(sg, exp_s)))


def test_brick_blast_crate():
    print("2. 炸砖变箱（corridor 构造）")
    key = jax.random.PRNGKey(1)
    st = _fresh(key)
    # 强制 corridor：直接重建，铺 brick 在玩家旁
    b = jnp.zeros((H, H), jnp.bool_).at[3, 1].set(True)   # (3,1) 玩家0 正下方 2 格（blast=2 覆盖）
    st = st._replace(brick=b, is_open=jnp.array(False),
                     wall=jnp.zeros((H, H), jnp.bool_))
    # 玩家0 放泡在 (2,1)（brick 上方），爆炸覆盖 (3,2)
    st = st._replace(pos=st.pos.at[0].set(jnp.array([2.5, 1.5])))
    obs0 = jax.device_get(make_obs(st, 0))
    check("obs ch7 显示砖位无宝箱", obs0[7, 3, 1] == 0.0)
    check("obs ch4 显示砖位为墙|砖", obs0[4, 3, 1] == 1.0)
    # 放泡动作：bomb=1 且不动（IDLE）
    acts = jnp.array([[4, 1], [4, 0]], jnp.int32)
    st2, done = step(st, acts, jax.random.PRNGKey(2), auto_reset=False)
    brick1 = jax.device_get(st2.brick)
    crate1 = jax.device_get(st2.crate)
    print(f"  tick0 后 brick[3,1]={brick1[3,1]} crate[3,1]={crate1[3,1]}")
    # 引信 FUSE=30 太长：把泡 fuse 设 1，下一 tick 爆炸
    st2 = st2._replace(fuse=st2.fuse.at[2, 1].set(1))
    st3, _ = step(st2, jnp.array([[4, 0], [4, 0]], jnp.int32),
                  jax.random.PRNGKey(3), auto_reset=False)
    brick3 = jax.device_get(st3.brick)
    crate3 = jax.device_get(st3.crate)
    print(f"  爆炸后 brick[3,1]={brick3[3,1]} crate[3,1]={crate3[3,1]}")
    check("爆炸摧毁 brick", not bool(brick3[3, 1]))
    check("炸掉的砖 → 宝箱", bool(crate3[3, 1]))
    # 玩家0 走到 (3.5,1.5) 踩箱：corridor 掷 CRATE_PROB=0.5
    stc = st3._replace(pos=st3.pos.at[0].set(jnp.array([3.5, 1.5])))
    grew = False
    for i in range(60):
        stc = stc._replace(crate=stc.crate.at[3, 1].set(True))  # 重置箱（踩过即消失）
        stc, _ = step(stc, jnp.array([[4, 0], [4, 0]], jnp.int32),
                      jax.random.PRNGKey(100 + i), auto_reset=False)
        if (jax.device_get(stc.bombs_cap[0]) > GROWTH_BOMBS_START
                or jax.device_get(stc.blast_cap[0]) > GROWTH_BLAST_START
                or jax.device_get(stc.spd_g[0]) > GROWTH_SPEED_START):
            grew = True
            break
    print(f"  60 次踩箱成长发生: {grew}")
    check("corridor 踩箱可成长", grew)


def test_pickup_growth():
    print("3. open 关拾取必升（crate_prob=1.0）")
    key = jax.random.PRNGKey(5)
    st = _fresh(key)
    # 强制 open：踩十字池里的箱，重复踩到成长
    st = st._replace(is_open=jnp.array(True))
    # 直接在地图中央放一个箱，玩家站上去
    crate = jnp.zeros((H, H), jnp.bool_).at[6, 6].set(True)
    st = st._replace(crate=crate, pos=st.pos.at[0].set(jnp.array([6.5, 6.5])))
    grew = False
    for i in range(30):
        st, _ = step(st._replace(crate=st.crate.at[6, 6].set(True)),
                     jnp.array([[4, 0], [4, 0]], jnp.int32),
                     jax.random.PRNGKey(200 + i), auto_reset=False)
        if bool(jax.device_get(st.crate[6, 6])):
            pass
        if (jax.device_get(st.bombs_cap[0]) > OPEN_GROWTH_BOMBS
                or jax.device_get(st.blast_cap[0]) > OPEN_GROWTH_BLAST
                or jax.device_get(st.spd_g[0]) > OPEN_GROWTH_SPEED):
            grew = True
            break
    check("open 关踩箱必升", grew)
    # 成长不超上限（多踩到上限）
    bc = jax.device_get(st.bombs_cap)
    zc = jax.device_get(st.blast_cap)
    sg = jax.device_get(st.spd_g)
    check("属性不超上限", bool((bc <= GROWTH_BOMBS_MAX).all())
          and bool((zc <= GROWTH_BLAST_MAX).all())
          and bool((sg <= GROWTH_SPEED_MAX).all()))


def test_hit_penalty_recycle():
    print("4. 掉血扣属性 + 回收守恒")
    key = jax.random.PRNGKey(6)
    st = _fresh(key)
    # 提升属性到有扣的空间：泡 6、威 5、速 1.6（corridor 起点 2/2/1.0）
    st = st._replace(bombs_cap=jnp.full((2,), 6.0), blast_cap=jnp.full((2,), 5.0),
                     spd_g=jnp.full((2,), 1.6), is_open=jnp.array(False),
                     crate=jnp.zeros((H, H), jnp.bool_),
                     brick=jnp.zeros((H, H), jnp.bool_))
    before = (jax.device_get(st.bombs_cap).copy(), jax.device_get(st.blast_cap).copy(),
              jax.device_get(st.spd_g).copy())
    # 玩家0 脚下格放火（爆炸覆盖）→ 掉血。构造 covered：放颗 fuse=1 的泡在脚下
    y, x = jax.device_get(st.pos[0]).astype(int)
    st = st._replace(fuse=st.fuse.at[y, x].set(1), owner=st.owner.at[y, x].set(0),
                     bomb_blast=st.bomb_blast.at[y, x].set(3),
                     invuln=jnp.zeros((2,), jnp.int32))
    st2, _ = step(st, jnp.array([[4, 0], [4, 0]], jnp.int32),
                  jax.random.PRNGKey(7), auto_reset=False)
    bc2, zc2, sg2 = (jax.device_get(x) for x in (st2.bombs_cap, st2.blast_cap,
                                                 st2.spd_g))
    hp2 = jax.device_get(st2.hp)
    crate2 = jax.device_get(st2.crate)
    lost = ((before[0] - bc2) + (before[1] - zc2)
            + jnp.round((before[2] - sg2) / GROWTH_SPEED_STEP))
    print(f"  掉血后 hp={hp2[0]} lost={lost[0]} crate 总数={int(crate2.sum())}")
    check("掉血扣 1 血", hp2[0] == MAX_HP - 1)
    check("泡/威各扣 2 层", bc2[0] == 6 - HIT_ATTR_PENALTY
          and zc2[0] == 5 - HIT_ATTR_PENALTY)
    check("速度扣 2 层", abs(sg2[0] - (1.6 - HIT_ATTR_PENALTY * GROWTH_SPEED_STEP)) < 1e-5)
    y0, x0 = jax.device_get(st.pos[0]).astype(int)
    check("回收箱数 = 扣层数", int(crate2.sum()) == int(lost[0]))
    # 未掉血的玩家 1 不扣
    check("未掉血玩家不扣属性", bc2[1] == before[0][1] and zc2[1] == before[1][1]
          and sg2[1] == before[2][1])
    # 回收箱可被踩升（rec_crate 必升）
    rec = jax.device_get(st2.rec_crate)
    check("回收箱标记 rec_crate", bool(rec.any()))


def test_hit_penalty_at_start():
    print("5. 起点掉血不扣不产箱（守恒边界）")
    key = jax.random.PRNGKey(8)
    st = _fresh(key)._replace(
        bombs_cap=jnp.full((2,), GROWTH_BOMBS_START, jnp.float32),
        blast_cap=jnp.full((2,), GROWTH_BLAST_START, jnp.float32),
        spd_g=jnp.full((2,), GROWTH_SPEED_START, jnp.float32),
        is_open=jnp.array(False), crate=jnp.zeros((H, H), jnp.bool_),
        brick=jnp.zeros((H, H), jnp.bool_))
    # 起点属性：2/2/1.0（corridor）→ clamp 到起点 = 不扣
    y, x = jax.device_get(st.pos[0]).astype(int)
    st = st._replace(fuse=st.fuse.at[y, x].set(1), owner=st.owner.at[y, x].set(0),
                     bomb_blast=st.bomb_blast.at[y, x].set(3),
                     invuln=jnp.zeros((2,), jnp.int32))
    st2, _ = step(st, jnp.array([[4, 0], [4, 0]], jnp.int32),
                  jax.random.PRNGKey(9), auto_reset=False)
    bc2, zc2 = (jax.device_get(x) for x in (st2.bombs_cap, st2.blast_cap))
    crate2 = jax.device_get(st2.crate)
    check("起点掉血不扣属性", bc2[0] == GROWTH_BOMBS_START and zc2[0] == GROWTH_BLAST_START)
    check("起点掉血不产箱", int(crate2.sum()) == 0)


def test_batch_and_reset():
    print("6. 批量 init + auto_reset")
    n = 64
    states = init_batch(jax.random.PRNGKey(10), n)
    mm, bm = jax.vmap(legal_mask)(states)
    obs = jax.vmap(lambda s: make_obs(s, 0))(states)
    check(f"批量 obs 形状 {obs.shape}", obs.shape == (n, N_OBS_CH, H, H))
    mm = jax.device_get(mm)
    bm = jax.device_get(bm)
    check("批量 mask 形状", mm.shape == (n, 2, 5) and bm.shape == (n, 2, 2))
    # 跑 10 步（含 auto_reset）
    keys = jax.random.split(jax.random.PRNGKey(11), n)
    for _ in range(10):
        acts = jnp.stack([jax.random.randint(jax.random.PRNGKey(12), (n, 2), 0, 5),
                          jax.random.randint(jax.random.PRNGKey(13), (n, 2), 0, 2)],
                         axis=-1)
        kk = jax.random.split(jax.random.PRNGKey(14), n)
        states, done = jax.vmap(step)(states, acts, kk)
    done = jax.device_get(done)
    check("批量 step 10 tick 无异常", done.shape == (n,))


if __name__ == "__main__":
    test_map_distribution()
    test_brick_blast_crate()
    test_pickup_growth()
    test_hit_penalty_recycle()
    test_hit_penalty_at_start()
    test_batch_and_reset()
    print(f"\n{'全部通过 ✔' if FAILS == 0 else f'{FAILS} 处失败 ✘'}")
    sys.exit(1 if FAILS else 0)
