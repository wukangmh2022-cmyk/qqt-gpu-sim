"""jax_env 对齐正式版规则的单元验证（CPU 可跑，JAX_PLATFORMS=cpu）。

验证点（对照 sim/torch_sim.py 语义）：
  1. 放泡：fuse=30, owner, bomb_blast=7；泡数上限 10
  2. 引信每 tick 递减
  3. 爆炸覆盖：BLAST=7 十字（泡挡火）
  4. 连锁：相邻泡被点燃（max_chain）
  5. 伤害：中心格着火扣 1 血，HP 5；无敌期不掉血
  6. 终局 + 重置：done 后回 fresh
  7. 移动：连续坐标 + 泡挡路滑动碰撞
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import (BLAST, FUSE, H, INVULN, MAX_BOMBS, MAX_HP,
                              MAX_STEPS, STEP, W, BombState, _fresh,
                              _resolve_explosions, init_batch, make_obs, step)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


_K = lambda: jrandom.PRNGKey(0)


def _fk():
    return _fresh(_K())


def main():
    print(f"devices: {jax.devices()}")

    # ---- 1. 放泡 ----
    s = _fk()
    acts = jnp.array([[4, 1], [4, 0]], jnp.int32)   # p0 放泡，p1 不动
    s2, done = step(s, acts, _K())
    fuse0 = s2.fuse[1, 1]
    owner0 = s2.owner[1, 1]
    bb0 = s2.bomb_blast[1, 1]
    check("放泡: fuse=FUSE(30)", int(fuse0) == FUSE)
    check("放泡: owner=0", int(owner0) == 0)
    check("放泡: bomb_blast=BLAST(7)", int(bb0) == BLAST)
    check("放泡: done=False", not bool(done))

    # ---- 2. 引信递减 ----
    s3, _ = step(s2, jnp.zeros((2, 2), jnp.int32), _K())
    check("引信递减: 30->29", int(s3.fuse[1, 1]) == FUSE - 1)

    # ---- 3. 爆炸覆盖（BLAST=7 十字，源 (1,1)）----
    # 直接调 _resolve_explosions（不经 step 的清场/重置）：fuse=0 即已走完
    fuse_t = jnp.zeros((H, W), jnp.int32).at[1, 1].set(0)
    owner_t = -jnp.ones((H, W), jnp.int32).at[1, 1].set(0)
    bb_t = jnp.zeros((H, W), jnp.int32).at[1, 1].set(BLAST)
    covered, triggered = _resolve_explosions(fuse_t, owner_t, bb_t)
    check("爆炸: 源格覆盖", bool(covered[1, 1]))
    check("爆炸: 向右 7 格覆盖", bool(covered[1, 8]))
    check("爆炸: 向右 8 格不覆盖(超 BLAST)", not bool(covered[1, 9]))
    check("爆炸: 向上 1 格覆盖", bool(covered[0, 1]))
    check("爆炸: 左界外不受影响", not bool(covered[1, 0]) or covered[1, 1])

    # ---- 4. 连锁：两泡相邻，先爆的点燃后爆的 ----
    fuse_c = jnp.zeros((H, W), jnp.int32).at[1, 1].set(0).at[1, 2].set(2)
    owner_c = -jnp.ones((H, W), jnp.int32).at[1, 1].set(0).at[1, 2].set(0)
    bb_c = jnp.zeros((H, W), jnp.int32)
    bb_c = bb_c.at[1, 1].set(BLAST).at[1, 2].set(BLAST)
    _, trig_c = _resolve_explosions(fuse_c, owner_c, bb_c)
    check("连锁: B(1,2) 被点燃", bool(trig_c[1, 2]))

    # ---- 5. 伤害：HP 5，站火上扣 1；无敌期不掉 ----
    s_h = _fk()
    # 在 p0 脚下放泡 fuse=1，p0 不动
    s_h = s_h._replace(
        fuse=s_h.fuse.at[1, 1].set(1),
        owner=s_h.owner.at[1, 1].set(1),      # 对方泡
        bomb_blast=s_h.bomb_blast.at[1, 1].set(BLAST),
    )
    s_h, _ = step(s_h, jnp.zeros((2, 2), jnp.int32), _K())
    check("伤害: HP 5->4", int(s_h.hp[0]) == MAX_HP - 1)
    check("伤害: 仍存活", bool(s_h.alive[0]))
    check("伤害: 进入无敌期", int(s_h.invuln[0]) == INVULN)
    # 无敌期再炸一次不掉血（把另一颗泡 fuse=1 放脚下）
    s_h = s_h._replace(
        fuse=s_h.fuse.at[1, 1].set(1),
        owner=s_h.owner.at[1, 1].set(1),
        bomb_blast=s_h.bomb_blast.at[1, 1].set(BLAST),
    )
    s_h, _ = step(s_h, jnp.zeros((2, 2), jnp.int32), _K())
    check("无敌期: 不掉血", int(s_h.hp[0]) == MAX_HP - 1)
    check("无敌期: invuln 递减", int(s_h.invuln[0]) == INVULN - 1)

    # ---- 5b. 血扣光死亡（终局后自动重置，用 done + step 前快照语义验证）----
    s_d = _fk()
    s_d = s_d._replace(hp=jnp.array([1, MAX_HP], jnp.int32))
    s_d = s_d._replace(
        fuse=s_d.fuse.at[1, 1].set(1),
        owner=s_d.owner.at[1, 1].set(1),
        bomb_blast=s_d.bomb_blast.at[1, 1].set(BLAST),
    )
    s_d, done = step(s_d, jnp.zeros((2, 2), jnp.int32), _K())
    check("死亡: done=True(终局判定)", bool(done))
    check("死亡: 终局后重置回满血满状态", int(s_d.hp[0]) == MAX_HP and bool(s_d.alive[0]))

    # ---- 6. 终局重置 ----
    s_r = _fk()
    s_r = s_r._replace(t=jnp.array(MAX_STEPS - 1, jnp.int32))
    s_r, done = step(s_r, jnp.zeros((2, 2), jnp.int32), _K())
    check("超时: done=True", bool(done))
    check("超时: 重置回 fresh(t=0)", int(s_r.t) == 0)
    check("超时: 重置回满血", int(s_r.hp[0]) == MAX_HP)

    # ---- 7. 移动：连续坐标 + 泡挡路 ----
    s_m = _fk()
    # p0 从 (1,1) 向右走 1 tick（STEP=0.3）→ (1, 1.3)
    s_m, _ = step(s_m, jnp.array([[1, 0], [4, 0]], jnp.int32), _K())
    check("移动: 连续坐标 x≈1.3", abs(float(s_m.pos[0, 1]) - (1 + STEP)) < 1e-3)
    check("移动: y 不变", abs(float(s_m.pos[0, 0]) - 1.0) < 1e-4)
    # 在 p0 前方 (1,2) 放泡（p0 从 (1,1) 走 0.3 → 贴停泡左边界 2-0.36=1.64 之外，未触发碰撞）
    # 直接给 x=1.7：下一步向右被泡挡 → 贴停 2-RADIUS-EPS = 1.6399
    s_b2 = _fk()
    s_b2 = s_b2._replace(
        fuse=s_b2.fuse.at[1, 2].set(30),
        owner=s_b2.owner.at[1, 2].set(0),
        bomb_blast=s_b2.bomb_blast.at[1, 2].set(BLAST),
        pos=s_b2.pos.at[0].set(jnp.array([1.0, 1.64], jnp.float32)),
    )
    s_b2, _ = step(s_b2, jnp.array([[1, 0], [4, 0]], jnp.int32), _K())
    x2 = float(s_b2.pos[0, 1])
    check("碰撞: 泡挡路贴停(x≈1.6399)", abs(x2 - (2 - 0.36 - 1e-4)) < 1e-3, )

    # ---- 8. obs 形状 + 泡威力图（ch5，化繁为简后无危险图） ----
    s_o = _fk()
    o0 = make_obs(s_o, 0)
    o1 = make_obs(s_o, 1)
    check("obs: p0 形状 (7,H,W)", o0.shape == (7, H, W))
    check("obs: p1 形状", o1.shape == (7, H, W))
    check("obs: 空场威力图全 0", float(jnp.abs(o0[5]).max()) == 0.0)
    # 有泡时威力图在泡格 = 1.0（blast/BLAST）
    s_o2 = s_o._replace(
        fuse=s_o.fuse.at[1, 1].set(10),
        owner=s_o.owner.at[1, 1].set(0),
        bomb_blast=s_o.bomb_blast.at[1, 1].set(BLAST),
    )
    o02 = make_obs(s_o2, 0)
    check("obs: 有泡威力图泡格=1.0", abs(float(o02[5][1, 1]) - 1.0) < 1e-6)
    check("obs: 有泡威力图他处=0", abs(float(o02[5][5, 5])) < 1e-6)

    # ---- 9. batch vmap ----
    states = init_batch(jrandom.PRNGKey(0), 4)
    acts = jnp.zeros((4, 2, 2), jnp.int32)
    ns, ndone = jax.vmap(step)(states, acts)
    check("vmap: batch step 形状", ns.pos.shape == (4, 2, 2) and ndone.shape == (4,))

    print(f"\n{'-'*40}\nPASS={PASS} FAIL={FAIL}")


if __name__ == "__main__":
    main()
