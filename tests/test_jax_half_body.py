"""JAX 半身位与水柱真实中轴线判定物理测试（对齐 Web Sim._isHitByExplosion）。"""

import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (
    H, W, BombState, _fresh, step, init_batch, N_MOVES, N_BOMB, RADIUS, BLAST
)


def _clean_state(key=0):
    s0 = _fresh(jax.random.PRNGKey(key))
    wall = jnp.zeros((H, W), jnp.bool_)
    brick = jnp.zeros((H, W), jnp.bool_)
    bush = jnp.zeros((H, W), jnp.bool_)
    crate = jnp.zeros((H, W), jnp.int8)
    return s0._replace(wall=wall, brick=brick, bush=bush, crate=crate,
                       invuln=jnp.zeros(2, jnp.int32))


def test_jax_single_column_half_body_safe():
    """单水柱场景：玩家在 x=2.0 处于半身位，未触及 x=1.5 水柱中轴线，完全无伤。"""
    s = _clean_state()
    pos = jnp.array([[3.5, 2.0], [10.5, 10.5]])
    fuse = jnp.zeros((H, W), jnp.int32).at[1, 1].set(1)
    owner = jnp.full((H, W), -1, jnp.int32).at[1, 1].set(1)
    bb = jnp.zeros((H, W), jnp.int32).at[1, 1].set(3)
    s = s._replace(pos=pos, fuse=fuse, owner=owner, bomb_blast=bb)

    actions = jnp.array([[4, 0], [4, 0]])
    next_s, _ = step(s, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(next_s.hp[0]) == 5, "半身位 x=2.0 必须无伤！"


def test_jax_single_column_direct_hit():
    """单水柱场景：玩家在 x=1.5 处于水柱中轴线上，必定扣血。"""
    s = _clean_state()
    pos = jnp.array([[3.5, 1.5], [10.5, 10.5]])
    fuse = jnp.zeros((H, W), jnp.int32).at[1, 1].set(1)
    owner = jnp.full((H, W), -1, jnp.int32).at[1, 1].set(1)
    bb = jnp.zeros((H, W), jnp.int32).at[1, 1].set(3)
    s = s._replace(pos=pos, fuse=fuse, owner=owner, bomb_blast=bb)

    actions = jnp.array([[4, 0], [4, 0]])
    next_s, _ = step(s, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(next_s.hp[0]) == 4, "正中 x=1.5 必须掉血！"


def test_jax_parallel_columns_broken_half_body():
    """双并排纵向水柱：两列分界缝隙 x=2.0 连通破半身，玩家在 x=2.0 必定受伤害。"""
    s = _clean_state()
    pos = jnp.array([[3.5, 2.0], [10.5, 10.5]])
    fuse = jnp.zeros((H, W), jnp.int32).at[1, 1].set(1).at[1, 2].set(1)
    owner = jnp.full((H, W), -1, jnp.int32).at[1, 1].set(1).at[1, 2].set(1)
    bb = jnp.zeros((H, W), jnp.int32).at[1, 1].set(3).at[1, 2].set(3)
    s = s._replace(pos=pos, fuse=fuse, owner=owner, bomb_blast=bb)

    actions = jnp.array([[4, 0], [4, 0]])
    next_s, _ = step(s, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(next_s.hp[0]) == 4, "并排双水柱分界缝隙处破半身，必须掉血！"


def test_jax_user_corner_crossroads():
    """十字路口对角角部半身位安全：第 5 行横向 + 第 8/9 列纵向，角色在 (6.0, 7.787) 绝对无伤。"""
    s = _clean_state()
    pos = jnp.array([[6.0, 7.787], [10.5, 10.5]])
    fuse = jnp.zeros((H, W), jnp.int32).at[5, 6].set(1).at[5, 8].set(1).at[5, 9].set(1)
    owner = jnp.full((H, W), -1, jnp.int32).at[5, 6].set(1).at[5, 8].set(1).at[5, 9].set(1)
    bb = jnp.zeros((H, W), jnp.int32).at[5, 6].set(5).at[5, 8].set(5).at[5, 9].set(5)
    s = s._replace(pos=pos, fuse=fuse, owner=owner, bomb_blast=bb)

    actions = jnp.array([[4, 0], [4, 0]])
    next_s, _ = step(s, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(next_s.hp[0]) == 5, "十字路口对角角部安全位必须完全无伤！"


def test_jax_blast_linger_damage():
    """爆炸余威伤害：后续 tick 进入余威区域同样受伤害。"""
    s = _clean_state()
    pos = jnp.array([[3.5, 2.0], [10.5, 10.5]])
    fuse = jnp.zeros((H, W), jnp.int32).at[1, 1].set(1)
    owner = jnp.full((H, W), -1, jnp.int32).at[1, 1].set(1)
    bb = jnp.zeros((H, W), jnp.int32).at[1, 1].set(3)
    s = s._replace(pos=pos, fuse=fuse, owner=owner, bomb_blast=bb)

    actions = jnp.array([[4, 0], [4, 0]])
    s1, _ = step(s, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(s1.hp[0]) == 5

    # 移动进入处于余威中的第 1 列中线
    s2 = s1._replace(pos=jnp.array([[3.5, 1.5], [10.5, 10.5]]), invuln=jnp.zeros(2, jnp.int32))
    s3, _ = step(s2, actions, jax.random.PRNGKey(0), auto_reset=False)
    assert int(s3.hp[0]) == 4, "踩入余威水柱中轴线必须掉血！"
