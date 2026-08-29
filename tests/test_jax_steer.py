import pytest

jnp = pytest.importorskip("jax.numpy")

import jax

from jax_bomb import levels
from jax_bomb.jax_env import H, W, _fresh, _steer, legal_mask, step


def test_steer_centers_instead_of_fixed_left_bias():
    blocked = jnp.zeros((H, W), dtype=jnp.bool_).at[4, 5].set(True)

    # 直上受阻且两侧状态相同：偏左时应向右归中。
    pos = jnp.array([5.36, 5.20], dtype=jnp.float32)
    out = _steer(pos, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out[1]) > float(pos[1])
    assert float(out[0]) == pytest.approx(float(pos[0]), abs=1e-6)

    # 镜像：偏右时应向左归中。
    pos = jnp.array([5.36, 5.80], dtype=jnp.float32)
    out = _steer(pos, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out[1]) < float(pos[1])
    assert float(out[0]) == pytest.approx(float(pos[0]), abs=1e-6)


def test_pushable_action_is_unmasked_and_does_not_side_step():
    blocked = jnp.zeros((H, W), dtype=jnp.bool_).at[5, 6].set(True)
    pushable = jnp.zeros((H, W), dtype=jnp.bool_).at[5, 6].set(True)
    pos = jnp.array([5.5, 5.5], dtype=jnp.float32)
    out = _steer(pos, jnp.int32(3), jnp.bool_(True), blocked, pushable=pushable)
    assert float(out[0]) == pytest.approx(5.5, abs=1e-6)
    assert 5.5 <= float(out[1]) < 6.0

    levels.clear()
    state = _fresh(jax.random.PRNGKey(0))
    state = state._replace(
        pos=jnp.array([[5.5, 5.5], [10.5, 10.5]], jnp.float32),
        fuse=jnp.zeros_like(state.fuse),
        wall=jnp.zeros_like(state.wall),
        brick=jnp.zeros_like(state.brick).at[5, 6].set(True),
        pushable=jnp.zeros_like(state.pushable).at[5, 6].set(True),
        push_t=jnp.zeros_like(state.push_t),
        crate=jnp.zeros_like(state.crate),
        alive=jnp.array([True, True]),
        spd_g=jnp.ones_like(state.spd_g),
    )
    move_mask, _ = legal_mask(state)
    assert bool(move_mask[0, 3])  # 向右推箱不能被 PPO mask 盖住

    actions = jnp.array([[3, 0], [4, 0]], jnp.int32)
    # 第 1 tick 先贴到箱边，之后连续 3 tick 累计到 PUSH_TIME。
    for tick in range(4):
        state, _ = step(state, actions, jax.random.PRNGKey(tick + 1), auto_reset=False)
    assert not bool(state.pushable[5, 6])
    assert bool(state.pushable[5, 7])


def test_partial_forward_collision_recenters_then_enters_corridor():
    blocked = jnp.zeros((H, W), dtype=jnp.bool_)
    blocked = blocked.at[6, 4].set(True).at[6, 6].set(True)
    pos = jnp.array([5.5, 5.12], dtype=jnp.float32)
    first = _steer(pos, jnp.int32(1), jnp.bool_(True), blocked)
    assert float(first[1]) > float(pos[1])
    assert float(first[0]) == pytest.approx(float(pos[0]), abs=1e-6)
    second = _steer(first, jnp.int32(1), jnp.bool_(True), blocked)
    assert float(second[0]) > float(first[0]) + 0.25
