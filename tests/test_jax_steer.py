import pytest

jnp = pytest.importorskip("jax.numpy")

import jax

from jax_bomb import levels
from jax_bomb.jax_env import H, W, _fresh, _steer, legal_mask, step


def test_steer_does_not_center_on_open_ground_obstacle():
    blocked = jnp.zeros((H, W), dtype=jnp.bool_).at[4, 5].set(True)

    # 开阔地单个障碍物（两侧均开阔）：直上受阻坚决不主动向中线归中，保持原身位
    pos = jnp.array([5.43, 5.20], dtype=jnp.float32)
    out = _steer(pos, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out[1]) == pytest.approx(5.20, abs=1e-4)
    out2 = _steer(out, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out2[0]) == pytest.approx(float(out[0]), abs=1e-4)
    assert float(out2[1]) == pytest.approx(5.20, abs=1e-4)

    # 镜像：偏右时同样保持原身位
    pos_r = jnp.array([5.43, 5.80], dtype=jnp.float32)
    out_r = _steer(pos_r, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out_r[1]) == pytest.approx(5.80, abs=1e-4)
    out_r2 = _steer(out_r, jnp.int32(0), jnp.bool_(True), blocked)
    assert float(out_r2[0]) == pytest.approx(float(out_r[0]), abs=1e-4)
    assert float(out_r2[1]) == pytest.approx(5.80, abs=1e-4)


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


def test_high_speed_boundary_steer_no_oscillation():
    from jax_bomb.jax_env import RADIUS, EPS
    blocked = jnp.zeros((H, W), dtype=jnp.bool_)
    # 场景 1：从 (1.0979, 5.7800) 高速 (spd=2.4) 向上直走，
    # 第一步直走 0.72 至 0.3779，第二步抵达上边界 y=RADIUS+EPS，不能左右在 5.78 和 5.06 互跳。
    pos = jnp.array([1.0979, 5.7800], dtype=jnp.float32)
    p1 = _steer(pos, jnp.int32(0), jnp.bool_(True), blocked, spd=2.4)
    assert float(p1[0]) == pytest.approx(1.0979 - 0.72, abs=1e-4)
    assert float(p1[1]) == pytest.approx(5.7800, abs=1e-4)
    p2 = _steer(p1, jnp.int32(0), jnp.bool_(True), blocked, spd=2.4)
    assert float(p2[0]) == pytest.approx(RADIUS + EPS, abs=1e-4)
    assert float(p2[1]) == pytest.approx(5.7800, abs=1e-4)

    # 场景 2：从 (5.1914, 13.9244) 高速 (spd=2.4) 向右直走，
    # 直达右边界 x=W-RADIUS-EPS，不能上下在 5.19 和 5.91 互跳。
    pos_r = jnp.array([5.1914, 13.9244], dtype=jnp.float32)
    p1_r = _steer(pos_r, jnp.int32(3), jnp.bool_(True), blocked, spd=2.4)
    assert float(p1_r[1]) == pytest.approx(W - RADIUS - EPS, abs=1e-4)
    assert float(p1_r[0]) == pytest.approx(5.1914, abs=1e-4)
    p2_r = _steer(p1_r, jnp.int32(3), jnp.bool_(True), blocked, spd=2.4)
    assert float(p2_r[1]) == pytest.approx(W - RADIUS - EPS, abs=1e-4)
    assert float(p2_r[0]) == pytest.approx(5.1914, abs=1e-4)

    # 场景 3：开阔地单个障碍物（高速 0.72 撞向障碍物坚决不主动向中线吸附，保持 5.20 原身位）
    blocked_mid = blocked.at[4, 5].set(True)
    pos_mid = jnp.array([5.43, 5.20], dtype=jnp.float32)
    p_center = _steer(pos_mid, jnp.int32(0), jnp.bool_(True), blocked_mid, spd=2.4)
    assert float(p_center[1]) == pytest.approx(5.20, abs=1e-4)
    p_center2 = _steer(p_center, jnp.int32(0), jnp.bool_(True), blocked_mid, spd=2.4)
    assert float(p_center2[1]) == pytest.approx(5.20, abs=1e-4)


def test_legal_mask_out_of_bounds_is_masked():
    from jax_bomb.jax_env import RADIUS, EPS
    state = _fresh(jax.random.PRNGKey(0))
    state = state._replace(
        wall=jnp.zeros((H, W), dtype=jnp.bool_),
        brick=jnp.zeros((H, W), dtype=jnp.bool_),
        fuse=jnp.zeros((H, W), dtype=jnp.int32),
        pushable=jnp.zeros((H, W), dtype=jnp.bool_),
        pos=jnp.array([[RADIUS + EPS, 5.5], [H - RADIUS - EPS, W - RADIUS - EPS]], jnp.float32),
        alive=jnp.array([True, True]),
    )
    mm, _ = legal_mask(state)
    # Player 0 at row 0 upper boundary: UP (0) must be False
    assert not bool(mm[0, 0]), "P0 抵住上边界时向上必须被 Mask 为非法"
    assert bool(mm[0, 1]), "P0 向下必须合法"
    assert bool(mm[0, 2]), "P0 向左必须合法"
    assert bool(mm[0, 3]), "P0 向右必须合法"
    assert bool(mm[0, 4]), "P0 IDLE 必须合法"

    # Player 1 at bottom-right boundary: DOWN (1) and RIGHT (3) must be False
    assert not bool(mm[1, 1]), "P1 抵住下边界时向下必须被 Mask 为非法"
    assert not bool(mm[1, 3]), "P1 抵住右边界时向右必须被 Mask 为非法"
    assert bool(mm[1, 0]), "P1 向上必须合法"
    assert bool(mm[1, 2]), "P1 向左必须合法"
    assert bool(mm[1, 4]), "P1 IDLE 必须合法"


def test_22debug_open_corridor_not_oscillating():
    # 场景：22debug 录像中 P0 从 (6.9001, 8.0999) 向左大踏步走向开阔列 7，
    # 隔列 (6, 6) 有障碍。向左迈出 0.6798 格（94.4% 步长）已成功跨入第 7 列，
    # 必须认定为有效直行，严禁清零并垂直侧滑震荡。
    blocked = jnp.zeros((H, W), dtype=jnp.bool_).at[6, 6].set(True)
    pos = jnp.array([6.9001, 8.0999], dtype=jnp.float32)
    out = _steer(pos, jnp.int32(2), jnp.bool_(True), blocked, spd=2.4)
    assert float(out[0]) == pytest.approx(6.9001, abs=1e-4), f"绝不应发生垂直超调侧滑，实际 y={float(out[0])}"
    assert float(out[1]) < 7.5, f"应成功跨入第 7 列 (x < 7.5)，实际 x={float(out[1])}"



