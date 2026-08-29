"""Focused JAX reward-regression tests for the self-play PPO environment."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

from jax_bomb import levels
from jax_bomb.jax_env import (
    BLAST,
    GROWTH_BOMBS_MAX,
    MAX_HP,
    MAX_STEPS,
    _fresh,
    step,
)
from jax_bomb.jax_train import (
    STEP_PENALTY,
    TIMEOUT_MAX_BONUS,
    WIN_BONUS,
    novelty_transition,
    reward_from_events,
)
from jax_bomb.multicard_train import (
    fixed_reward_alpha,
    reward_schedule_global_steps,
    shared_reward_anneal_steps,
)


def _empty_state(key):
    levels.clear()
    state = _fresh(key)
    return state._replace(
        pos=jnp.array([[2.5, 2.5], [10.5, 10.5]], jnp.float32),
        fuse=jnp.zeros_like(state.fuse),
        owner=-jnp.ones_like(state.owner),
        bomb_blast=jnp.zeros_like(state.bomb_blast),
        wall=jnp.zeros_like(state.wall),
        brick=jnp.zeros_like(state.brick),
        pushable=jnp.zeros_like(state.pushable),
        push_t=jnp.zeros_like(state.push_t),
        bush=jnp.zeros_like(state.bush),
        crate=jnp.zeros_like(state.crate),
        rec_crate=jnp.zeros_like(state.rec_crate),
        alive=jnp.array([True, True]),
        hp=jnp.array([MAX_HP, MAX_HP], jnp.int32),
        invuln=jnp.zeros_like(state.invuln),
        bombs_cap=jnp.array([2.0, 2.0], jnp.float32),
        blast_cap=jnp.array([2.0, 2.0], jnp.float32),
        spd_g=jnp.array([1.0, 1.0], jnp.float32),
        t=jnp.array(0, jnp.int32),
        level_id=jnp.array(-1, jnp.int32),
    )


def _idle_actions():
    return jnp.array([[4, 0], [4, 0]], jnp.int32)


def _reward(dmg, alive_before, alive_after, hp_after, done, timeout_alpha=1.0):
    return reward_from_events(
        jnp.asarray(dmg),
        jnp.asarray(alive_before),
        jnp.asarray(alive_after),
        jnp.asarray(hp_after),
        jnp.asarray(done),
        jnp.zeros((1, 2), jnp.bool_),
        jnp.zeros((1, 2), jnp.bool_),
        jnp.zeros((1,), jnp.int32),
        0.0,
        0.0,
        0.0,
        timeout_alpha,
    )


def test_crate_reward_event_requires_actual_growth():
    key = jax.random.PRNGKey(0)
    state = _empty_state(key)
    crate = state.crate.at[2, 2].set(1)

    grown_state, done, info = step(
        state._replace(crate=crate, bombs_cap=jnp.array([2.0, 2.0], jnp.float32)),
        _idle_actions(),
        jax.random.PRNGKey(1),
        auto_reset=False,
        return_info=True,
    )
    assert not bool(done)
    assert bool(info["crate"][0])
    assert int(grown_state.crate[2, 2]) == 0
    assert float(grown_state.bombs_cap[0]) == 3.0

    capped_state, done, info = step(
        state._replace(
            crate=crate,
            bombs_cap=jnp.array([GROWTH_BOMBS_MAX, 2.0], jnp.float32),
        ),
        _idle_actions(),
        jax.random.PRNGKey(2),
        auto_reset=False,
        return_info=True,
    )
    assert not bool(done)
    assert not bool(info["crate"][0])
    assert int(capped_state.crate[2, 2]) == 0
    assert float(capped_state.bombs_cap[0]) == float(GROWTH_BOMBS_MAX)


def test_shared_crate_simultaneous_pickup_grows_only_once():
    state = _empty_state(jax.random.PRNGKey(20))._replace(
        pos=jnp.array([[2.5, 2.5], [2.5, 2.5]], jnp.float32),
    )
    crate = state.crate.at[2, 2].set(1)
    next_state, done, info = step(
        state._replace(crate=crate),
        _idle_actions(),
        jax.random.PRNGKey(21),
        auto_reset=False,
        return_info=True,
    )
    assert not bool(done)
    np.testing.assert_array_equal(np.asarray(info["crate"]), [True, False])
    np.testing.assert_allclose(np.asarray(next_state.bombs_cap), [3.0, 2.0])


def test_novelty_is_first_visit_shared_and_resets_after_terminal():
    visited = jnp.zeros((1, 13, 15), jnp.bool_)
    cells = jnp.array([[[2, 2], [2, 3]]], jnp.int32)

    first, visited = novelty_transition(visited, cells, jnp.array([False]))
    np.testing.assert_array_equal(np.asarray(first), [[True, True]])

    repeated, visited = novelty_transition(visited, cells, jnp.array([False]))
    np.testing.assert_array_equal(np.asarray(repeated), [[False, False]])

    # P0 has already explored (4, 4); P1 entering it later gets no shared-map credit.
    visited = jnp.zeros((1, 13, 15), jnp.bool_).at[0, 4, 4].set(True)
    later_cells = jnp.array([[[5, 5], [4, 4]]], jnp.int32)
    later, _ = novelty_transition(visited, later_cells, jnp.array([False]))
    np.testing.assert_array_equal(np.asarray(later), [[True, False]])

    # Same-tick arrival at one fresh shared cell pays exactly once, deterministically to P0.
    same_cells = jnp.array([[[6, 6], [6, 6]]], jnp.int32)
    simultaneous, _ = novelty_transition(
        jnp.zeros((1, 13, 15), jnp.bool_), same_cells, jnp.array([False]))
    np.testing.assert_array_equal(np.asarray(simultaneous), [[True, False]])

    terminal, after_terminal = novelty_transition(
        jnp.zeros((1, 13, 15), jnp.bool_), cells, jnp.array([True]))
    np.testing.assert_array_equal(np.asarray(terminal), [[True, True]])
    assert not bool(after_terminal.any())


def test_terminal_info_cell_is_not_next_episode_spawn():
    state = _empty_state(jax.random.PRNGKey(3))._replace(
        t=jnp.array(MAX_STEPS - 1, jnp.int32),
    )
    reset_state, done, info = step(
        state,
        _idle_actions(),
        jax.random.PRNGKey(4),
        auto_reset=True,
        return_info=True,
    )
    assert bool(done)
    np.testing.assert_array_equal(np.asarray(info["cell"]), [[2, 2], [10, 10]])
    assert int(reset_state.t) == 0
    assert not np.array_equal(np.asarray(reset_state.pos.astype(jnp.int32)),
                              np.asarray(info["cell"]))


def test_timeout_bonus_is_capped_and_annealed():
    kwargs = dict(
        dmg=[[0, 0]],
        alive_before=[[True, True]],
        alive_after=[[True, True]],
        hp_after=[[MAX_HP, 1]],
        done=[True],
    )
    full = np.asarray(_reward(**kwargs, timeout_alpha=1.0))[0]
    quarter = np.asarray(_reward(**kwargs, timeout_alpha=0.25))[0]
    zero = np.asarray(_reward(**kwargs, timeout_alpha=0.0))[0]

    np.testing.assert_allclose(full, [TIMEOUT_MAX_BONUS - STEP_PENALTY,
                                      -TIMEOUT_MAX_BONUS - STEP_PENALTY])
    np.testing.assert_allclose(quarter, [0.5 - STEP_PENALTY,
                                         -0.5 - STEP_PENALTY])
    np.testing.assert_allclose(zero, [-STEP_PENALTY, -STEP_PENALTY])


def test_death_reward_stays_fixed_independent_of_health_or_timeout_alpha():
    for winner_hp in (1, MAX_HP):
        reward = np.asarray(_reward(
            dmg=[[0, 0]],
            alive_before=[[True, True]],
            alive_after=[[True, False]],
            hp_after=[[winner_hp, 0]],
            done=[True],
            timeout_alpha=0.0,
        ))[0]
        np.testing.assert_allclose(reward, [WIN_BONUS - STEP_PENALTY,
                                            -WIN_BONUS - STEP_PENALTY])


def test_shared_reward_anneal_window_is_unambiguous():
    assert shared_reward_anneal_steps(0.5, 30, 0.01, 30) == 30
    assert shared_reward_anneal_steps(0.0, 0, 0.01, 40) == 40
    with pytest.raises(ValueError, match="must match"):
        shared_reward_anneal_steps(0.5, 30, 0.01, 40)


def test_parameter_warm_start_preserves_fixed_schedule_progress():
    source_steps = 7_457_472_512
    steps_per_iteration = 2 * 32_760 * 256
    assert reward_schedule_global_steps(1, steps_per_iteration, source_steps) == source_steps
    assert fixed_reward_alpha(1, steps_per_iteration, 30_000_000_000, source_steps) == pytest.approx(
        0.7514175829333333
    )
