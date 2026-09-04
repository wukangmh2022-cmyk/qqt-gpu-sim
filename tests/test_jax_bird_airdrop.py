import jax
import jax.numpy as jnp
import pytest
from jax_bomb.jax_env import (H, W, BombState, _fresh, step, make_obs, N_OBS_CH,
                              N_OBS_CH_V2, init_batch)

def test_graveyard_collection_and_ch14():
    key = jax.random.PRNGKey(101)
    st = _fresh(key)
    # Clear map crates
    st = st._replace(crate=jnp.zeros((H, W), jnp.int8))
    assert int(st.graveyard) == 0

    # ch14 should be 0 when graveyard is 0
    obs15 = make_obs(st, 0, channels=15)
    assert jnp.all(obs15[14] == 0.0)

    # Put crate and bomb to destroy it
    st = st._replace(
        crate=st.crate.at[4, 4].set(jnp.int8(1)),
        fuse=st.fuse.at[4, 3].set(1),
        bomb_blast=st.bomb_blast.at[4, 3].set(2),
        owner=st.owner.at[4, 3].set(0)
    )
    actions = jnp.zeros((2, 2), jnp.int32)
    key, subkey = jax.random.split(key)
    st, done = step(st, actions, subkey, auto_reset=False)
    assert int(st.crate[4, 4]) == 0
    assert int(st.graveyard) == 1

def test_bird_airdrop_flight_and_columns():
    key = jax.random.PRNGKey(202)
    st = _fresh(key)
    st = st._replace(crate=jnp.zeros((H, W), jnp.int8), graveyard=jnp.int32(5), t=jnp.int32(270))
    actions = jnp.zeros((2, 2), jnp.int32)

    drops = []
    for _ in range(30):
        key, subkey = jax.random.split(key)
        pre_crates = int((st.crate > 0).sum())
        pre_t = int(st.t)
        st, done = step(st, actions, subkey, auto_reset=False)
        post_crates = int((st.crate > 0).sum())
        if post_crates > pre_crates:
            coords = jnp.argwhere(st.crate > 0)
            col = int(coords[-1, 1])
            drops.append((pre_t, col))

    # All 5 items must drop
    assert len(drops) == 5
    # All drops must be within columns [1, 13]
    for t_step, col in drops:
        assert 1 <= col <= 13
    # First drop should be on the right side, last drop on the left side
    assert drops[0][1] >= drops[-1][1]

def test_jit_performance():
    key = jax.random.PRNGKey(303)
    states = init_batch(key, 64)
    actions = jnp.zeros((64, 2, 2), jnp.int32)
    keys = jax.random.split(key, 64)

    step_vmap = jax.jit(jax.vmap(step, in_axes=(0, 0, 0)))
    # Warmup
    states, _ = step_vmap(states, actions, keys)
    # Run 50 steps
    for _ in range(50):
        keys = jax.random.split(keys[0], 64)
        states, _ = step_vmap(states, actions, keys)
