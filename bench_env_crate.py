"""新版 jax 环境纯 step 吞吐微基准（定位地图生成/宝箱开销）。

- 现版：_fresh 带地图生成 + 宝箱链路（8 通道 obs 已含）
- A/B：把 step 的 auto_reset 分支换成"空场 _fresh"（无地图生成），对比
  env-step/s —— 量化地图生成占 step 的比例。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax_bomb import jax_env
from jax_bomb.jax_env import (H, W, N_OBS_CH, BombState, _fresh, init_batch,
                              legal_mask, make_obs, step)

MODE = os.environ.get("CRATE_BENCH_MODE", "full")   # full | plain


def plain_fresh(key):
    """旧版空场 _fresh（无地图/宝箱）：只用于 A/B 对比。"""
    wall = jnp.zeros((H, W), jnp.bool_)
    brick = jnp.zeros((H, W), jnp.bool_)
    crate = jnp.zeros((H, W), jnp.bool_)
    rec = jnp.zeros((H, W), jnp.bool_)
    pos = jnp.array([[8.5, 5.5], [8.5, 8.5]], jnp.float32)
    return BombState(pos, jnp.zeros((H, W), jnp.int32),
                     -jnp.ones((H, W), jnp.int32),
                     jnp.zeros((H, W), jnp.int32), wall, brick, crate, rec,
                     jnp.ones((2,), jnp.bool_), jnp.full((2,), 5, jnp.int32),
                     jnp.zeros((2,), jnp.int32),
                     jnp.full((2,), 2.0, jnp.float32),
                     jnp.full((2,), 2.0, jnp.float32),
                     jnp.full((2,), 1.0, jnp.float32), jnp.array(False),
                     jnp.zeros((), jnp.int32))


def bench():
    n, steps = 4096, 200
    key = jrandom.PRNGKey(0)
    states = init_batch(key, n)
    keys = jrandom.split(jrandom.PRNGKey(1), n)
    acts = jrandom.randint(jrandom.PRNGKey(2), (n, 2, 2), 0, 5)
    acts = acts.at[:, :, 1].set(jrandom.randint(jrandom.PRNGKey(3), (n, 2), 0, 2))

    if MODE == "plain":
        states = jax.vmap(plain_fresh)(jrandom.split(key, n))

    def one_step(states, keys):
        new_states, done = jax.vmap(step)(states, acts, keys)
        return new_states, keys

    one_step_j = jax.jit(one_step)
    new_states, _ = one_step_j(states, keys)
    jax.block_until_ready(new_states)

    # 计时
    t0 = time.time()
    for _ in range(20):
        new_states, _ = one_step_j(new_states, keys)
        jax.block_until_ready(new_states)
    dt = time.time() - t0
    envs = n * 20
    print(f"[{MODE}] {envs} env-steps in {dt:.2f}s → {envs/dt:,.0f} env-step/s")


if __name__ == "__main__":
    bench()
