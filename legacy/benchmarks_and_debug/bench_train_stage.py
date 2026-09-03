"""端到端 PPO 分阶段计时：collect_rollout（环境+前向） vs ppo_update。

同生产配置（mlp_bf16 768, N=4096, steps=256, minibatch 8192, epochs 1），
量化 0.38M sps 里 collect / update 各占多少。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jax_bomb import jax_env as jax_bomb_env
from jax_bomb.jax_env import H, W, N_OBS_CH, init_batch
from jax_bomb.jax_net import init_net, net_forward as jax_bomb_jax_net_forward
from jax_bomb.jax_train import (both_masks, both_perspectives, both_states,
                                collect_rollout, compute_gae, ppo_update,
                                sample_actions)


def main():
    n, steps = 4096, 256
    hidden, mb, epochs = 768, 8192, 1
    arch = "mlp_bf16"
    key = jrandom.PRNGKey(0)
    states = init_batch(key, n)
    key, nk = jrandom.split(key)
    params = init_net(nk, arch, N_OBS_CH, H, W, hidden=hidden)
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)

    # --- collect 计时 ---
    collect_j = jax.jit(
        lambda p, s, k: collect_rollout(p, arch, s, k, steps))
    new_states, batch = collect_j(params, states, key)
    jax.block_until_ready(new_states)
    t0 = time.time()
    for _ in range(5):
        new_states, batch = collect_j(params, new_states, key)
        jax.block_until_ready(new_states)
    dt = time.time() - t0
    print(f"collect_rollout: {dt/5:.2f}s/iter → {2*n*steps/(dt/5):,.0f} sps",
          flush=True)

    # --- danger 阶段 A 占比（max_chain=16 vs 1）---
    def _dng(s, mc):
        return jax_bomb_env._danger_map(s.fuse, s.wall, s.bomb_blast,
                                        s.brick, max_chain=mc)
    dng16 = jax.jit(lambda s: jax.vmap(lambda st: _dng(st, 16))(s))
    dng1 = jax.jit(lambda s: jax.vmap(lambda st: _dng(st, 1))(s))
    a0 = dng16(states); jax.block_until_ready(a0)
    a1 = dng1(states); jax.block_until_ready(a1)
    t0 = time.time()
    for _ in range(10):
        a0 = dng16(new_states); jax.block_until_ready(a0)
    dt16 = (time.time() - t0) / 10
    t0 = time.time()
    for _ in range(10):
        a1 = dng1(new_states); jax.block_until_ready(a1)
    dt1 = (time.time() - t0) / 10
    print(f"_danger_map mc16={dt16*1000:.1f}ms mc1={dt1*1000:.1f}ms "
          f"阶段A占比={(dt16-dt1)/dt16*100:.0f}%", flush=True)
    dng_j = jax.jit(lambda s: jax.vmap(lambda st: jax_bomb_env._danger_map(
        st.fuse, st.wall, st.bomb_blast, st.brick))(s))
    d0 = dng_j(states)
    jax.block_until_ready(d0)
    t0 = time.time()
    for _ in range(10):
        d0 = dng_j(new_states)
        jax.block_until_ready(d0)
    dt = time.time() - t0
    print(f"_danger_map(4096): {dt/10*1000:.1f} ms/tick → 每 iter(256) "
          f"{dt/10*256:.2f}s", flush=True)
    obs_j = jax.jit(lambda s: both_perspectives(s))
    o0 = obs_j(states)
    jax.block_until_ready(o0)
    t0 = time.time()
    for _ in range(10):
        o0 = obs_j(new_states)
        jax.block_until_ready(o0)
    dt = time.time() - t0
    print(f"both_perspectives(8192 obs): {dt/10*1000:.1f} ms/tick → "
          f"每 iter(256) {dt/10*256:.2f}s", flush=True)
    fwd_j = jax.jit(lambda p, o: jax_bomb_jax_net_forward(p, arch, o))
    v0 = fwd_j(params, o0)
    jax.block_until_ready(v0)
    t0 = time.time()
    for _ in range(10):
        v0 = fwd_j(params, o0)
        jax.block_until_ready(v0)
    dt = time.time() - t0
    print(f"net_forward(8192): {dt/10*1000:.1f} ms/tick → 每 iter(256) "
          f"{dt/10*256:.2f}s", flush=True)

    # --- update 计时 ---
    obs, state, acts, lps, vals, rew, done, masks = batch
    fobs = both_perspectives(new_states)
    fmasks = both_masks(new_states)
    fstate = both_states(new_states)
    _, _, fval = sample_actions(params, arch, fobs, fmasks, key, state=fstate)
    next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
    advs = compute_gae(rew, vals, next_val, done, 0.99, 0.95)
    rets = advs + vals
    up_j = jax.jit(lambda p, os_, b, k: ppo_update(
        p, opt, os_, arch, b, k, mb, 0.2, 0.5, 0.01, epochs))
    p2, os2 = up_j(params, opt_state, (obs, state, acts, lps, advs, rets, masks),
                   key)
    jax.block_until_ready(p2)
    t0 = time.time()
    for _ in range(5):
        p2, os2 = up_j(p2, os2,
                       (obs, state, acts, lps, advs, rets, masks), key)
        jax.block_until_ready(p2)
    dt = time.time() - t0
    print(f"ppo_update: {dt/5:.2f}s/iter ({2*n*steps*epochs:,} frames)",
          flush=True)


if __name__ == "__main__":
    main()
