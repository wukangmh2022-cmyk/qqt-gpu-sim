import pickle
import sys
import jax
import jax.numpy as jnp
import numpy as np

from jax_bomb import levels
from jax_bomb.jax_env import step, _fresh
from jax_bomb.jax_train import sample_actions, both_perspectives, both_masks, both_states

def eval_policy(ckpt_path, num_envs=64, steps=1800):
    levels.set_active("levels.json")
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    params = ck.get("params", ck)
    
    # 1. Evaluate Self-Play (P0 vs P1 both use policy)
    key = jax.random.PRNGKey(42)
    k_init, key = jax.random.split(key)
    states = jax.vmap(_fresh)(jax.random.split(k_init, num_envs))
    
    @jax.jit
    def run_self(states, key):
        def step_fn(carry, _):
            states, key = carry
            key, k0, k1, kstep = jax.random.split(key, 4)
            obs = both_perspectives(states)
            masks = both_masks(states)
            gv = both_states(states)
            a0, _, _ = sample_actions(params, "transformer", obs[:num_envs],
                                      (masks[0][:num_envs], masks[1][:num_envs]), k0,
                                      state=gv[:num_envs])
            a1, _, _ = sample_actions(params, "transformer", obs[num_envs:],
                                      (masks[0][num_envs:], masks[1][num_envs:]), k1,
                                      state=gv[num_envs:])
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jax.random.split(kstep, num_envs)
            new_states, done, info = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
            
            p0_win = done & info["alive"][:, 0] & (~info["alive"][:, 1])
            p1_win = done & info["alive"][:, 1] & (~info["alive"][:, 0])
            both_die = done & (~info["alive"][:, 0]) & (~info["alive"][:, 1])
            bomb_count = (a0[:, 1] == 1).sum() + (a1[:, 1] == 1).sum()
            return (new_states, key), (p0_win, p1_win, both_die, done, bomb_count)
        _, (w0, w1, bd, dones, b_cnt) = jax.lax.scan(step_fn, (states, key), None, length=steps)
        return w0.sum(), w1.sum(), bd.sum(), dones.sum(), b_cnt.sum()
    
    w0, w1, bd, tot_s, b_cnt = run_self(states, key)
    print(f"\n==========================================")
    print(f"Model: {ckpt_path}")
    print(f"--- 1. Self-Play (P0 vs P1) [{num_envs} envs x {steps} steps] ---")
    print(f"  Total Bombs Placed: {int(b_cnt)}")
    print(f"  Total Games Ended: {int(tot_s)}")
    print(f"  P0 Wins: {int(w0)} ({float(w0)/max(1, float(tot_s))*100:.1f}%)")
    print(f"  P1 Wins: {int(w1)} ({float(w1)/max(1, float(tot_s))*100:.1f}%)")
    print(f"  Both Died / Draw: {int(bd)} ({float(bd)/max(1, float(tot_s))*100:.1f}%)")
    
    # 2. Evaluate vs IDLE (P0 uses policy, P1 does nothing)
    k_init2, key = jax.random.split(key)
    states2 = jax.vmap(_fresh)(jax.random.split(k_init2, num_envs))
    
    @jax.jit
    def run_idle(states, key):
        def step_fn(carry, _):
            states, key = carry
            key, k0, kstep = jax.random.split(key, 3)
            obs = both_perspectives(states)
            masks = both_masks(states)
            gv = both_states(states)
            a0, _, _ = sample_actions(params, "transformer", obs[:num_envs],
                                      (masks[0][:num_envs], masks[1][:num_envs]), k0,
                                      state=gv[:num_envs])
            a1 = jnp.zeros((num_envs, 2), dtype=jnp.int32).at[:, 0].set(4) # IDLE
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jax.random.split(kstep, num_envs)
            new_states, done, info = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
            
            p0_win = done & info["alive"][:, 0] & (~info["alive"][:, 1])
            p0_die = done & (~info["alive"][:, 0])
            return (new_states, key), (p0_win, p0_die, done)
        _, (wins, suicides, dones) = jax.lax.scan(step_fn, (states, key), None, length=steps)
        return wins.sum(), suicides.sum(), dones.sum()
    
    w_i, d_i, tot_i = run_idle(states2, key)
    print(f"--- 2. Against Static IDLE Opponent ---")
    print(f"  Total Games Ended: {int(tot_i)}")
    print(f"  P0 Kills Opponent (Win): {int(w_i)} ({float(w_i)/max(1, float(tot_i))*100:.1f}%)")
    print(f"  P0 Suicides (Died): {int(d_i)} ({float(d_i)/max(1, float(tot_i))*100:.1f}%)")
    print(f"==========================================\n")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        eval_policy(p)
