import pickle
import sys
import jax
import jax.numpy as jnp

from jax_bomb import levels
from jax_bomb.jax_env import step, _fresh
from jax_bomb.jax_train import sample_actions, both_perspectives, both_masks, both_states

def evaluate(ckpt_path, n_envs=128, steps=300):
    levels.set_active("levels.json")
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    params = ck.get("params", ck)
    
    key = jax.random.PRNGKey(0)
    k_init, key = jax.random.split(key)
    states = jax.vmap(_fresh)(jax.random.split(k_init, n_envs))
    
    @jax.jit
    def run_eval(states, key):
        def step_fn(carry, _):
            states, key = carry
            key, k0, k1, kstep = jax.random.split(key, 4)
            obs = both_perspectives(states)
            masks = both_masks(states)
            gv = both_states(states)
            
            # P0 acts with policy
            a0, _, _ = sample_actions(params, "transformer", obs[:n_envs],
                                      (masks[0][:n_envs], masks[1][:n_envs]), k0,
                                      state=gv[:n_envs])
            # P1 does IDLE (move=4, bomb=0)
            a1_idle = jnp.zeros((n_envs, 2), dtype=jnp.int32).at[:, 0].set(4)
            
            # Self play
            a1_self, _, _ = sample_actions(params, "transformer", obs[n_envs:],
                                           (masks[0][n_envs:], masks[1][n_envs:]), k1,
                                           state=gv[n_envs:])
            
            # Step IDLE
            env_acts_idle = jnp.stack([a0, a1_idle], axis=1)
            keys = jax.random.split(kstep, n_envs)
            new_states_idle, done_i, info_i = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts_idle, keys)
            
            # Step SELF
            env_acts_self = jnp.stack([a0, a1_self], axis=1)
            new_states_self, done_s, info_s = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts_self, keys)
            
            # Record events
            # In IDLE: P0 suicide is done & ~p0_alive
            p0_alive_i = info_i["alive"][:, 0]
            p1_alive_i = info_i["alive"][:, 1]
            p0_win_i = done_i & p0_alive_i & (~p1_alive_i)
            p0_die_i = done_i & (~p0_alive_i)
            
            # In SELF:
            p0_alive_s = info_s["alive"][:, 0]
            p1_alive_s = info_s["alive"][:, 1]
            p0_win_s = done_s & p0_alive_s & (~p1_alive_s)
            p1_win_s = done_s & p1_alive_s & (~p0_alive_s)
            
            return (new_states_idle, key), (p0_win_i, p0_die_i, done_i, p0_win_s, p1_win_s, done_s)
        
        _, (w_i, d_i, tot_i, w_s, l_s, tot_s) = jax.lax.scan(step_fn, (states, key), None, length=steps)
        return (w_i.sum(), d_i.sum(), tot_i.sum(), w_s.sum(), l_s.sum(), tot_s.sum())
    
    w_i, d_i, tot_i, w_s, l_s, tot_s = run_eval(states, key)
    print(f"=== Evaluation of {ckpt_path} ({n_envs} envs x {steps} steps) ===")
    print(f"Against IDLE opponent (Total completed: {int(tot_i)}):")
    print(f"  P0 Kills (Win): {int(w_i)} ({float(w_i)/max(1, float(tot_i))*100:.1f}%)")
    print(f"  P0 Suicides (Death): {int(d_i)} ({float(d_i)/max(1, float(tot_i))*100:.1f}%)")
    print(f"Self-Play P0 vs P1 (Total completed: {int(tot_s)}):")
    print(f"  P0 Wins: {int(w_s)} ({float(w_s)/max(1, float(tot_s))*100:.1f}%)")
    print(f"  P1 Wins: {int(l_s)} ({float(l_s)/max(1, float(tot_s))*100:.1f}%)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/params_Pre-Train_Test.pkl"
    evaluate(path)
