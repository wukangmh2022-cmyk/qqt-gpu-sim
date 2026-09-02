import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def eval_ckpt(ckpt_path, label=""):
    print(f"\n==========================================")
    print(f"📊 快照对局评测: {os.path.basename(ckpt_path)} {label}")
    print(f"==========================================")
    with open(ckpt_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    for map_name, w_cfg in [("空旷场景道场", "empty=1.0"), ("全池241复杂地图", "")]:
        _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights=w_cfg)
        
        for mode_name, is_greedy in [("随机采样 (Sample)", False), ("确定性 (Argmax)", True)]:
            n_envs = 64
            steps = 400
            states = init_batch(jrandom.PRNGKey(42), n_envs)
            key = jrandom.PRNGKey(101)
            
            def one_step(carry, _):
                states, key = carry
                key, k0, kstep = jrandom.split(key, 3)
                obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
                
                if is_greedy:
                    mv, bm_, _, _ = net_forward(chal, "transformer", obs[:n_envs], gv[:n_envs])
                    mm, bm = masks[0][:n_envs], masks[1][:n_envs]
                    mv = jnp.where(mm, mv, -1e9)
                    bm_ = jnp.where(bm, bm_, -1e9)
                    a0 = jnp.stack([jnp.argmax(mv, axis=-1), jnp.argmax(bm_, axis=-1)], axis=-1)
                else:
                    a0 = sample_actions(chal, "transformer", obs[:n_envs], (masks[0][:n_envs], masks[1][:n_envs]), k0, state=gv[:n_envs])[0]
                
                # 对手 P1 = 静止靶 IDLE
                a1 = jnp.full((n_envs, 2), 4, jnp.int32).at[:, 1].set(0)
                env_acts = jnp.stack([a0, a1], axis=1)
                keys = jrandom.split(kstep, n_envs)
                new_states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
                
                n_alive = info["alive"].sum(-1)
                death_done = done & (n_alive == 1)
                kill = death_done & info["alive"][:, 0]
                solo = death_done & info["alive"][:, 1]
                both_d = done & (n_alive == 0)
                timeout = done & (n_alive == 2)
                return (new_states, key), (kill.sum(), solo.sum(), both_d.sum(), timeout.sum(),
                                           (env_acts[:, 0, 1] == 1).sum(), done.sum())
            
            (states, key), outs = jax.lax.scan(one_step, (states, key), None, length=steps)
            tot = np.array([int(o.sum()) for o in outs])
            kill, solo, both, timeout, bombs, eps = tot
            eps = max(int(eps), 1)
            kill_pct = (kill / eps) * 100
            solo_pct = (solo / eps) * 100
            both_pct = (both / eps) * 100
            timeout_pct = (timeout / eps) * 100
            b_per_ep = float(bombs) / eps
            
            print(f"[{map_name} | {mode_name:12s}] 局数={eps:3d} | "
                  f"击杀静止靶={kill:2d} ({kill_pct:5.1f}%) | "
                  f"自杀={solo:2d} ({solo_pct:5.1f}%) | "
                  f"同归={both:2d} ({both_pct:5.1f}%) | "
                  f"超时={timeout:2d} ({timeout_pct:5.1f}%) | "
                  f"泡/局={b_per_ep:4.1f}", flush=True)

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "30"
    ckpt = f"/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local/params_it{int(it):08d}.pkl"
    eval_ckpt(ckpt, f"(Iter {it})")
