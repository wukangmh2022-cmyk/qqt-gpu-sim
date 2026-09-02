import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, net_forward

def eval_greedy(model_path, weights_cfg, map_label, n_envs=32, steps=1800):
    _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights=weights_cfg)
    with open(model_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    tot = np.zeros(6, np.int64)
    for seed in (11, 22):
        states = init_batch(jrandom.PRNGKey(seed), n_envs)
        key = jrandom.PRNGKey(seed * 7)
        n = n_envs
        def one_step(carry, _):
            states, key = carry
            key, kstep = jrandom.split(key)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            mv, bm_, _, _ = net_forward(chal, "transformer", obs[:n], gv[:n])
            mm, bm = masks[0][:n], masks[1][:n]
            mv = jnp.where(mm, mv, -1e9)
            bm_ = jnp.where(bm, bm_, -1e9)
            # Greedy argmax
            a0_move = jnp.argmax(mv, axis=-1)
            a0_bomb = jnp.argmax(bm_, axis=-1)
            a0 = jnp.stack([a0_move, a0_bomb], axis=-1)
            
            a1 = jnp.full((n, 2), 4, jnp.int32).at[:, 1].set(0)
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jrandom.split(kstep, n)
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
        tot += np.array([int(o.sum()) for o in outs])
        
    kill, solo, both, timeout, bombs, eps = tot
    eps = max(int(eps), 1)
    kill_pct = (kill / eps) * 100
    solo_pct = (solo / eps) * 100
    both_pct = (both / eps) * 100
    timeout_pct = (timeout / eps) * 100
    b_per_ep = float(bombs) / eps
    
    print(f"[{map_label} - 确定性 Argmax] 完局数={eps} 击杀木桩={kill}({kill_pct:.1f}%) "
          f"自杀={solo}({solo_pct:.1f}%) 同归={both}({both_pct:.1f}%) 超时={timeout}({timeout_pct:.1f}%) "
          f"泡/局={b_per_ep:.1f}", flush=True)

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "30"
    ckpt = f"/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local/params_it{int(it):08d}.pkl"
    print(f"=== 评测快照: {ckpt} ===")
    eval_greedy(ckpt, "empty=1.0", "1. 空旷场景道场")
    eval_greedy(ckpt, "", "2. 全关卡复杂地图")
