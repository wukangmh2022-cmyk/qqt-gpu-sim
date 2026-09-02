import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, net_forward, sample_actions

def run_eval(model_path, weights_cfg, map_label, greedy=False, n_envs=64, steps=1800):
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "levels.json")
    _levels.set_active(json_path, weights=weights_cfg)
    with open(model_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    tot = np.zeros(6, np.int64)
    for seed in (11, 22, 33):
        states = init_batch(jrandom.PRNGKey(seed), n_envs)
        key = jrandom.PRNGKey(seed * 7)
        n = n_envs
        def one_step(carry, _):
            states, key = carry
            key, k0, kstep = jrandom.split(key, 3)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            if greedy:
                mv, bm_, _, _ = net_forward(chal, "transformer", obs[:n], gv[:n])
                mm, bm = masks[0][:n], masks[1][:n]
                mv = jnp.where(mm, mv, -1e9)
                bm_ = jnp.where(bm, bm_, -1e9)
                a0_move = jnp.argmax(mv, axis=-1)
                a0_bomb = jnp.argmax(bm_, axis=-1)
                a0 = jnp.stack([a0_move, a0_bomb], axis=-1)
            else:
                a0 = sample_actions(chal, "transformer", obs[:n], (masks[0][:n], masks[1][:n]), k0, state=gv[:n])[0]
            
            # P1 = IDLE
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
    
    tag = "确定性 Argmax" if greedy else "随机采样 Sample"
    print(f"[{map_label} - {tag}] 局数={eps} 击杀静止靶={kill}({kill_pct:.1f}%) "
          f"自杀={solo}({solo_pct:.1f}%) 同归={both}({both_pct:.1f}%) 超时={timeout}({timeout_pct:.1f}%) "
          f"泡/局={b_per_ep:.1f}", flush=True)

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "30"
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(base, "ckpt_local", f"params_it{int(it):08d}.pkl")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(base, "ckpt", f"ckpt_{int(it):08d}_r0.pkl")
    print(f"=== 评测快照: {ckpt} ===")
    print("--- [1] 空旷道场对局 ---")
    run_eval(ckpt, "empty=1.0", "空旷道场", greedy=False)
    run_eval(ckpt, "empty=1.0", "空旷道场", greedy=True)
    print("--- [2] 全池 241 地图对局 ---")
    run_eval(ckpt, None, "全池241图", greedy=False)
    run_eval(ckpt, None, "全池241图", greedy=True)
