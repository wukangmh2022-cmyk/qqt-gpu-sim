import sys, os, subprocess, time, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions

def pull_ckpt(iter_num):
    fname = f"params_it{iter_num:08d}.pkl"
    local_dir = "/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fname)
    remote_path = f"/root/private_data/qqt-gpu-sim/ckpt_local/{fname}"
    cmd = f"/tmp/ndrun/pull_0 {remote_path} {local_path}"
    print(f"Pulling {fname} from Node 0...", flush=True)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if os.path.exists(local_path):
        print(f"  ✓ Successfully downloaded {local_path} ({os.path.getsize(local_path):,} bytes)", flush=True)
        return local_path
    else:
        print(f"  ✗ Failed to pull {remote_path}", flush=True)
        return None

def eval_idle(model_path, weights_cfg, map_label, n_envs=32, steps=1800):
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
            key, k0, kstep = jrandom.split(key, 3)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            a0 = sample_actions(chal, "transformer", obs[:n], (masks[0][:n], masks[1][:n]), k0, state=gv[:n])[0]
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
                                       (env_acts[:, :, 1] == 1).sum(), done.sum())
        (states, key), outs = jax.lax.scan(one_step, (states, key), None, length=steps)
        tot += np.array([int(o.sum()) for o in outs])
        
    kill, solo, both, timeout, bombs, eps = tot
    eps = max(int(eps), 1)
    kill_pct = (kill / eps) * 100
    solo_pct = (solo / eps) * 100
    both_pct = (both / eps) * 100
    timeout_pct = (timeout / eps) * 100
    bombs_per_ep = bombs / eps
    
    print(f"[{map_label}] 完局数={eps} 击杀木桩={kill}({kill_pct:.1f}%) 自杀={solo}({solo_pct:.1f}%) 同归={both}({both_pct:.1f}%) 超时={timeout}({timeout_pct:.1f}%) 泡/局={bombs_per_ep:.1f}", flush=True)
    return {
        "eps": eps, "kill_pct": kill_pct, "solo_pct": solo_pct,
        "both_pct": both_pct, "timeout_pct": timeout_pct, "bombs_per_ep": bombs_per_ep
    }

if __name__ == "__main__":
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 34
    p = pull_ckpt(it)
    if p:
        print(f"\n=== 测试 ①: IDLE 分地形基准评测 (Iter {it}) ===")
        res_empty = eval_idle(p, "empty=1.0", "1. 空旷场景道场")
        res_full = eval_idle(p, "", "2. 全关卡复杂地图")
