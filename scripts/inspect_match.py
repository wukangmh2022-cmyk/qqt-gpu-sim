import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def inspect_match(ckpt_path):
    with open(ckpt_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights="empty=1.0")
    key = jrandom.PRNGKey(42)
    state = init_batch(key, 1)
    
    print(f"Initial Pos: P0={state.pos[0, 0]}, P1={state.pos[0, 1]}")
    
    for t in range(200):
        obs = both_perspectives(state)
        masks = both_masks(state)
        gv = both_states(state)
        
        # Sample action
        key, k0, kstep = jrandom.split(key, 3)
        acts = sample_actions(chal, "transformer", obs, masks, k0, state=gv)[0]
        a0 = acts[0] # P0 action
        a1 = jnp.array([4, 0], jnp.int32) # P1 IDLE
        
        env_acts = jnp.stack([a0, a1])[None, :]
        new_state, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, auto_reset=False, return_info=True))(
            state, env_acts, jrandom.split(kstep, 1)
        )
        
        dir_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        d0, b0 = int(a0[0]), int(a0[1])
        p0_pos = new_state.pos[0, 0]
        p1_pos = new_state.pos[0, 1]
        
        bombs_on_field = np.argwhere(np.asarray(new_state.fuse[0] > 0))
        bombs_str = ", ".join([f"({r},{c}:fuse={int(new_state.fuse[0, r, c])})" for r, c in bombs_on_field])
        
        if b0 == 1 or not bool(info['alive'][0, 0]) or len(bombs_on_field) > 0:
            print(f"t={t:03d} | P0 Act: {dir_names[d0]:5s} Bomb={b0} | P0 Pos: ({p0_pos[0]:.2f}, {p0_pos[1]:.2f}) | P0 Alive: {bool(info['alive'][0, 0])} | Bombs: [{bombs_str}]")
        
        if not bool(info['alive'][0, 0]):
            print(f"💀 P0 DIED at t={t}!")
            break
        state = new_state

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt_local/params_it00000050.pkl"
    inspect_match(ckpt)
