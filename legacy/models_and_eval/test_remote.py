import pickle, jax, jax.numpy as jnp, numpy as np
from jax_bomb.jax_train import sample_actions, both_perspectives, both_masks, both_states
from jax_bomb.jax_env import step, _fresh, N_MOVES, N_BOMB
from jax_bomb import levels

with open("ckpt/params_it00000057_ema.pkl", "rb") as f:
    ck = pickle.load(f)
params = ck.get("params", ck)

def test_map(map_name, weights, num_envs=100, num_steps=600):
    levels.set_active("levels.json", weights=weights)
    key = jax.random.PRNGKey(42)
    k_init, key = jax.random.split(key)
    states = jax.vmap(_fresh)(jax.random.split(k_init, num_envs))
    
    p0_wins = 0
    p1_wins = 0
    p0_suicides = 0
    draws = 0
    
    for t in range(num_steps):
        key, k0, kstep = jax.random.split(key, 3)
        obs = both_perspectives(states)
        masks = both_masks(states)
        gv = both_states(states)
        
        a0, _, _ = sample_actions(params, "transformer", obs[:num_envs],
                                  (masks[0][:num_envs], masks[1][:num_envs]), k0,
                                  state=gv[:num_envs])
        a1 = jnp.zeros((num_envs, 2), dtype=jnp.int32)
        a1 = a1.at[:, 0].set(4)
        
        env_acts = jnp.stack([a0, a1], axis=1)
        keys = jax.random.split(kstep, num_envs)
        
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
        
        dones_np = np.array(done)
        alives = np.array(info["alive"])
        
        for i in range(num_envs):
            if dones_np[i]:
                p0_alive = alives[i, 0]
                p1_alive = alives[i, 1]
                if p0_alive and not p1_alive:
                    p0_wins += 1
                elif not p0_alive and p1_alive:
                    p1_wins += 1
                    p0_suicides += 1
                elif not p0_alive and not p1_alive:
                    draws += 1
                    p0_suicides += 1
                else:
                    hp = np.array(info["hp"][i])
                    if hp[0] > hp[1]: p0_wins += 1
                    elif hp[1] > hp[0]: p1_wins += 1
                    else: draws += 1
        states = new_states
        
    tot = p0_wins + p1_wins + draws
    print(f"=== {map_name} (打静止靶子, 完局数={tot}) ===")
    print(f"  P0 击杀靶子胜利: {p0_wins} ({p0_wins/max(1,tot)*100:.1f}%)")
    print(f"  P0 炸死自己(自杀): {p0_suicides} ({p0_suicides/max(1,tot)*100:.1f}%)")

print("1. 纯空场景道场 (Stage 0 地图):")
test_map("纯空场景道场", weights="empty=1.0")

print("\n2. 包含大量宝箱的复杂地图 (用户截图中的场景):")
test_map("复杂宝箱地图", weights="")
