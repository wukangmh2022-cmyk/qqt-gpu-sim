import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def run_fast_tournament(ckpt_path, n_games=128, max_ticks=800, greedy=False):
    print(f"\n=======================================================")
    print(f"🥊 战力全面大考: {os.path.basename(ckpt_path)} (对局数={n_games}, 模式={'确定性 Argmax' if greedy else '随机采样 Sample'})")
    print(f"=======================================================")
    
    with open(ckpt_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    for map_name, w_cfg in [("空旷场景道场", "empty=1.0"), ("全池241复杂地图", "")]:
        _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights=w_cfg)
        
        states = init_batch(jrandom.PRNGKey(42), n_games)
        key = jrandom.PRNGKey(101)
        
        # carry: (states, key, alive_mask_0, alive_mask_1, has_ended, end_tick, bomb_counts)
        carry_init = (
            states,
            key,
            jnp.ones(n_games, dtype=bool),
            jnp.ones(n_games, dtype=bool),
            jnp.zeros(n_games, dtype=bool),
            jnp.full(n_games, max_ticks, dtype=jnp.int32),
            jnp.zeros(n_games, dtype=jnp.int32)
        )
        
        def step_fn(carry, tick_idx):
            states, key, a0_alive, a1_alive, ended, end_t, bombs = carry
            key, k0, kstep = jrandom.split(key, 3)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            
            if greedy:
                mv, bm_, _, _ = net_forward(chal, "transformer", obs[:n_games], gv[:n_games])
                mm, bm = masks[0][:n_games], masks[1][:n_games]
                mv = jnp.where(mm, mv, -1e9)
                bm_ = jnp.where(bm, bm_, -1e9)
                a0 = jnp.stack([jnp.argmax(mv, axis=-1), jnp.argmax(bm_, axis=-1)], axis=-1)
            else:
                a0 = sample_actions(chal, "transformer", obs[:n_games], (masks[0][:n_games], masks[1][:n_games]), k0, state=gv[:n_games])[0]
            
            # P1 = 静止靶 (IDLE)
            a1 = jnp.full((n_games, 2), 4, jnp.int32).at[:, 1].set(0)
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jrandom.split(kstep, n_games)
            
            # Step environment
            new_states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
            
            is_bomb = (env_acts[:, 0, 1] == 1) & ~ended
            new_bombs = bombs + is_bomb.astype(jnp.int32)
            
            cur_a0 = info["alive"][:, 0]
            cur_a1 = info["alive"][:, 1]
            
            just_ended = (~cur_a0 | ~cur_a1) & ~ended
            new_ended = ended | just_ended
            new_end_t = jnp.where(just_ended, tick_idx + 1, end_t)
            
            new_a0_alive = jnp.where(just_ended, cur_a0, a0_alive)
            new_a1_alive = jnp.where(just_ended, cur_a1, a1_alive)
            
            return (new_states, key, new_a0_alive, new_a1_alive, new_ended, new_end_t, new_bombs), None
        
        ticks = jnp.arange(max_ticks)
        (final_s, _, final_a0, final_a1, final_ended, final_end_t, final_bombs), _ = jax.lax.scan(
            step_fn, carry_init, ticks
        )
        
        # 结果判定
        win = final_a0 & ~final_a1
        suicide = ~final_a0 & final_a1
        both = ~final_a0 & ~final_a1
        timeout = final_a0 & final_a1
        
        n = float(n_games)
        win_pct = float(jnp.mean(win.astype(jnp.float32))) * 100
        suicide_pct = float(jnp.mean(suicide.astype(jnp.float32))) * 100
        both_pct = float(jnp.mean(both.astype(jnp.float32))) * 100
        timeout_pct = float(jnp.mean(timeout.astype(jnp.float32))) * 100
        avg_b = float(jnp.mean(final_bombs.astype(jnp.float32)))
        avg_surv = float(jnp.mean(final_end_t.astype(jnp.float32)))
        
        print(f"\n🗺️  【{map_name}】(总对局数={n_games})")
        print(f"   🏆 击杀木桩胜率: {win_pct:5.1f}% ({int(jnp.sum(win))}/{n_games})")
        print(f"   💀 自杀败局率  : {suicide_pct:5.1f}% ({int(jnp.sum(suicide))}/{n_games})")
        print(f"   💥 同归于尽率  : {both_pct:5.1f}% ({int(jnp.sum(both))}/{n_games})")
        print(f"   ⏱️ 超时未死率  : {timeout_pct:5.1f}% ({int(jnp.sum(timeout))}/{n_games})")
        print(f"   💣 平均放泡量  : {avg_b:5.1f} 颗/局")
        print(f"   ⏳ 平均存活时间: {avg_surv:5.1f} ticks")

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "39"
    greedy_flag = "--greedy" in sys.argv
    ckpt = f"/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local/params_it{int(it):08d}.pkl"
    run_fast_tournament(ckpt, n_games=128, max_ticks=800, greedy=greedy_flag)
