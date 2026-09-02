import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def run_battle(ckpt_path, n_games=100, max_ticks=800, greedy=False):
    print(f"\n=======================================================")
    print(f"🥊 战力全面大考: {os.path.basename(ckpt_path)} (对局数={n_games}, 模式={'确定性 Argmax' if greedy else '随机采样 Sample'})")
    print(f"=======================================================")
    
    with open(ckpt_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    for map_name, w_cfg in [("空旷场景道场", "empty=1.0"), ("全池241复杂地图", "")]:
        _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights=w_cfg)
        
        states = init_batch(jrandom.PRNGKey(42), n_games)
        key = jrandom.PRNGKey(101)
        
        # 追踪每局比赛的终局状态
        p0_won = np.zeros(n_games, dtype=bool)
        p0_died = np.zeros(n_games, dtype=bool)
        both_died = np.zeros(n_games, dtype=bool)
        game_ended = np.zeros(n_games, dtype=bool)
        total_bombs_p0 = np.zeros(n_games, dtype=int)
        survival_ticks = np.full(n_games, max_ticks, dtype=int)
        
        for t in range(max_ticks):
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
            
            # 对手 P1 = IDLE (静止靶)
            a1 = jnp.full((n_games, 2), 4, jnp.int32).at[:, 1].set(0)
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jrandom.split(kstep, n_games)
            
            states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
            
            alive_p0 = np.array(info["alive"][:, 0])
            alive_p1 = np.array(info["alive"][:, 1])
            is_bomb_p0 = np.array(env_acts[:, 0, 1] == 1)
            
            for g in range(n_games):
                if not game_ended[g]:
                    if is_bomb_p0[g]:
                        total_bombs_p0[g] += 1
                    
                    # 判定胜负
                    if not alive_p0[g] and not alive_p1[g]:
                        both_died[g] = True
                        game_ended[g] = True
                        survival_ticks[g] = t + 1
                    elif not alive_p0[g] and alive_p1[g]:
                        p0_died[g] = True
                        game_ended[g] = True
                        survival_ticks[g] = t + 1
                    elif alive_p0[g] and not alive_p1[g]:
                        p0_won[g] = True
                        game_ended[g] = True
                        survival_ticks[g] = t + 1
            
            if np.all(game_ended):
                break
        
        # 统计汇总
        timeouts = ~(p0_won | p0_died | both_died)
        win_rate = np.mean(p0_won) * 100
        suicide_rate = np.mean(p0_died) * 100
        both_rate = np.mean(both_died) * 100
        timeout_rate = np.mean(timeouts) * 100
        avg_bombs = np.mean(total_bombs_p0)
        avg_survival = np.mean(survival_ticks)
        
        print(f"\n🗺️  【{map_name}】(总对局数={n_games})")
        print(f"   🏆 击杀木桩胜率: {win_rate:5.1f}% ({np.sum(p0_won)}/{n_games})")
        print(f"   💀 自杀败局率  : {suicide_rate:5.1f}% ({np.sum(p0_died)}/{n_games})")
        print(f"   💥 同归于尽率  : {both_rate:5.1f}% ({np.sum(both_died)}/{n_games})")
        print(f"   ⏱️ 超时未死率  : {timeout_rate:5.1f}% ({np.sum(timeouts)}/{n_games})")
        print(f"   💣 平均放泡量  : {avg_bombs:5.1f} 颗/局")
        print(f"   ⏳ 平均存活时间: {avg_survival:5.1f} ticks")

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "39"
    greedy_flag = "--greedy" in sys.argv
    ckpt = f"/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local/params_it{int(it):08d}.pkl"
    run_battle(ckpt, n_games=100, max_ticks=800, greedy=greedy_flag)
