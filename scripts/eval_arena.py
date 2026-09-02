import sys, os, pickle
import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def hunter_bot_policy(states, masks, key):
    """JAX 全向量化 AI Hunter 规则机器人:
    - 朝对手方向移动（若合法）
    - 避开危险
    - 贴身时（距离<=2）高概率放泡
    """
    n = states.pos.shape[0]
    p0_pos = states.pos[:, 0] # (n, 2) y, x
    p1_pos = states.pos[:, 1] # (n, 2) y, x
    
    # 相对位置
    dy = p0_pos[:, 0] - p1_pos[:, 0]
    dx = p0_pos[:, 1] - p1_pos[:, 1]
    dist = jnp.abs(dy) + jnp.abs(dx)
    
    # 移动优先级: 0上 1下 2左 3右 4停
    # dy < 0: 上(0), dy > 0: 下(1), dx < 0: 左(2), dx > 0: 右(3)
    pref_move = jnp.where(jnp.abs(dy) >= jnp.abs(dx),
                          jnp.where(dy < 0, 0, 1),
                          jnp.where(dx < 0, 2, 3))
    
    # 合法掩码检查
    mmask = masks[0][n:] # P1 move mask
    bmask = masks[1][n:] # P1 bomb mask
    
    pref_legal = jnp.take_along_axis(mmask, pref_move[:, None], axis=1).squeeze(1)
    
    # 若首选不合法，从合法方向中选择
    move_act = jnp.where(pref_legal, pref_move, 4) # 兜底停
    
    # 放泡策略: 贴身 (dist <= 2) 且有放泡配额
    bomb_act = (dist <= 2) & bmask[:, 1]
    
    return jnp.stack([move_act, bomb_act.astype(jnp.int32)], axis=1)

def run_tournament(ckpt_path, opponent_type="idle", n_games=128, max_ticks=800, greedy=False):
    with open(ckpt_path, "rb") as f:
        chal = jax.tree.map(jnp.asarray, pickle.load(f))
    
    opp_names = {
        "idle": "静止木桩 (IDLE)",
        "random": "随机游走 (Random)",
        "hunter": "追猎者 (AI Hunter)"
    }
    opp_name = opp_names.get(opponent_type, opponent_type)
    
    print(f"\n==========================================================================")
    print(f"🥊 对局评测: [{os.path.basename(ckpt_path)}] vs [{opp_name}]")
    print(f"   对局总数: {n_games} 局 | 动作策略: {'确定性 (Argmax)' if greedy else '随机采样 (Sample)'}")
    print(f"==========================================================================")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lvl_json = os.path.join(base_dir, "levels.json")
    for map_name, w_cfg in [("空旷场景道场", "empty=1.0"), ("全池241复杂地图", "")]:
        _levels.set_active(lvl_json, weights=w_cfg)
        
        states = init_batch(jrandom.PRNGKey(42), n_games)
        key = jrandom.PRNGKey(101)
        
        # carry: (states, key, a0_alive, a1_alive, ended, end_t, p0_bombs, p1_bombs)
        carry_init = (
            states,
            key,
            jnp.ones(n_games, dtype=bool),
            jnp.ones(n_games, dtype=bool),
            jnp.zeros(n_games, dtype=bool),
            jnp.full(n_games, max_ticks, dtype=jnp.int32),
            jnp.zeros(n_games, dtype=jnp.int32),
            jnp.zeros(n_games, dtype=jnp.int32)
        )
        
        def step_fn(carry, tick_idx):
            states, key, a0_alive, a1_alive, ended, end_t, b0_cnt, b1_cnt = carry
            key, k0, k1, kstep = jrandom.split(key, 4)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            
            # P0: Agent Policy
            if greedy:
                mv, bm_, _, _ = net_forward(chal, "transformer", obs[:n_games], gv[:n_games])
                mm, bm = masks[0][:n_games], masks[1][:n_games]
                mv = jnp.where(mm, mv, -1e9)
                bm_ = jnp.where(bm, bm_, -1e9)
                a0 = jnp.stack([jnp.argmax(mv, axis=-1), jnp.argmax(bm_, axis=-1)], axis=-1)
            else:
                a0 = sample_actions(chal, "transformer", obs[:n_games], (masks[0][:n_games], masks[1][:n_games]), k0, state=gv[:n_games])[0]
            
            # P1: Opponent Policy
            if opponent_type == "idle":
                a1 = jnp.full((n_games, 2), 4, jnp.int32).at[:, 1].set(0)
            elif opponent_type == "random":
                mmask1, bmask1 = masks[0][n_games:], masks[1][n_games:]
                m1 = jrandom.choice(k1, 5, shape=(n_games,))
                b1 = (jrandom.uniform(k0, (n_games,)) < 0.1) & bmask1[:, 1]
                a1 = jnp.stack([m1, b1.astype(jnp.int32)], axis=1)
            elif opponent_type == "hunter":
                a1 = hunter_bot_policy(states, masks, k1)
            else:
                a1 = jnp.full((n_games, 2), 4, jnp.int32).at[:, 1].set(0)
            
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jrandom.split(kstep, n_games)
            
            new_states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, auto_reset=False, return_info=True))(states, env_acts, keys)
            
            cur_a0 = info["alive"][:, 0]
            cur_a1 = info["alive"][:, 1]
            
            is_b0 = (env_acts[:, 0, 1] == 1) & ~ended
            is_b1 = (env_acts[:, 1, 1] == 1) & ~ended
            
            just_ended = (~cur_a0 | ~cur_a1) & ~ended
            new_ended = ended | just_ended
            new_end_t = jnp.where(just_ended, tick_idx + 1, end_t)
            
            new_a0_alive = jnp.where(just_ended, cur_a0, a0_alive)
            new_a1_alive = jnp.where(just_ended, cur_a1, a1_alive)
            
            return (new_states, key, new_a0_alive, new_a1_alive, new_ended, new_end_t,
                    b0_cnt + is_b0.astype(jnp.int32), b1_cnt + is_b1.astype(jnp.int32)), None
        
        ticks = jnp.arange(max_ticks)
        (final_s, _, final_a0, final_a1, final_ended, final_end_t, final_b0, final_b1), _ = jax.lax.scan(
            step_fn, carry_init, ticks
        )
        
        win = final_a0 & ~final_a1
        loss = ~final_a0 & final_a1
        both = ~final_a0 & ~final_a1
        timeout = final_a0 & final_a1
        
        win_pct = float(jnp.mean(win.astype(jnp.float32))) * 100
        loss_pct = float(jnp.mean(loss.astype(jnp.float32))) * 100
        both_pct = float(jnp.mean(both.astype(jnp.float32))) * 100
        timeout_pct = float(jnp.mean(timeout.astype(jnp.float32))) * 100
        avg_b0 = float(jnp.mean(final_b0.astype(jnp.float32)))
        avg_surv = float(jnp.mean(final_end_t.astype(jnp.float32)))
        
        print(f"\n🗺️  【{map_name}】", flush=True)
        print(f"   🏆 击败对手胜率: {win_pct:5.1f}% ({int(jnp.sum(win))}/{n_games})", flush=True)
        print(f"   💀 败局/自杀率  : {loss_pct:5.1f}% ({int(jnp.sum(loss))}/{n_games})", flush=True)
        print(f"   💥 同归于尽率  : {both_pct:5.1f}% ({int(jnp.sum(both))}/{n_games})", flush=True)
        print(f"   ⏱️ 超时保命平局: {timeout_pct:5.1f}% ({int(jnp.sum(timeout))}/{n_games})", flush=True)
        print(f"   💣 模型放泡量  : {avg_b0:5.1f} 颗/局", flush=True)
        print(f"   ⏳ 平均存活时间: {avg_surv:5.1f} ticks", flush=True)

if __name__ == "__main__":
    it = sys.argv[1] if len(sys.argv) > 1 else "51"
    opp = sys.argv[2] if len(sys.argv) > 2 else "idle"
    greedy_flag = "--greedy" in sys.argv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(base_dir, f"ckpt_local/params_it{int(it):08d}.pkl")
    levels_file = os.path.join(base_dir, "levels.json")
    run_tournament(ckpt, opponent_type=opp, n_games=128, max_ticks=800, greedy=greedy_flag)
