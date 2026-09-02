"""验证两大核心假设:
1. 初始化 Step 0 网络的放泡先验概率 (Baseline Bias)
2. 放泡后 1~3 步内模型是否具有远离炸弹的时序耦合 (Post-bomb Evasion Correlation)
"""
import os, sys, pickle
import jax, jax.numpy as jnp
import numpy as np

from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step, legal_mask, make_obs, _danger_map, global_vec
from jax_bomb.jax_net import init_net, net_forward
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions

def check_step0_prior():
    key = jax.random.PRNGKey(42)
    k_init, k_run = jax.random.split(key)
    # 初始化一个全新随机网络 (Step 0)
    p0 = init_net(k_init, "transformer", c=14, h=13, w=15, embed=392, depth=4, patch=4, heads=4, ff_factor=4)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _levels.set_active(os.path.join(base_dir, "levels.json"), weights="empty=1.0")
    states = init_batch(k_run, 512)
    
    obs = both_perspectives(states)[:512]
    masks = both_masks(states)
    st = both_states(states)[:512]
    m0, b0 = masks[0][:512], masks[1][:512]
    
    mv, bm, v, _ = net_forward(p0, "transformer", obs, st)
    mv_m = jnp.where(m0, mv, -1e9)
    bm_m = jnp.where(b0, bm, -1e9)
    
    pb = np.array(jax.nn.softmax(bm_m, axis=-1))
    avg_bomb_p = np.mean(pb[:, 1])
    print(f"==================== 🔬 假设 1: 初始未训练网络 (Step 0) 先验 ====================", flush=True)
    print(f"  💣 未训练网络开局放泡动作概率: {avg_bomb_p*100:.2f}% (理论均匀为 50.00%)", flush=True)
    print(f"===============================================================================\n", flush=True)

def check_post_bomb_evasion(ckpt_path, n_games=32, max_ticks=200):
    with open(ckpt_path, "rb") as f:
        params = jax.tree.map(jnp.asarray, pickle.load(f))
    arch = "transformer"
    
    key = jax.random.PRNGKey(999)
    k_init, k_run = jax.random.split(key)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _levels.set_active(os.path.join(base_dir, "levels.json"), weights="empty=1.0")
    states = init_batch(k_init, n_games)
    
    # 统计放泡后下一拍的行为: 移动 vs 停留在炸弹上
    stayed_on_bomb_count = 0
    moved_away_count = 0
    total_bomb_events = 0
    
    for t in range(max_ticks):
        k_run, k_step = jax.random.split(k_run)
        obs_curr = both_perspectives(states)
        masks_curr = both_masks(states)
        st_curr = both_states(states)
        
        acts0, _, _ = sample_actions(params, arch, obs_curr[:n_games],
                                    (masks_curr[0][:n_games], masks_curr[1][:n_games]),
                                    k_step, st_curr[:n_games])
        acts1 = jnp.zeros((n_games, 2), dtype=jnp.int32)
        acts = jnp.stack([acts0, acts1], axis=1)
        
        pos_before = np.array(states.pos[:, 0]) # (N, 2)
        placed_bomb = np.array(acts0[:, 1] == 1) # 本拍放了炸弹的 env 掩码
        
        k_s1, k_s2 = jax.random.split(k_step)
        step_keys = jax.random.split(k_s1, n_games)
        states, _ = jax.vmap(step)(states, acts, step_keys)
        pos_after = np.array(states.pos[:, 0]) # (N, 2)
        
        # 对于放了炸弹的事件，检查位移
        dist = np.linalg.norm(pos_after - pos_before, axis=1)
        for i in range(n_games):
            if placed_bomb[i] and states.alive[i, 0]:
                total_bomb_events += 1
                if dist[i] < 0.1: # 几乎没有位移 (停在原地 / 撞墙)
                    stayed_on_bomb_count += 1
                else: # 发生了有效移动
                    moved_away_count += 1
                    
        if bool(jnp.all(~states.alive[:, 0])):
            break
            
    print(f"==================== 🏃 假设 2: 放泡后时序逃跑耦合度分析 ({os.path.basename(ckpt_path)}) ====================")
    print(f"  💣 观测到的放泡总事件数: {total_bomb_events} 次")
    if total_bomb_events > 0:
        stay_p = stayed_on_bomb_count / total_bomb_events * 100
        move_p = moved_away_count / total_bomb_events * 100
        print(f"  🛑 放泡后第 1 拍【留在原地/撞墙】: {stay_p:.1f}% ({stayed_on_bomb_count}/{total_bomb_events})")
        print(f"  🏃 放泡后第 1 拍【发生位移逃跑】: {move_p:.1f}% ({moved_away_count}/{total_bomb_events})")
    print(f"====================================================================================================\n")

if __name__ == "__main__":
    check_step0_prior()
    if len(sys.argv) > 1:
        check_post_bomb_evasion(sys.argv[1])
