"""深入诊断 Checkpoint: 策略熵 (Entropy) + 真实死因解构 (自杀率 vs 被杀率 vs 击杀率)。"""
import os, sys, pickle
import jax, jax.numpy as jnp
import numpy as np

from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions, net_forward

def diagnose(ckpt_path, n_games=256, max_ticks=600):
    with open(ckpt_path, "rb") as f:
        params = jax.tree.map(jnp.asarray, pickle.load(f))
    arch = "transformer"
    
    # 1. 测量空旷场景初始观测下的策略熵 (Entropy)
    key = jax.random.PRNGKey(123)
    k_init, k_run = jax.random.split(key)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _levels.set_active(os.path.join(base_dir, "levels.json"), weights="empty=1.0")
    states = init_batch(k_init, n_games)
    
    obs = both_perspectives(states)
    masks = both_masks(states)
    st = both_states(states)
    
    mv, bm, v, _ = net_forward(params, arch, obs[:n_games], st[:n_games])
    m0, b0 = masks[0][:n_games], masks[1][:n_games]
    
    mv_m = jnp.where(m0, mv, -1e9)
    bm_m = jnp.where(b0, bm, -1e9)
    
    pm = jax.nn.softmax(mv_m, axis=-1)
    pb = jax.nn.softmax(bm_m, axis=-1)
    lsm = jax.nn.log_softmax(mv_m, axis=-1)
    lsb = jax.nn.log_softmax(bm_m, axis=-1)
    
    ent_m = float(-jnp.mean(jnp.sum(jnp.where(pm > 0, pm * lsm, 0.0), axis=-1)))
    ent_b = float(-jnp.mean(jnp.sum(jnp.where(pb > 0, pb * lsb, 0.0), axis=-1)))
    total_ent = ent_m + ent_b
    max_ent = np.log(5) + np.log(2) # 1.609 + 0.693 = 2.302
    
    bomb_prob = float(jnp.mean(pb[:, 1])) # 放泡动作 (action=1) 的初始概率
    
    # 2. 仿真 256 局空旷场景对战 (vs 静止木桩)
    p0_alive = states.alive[:, 0]
    p1_alive = states.alive[:, 1]
    
    p0_suicide = jnp.zeros(n_games, dtype=jnp.bool_)
    p0_win_kill = jnp.zeros(n_games, dtype=jnp.bool_)
    finished = jnp.zeros(n_games, dtype=jnp.bool_)
    b0_cnt = jnp.zeros(n_games, dtype=jnp.int32)
    
    for step_i in range(max_ticks):
        k_run, k_step = jax.random.split(k_run)
        obs_curr = both_perspectives(states)
        masks_curr = both_masks(states)
        st_curr = both_states(states)
        
        # P0 策略采样
        acts0, _, _ = sample_actions(params, arch, obs_curr[:n_games],
                                    (masks_curr[0][:n_games], masks_curr[1][:n_games]),
                                    k_step, st_curr[:n_games])
        
        # P1 静止木桩 (不动、不放泡)
        acts1 = jnp.zeros((n_games, 2), dtype=jnp.int32)
        acts = jnp.stack([acts0, acts1], axis=1)
        
        b0_cnt += acts0[:, 1]
        
        prev_p0 = states.alive[:, 0]
        prev_p1 = states.alive[:, 1]
        k_step1, k_step2 = jax.random.split(k_step)
        step_keys = jax.random.split(k_step1, n_games)
        states, _ = jax.vmap(step)(states, acts, step_keys)
        curr_p0 = states.alive[:, 0]
        curr_p1 = states.alive[:, 1]
        
        p0_died_now = prev_p0 & ~curr_p0
        p1_died_now = prev_p1 & ~curr_p1
        
        # 因为静止木桩从不放炸弹，所以只要 p0 死亡，100% 是死于自己放的炸弹（自杀）！
        p0_suicide = jnp.where(~finished & p0_died_now, True, p0_suicide)
        p0_win_kill = jnp.where(~finished & p1_died_now & ~p0_died_now, True, p0_win_kill)
        
        finished = finished | p0_died_now | p1_died_now
        if bool(jnp.all(finished)):
            break

    suicide_rate = float(jnp.mean(p0_suicide.astype(jnp.float32))) * 100
    win_rate = float(jnp.mean(p0_win_kill.astype(jnp.float32))) * 100
    timeout_rate = 100.0 - suicide_rate - win_rate
    avg_bombs = float(jnp.mean(b0_cnt.astype(jnp.float32)))

    print(f"\n==================== 🩺 【{os.path.basename(ckpt_path)}】深度诊断 ====================")
    print(f"📊 策略熵 (Entropy):")
    print(f"   - 移动熵 (Move):   {ent_m:.4f} / {np.log(5):.4f} (理论满值 1.6094)")
    print(f"   - 放泡熵 (Bomb):   {ent_b:.4f} / {np.log(2):.4f} (理论满值 0.6931)")
    print(f"   - 总策略熵:        {total_ent:.4f} / {max_ent:.4f} (占比: {total_ent/max_ent*100:.1f}%)")
    print(f"   - 开局放泡概率:    {bomb_prob*100:.1f}%\n")
    print(f"🎯 {n_games} 局空旷场景实测 (vs 静止木桩):")
    print(f"   💀 真实自杀率:     {suicide_rate:5.1f}% ({int(jnp.sum(p0_suicide))}/{n_games}) -> 纯自己放泡炸死自己")
    print(f"   🏆 击杀静止靶:     {win_rate:5.1f}% ({int(jnp.sum(p0_win_kill))}/{n_games}) -> 成功炸死对手")
    print(f"   ⏱️ 超时苟活局:     {timeout_rate:5.1f}%")
    print(f"   💣 平均放泡量:     {avg_bombs:5.1f} 颗/局")
    print("=========================================================================\n")

if __name__ == "__main__":
    diagnose(sys.argv[1])
