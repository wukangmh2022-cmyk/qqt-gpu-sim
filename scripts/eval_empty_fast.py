"""超快速空旷道场盲测 (纯针对空场景 128 局，秒级输出)。"""
import os, sys, pickle
import jax, jax.numpy as jnp
import numpy as np

from jax_bomb.jax_env import (
    BombState, legal_mask, make_obs, step, init_env, H, W, N_BOMB, N_MOVES
)
from jax_bomb.jax_train import sample_actions, both_states

def eval_ckpt(ckpt_path, n_games=128, max_ticks=600):
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    arch = data.get("arch", "transformer")
    
    key = jax.random.PRNGKey(42)
    k_init, k_run = jax.random.split(key)
    
    # 纯空旷地图
    init_fn = jax.vmap(lambda k: init_env(k, 0, 1.0))
    states = init_fn(jax.random.split(k_init, n_games))
    
    b0_cnt = jnp.zeros(n_games, dtype=jnp.int32)
    b1_cnt = jnp.zeros(n_games, dtype=jnp.int32)
    finished = jnp.zeros(n_games, dtype=jnp.bool_)
    win = jnp.zeros(n_games, dtype=jnp.bool_)
    loss = jnp.zeros(n_games, dtype=jnp.bool_)
    both = jnp.zeros(n_games, dtype=jnp.bool_)
    
    for t in range(max_ticks):
        k_run, k_step = jax.random.split(k_run)
        m, b = jax.vmap(legal_mask)(states)
        m0, b0 = m[:, 0], b[:, 0]
        
        # p0: 模型策略
        st0 = jax.vmap(lambda s: jax_train_global_vec(s, 0))(states)
        danger = jax.vmap(lambda s: jax_env_danger(s.fuse, s.wall, s.bomb_blast, s.brick))(states)
        obs0 = jax.vmap(lambda s, d: make_obs(s, 0, d))(states, danger)
        act0, _, _ = sample_actions(params, arch, obs0, (m0, b0), k_step, st0)
        
        # p1: 静止对手 (不动，不放泡)
        act1 = jnp.zeros((n_games, 2), dtype=jnp.int32)
        
        acts = jnp.stack([act0, act1], axis=1) # (N, 2, 2)
        
        # 统计放泡
        b0_cnt += act0[:, 1]
        
        alive_prev = states.alive
        states = jax.vmap(step)(states, acts)
        alive_curr = states.alive
        
        # 判定新死亡
        newly_dead = alive_prev & ~alive_curr
        p0_died = newly_dead[:, 0]
        p1_died = newly_dead[:, 1]
        
        p0_win_now = ~finished & p1_died & ~p0_died
        p0_loss_now = ~finished & p0_died & ~p1_died
        both_now = ~finished & p0_died & p1_died
        
        win = jnp.where(p0_win_now, True, win)
        loss = jnp.where(p0_loss_now, True, loss)
        both = jnp.where(both_now, True, both)
        
        finished = finished | p0_died | p1_died
        if bool(jnp.all(finished)):
            break

    win_p = float(jnp.mean(win.astype(jnp.float32))) * 100
    loss_p = float(jnp.mean(loss.astype(jnp.float32))) * 100
    both_p = float(jnp.mean(both.astype(jnp.float32))) * 100
    timeout_p = 100.0 - win_p - loss_p - both_p
    avg_b = float(jnp.mean(b0_cnt.astype(jnp.float32)))
    
    print(f"\n==================== 【{os.path.basename(ckpt_path)}】128 局空场景实测 ====================")
    print(f"  🏆 击杀对手胜率 : {win_p:5.1f}% ({int(jnp.sum(win))}/{n_games})")
    print(f"  💀 败局/自杀率   : {loss_p:5.1f}% ({int(jnp.sum(loss))}/{n_games})")
    print(f"  💥 同归于尽率   : {both_p:5.1f}% ({int(jnp.sum(both))}/{n_games})")
    print(f"  ⏱️ 超时未分胜负 : {timeout_p:5.1f}%")
    print(f"  💣 平均放泡量   : {avg_b:5.1f} 颗/局")
    print("========================================================================\n")

from jax_bomb.jax_env import global_vec as jax_train_global_vec, _danger_map as jax_env_danger

if __name__ == "__main__":
    eval_ckpt(sys.argv[1])
