#!/usr/bin/env python3
"""
针对 Stage 5 (5Hz) 检查点的精确对战实测脚本：
1. AI (5Hz) vs 10Hz 逃跑风筝怪 (FleeBot): 241 全图评测
2. AI (5Hz) vs 静止木桩 (IDLE): 检验死角自杀率
3. AI (5Hz) vs 10Hz it831 基模: 新老模型直接对决
"""
import sys
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np

from jax_bomb import levels
from jax_bomb.jax_env import step, _fresh
from jax_bomb.jax_train import (
    sample_actions, both_perspectives, both_masks, both_states, flee_bot_actions
)

def run_eval(ckpt_eval_path, ckpt_base_path=None, num_envs=128, max_macro_steps=500):
    levels.set_active("levels.json")
    
    print(f"Loading evaluated model: {ckpt_eval_path}")
    with open(ckpt_eval_path, "rb") as f:
        ck_eval = pickle.load(f)
    params_eval = ck_eval.get("params", ck_eval)
    
    # -------------------------------------------------------------
    # 1. AI (5Hz) vs 10Hz FleeBot (逃跑风筝怪)
    # -------------------------------------------------------------
    print(f"\n==================================================")
    print(f"--- 1. AI (5Hz) vs 10Hz 逃跑拉扯 Bot (FleeBot) ---")
    print(f"环境数: {num_envs} | 单局上限: {max_macro_steps * 2} 物理 ticks (100秒)")
    
    key = jax.random.PRNGKey(2026)
    k_init, key = jax.random.split(key)
    states = jax.vmap(_fresh)(jax.random.split(k_init, num_envs))
    
    @jax.jit
    def eval_vs_fleebot(states, key):
        def macro_step(carry, _):
            states, key = carry
            key, k_ai, k_flee1, k_flee2, k_step1, k_step2 = jax.random.split(key, 6)
            
            # --- Tick 1 (AI 5Hz 决策 + FleeBot 10Hz 决策) ---
            obs = both_perspectives(states)
            masks = both_masks(states)
            gv = both_states(states)
            
            # P0 (AI) 5Hz 决策
            a0, _, _ = sample_actions(
                params_eval, "transformer", obs[:num_envs],
                (masks[0][:num_envs], masks[1][:num_envs]), k_ai,
                state=gv[:num_envs]
            )
            # P1 (FleeBot) 10Hz 避险
            a1_t1 = flee_bot_actions(
                states.pos[:, 1], states.pos[:, 0],
                masks[0][num_envs:], masks[1][num_envs:], k_flee1
            )
            
            env_acts1 = jnp.stack([a0, a1_t1], axis=1)
            keys1 = jax.random.split(k_step1, num_envs)
            st1, done1, info1 = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts1, keys1)
            
            # --- Tick 2 (AI 保持移动惯性不放雷 + FleeBot 100ms 敏捷刷新避险) ---
            a0_t2 = jnp.stack([a0[:, 0], jnp.zeros_like(a0[:, 1])], axis=-1)
            mm2, bm2 = both_masks(st1)
            a1_t2 = flee_bot_actions(
                st1.pos[:, 1], st1.pos[:, 0],
                mm2[num_envs:], bm2[num_envs:], k_flee2
            )
            
            env_acts2 = jnp.stack([a0_t2, a1_t2], axis=1)
            keys2 = jax.random.split(k_step2, num_envs)
            st2, done2, info2 = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(st1, env_acts2, keys2)
            
            # 统计结算 (两个 tick 任一 tick 触发胜负均结算)
            p0_win = (done1 & info1["alive"][:, 0] & (~info1["alive"][:, 1])) | \
                     (done2 & info2["alive"][:, 0] & (~info2["alive"][:, 1]))
            p1_win = (done1 & info1["alive"][:, 1] & (~info1["alive"][:, 0])) | \
                     (done2 & info2["alive"][:, 1] & (~info2["alive"][:, 0]))
            both_die = (done1 & (~info1["alive"][:, 0]) & (~info1["alive"][:, 1])) | \
                       (done2 & (~info2["alive"][:, 0]) & (~info2["alive"][:, 1]))
            dones = done1 | done2
            
            return (st2, key), (p0_win, p1_win, both_die, dones)
        
        _, (w0, w1, bd, dones) = jax.lax.scan(macro_step, (states, key), None, length=max_macro_steps)
        return w0.sum(), w1.sum(), bd.sum(), dones.sum()
    
    w0, w1, bd, tot = eval_vs_fleebot(states, key)
    tot = max(1, int(tot))
    w0, w1, bd = int(w0), int(w1), int(bd)
    print(f"  对局结束总数: {tot}")
    print(f"  AI 击杀胜局: {w0} ({w0/tot*100:.1f}%)")
    print(f"  FleeBot 战胜: {w1} ({w1/tot*100:.1f}%)")
    print(f"  同归于尽/平局: {bd} ({bd/tot*100:.1f}%)")
    print(f"  AI 不败率 (胜+平): {(w0+bd)/tot*100:.1f}%")
    
    # -------------------------------------------------------------
    # 2. AI (5Hz) vs 静止木桩 (自杀率检验)
    # -------------------------------------------------------------
    print(f"\n--- 2. AI (5Hz) vs 静止木桩 (检验防自杀掩码) ---")
    k_init2, key = jax.random.split(key)
    states2 = jax.vmap(_fresh)(jax.random.split(k_init2, num_envs))
    
    @jax.jit
    def eval_vs_idle(states, key):
        def macro_step(carry, _):
            states, key = carry
            key, k_ai, k_step1, k_step2 = jax.random.split(key, 4)
            obs = both_perspectives(states)
            masks = both_masks(states)
            gv = both_states(states)
            a0, _, _ = sample_actions(
                params_eval, "transformer", obs[:num_envs],
                (masks[0][:num_envs], masks[1][:num_envs]), k_ai,
                state=gv[:num_envs]
            )
            a1_idle = jnp.zeros((num_envs, 2), dtype=jnp.int32).at[:, 0].set(4)
            env_acts1 = jnp.stack([a0, a1_idle], axis=1)
            keys1 = jax.random.split(k_step1, num_envs)
            st1, done1, info1 = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts1, keys1)
            
            a0_t2 = jnp.stack([a0[:, 0], jnp.zeros_like(a0[:, 1])], axis=-1)
            env_acts2 = jnp.stack([a0_t2, a1_idle], axis=1)
            keys2 = jax.random.split(k_step2, num_envs)
            st2, done2, info2 = jax.vmap(
                lambda s, a, kk: step(s, a, kk, return_info=True))(st1, env_acts2, keys2)
            
            p0_win = (done1 & info1["alive"][:, 0] & (~info1["alive"][:, 1])) | \
                     (done2 & info2["alive"][:, 0] & (~info2["alive"][:, 1]))
            p0_suicide = (done1 & (~info1["alive"][:, 0])) | (done2 & (~info2["alive"][:, 0]))
            dones = done1 | done2
            return (st2, key), (p0_win, p0_suicide, dones)
        
        _, (wins, suicides, dones) = jax.lax.scan(macro_step, (states, key), None, length=max_macro_steps)
        return wins.sum(), suicides.sum(), dones.sum()
    
    w_i, s_i, tot_i = eval_vs_idle(states2, key)
    tot_i = max(1, int(tot_i))
    w_i, s_i = int(w_i), int(s_i)
    print(f"  对局结束总数: {tot_i}")
    print(f"  AI 击杀获胜: {w_i} ({w_i/tot_i*100:.1f}%)")
    print(f"  AI 自灭/阵亡率: {s_i} ({s_i/tot_i*100:.1f}%)")
    print(f"==================================================\n")

if __name__ == "__main__":
    eval_p = sys.argv[1]
    base_p = sys.argv[2] if len(sys.argv) > 2 else None
    run_eval(eval_p, base_p)
