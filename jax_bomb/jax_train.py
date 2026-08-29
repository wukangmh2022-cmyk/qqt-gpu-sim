"""Self-play PPO for the naive JAX bomberman + end-to-end SPS bench.

Structure mirrors Average Joe (rollout via lax.scan, 2N batch, scan-based
minibatch PPO) but single-device and dependency-light (optax only).
--distill-data 提供离线蒸馏阶段：teacher（torch collect_distill 收集的
obs7/logits/masks）KL 蒸馏，可接 --distill-then-ppo 继续自对弈 PPO。
"""

import argparse
import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from .jax_env import (H, W, MAX_HP, MAX_STEPS, N_BOMB, N_MOVES, N_OBS_CH,
                      _danger_map, global_vec, init_batch, legal_mask,
                      make_obs, step)
from .jax_net import (BIN_CENTERS, NUM_VALUE_BINS, V_MAX, V_MIN,
                      count_params, init_net, net_forward)
from .platform import device_summary, setup_platform

# ---------------- 极简归一化零和生命演进奖励（Bitter Lesson 范式） ----------------
# 彻底废除人工稠密惩罚（danger_penalty / step_penalty / timeout_diff）。
# 严格零和：每损失 1 滴血转移 1/MAX_HP = 0.2 胜负份额，打满 5 滴血击杀累计 +1.0 / -1.0。
# 严格保证 sum(r_t) == 0，天然无偏，杜绝刷分。


def novelty_transition(visited, cells, done):
    """Return first-visit events and the next shared per-episode visit map."""
    was_visited = jax.vmap(
        lambda v, rc: v[rc[:, 0], rc[:, 1]])(visited, cells)
    newly = ~was_visited
    same_cell = ((cells[:, 0, 0] == cells[:, 1, 0])
                 & (cells[:, 0, 1] == cells[:, 1, 1]))
    newly = newly.at[:, 1].set(newly[:, 1] & ~same_cell)
    next_visited = jax.vmap(
        lambda v, rc: v.at[rc[:, 0], rc[:, 1]].set(True))(visited, cells)
    next_visited = jnp.where(done[:, None, None],
                             jnp.zeros_like(next_visited), next_visited)
    return newly, next_visited


def reward_from_events(dmg, alive_before, alive_after, hp_after, done,
                       crate_grew, newly, walls_destroyed, crate_coef=0.0,
                       explore_coef=0.0, brick_coef=0.0, timeout_alpha=1.0):
    """Compute normalized zero-sum life progression reward: (dealt - taken) / MAX_HP."""
    dmg = dmg.astype(jnp.float32)
    dealt = dmg.sum(axis=-1, keepdims=True) - dmg
    rew = (dealt - dmg) / float(MAX_HP)
    # Legacy shaping hooks (0.0 by default in zero-shaping mode)
    rew = rew + crate_coef * crate_grew.astype(jnp.float32)
    rew = rew + explore_coef * newly.astype(jnp.float32)
    rew = rew + brick_coef * walls_destroyed.astype(jnp.float32)[:, None] / 2.0
    return rew


def hl_gauss_value_loss(v_logits, targets, v_min=V_MIN, v_max=V_MAX,
                        num_bins=NUM_VALUE_BINS, sigma=0.04):
    """HL-Gauss categorical cross-entropy loss over [-1.0, 1.0]."""
    bin_centers = jnp.linspace(v_min, v_max, num_bins)
    half_width = (v_max - v_min) / (num_bins - 1) / 2.0
    upper = (bin_centers + half_width - targets[:, None]) / sigma
    lower = (bin_centers - half_width - targets[:, None]) / sigma
    target_probs = jax.scipy.stats.norm.cdf(upper) - jax.scipy.stats.norm.cdf(lower)
    target_probs = target_probs / jnp.maximum(jnp.sum(target_probs, axis=-1, keepdims=True), 1e-8)
    log_probs = jax.nn.log_softmax(v_logits, axis=-1)
    return -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))


# ---------------- policy head ----------------


def sample_actions(params, arch, obs, masks, key, state=None):
    """obs (N,C,H,W)，masks = (move_mask (N,5), bomb_mask (N,2)) bool。

    返回 (act (N,2), logp (N,), val (N,))。非法动作 logits 置 -inf 后采样
    （与 torch masked_dist 同语义：softmax 概率归零）。state (N,G) 为全局
    状态向量（transformer 的 state token），None=无（旧路径/对拍）。"""
    mv, bm, v_scalar, _ = net_forward(params, arch, obs, state)
    mm, bmb = masks
    mv_m = jnp.where(mm, mv, jnp.full_like(mv, -jnp.inf))
    bm_m = jnp.where(bmb, bm, jnp.full_like(bm, -jnp.inf))
    k1, k2 = jrandom.split(key)
    a_m = jrandom.categorical(k1, mv_m)
    a_b = jrandom.categorical(k2, bm_m)
    n = obs.shape[0]
    lp = (jax.nn.log_softmax(mv_m)[jnp.arange(n), a_m]
          + jax.nn.log_softmax(bm_m)[jnp.arange(n), a_b])
    return jnp.stack([a_m, a_b], axis=-1), lp, v_scalar


def both_perspectives(states):
    """返回 (2N, C, H, W)：p0 视角 + p1 视角拼接。

    危险图与视角无关 → 两个视角共享同一份（make_obs 传预计算 danger），
    每 tick 的 danger_map 计算减半（collect_rollout 热点，实测 2 遍版本
    每 iter 4.76s → 共享后 ≈2.4s）。
    """
    danger = jax.vmap(lambda s: _danger_map(s.fuse, s.wall, s.bomb_blast,
                                            s.brick))(states)
    obs0 = jax.vmap(lambda s, d: make_obs(s, 0, d))(states, danger)
    obs1 = jax.vmap(lambda s, d: make_obs(s, 1, d))(states, danger)
    return jnp.concatenate([obs0, obs1], axis=0)


def both_masks(states):
    """返回 ((2N,5), (2N,2))：与 both_perspectives 的 obs 行序对齐。

    legal_mask 的 vmap 结果是 (N,2,5)/(N,2,2)（每 state 两玩家的 mask）；
    obs 行序 = [N 个 p0 视角, N 个 p1 视角]，所以 p0 视角帧配玩家 0 的
    mask、p1 视角帧配玩家 1 的 —— 按玩家拆开拼接，不是整块 concat。
    """
    m, b = jax.vmap(legal_mask)(states)          # (N,2,5), (N,2,2)
    m0, m1 = m[:, 0], m[:, 1]
    b0, b1 = b[:, 0], b[:, 1]
    return (jnp.concatenate([m0, m1]), jnp.concatenate([b0, b1]))


def both_states(states):
    """返回 (2N, G) 全局状态向量，行序与 both_perspectives 对齐（p0 视角配
    玩家 0 的全局量、p1 配玩家 1 的）。血量/成长属性/存活/进度 —— 论文式
    双序列输入的第二路（transformer 的 state token）。"""
    v0 = jax.vmap(lambda s: global_vec(s, 0))(states)
    v1 = jax.vmap(lambda s: global_vec(s, 1))(states)
    return jnp.concatenate([v0, v1], axis=0)


# ---------------- rollout ----------------


def collect_rollout(params, arch, states, key, num_steps, no_mask=False,
                    obs_quant=False, checkpoint=False, crate_coef=0.0,
                    explore_coef=0.0, brick_coef=0.0, timeout_alpha=1.0):
    """自对弈：同一网络打两边。states (N, ...)。返回 (new_states, batch, nov, kills)。

    nov：每 env/玩家的 novelty 计数（未加权，与 batch.rew 同口径窗口累计）。
    训练侧除以 num_steps × coef 即得"探索分/帧"，与 rew 均值直接对比——
    探索分单局天然封顶 coef×可达格数（195 格地图 ≈ coef×195），coef=0.01 时
    全图逛完 1.95 分，远低于单次伤害 1.5×N 与击杀 10，不会压过胜负信号。

    step 在终局后**就地重置**（对齐正式版 auto_reset），所以胜负判定用
    step 前的 alive 快照（重置后 alive 恒全 True，无法区分谁死）。
    batch 含每 tick 的 (move_mask, bomb_mask)——PPO loss 用它屏蔽非法动作。
    no_mask=True：mask 全放开（性能 A/B 用，行为=无 mask 旧版）。
    obs_quant=True：obs buffer 存 uint8（×255 量化，PPO 反量化后进网络）。
    obs 8 通道都是低精度值（二值/离散/0-1），uint8 精度 1/255 ≈ 0.4% 优于
    bf16 尾数，buffer 从 fp32 45GB 降到 11GB（8192×512 OOM 的解法）。
    checkpoint=True：scan body 用 jax.checkpoint 包裹——反向重算中间量，
    不保留每 tick 的 obs/激活（8192×512 下 scan 中间量 44.8GB 是量化后
    剩余瓶颈，checkpoint 可进一步降到 buffer 本身大小）。
    crate_coef：开箱成长奖励系数（0=关）。关卡模式多数地图出生点被砖隔开，
    前期无交战通道，正信号只有破砖吃箱——bootstrap 奖励（长退火）加速
    前期学习；退火后只剩真胜负（参考实现 stage0 composite_reward 同款思路）。
    explore_coef：探索 novelty 奖励系数（0=关）。每 tick 玩家中心格若是
    **本局首次到达**（共享 visited 掩码，done 清零）→ +explore_coef。这是
    "整局只走几格就重罚"的稠密版：走过的格不再给分，坐桩/困在出生点几乎
    零探索分，破砖开路才拿分。与 crate 同款长退火。掩码在 scan carry 里，
    不进 BombState/ckpt（断点接续零兼容问题）。传入 jnp 标量（随 iter 变化
    不触发重编译）。
    brick_coef：炸墙奖励系数（0=关）。每炸毁一块砖（含灌木）双方各
    +brick_coef/2。治"出生点 3 格死锁"：crate 奖励的链路（炸→掷爆率→
    吃到）太长太弱学不会，给"炸墙"本身即时正反馈，破墙开路才有后续探索/
    吃箱/交手。同上乘统一退火（headless 实测 500 iter 模型在隔离图仍
    放炮少，炸墙是冷启动关键）。
    timeout_alpha：双方存活超时的血差 shaping 退火系数。最大血差 4 时才
    +2/-2，远低于固定死亡击杀 +10/-10；设为 0 后超时不再给血差奖励。
    返回 (final_states, batch, nov, kills)：nov 每 env/玩家 novelty 累计；
    kills 每 env 窗口内击杀局数（death_done 累计）——动态退火 α=1-tanh(k·x)
    的 x 来源（每局击杀率 = mean(kills)/n_episodes）。
    """
    n = states.pos.shape[0]
    ones_m = jnp.ones((2 * n, N_MOVES), jnp.bool_)
    ones_b = jnp.ones((2 * n, N_BOMB), jnp.bool_)
    visited0 = jnp.zeros((n, H, W), jnp.bool_)      # 探索掩码（scan carry）
    nov0 = jnp.zeros((n, 2), jnp.float32)           # 每 env/玩家 未加权 novelty 累计
    kills0 = jnp.zeros((n,), jnp.float32)           # 每 env 击杀局数（动态退火 x 来源）

    def one_step(carry, _):
        states, key, visited, nov, kills = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)               # (2N, C, H, W)
        masks = (ones_m, ones_b) if no_mask else both_masks(states)
        gv = both_states(states)                      # (2N, G) 全局状态向量
        acts, lps, vals = sample_actions(params, arch, obs, masks, key,
                                         state=gv)
        a0, a1 = acts[:n], acts[n:]
        env_acts = jnp.stack([a0, a1], axis=1)        # (N, 2, 2)
        keys = jrandom.split(kstep, n)                # 每 env 一步的 RNG（地图/宝箱）
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts,
                                                               keys)
        # Use the physical post-step cells captured before auto-reset. On a
        # terminal tick `new_states` is already the next episode's spawn map.
        newly, new_visited = novelty_transition(visited, info["cell"], done)
        # 稠密奖励（对齐 torch step 的 hit/step/win 段）：
        #   - 掉 1 血 -HIT_REWARD / 造成 1 伤害 +HIT_REWARD（info.dmg 结算后快照，
        #     auto_reset 前取值 —— 1v1 里对方掉血 = 我的泡干的）；
        #   - 每 tick -STEP_PENALTY（防磨洋工；**用 step 前 alive0**，对齐 torch
        #     死亡 tick 死者也扣步罚 —— info.alive 是结算后，死者已 False）；
        #   - 终局：死亡（n_alive==1）击杀方 ±WIN_BONUS 固定值；超时全员存活
        #     （n_alive==2）按血差 × TIMEOUT_PER_HP × timeout_alpha（退火）。
        #     info.alive/hp 是结算后值，不受 auto_reset 重置污染。
        rew = reward_from_events(
            info["dmg"], states.alive, info["alive"], info["hp"], done,
            info["crate"], newly, info["walls"], crate_coef, explore_coef,
            brick_coef, timeout_alpha)
        nov = nov + newly.astype(jnp.float32)       # 统计用：探索分/帧可监控
        n_alive = info["alive"].sum(axis=-1)          # (N,)
        death_done = done & (n_alive == 1)
        kills = kills + death_done.astype(jnp.float32)   # 击杀局数（动态退火 x）
        d = jnp.concatenate([done, done])
        rew = jnp.concatenate([rew[:, 0], rew[:, 1]])
        obs_s = (jnp.round(obs * 255.0).astype(jnp.uint8)
                 if obs_quant else obs)
        state_s = (jnp.round(gv * 255.0).astype(jnp.uint8)
                   if obs_quant else gv)
        data = (obs_s, state_s, acts, lps, vals, rew, d, masks)
        return (new_states, key, new_visited, nov, kills), data
    body = (jax.checkpoint(one_step) if checkpoint else one_step)
    (final_states, _, _, nov, kills), data = jax.lax.scan(
        body, (states, key, visited0, nov0, kills0), None, length=num_steps)
    obs, state, acts, lps, vals, rew, done, masks = data
    return final_states, (obs, state, acts, lps, vals, rew, done, masks), nov, kills


def collect_rollout_two(params_a, params_b, arch, states, key, num_steps,
                        no_mask=False, obs_quant=False):
    """两策略自对弈 rollout（评估用）：p0 视角用 params_a、p1 用 params_b。

    与 collect_rollout 同一环境语义（auto_reset/稠密奖励），但**不存 obs
    buffer**（评估不训练），只返回胜率计数：
      win_stats = (p0_wins, p0_losses) —— 终局击杀（死亡 tick 存活者胜）与
    超时（血高者胜）各计一局；episode 就地重置，每个终局 tick 计一次。
    p0 胜率 = p0_wins / (p0_wins + p0_losses)。"""
    n = states.pos.shape[0]
    ones_m = jnp.ones((2 * n, N_MOVES), jnp.bool_)
    ones_b = jnp.ones((2 * n, N_BOMB), jnp.bool_)

    def one_step(carry, _):
        states, key = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)
        masks = (ones_m, ones_b) if no_mask else both_masks(states)
        gv = both_states(states)
        # p0 帧（obs[:n]）用 params_a，p1 帧（obs[n:]）用 params_b
        a0, _, _ = sample_actions(params_a, arch, obs[:n],
                                  (masks[0][:n], masks[1][:n]), k0,
                                  state=gv[:n])
        a1, _, _ = sample_actions(params_b, arch, obs[n:],
                                  (masks[0][n:], masks[1][n:]), k1,
                                  state=gv[n:])
        env_acts = jnp.stack([a0, a1], axis=1)
        keys = jrandom.split(kstep, n)
        new_states, done, info = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts,
                                                               keys)
        # 胜率计数（与 collect_rollout 的 win/lose 奖励同口径）
        n_alive = info["alive"].sum(axis=-1)
        death_done = done & (n_alive == 1)
        p0_win = death_done & info["alive"][:, 0]
        p0_lose = death_done & ~info["alive"][:, 0]
        all_alive = done & (n_alive == 2)
        hp_f = info["hp"]
        p0_win = p0_win | (all_alive & (hp_f[:, 0] > hp_f[:, 1]))
        p0_lose = p0_lose | (all_alive & (hp_f[:, 0] < hp_f[:, 1]))
        return (new_states, key), (p0_win.sum(), p0_lose.sum())

    (final_states, _), (w, l) = jax.lax.scan(
        one_step, (states, key), None, length=num_steps)
    return final_states, (w.sum(), l.sum())

# ---------------- GAE ----------------

def compute_gae(rew, val, next_val, done, gamma, lam):
    def scan_fn(adv_prev, inputs):
        r, v, nv, d = inputs
        bootstrap = jnp.where(d, 0.0, nv)
        delta = r + gamma * bootstrap - v
        adv = delta + gamma * lam * (1.0 - d) * adv_prev
        return adv, adv

    _, advs = jax.lax.scan(scan_fn, jnp.zeros_like(val[0]),
                           (rew[::-1], val[::-1], next_val[::-1], done[::-1]))
    return advs[::-1]


# ---------------- PPO update ----------------


def ppo_update(params, opt, opt_state, arch, batch, key, minibatch,
               clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
               return_loss=False, adv_top_frac=0.25):
    """PPO 参数更新（Scheme 1: Critic 均匀无偏抽样 + Actor Top-Advantage 抽样 + 优势标准化）。

    Critic: 在均匀无偏的随机采样样本上优化 HL-Gauss 价值预测，0 选择偏差，价值基线稳定无漂移；
    Actor: 在 |Advantage| 排名前 adv_top_frac 的高信噪比决策帧上优化策略梯度，过滤冗余游走；
    计算量严格保持在 128 次 Minibatch 梯度更新，SPS 恢复至最高吞吐。
    """
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)

    # Batch 优势标准化 (Advantage Normalization)
    adv_mean = jnp.mean(adv_f)
    adv_std = jnp.std(adv_f)
    adv_norm = (adv_f - adv_mean) / (adv_std + 1e-8)

    mb_half = max(minibatch // 2, 1)

    # 提取 Actor Top-Advantage 索引
    if 0.0 < adv_top_frac < 1.0:
        n_keep = int(total * adv_top_frac)
        n_keep = max((n_keep // mb_half) * mb_half, mb_half)
        _, actor_idx = jax.lax.top_k(jnp.abs(adv_norm), n_keep)
    else:
        actor_idx = jnp.arange(total)
        n_keep = (total // mb_half) * mb_half

    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])
    st_f = state.reshape(total, -1)
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)

    n_mb = n_keep // mb_half

    def one_epoch(params, opt_state, key):
        ka, kc = jrandom.split(key)
        perm_a = jrandom.permutation(ka, n_keep)
        idx_a = actor_idx[perm_a].reshape(n_mb, mb_half)

        # Critic 均匀无偏抽样（覆盖平静与激烈全分布，0 选择偏差）
        critic_perm = jrandom.permutation(kc, total)[:n_keep]
        idx_c = critic_perm.reshape(n_mb, mb_half)

        # 拼成 (n_mb, minibatch)：前半段给 Actor，后半段给 Critic
        idx = jnp.concatenate([idx_a, idx_c], axis=1)

        def body(carry, mb):
            params, opt_state = carry
            o, a, ol, ad, rt, mm, bm = (
                obs_f[mb], acts_f[mb], old_f[mb], adv_norm[mb], ret_f[mb],
                mm_f[mb], bm_f[mb])
            st = st_f[mb]
            if obs_q:
                o = o.astype(jnp.float32) / 255.0   # 延迟反量化（minibatch 级）
                st = st.astype(jnp.float32) / 255.0

            loss_val, grads = jax.value_and_grad(
                lambda p: _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                    clip_eps, vf_coef, ent_coef))(params)
            if axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_val

        (params, opt_state), losses = jax.lax.scan(
            body, (params, opt_state), idx)
        return params, opt_state, jnp.mean(losses)

    last_loss = None
    for _ in range(epochs):
        key, ek = jrandom.split(key)
        params, opt_state, last_loss = one_epoch(params, opt_state, ek)
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


def _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt, clip_eps, vf_coef,
              ent_coef):
    """PPO loss：前半段优化 Actor（Top-Advantage），后半段优化 Critic（均匀无偏抽样）。"""
    mv, bm_, v_scalar, v_logits = net_forward(p, arch, o, st)
    n = o.shape[0]
    nh = n // 2

    # --- 1. Actor (前一半样本：来自 Top-Advantage 高信噪比决策帧) ---
    mv_a = jnp.where(mm[:nh], mv[:nh], jnp.full_like(mv[:nh], -jnp.inf))
    bm_a = jnp.where(bm[:nh], bm_[:nh], jnp.full_like(bm_[:nh], -jnp.inf))
    lsm_a = jax.nn.log_softmax(mv_a)
    lsb_a = jax.nn.log_softmax(bm_a)
    lp_a = (lsm_a[jnp.arange(nh), a[:nh, 0]]
            + lsb_a[jnp.arange(nh), a[:nh, 1]])
    ratio = jnp.exp(lp_a - ol[:nh])
    pg1 = -ad[:nh] * ratio
    pg2 = -ad[:nh] * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pol = jnp.maximum(pg1, pg2).mean()
    pm = jnp.exp(lsm_a)
    pb = jnp.exp(lsb_a)
    ent = (-(pm * jnp.where(pm > 0, lsm_a, 0.0)).sum(-1).mean()
           - (pb * jnp.where(pb > 0, lsb_a, 0.0)).sum(-1).mean())

    # --- 2. Critic (后一半样本：来自全量状态无偏均匀抽样) ---
    val_l = hl_gauss_value_loss(v_logits[nh:], rt[nh:])

    return pol + vf_coef * val_l - ent_coef * ent


def ppo_update_lsgd(params, opt, opt_state, arch, batch, key, minibatch,
                    clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
                    sync_k=128, bf16_sync=False, sync_state=False,
                    return_loss=False, adv_top_frac=0.25):
    """Local SGD 版 PPO（Scheme 1: Critic 均匀无偏抽样 + Actor Top-Advantage + 优势归一化）。"""
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)

    # Batch 优势标准化 (Advantage Normalization)
    adv_mean = jnp.mean(adv_f)
    adv_std = jnp.std(adv_f)
    adv_norm = (adv_f - adv_mean) / (adv_std + 1e-8)

    mb_half = max(minibatch // 2, 1)

    # 提取 Actor Top-Advantage 索引
    if 0.0 < adv_top_frac < 1.0:
        n_keep = int(total * adv_top_frac)
        n_keep = max((n_keep // mb_half) * mb_half, mb_half)
        _, actor_idx = jax.lax.top_k(jnp.abs(adv_norm), n_keep)
    else:
        actor_idx = jnp.arange(total)
        n_keep = (total // mb_half) * mb_half

    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])
    st_f = state.reshape(total, -1)
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)

    n_mb = n_keep // mb_half

    def local_step(carry, mb):
        params, opt_state = carry
        o, a, ol, ad, rt, mm, bm = (
            obs_f[mb], acts_f[mb], old_f[mb], adv_norm[mb], ret_f[mb],
            mm_f[mb], bm_f[mb])
        st = st_f[mb]
        if obs_q:
            o = o.astype(jnp.float32) / 255.0
            st = st.astype(jnp.float32) / 255.0

        loss_val, grads = jax.value_and_grad(
            lambda p: _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                clip_eps, vf_coef, ent_coef))(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    def _sync(x):
        """pmean 全量同步（可选 bf16 半精度传输）。axis_name=None 时 no-op。
        bf16 只作用于 fp32 叶子；int 叶子（Adam 的 count）各副本本就相同，
        直接透传（pmean 的 1/axis_size 缩放会把 int 变 float）。"""
        if axis_name is None:
            return x
        if bf16_sync:
            x = jax.tree.map(
                lambda t: (t.astype(jnp.bfloat16)
                           if t.dtype == jnp.float32 else t), x)
        x = jax.tree.map(
            lambda t: (jax.lax.pmean(t, axis_name)
                       if t.dtype in (jnp.float32, jnp.bfloat16) else t), x)
        if bf16_sync:
            x = jax.tree.map(
                lambda t: (t.astype(jnp.float32)
                           if t.dtype == jnp.bfloat16 else t), x)
        return x

    last_loss = None
    for _ in range(epochs):
        key, ek = jrandom.split(key)
        perm = jrandom.permutation(ek, total)
        idx = perm.reshape(-1, minibatch)
        n_mb = idx.shape[0]
        n_full, rem = divmod(n_mb, sync_k)
        chunk_means = []

        if n_full > 0:
            idx_c = idx[:n_full * sync_k].reshape(n_full, sync_k, minibatch)

            def run_chunk(carry, idx_k):
                """sync_k 个本地 minibatch + 一次全量同步。"""
                (params, opt_state), losses = jax.lax.scan(
                    local_step, carry, idx_k)
                params = _sync(params)
                if sync_state:
                    opt_state = _sync(opt_state)
                return (params, opt_state), jnp.mean(losses)

            (params, opt_state), cl = jax.lax.scan(
                run_chunk, (params, opt_state), idx_c)
            chunk_means.append(cl)
        if rem > 0:
            # 末尾不足 sync_k 个的余段：本地更新完也同步一次
            (params, opt_state), losses = jax.lax.scan(
                local_step, (params, opt_state), idx[n_full * sync_k:])
            params = _sync(params)
            if sync_state:
                opt_state = _sync(opt_state)
            chunk_means.append(jnp.mean(losses)[None])
        last_loss = jnp.mean(jnp.concatenate(chunk_means))
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


def ppo_update_gradsync(params, opt, opt_state, arch, batch, key, minibatch,
                        clip_eps, vf_coef, ent_coef, epochs, axis_name=None,
                        sync_k=128, bf16_sync=False, return_loss=False,
                        adv_top_frac=0.25):
    """梯度累积 + 周期同步（Scheme 1: Critic 均匀无偏抽样 + Actor Top-Advantage + 优势归一化）。"""
    obs, state, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    adv_f = advs.reshape(-1)

    # Batch 优势标准化 (Advantage Normalization)
    adv_mean = jnp.mean(adv_f)
    adv_std = jnp.std(adv_f)
    adv_norm = (adv_f - adv_mean) / (adv_std + 1e-8)

    mb_half = max(minibatch // 2, 1)

    # 提取 Actor Top-Advantage 索引
    if 0.0 < adv_top_frac < 1.0:
        n_keep = int(total * adv_top_frac)
        n_keep = max((n_keep // mb_half) * mb_half, mb_half)
        _, actor_idx = jax.lax.top_k(jnp.abs(adv_norm), n_keep)
    else:
        actor_idx = jnp.arange(total)
        n_keep = (total // mb_half) * mb_half

    obs_q = (obs.dtype == jnp.uint8)
    obs_f = obs.reshape(total, *obs.shape[2:])
    st_f = state.reshape(total, -1)
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)
    big = sync_k * minibatch
    GRAD_MAX_SAMPLES = 65536
    if big > GRAD_MAX_SAMPLES:
        raise SystemExit(
            f"grad 模式内存护栏：sync_k×minibatch = {big} 样本 > "
            f"{GRAD_MAX_SAMPLES}（64GB DCU 实测 131K 样本 OOM）。"
            f"请减小 --lsgd-k（本配置 K ≤ {GRAD_MAX_SAMPLES // minibatch}）"
            f"或改用 param 模式（--lsgd-mode param，无此限制）")

    n_mb = n_keep // mb_half

    def update_big(carry, mb):
        """一个 sync_k×minibatch 的大批：单次 value_and_grad → pmean → update。
        mb 是 (big,) 的行下标（scan 内静态长度切片）。"""
        params, opt_state = carry
        o, a, ol, ad, rt, mm, bm = (
            obs_f[mb], acts_f[mb], old_f[mb], adv_norm[mb], ret_f[mb],
            mm_f[mb], bm_f[mb])
        st = st_f[mb]
        if obs_q:
            o = o.astype(jnp.float32) / 255.0
            st = st.astype(jnp.float32) / 255.0
        loss_val, g = jax.value_and_grad(
            lambda p: _ppo_loss(p, arch, o, st, mm, bm, a, ol, ad, rt,
                                clip_eps, vf_coef, ent_coef))(params)
        if axis_name is not None:
            if bf16_sync:
                g = jax.tree.map(
                    lambda t: (t.astype(jnp.bfloat16)
                               if t.dtype == jnp.float32 else t), g)
            g = jax.tree.map(
                lambda t: (jax.lax.pmean(t, axis_name)
                           if t.dtype in (jnp.float32, jnp.bfloat16) else t),
                g)
            if bf16_sync:
                g = jax.tree.map(
                    lambda t: (t.astype(jnp.float32)
                               if t.dtype == jnp.bfloat16 else t), g)
        updates, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss_val

    last_loss = None
    for _ in range(epochs):
        ka, kc = jrandom.split(key)
        perm_a = jrandom.permutation(ka, n_keep)
        idx_a = actor_idx[perm_a].reshape(n_mb, mb_half)
        critic_perm = jrandom.permutation(kc, total)[:n_keep]
        idx_c = critic_perm.reshape(n_mb, mb_half)
        idx = jnp.concatenate([idx_a, idx_c], axis=1)
        n_full, rem = divmod(n_mb, sync_k)
        chunk_means = []

        if n_full > 0:
            idx_big = idx[:n_full * sync_k].reshape(n_full, big)
            (params, opt_state), cl = jax.lax.scan(
                update_big, (params, opt_state), idx_big)
            chunk_means.append(cl)
        if rem > 0:
            idx_rem = idx[n_full * sync_k:].reshape(-1)
            (params, opt_state), lv = update_big((params, opt_state), idx_rem)
            chunk_means.append(lv[None])
        last_loss = jnp.mean(jnp.concatenate(chunk_means))
    if return_loss:
        return params, opt_state, last_loss
    return params, opt_state


# ---------------- 离线蒸馏（--distill-data） ----------------

NEG = jnp.finfo(jnp.float32).min   # 非法动作 logits 占位（与 torch masked_dist 同语义）


def load_distill_data(pattern: str, max_frames: int):
    """加载 collect_distill 的 npz（每文件 obs7 (T,2,7,13,13) uint8×255 /
    logits (T,2,7) fp32 / move_mask (T,2,5) / bomb_mask (T,2,2)）。

    展平成帧（env×view → F）：每局面 2 视角各 1 帧、各自 logits/masks。
    超过 max_frames 时按文件 stride 均匀抽稀（保住混合地图构成比例）。
    返回 (obs_u8 (F,7,H,W) uint8 host, teacher_probs (F,7) fp32 host,
          move_mask (F,5), bomb_mask (F,2))。
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"--distill-data {pattern} 无文件")
    total = 0
    parts = []
    for p in paths:
        d = np.load(p)
        o = d["obs7"]                      # (T,2,7,H,W)
        lg = d["logits"]                   # (T,2,7)
        mm = d["move_mask"]                # (T,2,5)
        bm = d["bomb_mask"]                # (T,2,2)
        F = o.shape[0] * o.shape[1]
        total += F
        parts.append((o, lg, mm, bm))
    print(f"distill: {len(paths)} 文件共 {total:,} 帧，上限 {max_frames:,}", flush=True)
    if total > max_frames:
        keep = max_frames / total
        for i, (o, lg, mm, bm) in enumerate(parts):
            stride = max(1, int(1.0 / keep))
            s = o.reshape(-1, *o.shape[2:])[::stride]
            parts[i] = (s, lg.reshape(-1, lg.shape[-1])[::stride],
                        mm.reshape(-1, mm.shape[-1])[::stride],
                        bm.reshape(-1, bm.shape[-1])[::stride])
    obs = np.concatenate([p[0] for p in parts]).astype(np.uint8)
    lg = np.concatenate([p[1] for p in parts]).astype(np.float32)
    mm = np.concatenate([p[2] for p in parts]).astype(np.bool_)
    bm = np.concatenate([p[3] for p in parts]).astype(np.bool_)
    # teacher 概率目标：logits 已 mask（非法 = -inf/finfo.min）→ softmax 只在合法上归一
    lg = np.where(lg <= -1e30, -np.inf, lg).astype(np.float32)
    ex = np.exp(lg - lg.max(-1, keepdims=True))
    p_t = (ex / ex.sum(-1, keepdims=True)).astype(np.float32)
    F = obs.shape[0]
    print(f"distill: 实际 {F:,} 帧 obs={obs.shape} probs={p_t.shape} "
          f"(≈{obs.nbytes/1e6:.0f}MB host)", flush=True)
    return obs, p_t, mm, bm


def build_distill_update(params, opt, opt_state, arch, batch, ent_coef):
    """离线蒸馏 one update（jitted）：从数据采样一批帧 → mask 后 logits →
    KL(student || teacher)（teacher 概率做目标）+ 熵奖励。返回 jitted fn。"""

    @jax.jit
    def upd(params, opt_state, obs, p_t, mm, bm, key):
        def loss_fn(p):
            mv, bmv, _v = net_forward(p, arch, obs)
            mv = jnp.where(mm, mv, jnp.full_like(mv, NEG))
            bmv = jnp.where(bm, bmv, jnp.full_like(bmv, NEG))
            lsm = jax.nn.log_softmax(mv)
            lsb = jax.nn.log_softmax(bmv)
            # KL = sum p_t(log p_t - log p_s)；H(teacher) 对参数恒量，最小化 CE 即可。
            # p_t 只在合法动作上归一，student log_softmax 也只在合法上 —— 一致。
            ce_m = -(p_t[:, :5] * lsm).sum(-1).mean()
            ce_b = -(p_t[:, 5:] * lsb).sum(-1).mean()
            pm = jnp.exp(lsm)
            pb = jnp.exp(lsb)
            ent = (-(pm * jnp.where(pm > 0, lsm, 0.0)).sum(-1).mean()
                   - (pb * jnp.where(pb > 0, lsb, 0.0)).sum(-1).mean())
            return ce_m + ce_b - ent_coef * ent

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return upd


def run_distill(params, opt, opt_state, args, key):
    """离线蒸馏阶段：--distill-iters 轮，每轮采样 --distill-batch 帧做一次
    KL 更新（host→device 分批搬，数据可远大于显存）。返回 (params, opt_state)。"""
    obs_u8, p_t, mm, bm = load_distill_data(args.distill_data,
                                            args.distill_max_frames)
    F = obs_u8.shape[0]
    upd = build_distill_update(params, opt, opt_state, args.arch,
                               args.distill_batch, args.ent_coef)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(args.distill_iters):
        idx = rng.integers(0, F, args.distill_batch)
        o = jnp.asarray(obs_u8[idx]).astype(jnp.float32) / 255.0
        pt = jnp.asarray(p_t[idx])
        m = jnp.asarray(mm[idx])
        b = jnp.asarray(bm[idx])
        key, k = jrandom.split(key)
        t1 = time.time()
        params, opt_state = upd(params, opt_state, o, pt, m, b, k)
        jax.block_until_ready(params)
        dt = time.time() - t1
        if it % 10 == 0 or it == args.distill_iters - 1:
            print(f"[distill {it}] {dt*1000:.0f}ms/批 "
                  f"({args.distill_batch*1000/max(dt,1e-6):,.0f} 帧/s)", flush=True)
    print(f"distill 完成：{args.distill_iters} 轮 × {args.distill_batch:,} 帧，"
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return params, opt_state


# ---------------- checkpoint（蒸馏产物 / PPO 续跑） ----------------

def save_params(params, path: str) -> None:
    """保存 params（嵌套 numpy 数组 pytree）到 pickle。device_get 先搬回 host。"""
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"params 已保存 -> {path}", flush=True)


def load_params(path: str):
    """从 pickle 加载 params 并放到设备（用于蒸馏初始权重 / PPO 续跑）。"""
    with open(path, "rb") as f:
        params = pickle.load(f)
    return jax.device_put(params)


# ---------------- one_iter（可复用，probe 直接测训练主循环） ----------------


def _lsgd_updater(args):
    """按 --lsgd-k/--lsgd-mode 选有损同步函数；--lsgd-k 0 = 返回 None
    （现状无损路径：每个 minibatch pmean 梯度）。"""
    if getattr(args, "lsgd_k", 0) <= 0:
        return None
    if getattr(args, "lsgd_mode", "param") == "grad":
        return ppo_update_gradsync
    return ppo_update_lsgd


def build_one_iter(params, opt, opt_state, states, key, args):
    """返回 jitted one_iter：(params, opt_state, states, key) -> 同形四元组。

    与训练主循环完全同构：collect_rollout → bootstrap → GAE → PPO update。
    """
    n = states.pos.shape[0]
    steps = args.num_steps

    def one_iter(params, opt_state, states, key):
        states, batch, _nov, _kills = collect_rollout(
            params, args.arch, states, key, steps,
            getattr(args, "no_mask", False),
            getattr(args, "obs_quant", False),
            getattr(args, "checkpoint", False))
        obs, state, acts, lps, vals, rew, done, masks = batch
        # bootstrap：rollout 尾部状态价值（全局状态向量同步传入）
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fstate = both_states(states)
        fkey = jrandom.split(key)[0]
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey,
                                    state=fstate)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        upd = _lsgd_updater(args)
        if upd is None:
            params, opt_state = ppo_update(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, args.minibatch, args.clip_eps, args.vf_coef,
                args.ent_coef, args.epochs)
        else:
            kw = dict(sync_k=args.lsgd_k,
                      bf16_sync=getattr(args, "lsgd_bf16", False))
            if upd is ppo_update_lsgd:
                kw["sync_state"] = getattr(args, "lsgd_sync_state", False)
            params, opt_state = upd(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, args.minibatch, args.clip_eps, args.vf_coef,
                args.ent_coef, args.epochs, **kw)
        key = jrandom.split(key)[0]
        return params, opt_state, states, key

    return jax.jit(one_iter)


def build_dp_one_iter(params, opt, opt_state, states, key, args, n_dev):
    """数据并行（DP）one_iter：每卡 envs 切片独立 collect，更新时梯度
    pmean allreduce。n_dev 为卡数，states/key 首维须为 n_dev（pmap 切片）。

    与 build_one_iter 逐位一致的条件：
      - 每卡 minibatch = args.minibatch // n_dev（等效全局 minibatch 不变
        → 梯度步数不变，样本不重叠）
      - states 按卡切片后每卡独立 rollout（collect 侧零通信）
      - 梯度 pmean 后每卡应用相同更新 → 参数逐卡一致
    n_dev=1 时退化为 build_one_iter 语义（pmap 单设备）。
    """
    steps = args.num_steps
    mb_local = args.minibatch // n_dev
    assert mb_local >= 1, "minibatch 必须 >= 卡数"

    def one_iter_shard(params, opt_state, states, key):
        states, batch, _nov, _kills = collect_rollout(
            params, args.arch, states, key, steps,
            getattr(args, "no_mask", False),
            getattr(args, "obs_quant", False),
            getattr(args, "checkpoint", False))
        obs, state, acts, lps, vals, rew, done, masks = batch
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fstate = both_states(states)
        fkey = jrandom.split(key)[0]
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey,
                                    state=fstate)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        upd = _lsgd_updater(args)
        if upd is None:
            params, opt_state = ppo_update(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, mb_local, args.clip_eps, args.vf_coef, args.ent_coef,
                args.epochs, axis_name="dev")
        else:
            kw = dict(sync_k=args.lsgd_k,
                      bf16_sync=getattr(args, "lsgd_bf16", False))
            if upd is ppo_update_lsgd:
                kw["sync_state"] = getattr(args, "lsgd_sync_state", False)
            params, opt_state = upd(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, mb_local, args.clip_eps, args.vf_coef, args.ent_coef,
                args.epochs, axis_name="dev", **kw)
        key = jrandom.split(key)[0]
        return params, opt_state, states, key

    return jax.pmap(one_iter_shard, axis_name="dev",
                    in_axes=(None, None, 0, 0))


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp4",
                    choices=["mlp", "mlp_bf16", "mlp4", "cnn", "transformer"])
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=None,
                    help="隐藏层宽。不传时按 arch 选：mlp=256 / mlp4=768 / cnn=256")
    ap.add_argument("--embed", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995,
                    help="生产对齐：一局最长 1800 tick，折扣要够长才看得到终局奖励")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    # ---- 离线蒸馏（--distill-data 提供则先跑蒸馏，再可选接 PPO）----
    ap.add_argument("--distill-data", default=None,
                    help="collect_distill 的 npz（支持 glob）。给定时先跑蒸馏")
    ap.add_argument("--distill-iters", type=int, default=200)
    ap.add_argument("--distill-batch", type=int, default=8192,
                    help="每轮采样的帧数（host→device 分批搬，可远小于总帧数）")
    ap.add_argument("--distill-max-frames", type=int, default=2_000_000,
                    help="总帧数上限（超出按文件 stride 均匀抽稀）")
    ap.add_argument("--distill-then-ppo", action="store_true",
                    help="蒸馏后继续自对弈 PPO 微调（不传则蒸馏完即停）")
    ap.add_argument("--no-mask", action="store_true",
                    help="性能 A/B：mask 全放开（行为=无 mask 旧版）")
    ap.add_argument("--obs-quant", action="store_true",
                    help="obs buffer 存 uint8（×255 量化，反量化进网络）："
                         "45GB→11GB，8192×512 OOM 的解法（精度 1/255 优于 "
                         "bf16 尾数，低精度 obs 通道无损）")
    ap.add_argument("--checkpoint", action="store_true",
                    help="scan body 用 jax.checkpoint：反向重算中间量，"
                         "配合 --obs-quant 让 8192×512 放下（省 scan "
                         "中间量 44.8GB），代价是 collect 变慢（重算）")
    # ---- Local SGD（有损同步，跨机降通信）----
    ap.add_argument("--lsgd-k", type=int, default=0,
                    help="Local SGD 同步周期：每 K 个 minibatch 同步一次参数"
                         "（0=现状：每个 minibatch pmean 梯度，逐位一致）。"
                         ">0 时 minibatch 循环内零通信、每 K 步 pmean 参数，"
                         "通信量降到 ~1/K。20 卡（10 机×2 卡）下 K=256 ≈ "
                         "4 次同步/迭代 ≈ 0.5-1.5s，K=128 ≈ 8 次 ≈ 1-3s")
    ap.add_argument("--lsgd-mode", default="param",
                    choices=["param", "grad"],
                    help="Local SGD 同步对象：param=每 K 步平均参数（保持 "
                         "1024 次更新/迭代，代价是 K 步本地漂移）；grad=冻结"
                         "参数上累加 K 个梯度、一次平均梯度同步、一次更新"
                         "（零漂移、参数始终逐位一致，代价是只有 1024/K 次"
                         "更新/迭代；K=1 时与现状逐位一致）")
    ap.add_argument("--lsgd-bf16", action="store_true",
                    help="Local SGD 同步时用 bf16 半精度传输（流量减半，"
                         "尾数损失可忽略）")
    ap.add_argument("--lsgd-sync-state", action="store_true",
                    help="Local SGD 同步时连 Adam 动量/方差一起平均"
                         "（防本地漂移，流量×3）")
    # ---- checkpoint ----
    ap.add_argument("--load", default=None,
                    help="初始权重 pickle（蒸馏出的 student / 续跑）")
    ap.add_argument("--save", default=None,
                    help="训练结束时保存 params 的路径")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 迭代存一次中间 ckpt（0=不存）。文件名 = "
                         "--save 去掉扩展名 + _it{N}，供中途评估/续跑")
    args = ap.parse_args()
    if args.hidden is None:
        args.hidden = 768 if args.arch == "mlp4" else 256

    key = jrandom.PRNGKey(0)
    devs = setup_platform()          # 平台层：精度/设备探测（切 CUDA 只动这里）
    print(f"devices: {device_summary(devs)}", flush=True)

    n, steps = args.num_envs, args.num_steps
    states = init_batch(key, n)
    key, net_key = jrandom.split(key)
    if args.arch in ("mlp", "mlp_bf16", "mlp4", "cnn"):
        kw = {"hidden": args.hidden}
    else:
        kw = {"embed": args.embed, "depth": args.depth}
    params = init_net(net_key, args.arch, N_OBS_CH, H, W, **kw)
    if args.load:
        params = load_params(args.load)
        print(f"已加载初始权重: {args.load} (params={count_params(params):,})",
              flush=True)
    else:
        print(f"arch={args.arch} params={count_params(params):,}", flush=True)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    # 离线蒸馏阶段（先于 PPO）：KL 到 teacher 分布，warm-start student。
    if args.distill_data:
        params, opt_state = run_distill(params, opt, opt_state, args, key)
        if not args.distill_then_ppo:
            if args.save:
                save_params(params, args.save)
            print("蒸馏阶段结束（未接 PPO，退出）", flush=True)
            return
        print("接自对弈 PPO 微调：", flush=True)

    one_iter_j = build_one_iter(params, opt, opt_state, states, key, args)

    # warmup（首次编译）
    t0 = time.time()
    for _ in range(2):
        params, opt_state, states, key = one_iter_j(params, opt_state, states, key)
    jax.block_until_ready(params)
    print(f"warmup done ({time.time()-t0:.1f}s)", flush=True)

    # 计时
    t0 = time.time()
    for it in range(args.iters):
        t1 = time.time()
        params, opt_state, states, key = one_iter_j(params, opt_state, states, key)
        jax.block_until_ready(params)
        dt = time.time() - t1
        sps = 2 * n * steps / dt
        if args.save and args.save_every and it and it % args.save_every == 0:
            mid = f"{os.path.splitext(args.save)[0]}_it{it}.pt"
            save_params(params, mid)
        print(f"[iter {it}] {dt:.2f}s  sps={sps:,.0f}", flush=True)
    tot = 2 * n * steps * args.iters / (time.time() - t0)
    print(f"FINAL end-to-end sps = {tot:,.0f} "
          f"({2*n*steps*args.iters:,} steps)", flush=True)
    if args.save:
        save_params(params, args.save)


if __name__ == "__main__":
    main()
