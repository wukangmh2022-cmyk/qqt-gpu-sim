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

from .jax_env import (H, W, MAX_STEPS, N_BOMB, N_MOVES, N_OBS_CH,
                      init_batch, legal_mask, make_obs, step)
from .jax_net import count_params, init_net, net_forward
from .platform import device_summary, setup_platform


# ---------------- policy head ----------------


def sample_actions(params, arch, obs, masks, key):
    """obs (N,C,H,W)，masks = (move_mask (N,5), bomb_mask (N,2)) bool。

    返回 (act (N,2), logp (N,), val (N,))。非法动作 logits 置 -inf 后采样
    （与 torch masked_dist 同语义：softmax 概率归零）。
    """
    mv, bm, v = net_forward(params, arch, obs)
    mm, bmb = masks
    mv_m = jnp.where(mm, mv, jnp.full_like(mv, -jnp.inf))
    bm_m = jnp.where(bmb, bm, jnp.full_like(bm, -jnp.inf))
    k1, k2 = jrandom.split(key)
    a_m = jrandom.categorical(k1, mv_m)
    a_b = jrandom.categorical(k2, bm_m)
    n = obs.shape[0]
    lp = (jax.nn.log_softmax(mv_m)[jnp.arange(n), a_m]
          + jax.nn.log_softmax(bm_m)[jnp.arange(n), a_b])
    return jnp.stack([a_m, a_b], axis=-1), lp, v


def both_perspectives(states):
    """返回 (2N, C, H, W)：p0 视角 + p1 视角拼接。"""
    obs0 = jax.vmap(lambda s: make_obs(s, 0))(states)
    obs1 = jax.vmap(lambda s: make_obs(s, 1))(states)
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


# ---------------- rollout ----------------


def collect_rollout(params, arch, states, key, num_steps, no_mask=False):
    """自对弈：同一网络打两边。states (N, ...)。返回 (new_states, batch)。

    step 在终局后**就地重置**（对齐正式版 auto_reset），所以胜负判定用
    step 前的 alive 快照（重置后 alive 恒全 True，无法区分谁死）。
    batch 含每 tick 的 (move_mask, bomb_mask)——PPO loss 用它屏蔽非法动作。
    no_mask=True：mask 全放开（性能 A/B 用，行为=无 mask 旧版）。
    """
    n = states.pos.shape[0]
    ones_m = jnp.ones((2 * n, N_MOVES), jnp.bool_)
    ones_b = jnp.ones((2 * n, N_BOMB), jnp.bool_)

    def one_step(carry, _):
        states, key = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)               # (2N, C, H, W)
        masks = (ones_m, ones_b) if no_mask else both_masks(states)
        acts, lps, vals = sample_actions(params, arch, obs, masks, key)
        a0, a1 = acts[:n], acts[n:]
        env_acts = jnp.stack([a0, a1], axis=1)        # (N, 2, 2)
        alive_prev = states.alive                     # 重置前快照
        new_states, done = jax.vmap(step)(states, env_acts)
        # 稀疏胜负奖励（终局 tick）：胜 +1 / 负 -1 / 双亡或超时平 0.5
        me_alive, opp_alive = alive_prev[:, 0], alive_prev[:, 1]
        w0 = jnp.where(me_alive & ~opp_alive, 1.0,
                       jnp.where(~me_alive & opp_alive, 0.0, 0.5))
        rew0 = jnp.where(done, w0, 0.0)
        rew1 = jnp.where(done, 1.0 - w0, 0.0)
        rew = jnp.concatenate([rew0, rew1])
        d = jnp.concatenate([done, done])
        data = (obs, acts, lps, vals, rew, d, masks)
        return (new_states, key), data

    (final_states, _), data = jax.lax.scan(one_step, (states, key), None,
                                           length=num_steps)
    obs, acts, lps, vals, rew, done, masks = data
    return final_states, (obs, acts, lps, vals, rew, done, masks)


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
               clip_eps, vf_coef, ent_coef, epochs):
    obs, acts, old_lps, advs, rets, masks = batch
    total = obs.shape[0] * obs.shape[1]
    obs_f = obs.reshape(total, *obs.shape[2:])
    acts_f = acts.reshape(total, -1)
    old_f = old_lps.reshape(-1)
    adv_f = advs.reshape(-1)
    ret_f = rets.reshape(-1)
    mm_f, bm_f = masks
    mm_f = mm_f.reshape(total, -1)
    bm_f = bm_f.reshape(total, -1)

    def one_epoch(params, opt_state, key):
        perm = jrandom.permutation(key, total)
        idx = perm.reshape(-1, minibatch)

        def body(carry, mb):
            params, opt_state = carry
            o, a, ol, ad, rt, mm, bm = (obs_f[mb], acts_f[mb], old_f[mb],
                                        adv_f[mb], ret_f[mb], mm_f[mb],
                                        bm_f[mb])

            def loss_fn(p):
                mv, bm_, v = net_forward(p, arch, o)
                # 非法动作 logits 置 -inf（与采样/rollout logp 同一分布）
                mv = jnp.where(mm, mv, jnp.full_like(mv, -jnp.inf))
                bm_ = jnp.where(bm, bm_, jnp.full_like(bm_, -jnp.inf))
                lsm = jax.nn.log_softmax(mv)
                lsb = jax.nn.log_softmax(bm_)
                n = o.shape[0]
                lp = (lsm[jnp.arange(n), a[:, 0]]
                      + lsb[jnp.arange(n), a[:, 1]])
                ratio = jnp.exp(lp - ol)
                pg1 = -ad * ratio
                pg2 = -ad * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                pol = jnp.maximum(pg1, pg2).mean()
                val_l = jnp.mean((v - rt) ** 2)
                ent = (-(jnp.exp(lsm) * lsm).sum(-1).mean()
                       - (jnp.exp(lsb) * lsb).sum(-1).mean())
                return pol + vf_coef * val_l - ent_coef * ent

            grads = jax.grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), ()

        (params, opt_state), _ = jax.lax.scan(body, (params, opt_state), idx)
        return params, opt_state

    for _ in range(epochs):
        key, ek = jrandom.split(key)
        params, opt_state = one_epoch(params, opt_state, ek)
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
            ent = (-(jnp.exp(lsm) * lsm).sum(-1).mean()
                   - (jnp.exp(lsb) * lsb).sum(-1).mean())
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


def build_one_iter(params, opt, opt_state, states, key, args):
    """返回 jitted one_iter：(params, opt_state, states, key) -> 同形四元组。

    与训练主循环完全同构：collect_rollout → bootstrap → GAE → PPO update。
    """
    n = states.pos.shape[0]
    steps = args.num_steps

    def one_iter(params, opt_state, states, key):
        states, batch = collect_rollout(params, args.arch, states, key, steps,
                                        getattr(args, "no_mask", False))
        obs, acts, lps, vals, rew, done, masks = batch
        # bootstrap：rollout 尾部状态价值
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fkey = jrandom.split(key)[0]
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        params, opt_state = ppo_update(
            params, opt, opt_state, args.arch,
            (obs, acts, lps, advs, rets, masks),
            key, args.minibatch, args.clip_eps, args.vf_coef, args.ent_coef,
            args.epochs)
        key = jrandom.split(key)[0]
        return params, opt_state, states, key

    return jax.jit(one_iter)


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
    ap.add_argument("--gamma", type=float, default=0.99)
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
