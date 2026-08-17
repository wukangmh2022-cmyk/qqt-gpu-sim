"""Self-play PPO for the naive JAX bomberman + end-to-end SPS bench.

Structure mirrors Average Joe (rollout via lax.scan, 2N batch, scan-based
minibatch PPO) but single-device and dependency-light (optax only).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
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
    """返回 ((2N,5), (2N,2))：双视角 mask 拼接（同 obs 顺序）。"""
    m0, b0 = jax.vmap(legal_mask)(states)
    m1, b1 = jax.vmap(legal_mask)(states)
    return (jnp.concatenate([m0, m1]), jnp.concatenate([b0, b1]))


# ---------------- rollout ----------------


def collect_rollout(params, arch, states, key, num_steps):
    """自对弈：同一网络打两边。states (N, ...)。返回 (new_states, batch)。

    step 在终局后**就地重置**（对齐正式版 auto_reset），所以胜负判定用
    step 前的 alive 快照（重置后 alive 恒全 True，无法区分谁死）。
    batch 含每 tick 的 (move_mask, bomb_mask)——PPO loss 用它屏蔽非法动作。
    """
    n = states.pos.shape[0]

    def one_step(carry, _):
        states, key = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        obs = both_perspectives(states)               # (2N, C, H, W)
        masks = both_masks(states)
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


# ---------------- one_iter（可复用，probe 直接测训练主循环） ----------------


def build_one_iter(params, opt, opt_state, states, key, args):
    """返回 jitted one_iter：(params, opt_state, states, key) -> 同形四元组。

    与训练主循环完全同构：collect_rollout → bootstrap → GAE → PPO update。
    """
    n = states.pos.shape[0]
    steps = args.num_steps

    def one_iter(params, opt_state, states, key):
        states, batch = collect_rollout(params, args.arch, states, key, steps)
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
    print(f"arch={args.arch} params={count_params(params):,}", flush=True)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

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
        print(f"[iter {it}] {dt:.2f}s  sps={sps:,.0f}", flush=True)
    tot = 2 * n * steps * args.iters / (time.time() - t0)
    print(f"FINAL end-to-end sps = {tot:,.0f} "
          f"({2*n*steps*args.iters:,} steps)", flush=True)


if __name__ == "__main__":
    main()
