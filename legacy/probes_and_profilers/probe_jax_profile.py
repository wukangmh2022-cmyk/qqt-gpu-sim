"""分段 profile：mlp4 化繁为简版 4.5s/iter 的时间去哪了。

拆成 5 段（全部 jit 后计时，warmup 排除编译）：
  A. env step only（vmap(step) ×256，无 obs/网络）—— 环境规则成本
  B. obs only（both_perspectives ×256）—— 观测生成成本
  C. rollout 网络前向（sample_actions mlp4 ×256，2N batch）
  D. GAE（一次）
  E. PPO update（minibatch scan 的 net fwd+bwd）

4096 envs × 256 steps 与 run_jax8 同配置，直接对比 4.51s/iter。
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W, N_OBS_CH, init_batch, step
from jax_bomb.jax_net import init_mlp4, mlp4_forward
from jax_bomb.platform import device_summary, setup_platform


def timed(name, f, *args, iters=20):
    out = f(*args)
    jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / iters * 1000
    print(f"{name:<44s} {ms:9.3f} ms", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp4", choices=["mlp", "mlp4"])
    ap.add_argument("--num-envs", type=int, default=4096)
    args = ap.parse_args()
    devs = setup_platform()
    print(f"devices: {device_summary(devs)}  arch={args.arch}", flush=True)
    n = args.num_envs
    steps = 256
    key = jrandom.PRNGKey(0)
    states = init_batch(key, n)
    key, net_key = jrandom.split(key)
    if args.arch == "mlp4":
        p = init_mlp4(net_key, N_OBS_CH, H, W, hidden=768)
        from jax_bomb.jax_train import sample_actions, ppo_update
    else:
        from jax_bomb.jax_net import init_mlp
        p = init_mlp(net_key, N_OBS_CH, H, W, hidden=256)
        from jax_bomb.jax_train import sample_actions, ppo_update
    acts = jrandom.randint(key, (n, 2, 2), 0, 5, dtype=jnp.int32)
    acts = acts.at[..., 1].set(jrandom.randint(key, (n, 2), 0, 2))

    # ---- A: env step only ----
    @jax.jit
    def step_only(s, a):
        return jax.vmap(step)(s, a)
    sA = timed(f"A: env step ×1", step_only, states, acts)[0]
    print(f"    → 256 步 env = {4.51 * 1000 * (256/256):>8.1f} ms 换算见下", flush=True)

    # 直接测 256 步 scan（和 rollout 同构）
    def _scan_step(s, a):
        def body(c, _):
            s2, d = step_only(c, a)
            return s2, d
        s2, ds = jax.lax.scan(body, s, None, length=steps)
        return s2
    ss = jax.jit(_scan_step)
    timed(f"A: env step ×{steps}（scan）", ss, states, acts)

    # ---- B: obs only（含两个视角） ----
    def _obs(s):
        o0 = jax.vmap(lambda x: jnp.stack([x.pos[0, 0], x.fuse[0, 0]]))(s)  # noqa
        return o0
    from jax_bomb.jax_train import both_perspectives
    obs = both_perspectives(states)
    bo = jax.jit(lambda s: both_perspectives(s))
    timed(f"B: obs both_persp ×1", bo, states)
    obs256 = jax.jit(lambda s: jax.lax.scan(
        lambda c, _: (c, both_perspectives(c)), s, None, length=steps)[1])
    timed(f"B: obs ×{steps}（scan）", obs256, states)

    # ---- C: rollout 网络前向（2N batch） ----
    from jax_bomb.jax_train import sample_actions
    n2 = o2n.shape[0] if 'o2n' in dir() else None

    @jax.jit
    def net_fwd(pp, o, k):
        # 全真 mask（探针不关心 mask 行为，只测网络前向）
        mm = jnp.ones((o.shape[0], 5), jnp.bool_)
        bm = jnp.ones((o.shape[0], 2), jnp.bool_)
        return sample_actions(pp, args.arch, o, (mm, bm), k)
    o2n = both_perspectives(states)
    timed(f"C: net fwd 2N={o2n.shape[0]} ×1", net_fwd, p, o2n, key)

    # ---- E: PPO update（minibatch scan） ----
    from jax_bomb.jax_train import ppo_update
    import optax
    opt = optax.adam(3e-4)
    opt_state = opt.init(p)
    obs_b = jnp.zeros((o2n.shape[0], 256, N_OBS_CH, H, W), jnp.float32)
    acts_b = jnp.zeros((o2n.shape[0], 256, 2), jnp.int32)
    lps_b = jnp.zeros((o2n.shape[0], 256), jnp.float32)
    advs_b = jnp.zeros((o2n.shape[0], 256), jnp.float32)
    rets_b = jnp.zeros((o2n.shape[0], 256), jnp.float32)
    mm_b = jnp.ones((o2n.shape[0], 256, 5), jnp.bool_)
    bm_b = jnp.ones((o2n.shape[0], 256, 2), jnp.bool_)
    batch = (obs_b, acts_b, lps_b, advs_b, rets_b, (mm_b, bm_b))
    pu = jax.jit(lambda pp, os_, b: ppo_update(
        pp, opt, os_, args.arch, b, key, 2048, 0.2, 0.5, 0.01, 2))
    timed("E: PPO update (epochs=2, mb=2048)", pu, p, opt_state, batch)

    # ---- F: 单次 minibatch fwd+bwd（算 PPO 里固定开销占比） ----
    @jax.jit
    def mb_fwd_bwd(pp, o):
        def loss_fn(ppp):
            mv, bm, v, _ = net_forward(ppp, args.arch, o)
            return mv.sum() + bm.sum() + v.sum()
        g = jax.grad(loss_fn)(pp)
        return g
    from jax_bomb.jax_net import net_forward
    o_mb = jnp.zeros((2048, N_OBS_CH, H, W), jnp.float32)
    timed("F: 单 mb(2048) fwd+bwd", mb_fwd_bwd, p, o_mb)

    # ---- G: collect_rollout 整体（真实 scan 依赖链，对照分段） ----
    from jax_bomb.jax_train import collect_rollout
    cr = jax.jit(lambda pp, s, k: collect_rollout(pp, args.arch, s, k, steps))
    timed("G: collect_rollout ×1（完整 256 步）", cr, p, states, key)

    # ---- H: GAE ----
    from jax_bomb.jax_train import compute_gae
    v = jnp.zeros((o2n.shape[0], 256), jnp.float32)
    rw = jnp.zeros((o2n.shape[0], 256), jnp.float32)
    dn = jnp.zeros((o2n.shape[0], 256), jnp.bool_)
    gae = jax.jit(lambda vv, r, d: compute_gae(r, vv, vv, d, 0.99, 0.95))
    timed("H: GAE ×1", gae, v, rw, dn)

    # ---- I: 完整 one_iter（训练主循环整体，对照分段之和） ----
    import argparse as _ap
    a = _ap.Namespace(arch=args.arch, num_steps=256, minibatch=2048,
                      epochs=2, gamma=0.99, lam=0.95, clip_eps=0.2,
                      vf_coef=0.5, ent_coef=0.01)
    from jax_bomb.jax_train import build_one_iter
    one_iter_j = build_one_iter(p, opt, opt_state, states, key, a)
    t0 = time.time()
    for _ in range(2):
        p, opt_state, states, key = one_iter_j(p, opt_state, states, key)
    jax.block_until_ready(p)
    print(f"I: one_iter 编译(含2次热身) {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    for it in range(3):
        t1 = time.time()
        p, opt_state, states, key = one_iter_j(p, opt_state, states, key)
        jax.block_until_ready(p)
        print(f"I: one_iter[{it}] {time.time()-t1:.3f}s", flush=True)
    print(f"I: one_iter 平均 {(time.time()-t0)/3*1000:9.1f} ms", flush=True)

    # ---- I2: 二分——rollout+bootstrap+GAE（不含 PPO），找合并程序开销在哪 ----
    from jax_bomb.jax_train import (collect_rollout, sample_actions,
                                    compute_gae, both_masks)

    def pre_ppo(params, states, key):
        states, batch = collect_rollout(params, args.arch, states, key, 256)
        obs, acts, lps, vals, rew, done, masks = batch
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, key)
        nv = jnp.concatenate([vals[1:], fval[None]], axis=0)
        return states, compute_gae(rew, vals, nv, done, 0.99, 0.95)
    pre_j = jax.jit(pre_ppo)
    timed("I2: rollout+bootstrap+GAE（无 PPO）×1", pre_j, p, states, key)


if __name__ == "__main__":
    main()
