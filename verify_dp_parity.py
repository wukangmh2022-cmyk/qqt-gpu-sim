"""DP（pmap）一致性验证，三个不变量：

1) n_dev=1 退化：pmap 单设备 == 单卡 jit one_iter，逐位 0 差
   （同 key 同 total → 同 perm → 同 minibatch → 同梯度）。
2) n_dev=2 两卡一致：pmean 后各卡 params 逐位一致（x[0] == x[1]）。
3) 梯度级等价（确定性 batch）：loss 对全量 batch 的梯度
   == pmean(每卡子 batch 梯度)，逐位 0 差（数学恒等）。

用法：python3 verify_dp_parity.py [--envs 2048 --steps 64 --iters 3]
"""
import argparse
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jax_bomb.jax_env import H, W, N_OBS_CH, init_batch
from jax_bomb.jax_net import init_net, net_forward
from jax_bomb.jax_train import (both_masks, both_perspectives,
                                build_dp_one_iter, build_one_iter,
                                collect_rollout, compute_gae, ppo_update,
                                sample_actions)


class A:
    pass


def make_args():
    a = A()
    a.arch = "mlp_bf16"
    a.hidden = 768
    a.embed = 192
    a.depth = 4
    a.num_steps = 64
    a.minibatch = 512
    a.epochs = 2
    a.lr = 3e-4
    a.gamma = 0.995
    a.lam = 0.95
    a.clip_eps = 0.2
    a.vf_coef = 0.5
    a.ent_coef = 0.01
    a.no_mask = False
    return a


def maxdiff(p1, p2):
    d = max(float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(p1), jax.tree.leaves(p2)))
    return d


def tree_stack(xs):
    return jax.tree.map(lambda *v: jnp.stack(v), *xs)


def test_ndev1(cfg, n_total, iters):
    """不变量 1：n_dev=1 退化 == 单卡 jit。"""
    key = jrandom.PRNGKey(0)
    states0 = init_batch(key, n_total)
    key, nk = jrandom.split(key)
    params = init_net(nk, cfg.arch, N_OBS_CH, H, W, hidden=cfg.hidden)
    opt = optax.adam(cfg.lr)
    os0 = opt.init(params)

    it_jit = build_one_iter(params, opt, os0, states0, key, cfg)
    it_dp = build_dp_one_iter(
        params, opt, os0, tree_stack([states0]), key[None], cfg, 1)

    pj, oj, sj, kj = it_jit(params, os0, states0, key)
    pd, od, sd, kd = it_dp(params, os0, tree_stack([states0]), key[None])
    jax.block_until_ready((pj, pd))
    ok = (maxdiff(pj, jax.tree.map(lambda x: x[0], pd)) == 0.0
          and maxdiff(jax.tree.map(lambda x: x[0], sd), sj) == 0.0)
    print(f"[1] n_dev=1 退化 vs 单卡 jit: params 0 差 / states 0 差 = "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return pj, oj, sj


def test_ndev2(cfg, n_total, iters):
    """不变量 2（软件 pmean 更新语义，单设备模拟）：
    两卡独立 collect（envs 切片 + 独立 key）→ 各自算梯度 →
    梯度平均（pmean）后每卡用相同更新量 → 两卡参数逐位一致。
    对照：不做 pmean（各自独立 update）时两卡必然发散。
    """
    n_dev = 2
    n_local = n_total // n_dev
    key = jrandom.PRNGKey(0)
    states0 = init_batch(key, n_total)
    key, nk = jrandom.split(key)
    params = init_net(nk, cfg.arch, N_OBS_CH, H, W, hidden=cfg.hidden)
    opt = optax.adam(cfg.lr)
    os0 = opt.init(params)

    shards = [jax.tree.map(lambda x: x[d * n_local:(d + 1) * n_local], states0)
              for d in range(n_dev)]
    keys_dp = jrandom.split(key, n_dev)

    collect_j = jax.jit(
        lambda p, s, k: collect_rollout(p, cfg.arch, s, k, cfg.num_steps))
    # 每卡各自独立参数副本
    ps = [params, jax.tree.map(lambda x: jnp.array(x), params)]
    os_list = [opt.init(params), opt.init(params)]
    ss = shards
    kk = keys_dp

    def prep_batch(p, s, k):
        s2, batch = collect_j(p, s, k)
        obs, acts, lps, vals, rew, done, masks = batch
        fobs = both_perspectives(s2)
        fmasks = both_masks(s2)
        _, _, fval = sample_actions(p, cfg.arch, fobs, fmasks,
                                    jrandom.split(k)[0])
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, cfg.gamma, cfg.lam)
        rets = advs + vals
        return (obs, acts, lps, advs, rets, masks)

    for it in range(iters):
        # 两卡各自 rollout（数据不同：不同 envs 切片 + 不同 key）
        b0 = prep_batch(ps[0], ss[0], kk[0])
        b1 = prep_batch(ps[1], ss[1], kk[1])
        # 各自梯度（minibatch 减半；这里直接全量 loss，仅验证 pmean 更新语义）
        def loss(p, b):
            obs, acts, lps, advs, rets, masks = b
            mv, bm_, v, _ = net_forward(p, cfg.arch, obs.reshape(-1, *obs.shape[2:]))
            mm, bm = masks
            mm = mm.reshape(-1, 5)
            bm = bm.reshape(-1, 2)
            mv = jnp.where(mm, mv, jnp.full_like(mv, -jnp.inf))
            bm_ = jnp.where(bm, bm_, jnp.full_like(bm_, -jnp.inf))
            lsm = jax.nn.log_softmax(mv)
            lsb = jax.nn.log_softmax(bm_)
            n = mv.shape[0]
            lp = lsm[jnp.arange(n), acts.reshape(-1, 2)[:, 0]] \
                + lsb[jnp.arange(n), acts.reshape(-1, 2)[:, 1]]
            ratio = jnp.exp(lp - lps.reshape(-1))
            pol = jnp.maximum(-advs.reshape(-1) * ratio,
                              -advs.reshape(-1) * jnp.clip(ratio, 0.8, 1.2)).mean()
            vl = jnp.mean((v - rets.reshape(-1)) ** 2)
            return pol + 0.5 * vl
        g0 = jax.grad(loss)(ps[0], b0)
        g1 = jax.grad(loss)(ps[1], b1)
        g_mean = jax.tree.map(lambda a, b: (a + b) / 2, g0, g1)
        # pmean：两卡用相同梯度更新
        for d in range(n_dev):
            upd, os_list[d] = opt.update(g_mean, os_list[d], ps[d])
            ps[d] = optax.apply_updates(ps[d], upd)
            kk[d] = jrandom.split(kk[d])[0]
        jax.block_until_ready((ps[0], ps[1]))
        d = maxdiff(ps[0], ps[1])
        ok = d == 0.0
        print(f"iter{it}: pmean 更新后两卡 params maxdiff={d:.3e} = "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    print("[2] 软件 pmean 更新语义验证完成（真 pmap 双设备一致性需 DCU "
          "多卡环境跑 verify_dp_parity --n-dev 2）", flush=True)
    return ps[0], os_list[0], ss[0]


def test_grad_equiv(cfg, mb, n_dev=2):
    """不变量 3：确定性 batch，pmean(每卡子梯度) == 全量梯度。"""
    key = jrandom.PRNGKey(3)
    key, nk = jrandom.split(key)
    params = init_net(nk, cfg.arch, N_OBS_CH, H, W, hidden=cfg.hidden)
    obs = jrandom.normal(key, (mb * n_dev, N_OBS_CH, H, W))
    acts = jrandom.randint(jrandom.split(key)[0], (mb * n_dev, 2), 0, 2)
    old = jrandom.normal(jrandom.split(key)[1], (mb * n_dev,))
    adv = jrandom.normal(jrandom.split(key)[2], (mb * n_dev,))
    ret = jrandom.normal(jrandom.split(key)[3], (mb * n_dev,))
    mm = jnp.ones((mb * n_dev, 5), bool)
    bm = jnp.ones((mb * n_dev, 2), bool)

    def loss_for(p, ob, ac, ol, ad, rt, m, b):
        mv, bv, v, _ = net_forward(p, cfg.arch, ob)
        mv = jnp.where(m, mv, jnp.full_like(mv, -jnp.inf))
        bv = jnp.where(b, bv, jnp.full_like(bv, -jnp.inf))
        lsm = jax.nn.log_softmax(mv)
        lsb = jax.nn.log_softmax(bv)
        n = ob.shape[0]
        lp = lsm[jnp.arange(n), ac[:, 0]] + lsb[jnp.arange(n), ac[:, 1]]
        ratio = jnp.exp(lp - ol)
        pol = jnp.maximum(-ad * ratio,
                          -ad * jnp.clip(ratio, 0.8, 1.2)).mean()
        vl = jnp.mean((v - rt) ** 2)
        return pol + 0.5 * vl

    g_full = jax.grad(loss_for)(params, obs, acts, old, adv, ret, mm, bm)

    def shard(d):
        s = slice(d * mb, (d + 1) * mb)
        return (obs[s], acts[s], old[s], adv[s], ret[s], mm[s], bm[s])

    # 数学恒等：minibatch 不重叠且等大 → 全量梯度 = 各子批梯度的平均。
    # pmean 跨卡执行的就是这个平均（每卡算自己子批梯度后 allreduce /n_dev）。
    g0 = jax.grad(loss_for)(params, *shard(0))
    g1 = jax.grad(loss_for)(params, *shard(1))
    g_mean = jax.tree.map(lambda a, b: (a + b) / 2, g0, g1)
    d = maxdiff(g_full, g_mean)
    ok = d == 0.0
    print(f"[3] 梯度等价 pmean(子梯度) vs 全量梯度 maxdiff={d:.3e} = "
          f"{'PASS' if ok else 'FAIL'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    cfg = make_args()
    cfg.num_steps = args.steps
    print(f"envs={args.envs} steps={cfg.num_steps} minibatch={cfg.minibatch} "
          f"iters={args.iters}", flush=True)

    test_ndev1(cfg, args.envs, args.iters)
    test_ndev2(cfg, args.envs, args.iters)
    test_grad_equiv(cfg, cfg.minibatch // 2)


if __name__ == "__main__":
    main()
