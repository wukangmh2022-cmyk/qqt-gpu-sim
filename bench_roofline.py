"""Roofline 分析：one_iter 阶段拆分 + DCU 峰值带宽/算力实测。

用法（远程 DCU2）：
    source ~/jax_env2.sh
    python3 bench_roofline.py [--num-envs 8192 --num-steps 256]

输出：
  1) one_iter 阶段拆分（collect/update/overhead）
  2) 纯 env step 吞吐（无网络）
  3) DCU 实测峰值内存带宽（大张量拷贝）
  4) DCU 实测峰值算力（稠密 GEMM fp32）
"""
import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W, N_OBS_CH, init_batch, step
from jax_bomb.jax_net import init_net
from jax_bomb.platform import device_summary, setup_platform
from jax_bomb.jax_train import (build_one_iter, collect_rollout, compute_gae,
                                both_perspectives, both_masks,
                                sample_actions, ppo_update)


def make_args(num_envs, num_steps):
    class A:
        pass
    a = A()
    a.arch = "mlp_bf16"
    a.num_envs = num_envs
    a.num_steps = num_steps
    a.hidden = 768
    a.embed = 192
    a.depth = 4
    a.minibatch = 8192
    a.epochs = 1
    a.lr = 3e-4
    a.gamma = 0.995
    a.lam = 0.95
    a.clip_eps = 0.2
    a.vf_coef = 0.5
    a.ent_coef = 0.01
    a.no_mask = False
    return a


def bench_one_iter(num_envs=8192, num_steps=256, iters=4):
    import optax
    args = make_args(num_envs, num_steps)
    key = jrandom.PRNGKey(0)
    states = init_batch(key, num_envs)
    key, net_key = jrandom.split(key)
    params = init_net(net_key, args.arch, N_OBS_CH, H, W, hidden=args.hidden)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    one_iter = build_one_iter(params, opt, opt_state, states, key, args)

    print(f"== one_iter（真实训练路径，整体 jit）n={num_envs} "
          f"steps={num_steps} ==", flush=True)
    frames = 2 * num_envs * num_steps   # 双视角帧
    for i in range(iters):
        t0 = time.perf_counter()
        params, opt_state, states, key = one_iter(params, opt_state, states, key)
        jax.block_until_ready(params)
        dt = time.perf_counter() - t0
        print(f"  iter{i}: {dt:.3f}s  -> {frames/dt/1e6:.2f}M sps", flush=True)


def bench_pure_env(num_envs=8192, ticks=512):
    print(f"\n== 纯 env step（无网络） n={num_envs} ticks={ticks} ==", flush=True)
    key = jrandom.PRNGKey(1)
    states = init_batch(key, num_envs)
    acts = jnp.zeros((num_envs, 2, 2), jnp.int32)
    step_j = jax.jit(jax.vmap(step))
    keys = jrandom.split(key, num_envs)
    states, _ = step_j(states, acts, keys)   # 预热
    t0 = time.perf_counter()
    for _ in range(ticks):
        keys = jrandom.split(key, num_envs)
        states, _ = step_j(states, acts, keys)
    dt = time.perf_counter() - t0
    print(f"  {num_envs*ticks/dt/1e6:.2f}M 步/s", flush=True)
    return states


def bench_bw_and_flops():
    import torch
    print("\n== DCU 实测峰值（torch） ==", flush=True)
    dev = "cuda"
    n = 1 << 28                      # 1Gi 元素 = 4GiB fp32
    a = torch.ones(n, dtype=torch.float32, device=dev)
    b = torch.ones(n, dtype=torch.float32, device=dev)
    c = torch.empty_like(a)
    c.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bw_copy = (2 * n * 4) / (dt / 10) / 1e9
    print(f"  copy 带宽: {bw_copy:.0f} GB/s", flush=True)
    t0 = time.perf_counter()
    for _ in range(10):
        torch.add(a, b, out=c)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bw_add = (3 * n * 4) / (dt / 10) / 1e9
    print(f"  add  带宽: {bw_add:.0f} GB/s", flush=True)
    del a, b, c
    torch.cuda.empty_cache()

    k = 4096
    x = torch.randn(k, k, device=dev)
    y = torch.randn(k, k, device=dev)
    z = torch.empty(k, k, device=dev)
    z.copy_(x @ y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        z.copy_(x @ y)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * k ** 3 * 5 / dt / 1e12
    print(f"  GEMM {k}^3 fp32: {tflops:.2f} TFLOPS", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    devs = setup_platform()
    print(f"devices: {device_summary(devs)}", flush=True)

    bench_pure_env(args.num_envs, 512)
    bench_one_iter(args.num_envs, args.num_steps)
    if not args.skip_torch:
        try:
            bench_bw_and_flops()
        except Exception as e:
            print(f"  torch bench 跳过: {e}", flush=True)