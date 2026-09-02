"""架构规模化 A/B：不同架构 × 参数量档位，测端到端吞吐 + GPU 功耗/利用率。

背景：当前 mlp_bf16 768（1.6M）端到端 2.1M sps，但稳态功耗仅 83W/1000W，
HCU 88.8% —— ALU 大量闲置（launch/访存受限）。目标验证更大架构
（CNN/ViT、3M/6M 参数）能否吃满算力，且单卡 sps 保持 ≥10 万（论文
端到端口径 ~107K/卡/s）。

用法（DCU2）：
    source ~/jax_env2.sh; cd /root/qqt-gpu-sim
    nohup python3 /root/bench_arch.py --iters 2 > /root/arch_out.txt 2>&1 &
    另开 shell 循环采样 rocm-smi 功耗/利用率。

输出每架构：params / one_iter s / sps / 相对基线倍数。
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from jax_bomb.jax_env import H, W, N_OBS_CH, init_batch
from jax_bomb.jax_net import count_params, init_net
from jax_bomb.jax_train import build_one_iter
from jax_bomb.platform import setup_platform


class A:
    pass


def make_args(arch, hidden, num_envs, num_steps):
    a = A()
    a.arch = arch
    a.hidden = hidden
    a.embed = 192
    a.depth = 4
    a.num_envs = num_envs
    a.num_steps = num_steps
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


# (name, arch, init_kw) —— init_kw 经 init_net 传给架构初始化
# 参数量档位：1.6M 基线 / 3M CNN / 4.8M / 8M ViT / 15M ViT
ARCHES = [
    ("mlp_bf16-1.6M", "mlp_bf16", {"hidden": 768}),
    ("mlp4-3.4M", "mlp4", {"hidden": 768}),
    ("cnn-2.8M", "cnn", {"ch1": 256, "ch2": 512, "hidden": 3072}),
    ("cnn-3.3M", "cnn", {"ch1": 256, "ch2": 512, "hidden": 4096}),
    ("tf-4.8M", "transformer", {"embed": 256, "depth": 6}),
    ("tf-7.5M", "transformer", {"embed": 320, "depth": 6}),
    ("tf-14.6M", "transformer", {"embed": 448, "depth": 6}),
]


def bench_arch(name, arch, kw, num_envs, num_steps, iters):
    cfg = make_args(arch, kw.get("hidden", 768), num_envs, num_steps)
    cfg.embed = kw.get("embed", 192)
    cfg.depth = kw.get("depth", 4)
    key = jrandom.PRNGKey(0)
    states = init_batch(key, num_envs)
    key, nk = jrandom.split(key)
    params = init_net(nk, arch, N_OBS_CH, H, W, **kw)
    n_params = count_params(params)
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    one_iter = build_one_iter(params, opt, opt_state, states, key, cfg)
    frames = 2 * num_envs * num_steps

    # 预热（编译）
    t0 = time.time()
    params, opt_state, states, key = one_iter(params, opt_state, states, key)
    jax.block_until_ready(params)
    compile_s = time.time() - t0

    times = []
    for i in range(iters):
        t0 = time.time()
        params, opt_state, states, key = one_iter(params, opt_state, states, key)
        jax.block_until_ready(params)
        times.append(time.time() - t0)
    dt = min(times)
    sps = frames / dt
    print(f"{name:16s} params={n_params:>9,}  one_iter={dt:.3f}s  "
          f"{sps/1e6:.2f}M sps  (compile {compile_s:.0f}s)",
          flush=True)
    return n_params, dt, sps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--only", default=None,
                    help="只跑指定架构（名字前缀），逗号分隔")
    args = ap.parse_args()

    setup_platform()
    only = set(x.strip() for x in (args.only or "").split(",") if x.strip())
    print(f"envs={args.num_envs} steps={args.num_steps} iters={args.iters} "
          f"minibatch=8192 epochs=1", flush=True)
    for name, arch, kw in ARCHES:
        if only and name.split("-")[0] not in only:
            continue
        try:
            bench_arch(name, arch, kw, args.num_envs, args.num_steps,
                       args.iters)
        except Exception as e:
            print(f"{name}: FAIL {type(e).__name__}: {str(e)[:120]}",
                  flush=True)


if __name__ == "__main__":
    main()
