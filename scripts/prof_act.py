#!/usr/bin/env python3
"""learner.act 的 DCU/HIP 开销定位（纯网络，不碰模拟器）。

只测 ActorCritic.act 的前向 + Categorical 采样，torch profiler 拿 kernel
分布。n 默认 1024（小规模，几秒，不抢训练 GPU 算力 —— 用户允许简单测试）。
用法：python -m scripts.prof_act --n 1024 --arch mlp
"""
import argparse
import time

import torch

from train.model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prof_act] device={dev} n={args.n} arch={args.arch}")
    net = ActorCritic((14, 13, 13), arch=args.arch, n_players=2).to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    obs = torch.randn(args.n, 14, 13, 13, device=dev, dtype=torch.float16)
    mm = torch.ones(args.n, 5, dtype=torch.bool, device=dev)
    bm = torch.ones(args.n, 2, dtype=torch.bool, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            net.act(obs, mm, bm, 0)
    if dev == "cuda":
        torch.cuda.synchronize()

    # 分段：前向 vs 采样
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            net.act(obs, mm, bm, 0)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"act 总: {(t1-t0)/20*1000:.2f} ms/次（n={args.n}）")

    if not args.profile:
        return
    import torch.profiler as prof
    acts = [prof.ProfilerActivity.CUDA] if dev == "cuda" else [prof.ProfilerActivity.CPU]
    with prof.profile(activities=acts, record_shapes=True) as p:
        with torch.no_grad():
            for _ in range(20):
                net.act(obs, mm, bm, 0)
        if dev == "cuda":
            torch.cuda.synchronize()
    print("\n=== top-30（self CPU）===")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
    if dev == "cuda":
        print("\n=== top-20（CUDA device）===")
        print(p.key_averages().table(sort_by="self_device_time_total", row_limit=20))


if __name__ == "__main__":
    main()
