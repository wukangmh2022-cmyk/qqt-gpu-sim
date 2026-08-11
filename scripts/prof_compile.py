#!/usr/bin/env python3
"""torch.compile 在 DCU/HIP 上的可行性 + 加速测试。

测一段"danger stage B 传播链"结构（4 方向 pad + maximum 循环）的 compile
融合效果。HIP inductor 支持是 100k 路径的关键未知。
用法：python -m scripts.prof_compile --n 512
"""
import argparse
import time

import torch


def spread_chain(x, passable, not_solid, max_b=7):
    out = x.clone()
    for _ in range(max_b):
        y = torch.nn.functional.pad(x[:, 1:, :], (0, 0, 1, 0)) * passable
        z = torch.nn.functional.pad(x[:, :, 1:], (0, 1, 0, 0)) * passable
        x = (y + z) * not_solid
        out = torch.maximum(out, x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prof_compile] device={dev} n={args.n}")
    x = torch.rand(args.n, 13, 13, device=dev)
    p = torch.rand(args.n, 13, 13, device=dev)
    ns = torch.rand(args.n, 13, 13, device=dev)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        spread_chain(x, p, ns)
    sync()
    t0 = time.perf_counter()
    for _ in range(30):
        spread_chain(x, p, ns)
    sync()
    t1 = time.perf_counter()
    print(f"eager: {(t1-t0)/30*1000:.2f} ms/次")

    try:
        cf = torch.compile(spread_chain)
        for _ in range(3):
            cf(x, p, ns)
        sync()
        t0 = time.perf_counter()
        for _ in range(30):
            cf(x, p, ns)
        sync()
        t1 = time.perf_counter()
        print(f"compile: {(t1-t0)/30*1000:.2f} ms/次")
        # 正确性：compile vs eager 输出一致
        with torch.no_grad():
            r1 = spread_chain(x, p, ns)
            r2 = cf(x, p, ns)
        print(f"  compile==eager: {bool(torch.equal(r1, r2))}")
    except Exception as e:
        print(f"compile FAILED: {type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
