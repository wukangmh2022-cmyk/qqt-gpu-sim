"""DCU 峰值带宽/算力实测（torch）。独立跑，不与 jax bench 并发抢显存。"""
import time
import torch


def main():
    dev = "cuda"
    print(torch.cuda.get_device_name(0), flush=True)
    n = 1 << 28                      # 1Gi 元素 = 4GiB fp32
    a = torch.ones(n, dtype=torch.float32, device=dev)
    b = torch.ones(n, dtype=torch.float32, device=dev)
    c = torch.empty_like(a)
    c.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        c.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bw_copy = (2 * n * 4) / (dt / 20) / 1e9
    print(f"copy 带宽: {bw_copy:.0f} GB/s", flush=True)
    t0 = time.perf_counter()
    for _ in range(20):
        torch.add(a, b, out=c)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bw_add = (3 * n * 4) / (dt / 20) / 1e9
    print(f"add  带宽: {bw_add:.0f} GB/s", flush=True)
    del a, b, c
    torch.cuda.empty_cache()

    for k in (2048, 4096):
        x = torch.randn(k, k, device=dev)
        y = torch.randn(k, k, device=dev)
        z = torch.empty(k, k, device=dev)
        z.copy_(x @ y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            z.copy_(x @ y)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tflops = 2 * k ** 3 * 10 / dt / 1e12
        print(f"GEMM {k}^3 fp32: {tflops:.2f} TFLOPS", flush=True)
        del x, y, z


if __name__ == "__main__":
    main()
