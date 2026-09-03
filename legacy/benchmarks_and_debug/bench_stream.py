"""量化 multistream 可行性 + host↔NPU 带宽成本。

1. 独立 op 链（模拟 4 方向传播）：单 stream vs 2/4 stream 真并行度
2. host→device / device→host 拷贝 (16384,13,13) 带宽
"""
import sys, time, torch, torch_npu
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
dev = pick_device()
N = 16384
h, w = 11, 13
shape = (N, h, w)
BYTES = N * h * w * 4

def sync():
    torch.npu.synchronize()

def chain(seed_t, iters=200, mul_shape=shape):
    x = seed_t
    for _ in range(iters):
        x = x * 1.0000001 + 0.0000001
    return x

print("== 1. multistream 并行度 ==")
a = torch.rand(shape, device=dev)
# 单 stream 4 条链（顺序）
t0 = time.perf_counter()
r = [chain(a) for _ in range(4)]
sync()
t_seq = (time.perf_counter() - t0) * 1000
print(f"单stream 4链顺序: {t_seq:7.2f} ms")

# 2 stream
try:
    s1 = torch.npu.Stream(); s2 = torch.npu.Stream()
    t0 = time.perf_counter()
    ev = torch.npu.Event()
    with torch.npu.stream(s1):
        r1 = chain(a, 200)
        s1.record_event(ev)
    with torch.npu.stream(s2):
        r2 = chain(a, 200)
        r3 = chain(a, 200)
        r4 = chain(a, 200)
    sync()
    t2 = (time.perf_counter() - t0) * 1000
    print(f"2stream(1+3链): {t2:7.2f} ms  speedup x{t_seq/t2:.2f}")
except Exception as ex:
    print(f"2stream FAIL: {type(ex).__name__}: {str(ex)[:150]}")

# 4 stream（各 1 链）
try:
    streams = [torch.npu.Stream() for _ in range(4)]
    t0 = time.perf_counter()
    for i, s in enumerate(streams):
        with torch.npu.stream(s):
            r = chain(a, 200)
    sync()
    t4 = (time.perf_counter() - t0) * 1000
    print(f"4stream(各1链): {t4:7.2f} ms  speedup x{t_seq/t4:.2f}")
except Exception as ex:
    print(f"4stream FAIL: {type(ex).__name__}: {str(ex)[:150]}")

print("\n== 2. host↔NPU 带宽（(16384,13,13) = %.1f MB）==" % (BYTES/1e6))
hst = torch.rand(shape, dtype=torch.float32)
dev_t = torch.empty(shape, dtype=torch.float32, device=dev)
# host→device
for _ in range(3): dev_t.copy_(hst)
sync()
t0 = time.perf_counter()
for _ in range(10): dev_t.copy_(hst)
sync()
t_h2d = (time.perf_counter() - t0) / 10 * 1000
print(f"host→device: {t_h2d:7.2f} ms ({BYTES/t_h2d/1e6*1e3/1e3:.0f} GB/s 理论 1.02e3)")

for _ in range(3): hst.copy_(dev_t)
sync()
t0 = time.perf_counter()
for _ in range(10): hst.copy_(dev_t)
sync()
t_d2h = (time.perf_counter() - t0) / 10 * 1000
print(f"device→host: {t_d2h:7.2f} ms ({BYTES/t_d2h/1e6*1e3/1e3:.0f} GB/s)")

# CPU 侧同形状 op 成本（对比：搬 CPU 算值不值）
cpu_t = torch.rand(shape, dtype=torch.float32)
t0 = time.perf_counter()
for _ in range(20):
    cpu_t = cpu_t * 1.0000001 + 0.0000001
print(f"CPU 200 op (16384,13,13): {(time.perf_counter()-t0)/20*1000:7.2f} ms")
