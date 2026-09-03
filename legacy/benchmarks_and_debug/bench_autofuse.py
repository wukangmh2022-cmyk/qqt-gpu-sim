"""AUTOFUSE_FLAGS（GE 自动融合） + torch.compile(backend='npu') 单段实测。

Hypothesis：之前 backend='npu' 单段 x0.89 慢是因为 GE 融合没开，
AutoFuse 打开后 Elemwise/Broadcast 链被融合 → 设备 kernel 数大降。
"""
import os, sys, time, torch
print("AUTOFUSE_FLAGS =", os.environ.get("AUTOFUSE_FLAGS", "(未设置)"))
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
import sim.blast as B

dev = pick_device()
N = 16384

# danger 单段（sync_free 纯张量，可编译）
torch.manual_seed(5)
fuse = torch.randint(0, 10, (N, 11, 13), device=dev)
wall = torch.rand(N, 11, 13, device=dev) > 0.7
brick = torch.rand(N, 11, 13, device=dev) > 0.85
blast = torch.randint(0, 8, (N, 11, 13), device=dev)
kw = dict(fuse_max=8, brick=brick, max_chain=3, chain_cap=3,
          blast_max_hint=4, exp=2.0)

def danger_wrap(fuse, wall, brick, blast):
    return B.danger_map(fuse, wall, blast, 8, brick=brick, max_chain=3,
                        chain_cap=3, blast_max_hint=4, exp=2.0)

def bench(fn, it=10):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

t_e = bench(lambda: danger_wrap(fuse, wall, brick, blast))
print(f"eager danger: {t_e:7.2f} ms")

try:
    torch._dynamo.reset()
    f = torch.compile(danger_wrap, backend="npu", dynamic=False)
    a = danger_wrap(fuse, wall, brick, blast)
    b = f(fuse, wall, brick, blast)
    torch.npu.synchronize()
    print(f"位级一致: {torch.equal(a, b)}")
    t_c = bench(lambda: f(fuse, wall, brick, blast))
    print(f"npu+autofuse danger: {t_c:7.2f} ms  x{t_e/t_c:.2f}")
except Exception as ex:
    print(f"compile npu FAIL {type(ex).__name__}: {str(ex)[:300]}")
