"""danger 分档编译：max_b∈{1..7} 每档 torch.compile(backend='npu') + AutoFuse。

验证每档位级一致 + 时间。编译按调用 shape 缓存（N=16384 一份）。
"""
import os, sys, time, torch
print("AUTOFUSE_FLAGS =", os.environ.get("AUTOFUSE_FLAGS", "(未设置)"))
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
import sim.blast as B

dev = pick_device()
N = 16384

torch.manual_seed(5)
fuse = torch.randint(0, 10, (N, 11, 13), device=dev)
wall = torch.rand(N, 11, 13, device=dev) > 0.7
brick = torch.rand(N, 11, 13, device=dev) > 0.85

FUSE_MAX, MAX_CHAIN, CAP, EXP = 8, 3, 3, 2.0

def eager_d(fuse, wall, brick, blast, mb):
    return B.danger_map(fuse, wall, blast, FUSE_MAX, brick=brick,
                        max_chain=MAX_CHAIN, chain_cap=CAP,
                        blast_max_hint=mb, exp=EXP)

def bench(fn, it=12):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

cache = {}
all_ok = True
for mb in (1, 2, 3, 4, 7):
    blast = torch.randint(0, mb + 1, (N, 11, 13), device=dev)
    # eager 基准 + 位级参照
    a = eager_d(fuse, wall, brick, blast, mb)
    t_e = bench(lambda: eager_d(fuse, wall, brick, blast, mb))
    # 编译
    torch._dynamo.reset()
    def _d(fuse, wall, brick, blast, _mb=mb):
        return B.danger_map(fuse, wall, blast, FUSE_MAX, brick=brick,
                            max_chain=MAX_CHAIN, chain_cap=CAP,
                            blast_max_hint=_mb, exp=EXP)
    try:
        f = torch.compile(_d, backend="npu", dynamic=False)
        b = f(fuse, wall, brick, blast)
        torch.npu.synchronize()
        eq = torch.equal(a, b)
        md = (a - b).abs().max().item() if not eq else 0.0
        t_c = bench(lambda: f(fuse, wall, brick, blast))
        print(f"max_b={mb}: eager {t_e:6.2f} ms | compiled {t_c:6.2f} ms | "
              f"x{t_e/t_c:.2f} | 位级一致={eq} maxdiff={md:.2e}")
        all_ok &= eq
    except Exception as ex:
        print(f"max_b={mb}: FAIL {type(ex).__name__}: {str(ex)[:150]}")
print("ALL:", "PASS" if all_ok else "FAIL")
