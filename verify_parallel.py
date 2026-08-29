"""danger_map parallel(4-stream) vs 单 stream 位级对拍 + 阶段计时。"""
import sys, time, torch, torch_npu
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
import sim.blast as B

dev = pick_device()
assert dev.startswith("npu"), "只在 910B 跑"

torch.manual_seed(11)
ok = True
for trial in range(8):
    n, h, w = 1024, 11, 13
    fuse = torch.randint(0, 10, (n, h, w), device=dev)
    wall = torch.rand(n, h, w, device=dev) > 0.7
    brick = torch.rand(n, h, w, device=dev) > 0.85
    blast = torch.randint(0, 8, (n, h, w), device=dev)
    for max_chain in (1, 2, 3):
        for cap in (3, 4):
            kw = dict(fuse_max=8, brick=brick, max_chain=max_chain,
                      chain_cap=cap, blast_max_hint=4, exp=2.0)
            a = B.danger_map(fuse, wall, blast, parallel=False, **kw)
            b = B.danger_map(fuse, wall, blast, parallel=True, **kw)
            if not torch.equal(a, b):
                print(f"trial={trial} chain={max_chain} cap={cap}: MISMATCH maxdiff={(a-b).abs().max().item()}")
                ok = False
print("函数级 parallel 对拍:", "PASS" if ok else "FAIL")

# 阶段计时（N=16384 全档 blast=7 场景）
N = 16384
torch.manual_seed(3)
fuse = torch.randint(0, 10, (N, 11, 13), device=dev)
wall = torch.rand(N, 11, 13, device=dev) > 0.7
brick = torch.rand(N, 11, 13, device=dev) > 0.85
blast = torch.full((N, 11, 13), 7, dtype=torch.int64, device=dev)
kw = dict(fuse_max=8, brick=brick, max_chain=3, chain_cap=3, blast_max_hint=7, exp=2.0)
def bench(fn, it=15):
    for _ in range(4): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.npu.synchronize()
    return (time.perf_counter()-t0)/it*1000
ts = bench(lambda: B.danger_map(fuse, wall, blast, parallel=False, **kw))
tp = bench(lambda: B.danger_map(fuse, wall, blast, parallel=True, **kw))
print(f"danger N={N} blast=7: 单stream {ts:6.2f} ms | 4-stream {tp:6.2f} ms | x{ts/tp:.2f}")
