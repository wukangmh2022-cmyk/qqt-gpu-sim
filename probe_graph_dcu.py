"""DCU 可行性探测：torch.cuda.CUDAGraph capture + replay 是否可用。

含 device RNG 是否阻塞 capture、capture 内 host 同步是否报错。
"""
import sys, time
sys.path.insert(0, ".")
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
dev = "cuda"

# 1) 基础 capture：纯张量链
def body(x, y):
    z = (x * y + x).relu()
    return z.sum()
x = torch.randn(4096, 13, 13, device=dev)
y = torch.randn(4096, 13, 13, device=dev)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
try:
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body(x, y)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        out = body(x, y)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ref = body(x, y)
    torch.cuda.synchronize()
    print("basic capture: OK, equal:", torch.equal(out, ref))
except Exception as e:
    print("basic capture FAILED:", type(e).__name__, str(e)[:300])

# 2) capture 内调用 torch.rand（device RNG）—— 模拟 step 的 crate RNG
x2 = torch.randn(4096, 13, 13, device=dev)
g2 = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g2):
        r = torch.rand(4096, device=dev)
        out2 = x2 * r.view(-1, 1, 1)
    torch.cuda.synchronize()
    g2.replay()
    torch.cuda.synchronize()
    print("capture with device rand: OK (replay deterministic?)")
    # replay 两次是否得到相同 rand？graph 回放会用捕获时固定的随机数
    g2.replay()
    torch.cuda.synchronize()
    g2.replay()
    torch.cuda.synchronize()
    print("  replay x2 equal:", torch.equal(x2 * torch.rand(4096, device=dev).view(-1,1,1), x2))
except Exception as e:
    print("capture with device rand FAILED:", type(e).__name__, str(e)[:300])

# 3) capture 内 .item()/bool()（host 同步）—— 模拟 step 的 blast_hint/placed.any()
x3 = torch.randn(4096, 13, 13, device=dev)
g3 = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g3):
        v = x3.max().item()   # host 同步
        out3 = x3 * v
    torch.cuda.synchronize()
    print("capture with .item(): OK")
except Exception as e:
    print("capture with .item() FAILED:", type(e).__name__, str(e)[:300])

# 4) 计时：eager 循环 vs graph replay（纯张量链，20 个 kernel 规模）
def chain(xx):
    a = xx * 2
    for _ in range(19):
        a = (a + xx).relu()
    return a
def bench(fn, reps=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000
x4 = torch.randn(4096, 13, 13, device=dev)
te = bench(lambda: chain(x4))
g4 = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g4):
        out4 = chain(x4)
    torch.cuda.synchronize()
    tg = bench(g4.replay)
    print(f"20-kernel chain: eager {te:.3f}ms  graph-replay {tg:.3f}ms  "
          f"speedup {te/tg:.2f}x")
except Exception as e:
    print("graph bench FAILED:", type(e).__name__, str(e)[:300])
