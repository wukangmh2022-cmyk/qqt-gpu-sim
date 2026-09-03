"""danger 阶段 B 4-stream 并行原型：位级 + 加速对比。

阶段 B 结构：4 方向各自从同一 fw/fd 出发传播 max_b 步，最后 max 合并
→ 天然独立，可 4 stream 并行（每方向写自己的 danger_i）。
"""
import sys, time, torch, torch_npu
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
import sim.blast as B

dev = pick_device()
N = 16384
h, w = 11, 13

torch.manual_seed(7)
fuse = torch.randint(0, 10, (N, h, w), device=dev)
wall = torch.rand(N, h, w, device=dev) > 0.7
brick = torch.rand(N, h, w, device=dev) > 0.85
blast = torch.randint(0, 8, (N, h, w), device=dev)
bombed = fuse > 0
passable = (~wall).float()
not_solid = (~(bombed | brick)).float()
one_buf = torch.ones_like(passable)

# weight（与 danger_map 同款）：fuse 权重
w_raw = 1.0 - (fuse.float() - 1.0) / 8.0
weight = torch.where(fuse > 0, w_raw.clamp_min(0.0).pow(2.0),
                     torch.zeros_like(fuse, dtype=torch.float32))
blast_f = blast.float()
max_b = max(1, int(blast.max()))

# ---- 单 stream 原版（阶段 B） ----
def stageB_single():
    seed = weight * passable
    danger = seed.clone()
    fw = seed.clone()
    fd = torch.where(bombed, blast_f, torch.zeros_like(seed))
    for drow, dcol in B._DIRS:
        fw_p, fd_p = fw, fd
        for _ in range(max_b):
            fw1 = B._shift(fw_p, drow, dcol) * passable
            fd1 = B._shift(fd_p, drow, dcol) * passable
            fd1 = fd1 - one_buf
            keep = fd1 >= 0
            fw1 = fw1 * keep
            danger = torch.maximum(danger, fw1, out=danger)
            fw1 = fw1 * not_solid
            fd1 = fd1 * not_solid
            fw_p, fd_p = fw1, fd1
    return danger

# ---- 4 stream 版本 ----
streams = [torch.npu.Stream() for _ in range(4)]
evs = [torch.npu.Event() for _ in range(4)]

def _dir_run(i, fw0, fd0, out):
    drow, dcol = B._DIRS[i]
    with torch.npu.stream(streams[i]):
        fw_p, fd_p = fw0, fd0
        for _ in range(max_b):
            fw1 = B._shift(fw_p, drow, dcol) * passable
            fd1 = B._shift(fd_p, drow, dcol) * passable
            fd1 = fd1 - one_buf
            keep = fd1 >= 0
            fw1 = fw1 * keep
            torch.maximum(out, fw1, out=out)
            fw_p = fw1 * not_solid
            fd_p = fd1 * not_solid
        streams[i].record_event(evs[i])

def stageB_parallel():
    seed = weight * passable
    danger = seed.clone()
    fw = seed.clone()
    fd = torch.where(bombed, blast_f, torch.zeros_like(seed))
    outs = [danger.clone() for _ in range(4)]
    cur = torch.npu.current_stream()
    for i in range(4):
        _dir_run(i, fw, fd, outs[i])
    for i in range(4):
        cur.wait_event(evs[i])
    d = outs[0]
    for i in range(1, 4):
        d = torch.maximum(d, outs[i])
    return d

# 位级对拍
a = stageB_single()
b = stageB_parallel()
torch.npu.synchronize()
print(f"位级一致: {torch.equal(a, b)}  (max_b={max_b})")

def bench(fn, it=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

t1 = bench(stageB_single)
t4 = bench(stageB_parallel)
print(f"阶段B 单stream: {t1:7.2f} ms | 4stream: {t4:7.2f} ms | x{t1/t4:.2f}")
