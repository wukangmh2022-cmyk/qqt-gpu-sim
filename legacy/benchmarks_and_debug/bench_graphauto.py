"""graph-autofusion（torch.npu.enable_graph_mode）实测：小算子链自动建图+融合。

测：
1. danger 单段（sync_free 纯张量，无 host 同步）graph 模式 vs eager
2. 完整 step graph 模式 vs eager（含 host 同步，看能否自动子图化）
位级对拍 + 稳态计时。
"""
import sys, time, torch, torch_npu
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.dev import pick_device
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
import sim.blast as B

dev = pick_device()
assert dev.startswith("npu")
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def bench(fn, it=8):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

# ---- 1. danger 单段 ----
torch.manual_seed(5)
fuse = torch.randint(0, 10, (N, 11, 13), device=dev)
wall = torch.rand(N, 11, 13, device=dev) > 0.7
brick = torch.rand(N, 11, 13, device=dev) > 0.85
blast = torch.randint(0, 8, (N, 11, 13), device=dev)
kw = dict(fuse_max=8, brick=brick, max_chain=3, chain_cap=3,
          blast_max_hint=4, exp=2.0, parallel=False)

a_eager = B.danger_map(fuse, wall, blast, **kw)
t_eager = bench(lambda: B.danger_map(fuse, wall, blast, **kw))
torch.npu.synchronize()

print("== 1. danger 单段 ==")
print(f"eager: {t_eager:7.2f} ms")
try:
    torch.npu.enable_graph_mode()
    # 首调用建图（编译）
    a_graph = B.danger_map(fuse, wall, blast, **kw)
    torch.npu.synchronize()
    t_graph = bench(lambda: B.danger_map(fuse, wall, blast, **kw))
    torch.npu.disable_graph_mode()
    torch.npu.synchronize()
    eq = torch.equal(a_eager, a_graph)
    md = (a_eager - a_graph).abs().max().item() if not eq else 0.0
    print(f"graph-autofusion: {t_graph:7.2f} ms 位级一致={eq} maxdiff={md:.2e} x{t_eager/t_graph:.2f}")
except Exception as ex:
    torch.npu.disable_graph_mode()
    print(f"graph danger FAIL {type(ex).__name__}: {str(ex)[:200]}")

# ---- 2. 完整 step ----
torch.manual_seed(0)
sim_e = BatchedSim(cfg, N, device=dev, seed=0); sim_e.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
t_step_e = bench(lambda: sim_e.step(acts))
torch.npu.synchronize()
re0, de0, _ = sim_e.step(acts)

print("\n== 2. 完整 step ==")
print(f"eager: {t_step_e:7.2f} ms ({N/t_step_e*1e3/1e4:.2f}万 SPS)")
try:
    torch.manual_seed(0)
    sim_g = BatchedSim(cfg, N, device=dev, seed=0); sim_g.reset_all()
    torch.npu.enable_graph_mode()
    r0, d0, _ = sim_g.step(acts)     # 建图
    torch.npu.synchronize()
    t_step_g = bench(lambda: sim_g.step(acts))
    torch.npu.disable_graph_mode()
    torch.npu.synchronize()
    eq = torch.equal(re0, r0) and torch.equal(de0, d0)
    md = (re0 - r0).abs().max().item() if not eq else 0.0
    print(f"graph-autofusion: {t_step_g:7.2f} ms 位级一致={eq} maxdiff={md:.2e} x{t_step_e/t_step_g:.2f}")
except Exception as ex:
    torch.npu.disable_graph_mode()
    print(f"graph step FAIL {type(ex).__name__}: {str(ex)[:200]}")
