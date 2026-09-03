"""AUTOFUSE_FLAGS + torch.compile(backend='npu') 整步编译实测。

move 临时换 torch 版（绕开 triton_kernel_wrapper_functional 转换失败），
graph-break 段用 eager，非 break 子图 GE 融合。位级对拍 + 稳态计时。
"""
import os, sys, time, torch
print("AUTOFUSE_FLAGS =", os.environ.get("AUTOFUSE_FLAGS", "(未设置)"))
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
import sim.torch_sim as TS

dev = pick_device()
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def make_sim():
    torch.manual_seed(0)
    s = TS.BatchedSim(cfg, N, device=dev, seed=0)
    s.reset_all()
    return s

mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

def bench(fn, it=8):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

# eager 基准（torch move）
TS._HAS_TRITON = False
sim_e = make_sim()
t_e = bench(lambda: sim_e.step(acts))
print(f"eager step (torch move): {t_e:7.2f} ms  ({N/t_e*1e3/1e4:.2f}万 SPS)")

# 整步编译
try:
    torch._dynamo.reset()
    sim_c = make_sim()
    compiled = torch.compile(sim_c.step, backend="npu", dynamic=False)
    # 位级对拍 3 tick
    torch.manual_seed(0)
    sim_a = make_sim()
    sim_b = make_sim()
    ok = True
    for t in range(3):
        ra, da, _ = sim_a.step(acts)
        rb, db, _ = compiled(acts)
        if not torch.equal(ra, rb) or not torch.equal(da, db):
            print(f"  t={t} MISMATCH ndiff={(ra!=rb).sum().item()}  maxdiff={(ra-rb).abs().max().item():.2e}")
            ok = False
            break
    print(f"整步编译 位级一致(3 tick) = {ok}")
    t_c = bench(lambda: compiled(acts))
    print(f"compiled step (npu+autofuse): {t_c:7.2f} ms  ({N/t_c*1e3/1e4:.2f}万 SPS)  x{t_e/t_c:.2f}")
except Exception as ex:
    print(f"整步编译 FAIL {type(ex).__name__}: {str(ex)[:250]}")
