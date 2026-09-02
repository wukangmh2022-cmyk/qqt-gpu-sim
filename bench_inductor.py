"""方案一：torch.compile(backend='inductor') 在 910B 上的实测。

对比 eager vs inductor：
1. 危险段 danger_map 单独编译（最大单项 ~10ms）
2. reward 段（纯张量部分）
3. 整个 step 编译（预期 host 分支 graph-break，看实际收益）
位级对拍 eager vs compiled。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
N = 16384

torch.manual_seed(0)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

# ---- 编译对象：danger_map（sync_free 纯张量段） ----
import sim.blast as B

def danger_wrap(fuse, wall, brick, blast, fuse_max, max_chain, cap, hint, exp):
    return B.danger_map(fuse, wall, blast, fuse_max, brick=brick,
                        max_chain=max_chain, chain_cap=cap,
                        blast_max_hint=hint, exp=exp)

def bench(fn, it=10):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

def bench_t(fn, it=10):
    return bench(fn, it)

# 0) eager danger 基准
def danger_eager():
    return danger_wrap(sim.fuse, sim.wall, sim.brick, sim.bomb_blast,
                       8, cfg.max_chain, 3, 4, 2.0)

print("== inductor 可用性检查 ==")
try:
    from torch._inductor import config as _ic  # noqa
    print("inductor import OK")
    # 昇腾 inductor 需要 torch_npu 适配；backend 列表
    for bk in ("inductor", "npu"):
        try:
            torch._dynamo.reset()
            f = torch.compile(danger_wrap, backend=bk, dynamic=False)
            _ = f(sim.fuse, sim.wall, sim.brick, sim.bomb_blast, 8,
                  cfg.max_chain, 3, 4, 2.0)
            torch.npu.synchronize()
            print(f"backend={bk}: compile OK")
        except Exception as ex:
            print(f"backend={bk}: FAIL {type(ex).__name__}: {str(ex)[:200]}")
except Exception as ex:
    print("inductor import FAIL:", str(ex)[:200])

# 1) danger 单段：eager vs inductor/npu
print("\n== danger_map 单段 ==")
t_e = bench_t(danger_eager, 8)
print(f"eager:    {t_e:7.2f} ms/step-danger")
for bk in ("inductor", "npu"):
    try:
        torch._dynamo.reset()
        f = torch.compile(danger_wrap, backend=bk, dynamic=False)
        # 预热 + 位级对拍
        a = danger_eager()
        b = f(sim.fuse, sim.wall, sim.brick, sim.bomb_blast, 8,
              cfg.max_chain, 3, 4, 2.0)
        torch.npu.synchronize()
        eq = torch.equal(a, b)
        t_c = bench_t(lambda: f(sim.fuse, sim.wall, sim.brick,
                                sim.bomb_blast, 8, cfg.max_chain, 3, 4, 2.0), 8)
        print(f"{bk:9s}: {t_c:7.2f} ms/step-danger  位级一致={eq}  x{t_e/t_c:.2f}")
    except Exception as ex:
        print(f"{bk:9s}: FAIL {type(ex).__name__}: {str(ex)[:160]}")

# 2) 整个 step 编译（backend='npu'，稳态 + 位级）
print("\n== 整个 step（backend='npu'）==")
import sim.torch_sim as TS

def step_eager(a):
    return sim.step(a)

try:
    torch._dynamo.reset()
    compiled_step = torch.compile(sim.step, backend="npu", dynamic=False)
    # 位级对拍：新 sim
    torch.manual_seed(0)
    s1 = TS.BatchedSim(cfg, N, device=dev, seed=0); s1.reset_all()
    torch.manual_seed(0)
    s2 = TS.BatchedSim(cfg, N, device=dev, seed=0); s2.reset_all()
    ok = True
    for t in range(3):
        r1, d1, i1 = s1.step(acts)
        r2, d2, i2 = compiled_step(acts)
        if not torch.equal(r1, r2) or not torch.equal(d1, d2):
            print(f"  t={t} MISMATCH reward ndiff={(r1!=r2).sum().item()}")
            ok = False
            break
    print(f"  位级一致(3 tick)={ok}")
    # 稳态计时（同一 sim 上）
    t_e2 = bench_t(lambda: sim.step(acts), 6)
    t_c2 = bench_t(lambda: compiled_step(acts), 6)
    print(f"  eager: {t_e2:7.2f} ms/step  |  npu: {t_c2:7.2f} ms/step  x{t_e2/t_c2:.2f}")
except Exception as ex:
    print(f"  step compile FAIL {type(ex).__name__}: {str(ex)[:200]}")
