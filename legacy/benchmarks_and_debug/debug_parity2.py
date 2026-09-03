"""调试2：两条同 seed sim（级联 vs 强制 eager）逐 tick 找第一个分歧点。

列出所有状态张量属性，每 tick 对比，打印第一个分歧的 (tick, attr, maxdiff)。
"""
import sys
sys.path.insert(0, ".")
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim

DEV = "cuda"
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=1.0, ring_fraction=0.0, hazard_fraction=0.0,
                crate_speed_only=False, timeout_draw=False,
                combo_reward=0.10, combo_gap_factor=0.9)
N = 2048
ticks = 8

torch.manual_seed(20260816)
orig_triton_ok = BatchedSim._triton_ok  # class 访问已解包 staticmethod → 裸函数
simA = BatchedSim(cfg, N, device=DEV, seed=0)
simA.reset_all()
BatchedSim._triton_ok = staticmethod(lambda: False)   # simB 强制 eager
simB = BatchedSim(cfg, N, device=DEV, seed=0)
simB.reset_all()
BatchedSim._triton_ok = staticmethod(orig_triton_ok)  # 还原（级联）
g = torch.Generator(device=DEV).manual_seed(99)

STATE_ATTRS = ["fuse", "wall", "pos", "alive", "hp", "bomb_blast", "owner",
               "brick", "crate", "recycle_crate", "bombs_cap", "blast_cap",
               "spd_g", "winner", "done", "step_count"]
STATEFUL = [a for a in STATE_ATTRS if hasattr(simA, a)]

def actions():
    mmask, bmask = simA.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)

diverge = None
for t in range(ticks):
    a = actions()
    ra, da, ia = simA.step(a)
    rb, db, ib = simB.step(a.clone())
    for attr in STATEFUL:
        va, vb = getattr(simA, attr), getattr(simB, attr)
        if va.dtype.is_floating_point:
            d = (va - vb).abs().max().item()
        elif va.dtype == torch.bool:
            d = 0.0 if torch.equal(va, vb) else 1.0
        else:
            d = 0.0 if torch.equal(va, vb) else 1.0
        if d > 1e-9:
            print(f"tick {t}: attr={attr} maxdiff={d:.3e} "
                  f"dtype={va.dtype} shape={tuple(va.shape)}")
            if diverge is None:
                diverge = (t, attr, d)
    if t < 3 or diverge is None:
        print(f"tick {t}: all equal" + ("" if diverge is None else " (after first divergence)"))
    if diverge is not None and t >= diverge[0]:
        pass
print("first divergence:", diverge)
