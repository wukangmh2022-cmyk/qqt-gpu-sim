"""调试3c：对照实验——两条 sim 相同配置（都 eager / 都编译），看 rng_eq。

若都 eager 时 rng_eq=True、都编译时 rng_eq=True，说明"编译 vs eager 混合"
才导致 RNG 流不同（编译路径消耗随机数）→ 单 sim 生产路径无影响。
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

def run(modeA, modeB, ticks=6, tag=""):
    torch.manual_seed(20260816)
    def mk(on):
        s = BatchedSim(cfg, N, device=DEV, seed=0)
        s.reset_all()
        return s
    orig = BatchedSim._triton_ok
    simA = mk(modeA)
    BatchedSim._triton_ok = staticmethod(lambda: not modeB)
    simB = mk(modeB)
    BatchedSim._triton_ok = staticmethod(orig)
    g = torch.Generator(device=DEV).manual_seed(99)
    def actions():
        mmask, bmask = simA.legal_mask()
        mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
        bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
        return torch.stack([mv, bm], dim=-1)
    line = f"[{tag} A={modeA} B={modeB}]"
    for t in range(ticks):
        a = actions()
        simA.step(a)
        torch.cuda.synchronize()
        sA = torch.cuda.get_rng_state().clone()
        simB.step(a.clone())
        torch.cuda.synchronize()
        sB = torch.cuda.get_rng_state().clone()
        eq = torch.equal(sA, sB) and torch.equal(simA.bombs_cap, simB.bombs_cap) \
            and torch.equal(simA.pos, simB.pos)
        print(f"{line} tick {t}: rng+state_eq={eq}", flush=True)
    print(line, "DONE", flush=True)

# 两组对照：eager/eager 与 compiled/compiled
run(True, True, tag="both-eager")
run(False, False, tag="both-compiled")
run(False, True, tag="mixed")
