"""调试3b：逐 tick 对比 simA/simB step 后 RNG 状态与状态一致性。"""
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
torch.manual_seed(20260816)
orig = BatchedSim._triton_ok
simA = BatchedSim(cfg, N, device=DEV, seed=0)
simA.reset_all()
BatchedSim._triton_ok = staticmethod(lambda: False)
simB = BatchedSim(cfg, N, device=DEV, seed=0)
simB.reset_all()
BatchedSim._triton_ok = staticmethod(orig)
g = torch.Generator(device=DEV).manual_seed(99)

def actions():
    mmask, bmask = simA.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)

def rng():
    return torch.cuda.get_rng_state()

for t in range(6):
    a = actions()
    ra, _, _ = simA.step(a)
    torch.cuda.synchronize()
    sA = rng().clone()
    rb, _, _ = simB.step(a.clone())
    torch.cuda.synchronize()
    sB = rng().clone()
    print(f"tick {t}: rng_eq={torch.equal(sA, sB)} "
          f"pos_eq={torch.equal(simA.pos, simB.pos)} "
          f"bombscap_eq={torch.equal(simA.bombs_cap, simB.bombs_cap)} "
          f"spd_eq={torch.equal(simA.spd_g, simB.spd_g)} "
          f"crate_eq={torch.equal(simA.crate, simB.crate)}", flush=True)
print("done")
