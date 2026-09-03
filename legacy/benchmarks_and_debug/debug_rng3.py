"""调试3：编译是否消耗默认设备 RNG？

tick 0 时 simA.step 触发 inductor 编译。对比 simA/simB 每 tick step 前后
默认设备 RNG 状态的推进量。若 simA 首次 step 后 RNG 状态推进 > simB，说明
编译路径消耗了随机数 → 解释两条 sim 分歧（RNG 流错位）。
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

def rng_state():
    return torch.cuda.get_rng_state()

# 预对齐：两条 sim 的 step 之前 RNG 状态必须相同（排除 reset 的影响）
s0 = rng_state().clone()
for t in range(3):
    a = actions()
    s_before = rng_state().clone()
    ra, _, _ = simA.step(a)
    torch.cuda.synchronize()
    s_mid = rng_state().clone()
    rb, _, _ = simB.step(a.clone())
    torch.cuda.synchronize()
    s_after = rng_state().clone()
    eq_before = torch.equal(s_before, s_mid)
    advA = s_mid[0].item()  # rng_state[0] = philox offset
    advB = s_after[0].item()
    print(f"tick {t}: eq(s_before,s_mid)={eq_before} "
          f"A_offset={advA} B_offset={advB} deltaA={advA - s0[0].item()} "
          f"deltaB={advB - s0[0].item()} "
          f"pos_eq={torch.equal(simA.pos, simB.pos)} "
          f"bombscap_eq={torch.equal(simA.bombs_cap, simB.bombs_cap)}", flush=True)
    if not eq_before:
        print("  >>> simA.step 推进 RNG 状态 ≠ simB.step 前（编译消耗 RNG 或步数不同）")
        break
print("done")
