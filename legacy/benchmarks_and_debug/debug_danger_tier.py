"""调试：N=2048 corridor+open 下 _danger_c（inductor）vs eager danger_map。

逐 tick：同一状态上直接对比两者输出（同 tensors，不算两条 sim），
打印 max_b（档位键）与 maxdiff；并打印 _dng_tier 的档位集合。
"""
import sys, time
sys.path.insert(0, ".")
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
import sim.blast as B

DEV = "cuda"
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=1.0, ring_fraction=0.0, hazard_fraction=0.0,
                crate_speed_only=False, timeout_draw=False,
                combo_reward=0.10, combo_gap_factor=0.9)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 30

torch.manual_seed(20260816)
sim = BatchedSim(cfg, N, device=DEV, seed=0)
sim.reset_all()
g = torch.Generator(device=DEV).manual_seed(99)

def acts():
    mmask, bmask = sim.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)

print(f"N={N} cfg.blast={cfg.blast} growth_blast_max={cfg.growth_blast_max} "
      f"max_chain={cfg.max_chain} chain_cap_rounds={cfg.chain_cap_rounds}", flush=True)
tier_seen = {}
first_compile_ms = None
for t in range(ticks):
    a = acts()
    if t == 2:
        t0 = time.time()
    sim.step(a)
    if t == 2:
        first_compile_ms = (time.time() - t0) * 1000
    mb = max(int(sim.bomb_blast.max()), int(cfg.blast))
    tier_seen[mb] = tier_seen.get(mb, 0) + 1
    # 同 tensors 上直接对拍：编译版 vs eager danger_map
    bm = sim._blast_map()
    dng_comp = sim._danger_c(sim.fuse, sim.wall, bm, sim.brick, mb)
    dng_eager = B.danger_map(sim.fuse, sim.wall, bm, cfg.fuse,
                             brick=sim.brick, max_chain=cfg.max_chain,
                             early_exit=False, blast_max_hint=mb,
                             chain_cap=cfg.chain_cap_rounds)
    md = (dng_comp - dng_eager).abs().max().item()
    if t < 5 or md > 1e-6:
        print(f"tick {t:2d} max_b={mb} danger_maxdiff={md:.3e} "
              f"tier_keys={sorted(sim._dng_tier.keys())}", flush=True)
print("tier_seen:", sorted(tier_seen.items()), flush=True)
print(f"first-compile-window(step@tick2)={first_compile_ms:.0f}ms", flush=True)
