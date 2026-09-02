"""验证：resolve/danger 固定轮 cap 在 DCU 训练分布内的逐位一致性，并扫描 danger 链深。

在真实 gameplay 的 ticks 上逐 tick 对比：
1) resolve_explosions：旧（early_exit=True, chain_cap=None）vs 新（固定 cap）
2) 观测 danger：新（_danger_c，cap 扫描）vs 旧（动态 early_exit）
找到 danger 无差异的最小 cap（DCU 分布实测链深）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.blast import danger_map, resolve_explosions

cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0,
                crate_speed_only=False, timeout_draw=False,
                combo_reward=0.10, combo_gap_factor=0.9)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5632
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 60
caps = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 4, 6, 8]
dev = "cuda"
torch.manual_seed(20260816)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
g = torch.Generator(device=dev).manual_seed(99)

def acts():
    mmask, bmask = sim.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)

hint = max(cfg.growth_blast_max, cfg.hazard_blast_max)
res_diff = 0; res_max = 0
dng_stats = {c: {"diff": 0, "max": 0.0} for c in caps}
for t in range(ticks):
    sim.step(acts())
    fuse, owner, wall, bm = sim.fuse, sim.owner, sim.wall, sim._blast_map()
    brick = sim.brick
    covered_new, trig_new = resolve_explosions(
        fuse, owner, wall, bm, cfg.max_chain, brick,
        early_exit=False, blast_max_hint=hint, chain_cap=cfg.chain_cap_rounds)
    covered_old, trig_old = resolve_explosions(
        fuse, owner, wall, bm, cfg.max_chain, brick,
        early_exit=True, blast_max_hint=hint, chain_cap=None)
    d = (covered_new.long() - covered_old.long()).abs().max().item()
    dt = (trig_new.long() - trig_old.long()).abs().max().item()
    if d or dt:
        res_diff += 1; res_max = max(res_max, d, dt)
    dng_old = danger_map(fuse, wall, bm, cfg.fuse, brick, cfg.max_chain,
                         early_exit=True)
    for c in caps:
        dng_new = danger_map(fuse, wall, bm, cfg.fuse, brick, cfg.max_chain,
                             early_exit=False, blast_max_hint=hint,
                             chain_cap=c)
        dd = (dng_new - dng_old).abs().max().item()
        if dd:
            dng_stats[c]["diff"] += 1
            dng_stats[c]["max"] = max(dng_stats[c]["max"], dd)
    if t % 15 == 0:
        print(f"tick {t}: resolve_diff={res_diff} "
              + " ".join(f"cap{c}={dng_stats[c]['diff']}" for c in caps), flush=True)
print(f"RESOLVE cap={cfg.chain_cap_rounds}: diff_ticks={res_diff}/{ticks} maxdiff={res_max}")
for c in caps:
    s = dng_stats[c]
    print(f"DANGER cap={c}: diff_ticks={s['diff']}/{ticks} maxdiff={s['max']}")
