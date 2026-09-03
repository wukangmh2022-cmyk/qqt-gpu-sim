"""位级验证：reward 段 _sc 标量 op 替换（old=HEAD vs new=工作区）。

同 seed 同 config 跑 600 tick，逐 tick 对比 reward/done/hp/状态位级。
"""
import sys
import torch

sys.path.insert(0, "/tmp/sim_old")
import sim.torch_sim as old_ts
sys.path.pop(0)

sys.path.insert(0, ".")
import sim.torch_sim as new_ts

from sim.config import SimConfig
from sim.dev import pick_device

dev = pick_device()
N = 1024
TICKS = 600
CFGS = [
    SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
              open_fraction=0.5, timeout_draw=True, combo_reward=0.10,
              hit_attr_penalty=2, place_cover_reward=0.05,
              place_chain_reward=0.20, chain_blast_bonus=0.08,
              danger_penalty=0.02, brick_reward=0.5,
              win_bonus=5.0, step_penalty=0.001, hit_reward=0.5),
    SimConfig(map_mode="open", max_steps=1800, combo_reward=0.0,
              hit_attr_penalty=0),
]


def gen_acts(N, ticks):
    torch.manual_seed(123)
    acts = []
    for _ in range(ticks):
        mv = torch.randint(0, 5, (N, 2), device=dev)
        bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
        acts.append(torch.stack([mv, bmb], dim=-1))
    return acts


def run(ts_mod, cfg, acts):
    torch.manual_seed(0)
    s = ts_mod.BatchedSim(cfg, N, device=dev, seed=0)
    s.reset_all()
    buf = []
    for a in acts:
        r, done, info = s.step(a)
        buf.append((r.clone(), done.clone(),
                    s.hp.clone(), s.alive.clone(), s._combo.clone()))
    return buf


def cmp(a, b, tag):
    ok = True
    for t, ((ra, da, ha, aa, ca), (rb, db, hb, ab, cb)) in enumerate(zip(a, b)):
        if not torch.equal(ra, rb):
            md = (ra - rb).abs().max().item()
            ndiff = (ra != rb).sum().item()
            print(f"  [{tag}] t={t} reward maxdiff={md:.3e} ndiff={ndiff}")
            ok = False
            break
        if not torch.equal(da, db) or not torch.equal(ha, hb) \
                or not torch.equal(aa, ab) or not torch.equal(ca, cb):
            print(f"  [{tag}] t={t} state mismatch")
            ok = False
            break
    print(f"  [{tag}] {'BITWISE IDENTICAL' if ok else 'MISMATCH'} over {min(len(a), len(b))} ticks")
    return ok


all_ok = True
for i, cfg in enumerate(CFGS):
    print(f"=== cfg#{i} map={cfg.map_mode} combo={cfg.combo_reward} ===")
    acts = gen_acts(N, TICKS)
    a = run(old_ts, cfg, acts)
    b = run(new_ts, cfg, acts)
    all_ok &= cmp(a, b, f"cfg#{i}")
print("ALL:", "PASS" if all_ok else "FAIL")
