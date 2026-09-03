"""legal_mask / encode_obs 分块编译（AUTOFUSE + backend='npu'）位级+时间。"""
import os, sys, time, torch
print("AUTOFUSE_FLAGS =", os.environ.get("AUTOFUSE_FLAGS", "(未设置)"))
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
from sim.torch_sim import BatchedSim
from sim.obs import legal_mask, encode_obs

dev = pick_device()
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(3):
    sim.step(acts)
torch.npu.synchronize()

def bench(fn, it=8):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

# ================= legal_mask =================
args = (cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
        sim.brick, sim.bombs_cap)
m0, b0 = legal_mask(*args)
t_e = bench(lambda: legal_mask(*args))
print(f"\nlegal_mask eager: {t_e:7.2f} ms")
try:
    torch._dynamo.reset()
    def _lm(cfg, wall, fuse, owner, pos, alive, brick, bombs_p):
        return legal_mask(cfg, wall, fuse, owner, pos, alive, brick, bombs_p)
    flm = torch.compile(_lm, backend="npu", dynamic=False)
    m1, b1 = flm(*args)
    torch.npu.synchronize()
    eq = torch.equal(m0, m1) and torch.equal(b0, b1)
    md = (m0.float() - m1.float()).abs().max().item() if not eq else 0.0
    t_c = bench(lambda: flm(*args))
    print(f"legal_mask compiled: {t_c:7.2f} ms 位级一致={eq} maxdiff={md:.2e} x{t_e/t_c:.2f}")
except Exception as ex:
    print(f"legal_mask compile FAIL {type(ex).__name__}: {str(ex)[:200]}")

# ================= encode_obs =================
dng = sim._dng_cache
ob = encode_obs(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
                sim.t, sim.brick, sim.bomb_blast, sim.crate, sim.invuln,
                sim.bombs_cap, danger_precomputed=dng)
t_oe = bench(lambda: encode_obs(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                                sim.alive, sim.t, sim.brick, sim.bomb_blast,
                                sim.crate, sim.invuln, sim.bombs_cap,
                                danger_precomputed=dng))
print(f"\nobs eager: {t_oe:7.2f} ms")
try:
    torch._dynamo.reset()
    def _eo(cfg, wall, fuse, owner, pos, alive, t, brick, bomb_blast,
            crate, invuln, bombs_p, dng):
        return encode_obs(cfg, wall, fuse, owner, pos, alive, t, brick,
                          bomb_blast, crate, invuln, bombs_p,
                          danger_precomputed=dng)
    feo = torch.compile(_eo, backend="npu", dynamic=False)
    ob2 = feo(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive, sim.t,
              sim.brick, sim.bomb_blast, sim.crate, sim.invuln, sim.bombs_cap, dng)
    torch.npu.synchronize()
    eq = torch.equal(ob, ob2)
    md = (ob - ob2).abs().max().item() if not eq else 0.0
    t_oc = bench(lambda: feo(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                             sim.alive, sim.t, sim.brick, sim.bomb_blast,
                             sim.crate, sim.invuln, sim.bombs_cap, dng))
    print(f"obs compiled: {t_oc:7.2f} ms 位级一致={eq} maxdiff={md:.2e} x{t_oe/t_oc:.2f}")
except Exception as ex:
    print(f"obs compile FAIL {type(ex).__name__}: {str(ex)[:200]}")
