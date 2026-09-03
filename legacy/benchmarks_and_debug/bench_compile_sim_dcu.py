"""DCU：observe/legal_mask/rays 的 inductor compile 探测。

三者都是纯张量函数（无 RNG）→ 可编译融合。DCU HIP launch ~1ms/算子，
融合减 launch 是主要收益。对拍 compiled vs eager（maxdiff）。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.obs import encode_obs, legal_mask
import sim.blast as B

torch.manual_seed(20260816)
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, hazard_fraction=0.0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
dev = "cuda"
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
g = torch.Generator(device=dev).manual_seed(99)

def acts():
    mmask, bmask = sim.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
    return torch.stack([mv, bm], dim=-1)

# 跑 10 tick 让状态丰富（有泡、有成长、有危险）
for _ in range(10):
    sim.step(acts())
torch.cuda.synchronize()

def bench(fn, reps=20, warmup=4):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000

# ---------- observe ----------
dng = sim._dng_cache.clone() if sim._dng_cache is not None else None
def obs_eager():
    return encode_obs(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
                      sim.t, sim.brick, sim.bomb_blast, crate=sim.crate,
                      invuln=sim.invuln, bombs_p=sim.bombs_cap,
                      danger_precomputed=dng, early_exit=True)
def obs_wrap(wall, fuse, owner, pos, alive, t, brick, bb, crate, invuln, bp, d):
    return encode_obs(cfg, wall, fuse, owner, pos, alive, t, brick, bb,
                      crate=crate, invuln=invuln, bombs_p=bp,
                      danger_precomputed=d, early_exit=True)
o_e = obs_eager()
torch.cuda.synchronize()
te = bench(obs_eager)
try:
    obs_c = torch.compile(obs_wrap, backend="inductor", dynamic=False)
    o_c = obs_c(sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive, sim.t,
                sim.brick, sim.bomb_blast, sim.crate, sim.invuln,
                sim.bombs_cap, dng)
    torch.cuda.synchronize()
    md = (o_e - o_c).abs().max().item()
    tc = bench(lambda: obs_c(sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
                             sim.t, sim.brick, sim.bomb_blast, sim.crate,
                             sim.invuln, sim.bombs_cap, dng))
    print(f"N={N} observe: eager {te:.1f}ms inductor {tc:.1f}ms "
          f"speedup {te/tc:.2f}x maxdiff {md:.2e}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

# ---------- legal_mask ----------
def lm_eager():
    return legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
                      sim.brick, sim.bombs_cap)
def lm_wrap(wall, fuse, owner, pos, alive, brick, bc):
    return legal_mask(cfg, wall, fuse, owner, pos, alive, brick, bc)
le = lm_eager()
te = bench(lm_eager)
try:
    lm_c = torch.compile(lm_wrap, backend="inductor", dynamic=False)
    lc = lm_c(sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive, sim.brick,
              sim.bombs_cap)
    torch.cuda.synchronize()
    md = max((le[0] - lc[0]).abs().max().item(), (le[1] - lc[1]).abs().max().item())
    tc = bench(lambda: lm_c(sim.wall, sim.fuse, sim.owner, sim.pos, sim.alive,
                            sim.brick, sim.bombs_cap))
    print(f"N={N} legal_mask: eager {te:.1f}ms inductor {tc:.1f}ms "
          f"speedup {te/tc:.2f}x maxdiff {md:.2e}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

# ---------- rays（int8 波前） ----------
bm = sim._blast_map()
src = (sim.fuse > 0) & ~sim.wall
blast_cell = torch.where(sim.bomb_blast > 0, sim.bomb_blast.long(), cfg.blast)
hint = max(int(blast_cell.max()), int(cfg.blast))
def rays_eager():
    return B.rays(src, sim.wall, sim.fuse > 0, sim.bomb_blast,
                  brick=sim.brick, blast_max_hint=hint)
re_e = rays_eager()
te = bench(rays_eager)
try:
    def rays_wrap(sources, wall, bombed, blast, brick, bmh):
        return B.rays(sources, wall, bombed, blast, brick=brick,
                      blast_max_hint=bmh)
    rays_c = torch.compile(rays_wrap, backend="inductor", dynamic=False)
    rc = rays_c(src, sim.wall, sim.fuse > 0, sim.bomb_blast, sim.brick, hint)
    torch.cuda.synchronize()
    md = (re_e.long() - rc.long()).abs().max().item()
    tc = bench(lambda: rays_c(src, sim.wall, sim.fuse > 0, sim.bomb_blast,
                              sim.brick, hint))
    print(f"N={N} rays: eager {te:.1f}ms inductor {tc:.1f}ms "
          f"speedup {te/tc:.2f}x maxdiff {md:.2e}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
