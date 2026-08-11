"""910B triton 逐 kernel 剖析：count/place/move/explode/danger 单独计时 + launch 开销。"""
import sys, time, torch
import triton, triton.language as tl
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_sim import (count_bombs_triton, place_bombs_triton,
                            move_players_triton, explode_triton, resolve_triton,
                            danger_triton)
from sim.blast import rays, resolve_explosions, danger_map
from sim.dev import pick_device


@triton.jit
def _trivial(ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, offs, mask=offs < N)


dev = pick_device()
N = 2048
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.0,
                hit_attr_penalty=0, place_cover_reward=0.0,
                place_chain_reward=0.0, place_dist_reward=0.0,
                chain_blast_bonus=0.0, growth_crate_prob=0.0,
                brick_reward=0.0, win_bonus=0.0, recycle_crate_prob=0.0)

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
acts = []
for i in range(30):
    mv = torch.randint(0, 5, (N, 2), device=dev)
    bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
    acts.append(torch.stack([mv, bmb], dim=-1))
for a in acts:
    sim.step(a, auto_reset=False)
sync()

def bench(name, fn, it=50, warm=5):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    dt = (time.perf_counter() - t0) / it * 1e3
    print(f"  {name:<26}: {dt:9.3f} ms")
    return dt

print(f"N={N} dev={dev} 富泡状态（泡数 {int((sim.fuse>0).sum().item())}）")
move = acts[-1][..., 0]
bomb = acts[-1][..., 1]
alive0 = sim.alive.clone()
blocked = sim.wall | sim.brick | (sim.fuse > 0)
bmap = sim._blast_map()

print("== triton 单 kernel ==")
t_cnt = bench("count", lambda: count_bombs_triton(sim.owner, sim.fuse))
t_pl = bench("place", lambda: place_bombs_triton(
    cfg, sim.fuse, sim.owner, sim.bomb_blast, sim.pos, sim.alive, bomb,
    sim.bombs_cap, sim.blast_cap, sim.brick, sim.wall))
t_mv = bench("move", lambda: move_players_triton(
    cfg, sim.pos, move, sim.alive, blocked, sim.spd_g))
t_ex = bench("explode x1", lambda: explode_triton(
    sim.fuse == 0, sim.wall, sim.fuse > 0, sim.brick, sim.bomb_blast))
t_res = bench("resolve x16", lambda: resolve_triton(
    sim.fuse, sim.owner, sim.wall, sim.bomb_blast, sim.brick, cfg.max_chain))
t_dng1 = bench("danger mc=1", lambda: danger_triton(
    sim.fuse, sim.wall, sim.fuse > 0, sim.brick, bmap, cfg.fuse, max_chain=1))
t_dng16 = bench("danger mc=16", lambda: danger_triton(
    sim.fuse, sim.wall, sim.fuse > 0, sim.brick, bmap, cfg.fuse,
    max_chain=cfg.max_chain))

print("== 对照 torch ==")
t_rays = bench("torch rays", lambda: rays(sim.fuse == 0, sim.wall, sim.fuse > 0,
                                          bmap, sim.brick))
t_res_t = bench("torch resolve", lambda: resolve_explosions(
    sim.fuse, sim.owner, sim.wall, bmap, cfg.max_chain, sim.brick))
t_dng_t = bench("torch danger_map(16)", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=16))

buf = torch.zeros(1024, dtype=torch.int32, device=dev)
t_triv = bench("trivial launch (grid=1)", lambda: _trivial[(1,)](
    buf, 1024, BLOCK=1024))

print(f"\n估算单 tick（core 段无 danger）: count+place+move+resolve16 "
      f"= {t_cnt+t_pl+t_mv+t_res:.1f} ms")
print(f"估算单 tick（full 含 danger16）: 上面 + danger16 "
      f"= {t_cnt+t_pl+t_mv+t_res+t_dng16:.1f} ms")
