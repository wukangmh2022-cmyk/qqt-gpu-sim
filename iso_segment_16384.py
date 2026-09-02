"""910B N=16384 的 torch step 分段剖析：精确锁定 100万 SPS 的目标块。

做法：独立 sim 上逐段计时（与 step 相同顺序调用，稳态），
reward 段用整步减去核心段推算。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim, center_cell
from sim.blast import resolve_explosions, danger_map
from sim.triton_sim import count_bombs_triton, place_bombs_triton, move_players_triton
from sim.dev import pick_device

dev = pick_device()
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
acts = torch.stack([
    torch.randint(0, 5, (N, 2), device=dev),
    (torch.rand(N, 2, device=dev) < 0.4).long()], dim=-1)
# 跑到富泡稳定状态（30 tick）
for _ in range(30):
    sim.step(acts)
sync()
print(f"N={N} dev={dev} 泡数 {int((sim.fuse>0).sum().item())}")

def bench(name, fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    dt = (time.perf_counter() - t0) / it * 1e3
    print(f"  {name:<30}: {dt:9.3f} ms")
    return dt

move, bomb = acts[..., 0], acts[..., 1]
alive0 = sim.alive.clone()
bmap = sim._blast_map()
blocked = sim.wall | sim.brick | (sim.fuse > 0)

print("== 核心段（可 triton）==")
t_place = bench("place（count+place triton）", lambda: place_bombs_triton(
    cfg, sim.fuse, sim.owner, sim.bomb_blast, sim.pos, sim.alive, bomb,
    sim.bombs_cap, sim.blast_cap, sim.brick, sim.wall))
t_move = bench("move（triton）", lambda: move_players_triton(
    cfg, sim.pos, move, sim.alive, blocked, sim.spd_g))
t_res = bench("resolve（torch，早退）", lambda: resolve_explosions(
    sim.fuse, sim.owner, sim.wall, bmap, cfg.max_chain, sim.brick))

print("== 危险图 ==")
t_dng = bench("danger_map（max_chain=16）", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=cfg.max_chain))
t_dng1 = bench("danger_map（max_chain=1）", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=1))
t_dngB = bench("danger 阶段B only（weight 扩散）", lambda: danger_map(
    sim.fuse, sim.wall, bmap, cfg.fuse, sim.brick, max_chain=1))

print("== 整步对照 ==")
t_full = bench("torch step", lambda: sim.step(acts))
print(f"\n核心段合计: {t_place+t_move+t_res:.1f} ms | danger: {t_dng:.1f} ms | "
      f"reward+其余 ≈ {t_full-(t_place+t_move+t_res+t_dng):.1f} ms")
print(f"当前 SPS: {N/t_full*1e3:.0f} | 100万目标 tick: {N/1e6:.2f} ms")
