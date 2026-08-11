"""验证 triton 计数/放泡/危险图 kernel（bitwise vs torch，DCU/本地通用）。"""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/root")  # DCU 部署优先（避免 opt_test 遮蔽）

from sim.triton_sim import (_HAS_TRITON, count_bombs_triton, place_bombs_triton,
                            danger_triton)
if not _HAS_TRITON:
    print("triton 不可用"); sys.exit(1)
from sim.config import SimConfig
from sim.torch_sim import BatchedSim, center_cell
from sim.dev import pick_device

dev = pick_device()
N, H, W = 4096, 13, 13
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10)

# ---- 富泡状态 ----
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(15):
    sim.step(acts, auto_reset=False)
def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()
sync()

print(f"=== 1. 在场泡数计数 ===")
owner, fuse = sim.owner, sim.fuse
ref_cnt = torch.stack([
    ((owner == me) & (fuse > 0)).flatten(1).sum(dim=1) for me in range(2)], dim=1)
cnt = count_bombs_triton(owner, fuse)
d = int((cnt != ref_cnt).sum())
print(f"计数: 不一致 {d}/{ref_cnt.numel()} {'✓' if d == 0 else '✗'}")

print(f"=== 2. 放泡 ===")
pos, alive = sim.pos.clone(), sim.alive.clone()
bomb = (torch.rand(N, 2, device=dev) < 0.5).long()
bombs_cap, blast_cap = sim.bombs_cap.clone(), sim.blast_cap.clone()
brick, wall = sim.brick, sim.wall

# torch 参考（复制状态 → _place_bombs）
fuse_t = fuse.clone(); owner_t = owner.clone(); blast_t = sim.bomb_blast.clone()
sim2 = BatchedSim(cfg, N, device=dev, seed=0)
sim2.fuse.copy_(fuse_t); sim2.owner.copy_(owner_t); sim2.bomb_blast.copy_(blast_t)
sim2.pos.copy_(pos); sim2.alive.copy_(alive)
sim2.bombs_cap.copy_(bombs_cap); sim2.blast_cap.copy_(blast_cap)
sim2.brick.copy_(brick); sim2.wall.copy_(wall)
placed_ref = sim2._place_bombs(bomb, alive, bombs_cap, blast_cap)
fuse_ref = sim2.fuse.clone(); owner_ref = sim2.owner.clone(); blast_ref = sim2.bomb_blast.clone()

# triton
fuse_t2 = fuse.clone(); owner_t2 = owner.clone(); blast_t2 = sim.bomb_blast.clone()
place_bombs_triton(cfg, fuse_t2, owner_t2, blast_t2, pos, alive, bomb,
                   bombs_cap, blast_cap, brick, wall)
sync()
d_f = int((fuse_t2 != fuse_ref).sum())
d_o = int((owner_t2 != owner_ref).sum())
d_b = int((blast_t2 != blast_ref).sum())
print(f"放泡: fuse差 {d_f} owner差 {d_o} blast差 {d_b} {'✓' if d_f+d_o+d_b == 0 else '✗'}")

print(f"=== 3. 危险图（阶段 B）===")
from sim.blast import danger_map
blast_map = sim._blast_map()
ref_d = danger_map(fuse, wall, blast_map, cfg.fuse, brick, max_chain=1)
got_d = danger_triton(fuse, wall, fuse > 0, brick, blast_map, cfg.fuse)
sync()
diff = (ref_d - got_d).abs().max().item()
pos_diff = int(((ref_d > 0) != (got_d > 0)).sum())
print(f"危险图: maxdiff {diff:.2e} 非零位置差 {pos_diff} {'✓' if diff < 1e-4 and pos_diff == 0 else '✗'}")

print("DONE")
