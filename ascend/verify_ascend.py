"""Ascend 910B kernel 对拍验证：triton kernel vs torch 参考（torch_npu 基准）。

用法（910B 上，setup_ascend.sh 之后）：python3 ascend/verify_ascend.py
覆盖：移动 / 爆炸传播 / 计数 / 放泡 / 危险图B —— 与 DCU/本地同一份 kernel 代码。
"""
import sys, time, torch
sys.path.insert(0, ".")

try:
    import torch_npu  # noqa: F401
    dev = "npu:0"
except ImportError:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {dev} | torch {torch.__version__}")

from sim.triton_sim import (move_players_triton, explode_triton,
                            count_bombs_triton, place_bombs_triton,
                            danger_triton, _HAS_TRITON)
assert _HAS_TRITON, "triton 不可用——检查 triton-ascend 安装"
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.move import move_players
from sim.blast import rays, danger_map

N = 4096
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10)
H, W = cfg.height, cfg.width

def sync():
    if dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    else:
        torch.mps.synchronize()

ok = True
def check(name, cond, extra=""):
    global ok
    mark = "✓" if cond else "✗"
    print(f"  {name}: {mark} {extra}")
    ok &= bool(cond)

# 富泡状态
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(10):
    sim.step(acts, auto_reset=False)
sync()

print("=== 1. 移动 ===")
pos, alive = sim.pos.clone(), sim.alive.clone()
move = torch.randint(0, 5, (N, 2), device=dev)
blocked = sim.wall | sim.brick | (sim.fuse > 0)
sm = torch.rand(N, 2, device=dev) * 2 + 0.5
ref = move_players(cfg, pos, move, alive, blocked, sm)
got = move_players_triton(cfg, pos, move, alive, blocked, sm)
sync()
check("move", (ref - got).abs().max().item() < 1e-3,
      f"maxdiff {(ref-got).abs().max().item():.2e}")

print("=== 2. 爆炸传播 ===")
src = torch.rand(N, H, W, device=dev) < 0.04
bombed = torch.rand(N, H, W, device=dev) < 0.07
brick = torch.rand(N, H, W, device=dev) < 0.15
blast = torch.randint(1, 8, (N, H, W), device=dev)
wall = sim.wall
ref = rays(src, wall, bombed, blast, brick)
got = explode_triton(src, wall, bombed, brick, blast)
sync()
check("explode", int((ref != got).sum()) == 0, f"diff {int((ref!=got).sum())}")

print("=== 3. 计数 + 放泡 ===")
owner, fuse = sim.owner, sim.fuse
ref_cnt = torch.stack([((owner == m) & (fuse > 0)).flatten(1).sum(dim=1)
                       for m in range(2)], dim=1)
cnt = count_bombs_triton(owner, fuse)
check("count", int((cnt != ref_cnt).sum()) == 0)
bomb = (torch.rand(N, 2, device=dev) < 0.5).long()
f_t = fuse.clone(); o_t = owner.clone(); b_t = sim.bomb_blast.clone()
place_bombs_triton(cfg, f_t, o_t, b_t, pos, alive, bomb,
                   sim.bombs_cap, sim.blast_cap, sim.brick, wall)
sync()
sim2 = BatchedSim(cfg, N, device=dev, seed=0)
sim2.fuse.copy_(fuse); sim2.owner.copy_(owner); sim2.bomb_blast.copy_(sim.bomb_blast)
sim2.pos.copy_(pos); sim2.alive.copy_(alive)
sim2.bombs_cap.copy_(sim.bombs_cap); sim2.blast_cap.copy_(sim.blast_cap)
sim2.brick.copy_(sim.brick); sim2.wall.copy_(wall)
sim2._place_bombs(bomb, alive, sim2.bombs_cap, sim2.blast_cap)
check("place", int((f_t != sim2.fuse).sum()) +
      int((o_t != sim2.owner).sum()) + int((b_t != sim2.bomb_blast).sum()) == 0)

print("=== 4. 危险图B ===")
bmap = sim._blast_map()
ref_d = danger_map(fuse, wall, bmap, cfg.fuse, brick, max_chain=1)
got_d = danger_triton(fuse, wall, fuse > 0, brick, bmap, cfg.fuse)
sync()
diff = (ref_d - got_d).abs().max().item()
posd = int(((ref_d > 0) != (got_d > 0)).sum())
check("dangerB", diff < 1e-4 and posd == 0, f"maxdiff {diff:.2e} posdiff {posd}")

print(f"\n{'全部通过 —— 可以进入完整 step 整合' if ok else '有失败 —— 检查输出'}")
