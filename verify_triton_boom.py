"""爆炸/连锁地图状态专用验证（本地 MPS / DCU 通用）。

聚焦：1) 爆炸后地图状态变更（fuse 清场/owner 清/brick 摧毁/crate 生成/pos 伤害）
2) 连锁爆炸（resolve_triton 多轮 vs torch）
3) 多 tick 完整 step 的地图状态对比（torch vs triton_step_full）
"""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/root")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_full
from sim.triton_sim import (_HAS_TRITON, resolve_triton, explode_triton)
assert _HAS_TRITON, "triton 不可用（本地编译未完成）"

from sim.blast import resolve_explosions, rays
from sim.dev import pick_device

dev = pick_device()
N = 1024
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10,
                hit_attr_penalty=2, place_cover_reward=0.05,
                place_chain_reward=0.20, chain_blast_bonus=0.08)
H, W = cfg.height, cfg.width

def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    else:
        torch.mps.synchronize()

ok = True
def check(name, cond, extra=""):
    global ok
    print(f"  {name}: {'✓' if cond else '✗'} {extra}")
    ok &= bool(cond)

print(f"设备 {dev} | 地图 {H}x{W}")
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
sim.reset_all()

# ---- 1. 连锁爆炸（密集泡：多颗引信不同的泡互相覆盖触发连锁）----
print("=== 1. 连锁爆炸 resolve ===")
mv = torch.randint(0, 5, (N, 2), device=dev)
bmb = torch.ones(N, 2, dtype=torch.long, device=dev)
acts = torch.stack([mv, bmb], dim=-1)
for _ in range(10):
    sim.step(acts, auto_reset=False)
sync()
fuse = sim.fuse.clone(); owner = sim.owner.clone()
wall = sim.wall.bool(); brick = sim.brick.bool(); bl = sim.bomb_blast.clone()
# 制造密集触发（相邻泡不同引信 → 连锁）
trig = (torch.rand(N, H, W, device=dev) < 0.04) & ~wall & (owner < 0)
fuse2 = torch.where(trig, torch.zeros_like(fuse), fuse)
owner2 = torch.where(trig, torch.full_like(owner, 1), owner)
rc, rt = resolve_explosions(fuse2, owner2, wall, bl, 16, brick, early_exit=False)
tc, tt = resolve_triton(fuse2, owner2, wall, bl, brick, 16)
sync()
check("连锁 covered", int((rc != tc).sum()) == 0, f"diff {int((rc!=tc).sum())}")
check("连锁 triggered", int((rt != tt).sum()) == 0)

# ---- 2. 多 tick 完整 step：爆炸后地图状态 ----
print("=== 2. 多 tick 地图状态（含爆炸/连锁/炸砖/宝箱）===")
T = 30
acts_seq = []
for i in range(T):
    acts_seq.append(torch.stack([
        torch.randint(0, 5, (N, 2), device=dev),
        (torch.rand(N, 2, device=dev) < 0.5).long()], dim=-1))
def mk():
    s = BatchedSim(cfg, N, device=dev, seed=0)
    s.bombs_cap[:] = 10; s.blast_cap[:] = 7; s.spd_g[:] = 2.1
    s.reset_all(); return s
sa, sb = mk(), mk()
torch.manual_seed(0)
for a in acts_seq:
    sa.step(a)
sync()
torch.manual_seed(0)
for a in acts_seq:
    triton_step_full(sb, a)
sync()
FIELDS = ["fuse", "owner", "bomb_blast", "pos", "alive", "hp",
          "brick", "crate", "_recycle_crate", "bombs_cap", "blast_cap",
          "spd_g", "t", "since_bomb"]
for f in FIELDS:
    d = int((getattr(sa, f) != getattr(sb, f)).sum())
    check(f"地图状态 {f}", d == 0, f"diff {d}")
print("=== 完成 ===")
print("全部通过" if ok else "有失败")
