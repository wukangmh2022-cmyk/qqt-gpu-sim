"""验证 triton_step_core vs torch step（cfg 关奖励——核心状态逐位一致）。"""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/root")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_core
from sim.triton_sim import _HAS_TRITON
from sim.dev import pick_device
assert _HAS_TRITON, "triton 不可用"

dev = pick_device()
N = 2048

# 关奖励副作用：place/hit_attr/combo/chain 全 0、不成长（crate_prob=0）、win_bonus=0
# open_fraction 必须 0：open 关强制 crate_prob=1.0（踩箱必拾取成长），而
# triton_step_core 是 Step 1（核心传播，不含 Step 2 的宝箱拾取/成长）——
# 混入 open 关会让 torch step 开箱、core 不动 → 成长字段发散（实测 910B/MPS
# 同款发散）。纯 corridor + growth_crate_prob=0 → 无箱无拾取，core 可对拍。
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.0,
                hit_attr_penalty=0, place_cover_reward=0.0,
                place_chain_reward=0.0, place_dist_reward=0.0,
                chain_blast_bonus=0.0, growth_crate_prob=0.0,
                brick_reward=0.0, win_bonus=0.0,
                recycle_crate_prob=0.0)   # 堵回收箱必升路径（Step 2 才做拾取）

torch.manual_seed(0)
sim_t = BatchedSim(cfg, N, device=dev, seed=0)   # torch 参考
sim_k = BatchedSim(cfg, N, device=dev, seed=0)   # triton 核心
sim_t.bombs_cap[:] = 10; sim_t.blast_cap[:] = 7; sim_t.spd_g[:] = 2.1
sim_k.bombs_cap[:] = 10; sim_k.blast_cap[:] = 7; sim_k.spd_g[:] = 2.1
sim_t.reset_all(); sim_k.reset_all()
assert (sim_t.fuse == sim_k.fuse).all() and (sim_t.pos == sim_k.pos).all()

def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    else:
        torch.mps.synchronize()

# Step 1 只验证核心状态（成长/拾取字段 bombs_cap/blast_cap/spd_g 属 Step 2 奖励段）
FIELDS = ["fuse", "owner", "bomb_blast", "pos", "alive", "hp",
          "invuln", "t", "since_bomb"]
T = 40
# 预生成固定动作序列（避免两 sim 交错消费全局设备 RNG 导致分叉）
acts_seq = []
for i in range(T):
    mv = torch.randint(0, 5, (N, 2), device=dev)
    bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
    acts_seq.append(torch.stack([mv, bmb], dim=-1))

# sim_t 完整跑（各自独立 RNG 序列：跑前 manual_seed 同一起点）
torch.manual_seed(0)
for a in acts_seq:
    sim_t.step(a)
sync()
# sim_k triton 核心跑（重置 RNG 到同一起点）
torch.manual_seed(0)
for a in acts_seq:
    pk, ck, tk, dk = triton_step_core(sim_k, a)
    if bool(dk.any()):
        sim_k.reset_(dk)                 # 与 torch step 的 auto_reset 对齐
sync()
worst = {}
for f in FIELDS:
    worst[f] = int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
worst["done"] = 0

ok = all(v == 0 for v in worst.values())
for f, v in worst.items():
    print(f"  {f:<12}: 最大差 {v} {'✓' if v == 0 else '✗'}")
print(f"\nStep1 核心状态: {'全部一致 ✓' if ok else '有差异 ✗'}")

# 速度（同步稳态——含伤害/清场/终局）
def bt(fn, it=15):
    for _ in range(3):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    return (time.perf_counter() - t0) / it * 1000

t_t = bt(lambda: sim_t.step(acts_seq[-1]))
t_k = bt(lambda: triton_step_core(sim_k, acts_seq[-1]))
print(f"step 同步稳态: torch {t_t:.1f} ms | triton核心 {t_k:.1f} ms | x{t_t/t_k:.1f}")
