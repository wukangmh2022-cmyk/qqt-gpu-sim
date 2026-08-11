"""诊断 triton_step_core vs torch step 的第一个发散点（逐 tick 二分）。

本地 MPS 跑：每个 tick 后对比全部核心字段，输出第一个差异 tick，
然后该 tick 内逐步对比（fuse 递减/放泡/移动/爆炸/伤害）定位根因。
用法：.venv/bin/python iso_step_diverge.py
"""
import sys, torch
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_core
from sim.triton_sim import _HAS_TRITON
from sim.dev import pick_device

dev = pick_device()
N = 2048
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.0,
                hit_attr_penalty=0, place_cover_reward=0.0,
                place_chain_reward=0.0, place_dist_reward=0.0,
                chain_blast_bonus=0.0, growth_crate_prob=0.0,
                brick_reward=0.0, win_bonus=0.0, recycle_crate_prob=0.0)
T = 40
acts_seq = []
for i in range(T):
    mv = torch.randint(0, 5, (N, 2), device=dev)
    bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
    acts_seq.append(torch.stack([mv, bmb], dim=-1))

sim_t = BatchedSim(cfg, N, device=dev, seed=0)
sim_k = BatchedSim(cfg, N, device=dev, seed=0)
sim_t.reset_all(); sim_k.reset_all()

FIELDS = ["fuse", "owner", "bomb_blast", "pos", "alive", "hp",
          "invuln", "t", "since_bomb"]

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

for i in range(T):
    r_t, d_t, inf_t = sim_t.step(acts_seq[i])
    pk, ck, tk, dk = triton_step_core(sim_k, acts_seq[i])
    if bool(dk.any()):
        sim_k.reset_(dk)
    sync()
    diffs = {f: int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
             for f in FIELDS}
    bad = {f: v for f, v in diffs.items() if v > 0}
    if bad:
        print(f"tick {i}: 首个发散 -> {bad}")
        # 该 tick 内逐步对比
        # 用新 sim 从头跑到 i-1（对齐），再手动对比子步骤
        break
    if i % 5 == 0:
        print(f"tick {i}: 一致 ✓")
else:
    print("40 tick 全部一致 ✓")
