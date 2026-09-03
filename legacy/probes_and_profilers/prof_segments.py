"""分段短路实验：把 step 内各段替换成廉价实现，量每段边际成本。

wall_short < wall_base → 该段贡献 (wall_base - wall_short)ms。
短路段不改变张量形状/类型，仅改变工作量，计时仍可比。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

import sim.torch_sim as TS
from sim.config import SimConfig
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
N = 16384

def make_sim():
    torch.manual_seed(0)
    s = TS.BatchedSim(cfg, N, device=dev, seed=0)
    s.reset_all()
    return s

mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

def bench(sim, it=10):
    for _ in range(4):
        sim.step(acts)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        sim.step(acts)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

# 保存原实现
_orig = {}
for name in ("danger_map", "resolve_explosions", "move_players"):
    _orig[name] = getattr(TS, name, None)
_orig_pp = TS.BatchedSim._place_predict_reward
_orig_move = TS.BatchedSim._move_triton if getattr(TS.BatchedSim, "_move_triton", None) else None

d = TS.BatchedSim(dict) if False else None  # noqa

def bench_with(label, patch_fn):
    # 建新 sim（patch 前）
    sim = make_sim()
    restore = patch_fn()
    try:
        ms = bench(sim)
        print(f"{label:28s}: {ms:7.2f} ms/tick  ({N/ms*1e3/1e4:.2f}万 SPS)")
    finally:
        for k, v in restore.items():
            setattr(TS, k, v)
        TS.BatchedSim._place_predict_reward = _orig_pp

def patch_danger():
    TS.danger_map = lambda *a, **k: torch.zeros(
        (a[0].shape[0], a[0].shape[-2], a[0].shape[-1]),
        dtype=torch.float32, device=a[0].device)
    return {"danger_map": _orig["danger_map"]}

def patch_resolve():
    TS.resolve_explosions = lambda *a, **k: (
        torch.zeros_like(a[0], dtype=torch.bool), a[0] == 0)
    return {"resolve_explosions": _orig["resolve_explosions"]}

def patch_pp():
    def fake_pp(self, placed, alive0):
        return torch.zeros(self.num_envs, self.cfg.n_players,
                           device=self.pos.device)
    TS.BatchedSim._place_predict_reward = fake_pp
    return {}

def patch_move():
    # move 已是 triton 1 kernel；短路成 copy（省 1 kernel 的执行时间）
    def fake_move(cfg, pos, move, alive0, blocked, sm):
        return pos
    TS._move_triton = fake_move
    return {"_move_triton": _orig_move if _orig_move is not None else TS._move_triton}

def patch_both():
    TS.danger_map = lambda *a, **k: torch.zeros(
        (a[0].shape[0], a[0].shape[-2], a[0].shape[-1]),
        dtype=torch.float32, device=a[0].device)
    TS.resolve_explosions = lambda *a, **k: (
        torch.zeros_like(a[0], dtype=torch.bool), a[0] == 0)
    return {"danger_map": _orig["danger_map"],
            "resolve_explosions": _orig["resolve_explosions"]}

print("== 分段边际成本（N=16384, corridor）==")
bench_with("baseline", lambda: {})
bench_with("danger→zeros", patch_danger)
bench_with("resolve→trivial", patch_resolve)
bench_with("place_predict→zeros", patch_pp)
bench_with("move→identity", patch_move)
bench_with("danger+resolve→trivial", patch_both)
