"""mock 分段测贡献：把各段替换为零成本实现，测 step 时间差。"""
import sys, time, torch
from unittest.mock import patch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
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
for _ in range(30):
    sim.step(acts)
sync()

def t_step(it=8):
    for _ in range(2):
        sim.step(acts)
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        sim.step(acts)
    sync()
    return (time.perf_counter() - t0) / it * 1e3

z16 = torch.zeros(N, cfg.n_players, device=dev)
z_map = torch.zeros(N, cfg.height, cfg.width, device=dev)

print(f"N={N} 泡数 {int((sim.fuse>0).sum().item())}")
t_full = t_step()
print(f"完整 step: {t_full:.2f} ms")

def dng_zero(*a, **k):
    return z_map.clone()
with patch("sim.torch_sim.danger_map", dng_zero):
    t_ndng = t_step()
print(f"去 danger: {t_ndng:.2f} ms → 贡献 {t_full-t_ndng:.1f} ms")

def pp_zero(self, *a, **k):
    return z16
with patch.object(BatchedSim, "_place_predict_reward", pp_zero):
    t_npp = t_step()
print(f"去 place_predict: {t_npp:.2f} ms → 贡献 {t_full-t_npp:.1f} ms")

def hw_none(self):
    return None
with patch.object(BatchedSim, "_hazard_wave", hw_none):
    t_nhw = t_step()
print(f"去 hazard: {t_nhw:.2f} ms → 贡献 {t_full-t_nhw:.1f} ms")

print(f"\n剩余（damage/clear/combo/win/pickup/终局/结算）≈ "
      f"{t_full - (t_full-t_ndng) - (t_full-t_npp) - (t_full-t_nhw):.1f} ms")
