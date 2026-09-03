"""隔离 910B 不确定性源头：同一输入重复跑 triton move / torch move，对比自洽性。

- triton twice: 同输入跑 2 次 move_players_triton → 若不同则 triton-ascend 不确定
- torch twice: 同输入跑 2 次 torch move_players → 若不同则 torch_npu 不确定
- cross: triton vs torch 逐位对比
dense 随机输入 × 多 seed，统计不一致次数。
"""
import sys, torch
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.move import move_players
from sim.triton_sim import move_players_triton
from sim.dev import pick_device

dev = pick_device()
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10)
N, H, W = 4096, 13, 13

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()

t_tot = t_tt = t_cross = 0
for seed in range(8):
    torch.manual_seed(seed)
    pos = torch.rand(N, 2, 2, device=dev) * (H - 2) + 0.5
    move = torch.randint(0, 5, (N, 2), device=dev)
    alive = torch.rand(N, 2, device=dev) > 0.2
    wall = torch.rand(N, H, W, device=dev) < 0.25
    brick = torch.rand(N, H, W, device=dev) < 0.25
    fuse = (torch.rand(N, H, W, device=dev) < 0.08) * torch.randint(1, 31, (N, H, W), device=dev)
    blocked = wall | brick | (fuse > 0)
    sm = torch.rand(N, 2, device=dev) * 2 + 0.5

    p1 = move_players_triton(cfg, pos, move, alive, blocked, sm)
    sync()
    p2 = move_players_triton(cfg, pos, move, alive, blocked, sm)
    sync()
    pt = move_players(cfg, pos, move, alive, blocked, sm)
    sync()
    # 自洽
    dd = int((p1 != p2).sum())
    t_tt += dd
    # triton vs torch
    dc = int((p1 != pt).sum())
    t_cross += dc
    print(f"seed {seed}: triton 自洽差 {dd} | triton-vs-torch 差 {dc}")
print(f"总计: triton 自洽差 {t_tt} | cross 差 {t_cross} (共 {N*2*8} 值)")
