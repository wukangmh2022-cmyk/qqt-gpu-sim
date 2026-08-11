"""本地验证 triton 模拟 kernel（MPS）—— bitwise vs torch 参考。

用法（triton 编译完成后）：.venv/bin/python3 verify_triton_sim.py
"""
import os, sys, time, torch
sys.path.insert(0, ".")      # 脚本所在目录
sys.path.insert(0, "/root")  # 确保 /root/sim（DCU 部署）优先于 opt_test 的遮蔽
from sim.triton_sim import _HAS_TRITON
if not _HAS_TRITON:
    print("triton 不可用（未编译好）"); sys.exit(1)

from sim.triton_sim import move_players_triton, explode_triton
from sim.move import move_players
from sim.blast import rays
from sim.config import SimConfig

torch.manual_seed(0)
if os.environ.get("TRITON_INTERPRET"):
    dev = "cpu"                    # interpreter 模式：解释执行，无 GPU 后端
    N = 64
else:
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
N, H, W = 4096, 13, 13
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10)

print(f"=== 1. 移动（AABB 滑动）===")
pos = torch.rand(N, 2, 2, device=dev) * (H - 2) + 0.5
move = torch.randint(0, 5, (N, 2), device=dev)
alive = torch.rand(N, 2, device=dev) > 0.2
wall = torch.rand(N, H, W, device=dev) < 0.2
brick = torch.rand(N, H, W, device=dev) < 0.15
fuse = torch.rand(N, H, W, device=dev) < 0.1
blocked = wall | brick | fuse
sm = torch.rand(N, 2, device=dev) * 2 + 0.5
ref = move_players(cfg, pos, move, alive, blocked, sm)
got = move_players_triton(cfg, pos, move, alive, blocked, sm)
d = (ref - got).abs().max().item()
print(f"move: 最大差 {d:.2e} {'✓' if d < 1e-3 else '✗ 不一致'}")

print(f"=== 2. 爆炸传播（rays）===")
src = torch.rand(N, H, W, device=dev) < 0.04
bombed = torch.rand(N, H, W, device=dev) < 0.07
brick2 = torch.rand(N, H, W, device=dev) < 0.15
blast = torch.randint(1, 8, (N, H, W), device=dev)
ref = rays(src, wall, bombed, blast, brick2)
got = explode_triton(src, wall, bombed, brick2, blast, b_max=7)
d = int((ref != got).sum())
print(f"explode: 不一致 {d}/{ref.numel()} {'✓' if d == 0 else '✗ 不一致'}")

print(f"=== 3. 速度（MPS 参考）===")
def bt(fn, it=20):
    sync = (torch.cuda.synchronize if dev == "cuda"
            else torch.mps.synchronize)
    for _ in range(3): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    sync()
    return (time.perf_counter() - t0) / it * 1000
t_mv = bt(lambda: move_players(cfg, pos, move, alive, blocked, sm))
t_tv = bt(lambda: move_players_triton(cfg, pos, move, alive, blocked, sm))
t_ex = bt(lambda: rays(src, wall, bombed, blast, brick2))
t_tx = bt(lambda: explode_triton(src, wall, bombed, brick2, blast))
print(f"move: torch {t_mv:.2f} ms | triton {t_tv:.2f} ms | x{t_mv/t_tv:.1f}")
print(f"explode: torch {t_ex:.2f} ms | triton {t_tx:.2f} ms | x{t_ex/t_tx:.1f}")
print("DONE")
