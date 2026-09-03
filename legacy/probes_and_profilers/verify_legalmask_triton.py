"""legal_mask triton（复用 move kernel 试探）vs torch 位级对拍 + 时间。"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
from sim.torch_sim import BatchedSim
from sim.obs import legal_mask as legal_mask_torch
from sim.triton_sim import legal_mask_triton

dev = pick_device()
assert dev.startswith("npu")
N = 16384
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)

def bench(fn, it=8):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / it * 1000

# 多组随机状态对拍（覆盖不同位置/墙分布）
ok = True
for trial in range(5):
    torch.manual_seed(100 + trial)
    sim = BatchedSim(cfg, N, device=dev, seed=trial)
    sim.reset_all()
    mv = torch.randint(0, 5, (N, 2), device=dev)
    acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
    for _ in range(trial * 7 + 1):
        sim.step(acts)
    torch.npu.synchronize()
    m0, b0 = legal_mask_torch(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                              sim.alive, sim.brick, sim.bombs_cap)
    m1, b1 = legal_mask_triton(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                               sim.alive, sim.brick, sim.bombs_cap)
    if not torch.equal(m0, m1):
        nd = (m0 != m1).sum().item()
        print(f"trial={trial}: move_mask MISMATCH ndiff={nd}/{m0.numel()}")
        ok = False
    if not torch.equal(b0, b1):
        print(f"trial={trial}: bomb_mask MISMATCH ndiff={(b0!=b1).sum().item()}")
        ok = False
print("位级对拍(5 组):", "PASS" if ok else "FAIL")

# 时间（用同一 sim 状态）
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
for _ in range(5):
    sim.step(acts)
torch.npu.synchronize()
t_t = bench(lambda: legal_mask_torch(cfg, sim.wall, sim.fuse, sim.owner,
                                     sim.pos, sim.alive, sim.brick,
                                     sim.bombs_cap))
t_tr = bench(lambda: legal_mask_triton(cfg, sim.wall, sim.fuse, sim.owner,
                                       sim.pos, sim.alive, sim.brick,
                                       sim.bombs_cap))
print(f"torch: {t_t:7.2f} ms | triton: {t_tr:7.2f} ms | x{t_t/t_tr:.2f}")
