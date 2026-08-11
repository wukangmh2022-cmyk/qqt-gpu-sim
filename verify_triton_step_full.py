"""验证 triton_step_full vs torch step（完整奖励——全部字段/reward/done/info bitwise）。"""
import sys, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/root")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_full
from sim.triton_sim import _HAS_TRITON
assert _HAS_TRITON

if torch.backends.mps.is_available():
    dev = "mps"
elif torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
N = 1024
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, timeout_draw=True, combo_reward=0.10,
                hit_attr_penalty=2, place_cover_reward=0.05,
                place_chain_reward=0.20, chain_blast_bonus=0.08)
torch.manual_seed(0)
sim_t = BatchedSim(cfg, N, device=dev, seed=0)
sim_k = BatchedSim(cfg, N, device=dev, seed=0)
sim_t.bombs_cap[:] = 10; sim_t.blast_cap[:] = 7; sim_t.spd_g[:] = 2.1
sim_k.bombs_cap[:] = 10; sim_k.blast_cap[:] = 7; sim_k.spd_g[:] = 2.1
sim_t.reset_all(); sim_k.reset_all()

def sync():
    torch.cuda.synchronize() if dev == "cuda" else torch.mps.synchronize()

FIELDS = ["fuse", "owner", "bomb_blast", "pos", "alive", "hp", "invuln",
          "t", "since_bomb", "bombs_cap", "blast_cap", "spd_g", "crate",
          "_combo", "_last_hit", "_recycle_crate"]
T = 30
acts_seq = []
for i in range(T):
    mv = torch.randint(0, 5, (N, 2), device=dev)
    bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
    acts_seq.append(torch.stack([mv, bmb], dim=-1))

# sim_t 完整
torch.manual_seed(0)
rews_t, dones_t, deads_t, wins_t = [], [], [], []
for a in acts_seq:
    r, d, inf = sim_t.step(a)
    rews_t.append(r); dones_t.append(d)
    deads_t.append(inf["died"]); wins_t.append(inf["winner"])
sync()
# sim_k triton full
torch.manual_seed(0)
rews_k, dones_k, deads_k, wins_k = [], [], [], []
for a in acts_seq:
    r, d, inf = triton_step_full(sim_k, a)
    rews_k.append(r); dones_k.append(d)
    deads_k.append(inf["died"]); wins_k.append(inf["winner"])
sync()

worst = {}
for f in FIELDS:
    worst[f] = int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
max_r = max((rews_t[i] - rews_k[i]).abs().max().item() for i in range(T))
n_d = sum(int((dones_t[i] != dones_k[i]).sum()) for i in range(T))
n_dead = sum(int((deads_t[i] != deads_k[i]).sum()) for i in range(T))
n_win = sum(int((wins_t[i] != wins_k[i]).sum()) for i in range(T))
print(f"reward maxdiff: {max_r:.2e} | done差 {n_d} | died差 {n_dead} | winner差 {n_win}")
ok = max_r < 1e-4 and n_d == 0 and n_dead == 0 and n_win == 0
for f, v in worst.items():
    print(f"  {f:<14}: 差 {v} {'✓' if v == 0 else '✗'}")
    ok &= v == 0
print(f"\nStep2 完整 step: {'全部一致 ✓' if ok else '有差异 ✗'}")

# 速度
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
t_k = bt(lambda: triton_step_full(sim_k, acts_seq[-1]))
print(f"step 同步稳态: torch {t_t:.1f} ms | triton_full {t_k:.1f} ms | x{t_t/t_k:.1f}")
