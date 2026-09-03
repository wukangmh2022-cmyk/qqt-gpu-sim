"""astar bot.act 单次计时（真实训练 warmup 对手）@N=2048 open。"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.bots import make_bot
from sim.obs import legal_mask

dev = torch.device("npu:0")
N = 2048
cfg = SimConfig(map_mode="open", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
bot = make_bot(sim, "astar", mode=True)
obs = sim.observe()
mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                    sim.alive, sim.brick, sim.bombs_cap)
for _ in range(3):
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
torch.npu.synchronize()
t = (time.perf_counter() - t0) / 10 * 1000
print(f"astar bot.act: {t:7.2f} ms/tick  ({N/t*1e3/1e4:.2f}万 act/s)")
