"""真实训练配置（open 13x13 N=2048）的 collect 分解：为什么真实 17k 比 bench 低 2 倍。

对比 corridor(11x13) vs open(13x13)：step/obs/mask/act 时间。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
from sim.torch_sim import BatchedSim
from sim.obs import legal_mask
from train.model import ActorCritic

dev = pick_device()
assert dev.startswith("npu")
N = 2048

def sync():
    torch.npu.synchronize()

def bench(fn, it=10):
    for _ in range(4):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    return (time.perf_counter() - t0) / it * 1000

for mode in ("open", "corridor"):
    cfg = SimConfig(map_mode=mode, speed=3.0, max_steps=1800,
                    open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
    print(f"\n=== {mode}  obs_shape={cfg.obs_shape} ===")
    sim = BatchedSim(cfg, N, device=dev, seed=0)
    sim.reset_all()
    mv = torch.randint(0, 5, (N, 2), device=dev)
    acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
    learner = ActorCritic(cfg.obs_shape, arch="mlp").to(dev).eval()
    for _ in range(3):
        sim.step(acts)
        obs = sim.observe()
        mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                            sim.alive, sim.brick, sim.bombs_cap)
        with torch.no_grad():
            learner.act(obs, mm[:, 0], bm[:, 0], 0)
    sync()
    t_step = bench(lambda: sim.step(acts))
    t_obs = bench(lambda: sim.observe())
    t_mask = bench(lambda: legal_mask(cfg, sim.wall, sim.fuse, sim.owner,
                                      sim.pos, sim.alive, sim.brick,
                                      sim.bombs_cap))
    obs = sim.observe()
    mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                        sim.alive, sim.brick, sim.bombs_cap)
    t_act = bench(lambda: learner.act(obs, mm[:, 0], bm[:, 0], 0), it=20)
    t_collect = bench(lambda: (sim.step(acts), sim.observe(),
                               legal_mask(cfg, sim.wall, sim.fuse, sim.owner,
                                          sim.pos, sim.alive, sim.brick,
                                          sim.bombs_cap)), it=6)
    print(f"step {t_step:6.2f} | obs {t_obs:5.2f} | mask {t_mask:5.2f} | "
          f"act(MLP) {t_act:5.2f} | 三者合计 {t_collect:6.2f} ms/tick")
    print(f"纯 step SPS: {N/t_step*1e3/1e4:.2f}万 | collect SPS: {N/t_collect*1e3/1e4:.2f}万")
