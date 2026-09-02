"""训练管线全分解：collect 每 tick（obs+mask+act+step+buf）vs PPO update。

回答"瓶颈在模拟器还是网络"：N=16384，CNN learner，rollout_steps=128。
"""
import sys, time, torch
torch.manual_seed(0)
sys.path.insert(0, ".")
from sim.config import SimConfig
from sim.dev import pick_device
from sim.torch_sim import BatchedSim
from sim.obs import legal_mask
from train.model import ActorCritic
from train.ppo import RolloutBuffer, PPOConfig, ppo_update
from train.ppo import compute_gae  # noqa

dev = pick_device()
assert dev.startswith("npu")
N = 4096
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
ppo_cfg = PPOConfig()

def sync():
    torch.npu.synchronize()

def bench(fn, it=6):
    for _ in range(3):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    return (time.perf_counter() - t0) / it * 1000

# ---- learner + sim ----
obs_shape = cfg.obs_shape
learner = ActorCritic(obs_shape, arch="cnn").to(dev).eval()
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
buf = RolloutBuffer(ppo_cfg.rollout_steps, N, obs_shape, dev,
                    obs_dtype=torch.float16 if cfg.obs_fp16 else torch.float32)
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)

# 预热
for _ in range(3):
    sim.step(acts)
    obs = sim.observe()
    mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                        sim.alive, sim.brick, sim.bombs_cap)
    with torch.no_grad():
        a, lp, v = learner.act(obs, mm[:, 0], bm[:, 0], 0)
sync()

# ---- 单组件 ----
t_step = bench(lambda: sim.step(acts))
t_obs = bench(lambda: sim.observe())
t_mask = bench(lambda: legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                                  sim.alive, sim.brick, sim.bombs_cap))
def act_once():
    obs = sim.observe()
    mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                        sim.alive, sim.brick, sim.bombs_cap)
    with torch.no_grad():
        learner.act(obs, mm[:, 0], bm[:, 0], 0)
t_act = bench(act_once)
print(f"单组件: step {t_step:6.2f} | obs {t_obs:5.2f} | mask {t_mask:5.2f} | "
      f"act(含obs+mask) {t_act:6.2f} ms")

# 纯 act（不含 obs/mask）
obs = sim.observe()
mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                    sim.alive, sim.brick, sim.bombs_cap)
def act_pure():
    with torch.no_grad():
        learner.act(obs, mm[:, 0], bm[:, 0], 0)
t_act_pure = bench(act_pure)
print(f"纯网络前向 act: {t_act_pure:6.2f} ms/次")

# ---- 完整 collect tick（含 buf.add + 统计杂项，模拟 ppo.collect 主体） ----
def collect_tick(i):
    obs = sim.observe()
    mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                        sim.alive, sim.brick, sim.bombs_cap)
    with torch.no_grad():
        a0, logp, value = learner.act(obs, mm[:, 0], bm[:, 0], 0)
    reward, done, info = sim.step(acts)
    buf.add(obs, mm[:, 0], bm[:, 0], a0, logp, value, reward[:, 0],
            done.float(), torch.zeros(N, dtype=torch.bool, device=dev))
t_tick = bench(lambda i=0: collect_tick(i), it=8)
print(f"\n完整 collect tick: {t_tick:6.2f} ms  → {N/t_tick*1e3/1e4:.2f}万 env-steps/s")
print(f"  分解: step {t_step:.1f} ({t_step/t_tick*100:.0f}%) | obs {t_obs:.1f} "
      f"({t_obs/t_tick*100:.0f}%) | mask {t_mask:.1f} ({t_mask/t_tick*100:.0f}%) | "
      f"act {t_act_pure:.1f} ({t_act_pure/t_tick*100:.0f}%) | 其余 "
      f"{t_tick-t_step-t_obs-t_mask-t_act_pure:.1f} ms")

# ---- ppo_update（用真实 buffer 填满后跑一次）----
# 显式 index 填满 128 tick（避开 buf.ptr 语义）
for i in range(ppo_cfg.rollout_steps):
    obs = sim.observe()
    mm, bm = legal_mask(cfg, sim.wall, sim.fuse, sim.owner, sim.pos,
                        sim.alive, sim.brick, sim.bombs_cap)
    with torch.no_grad():
        a0, logp, value = learner.act(obs, mm[:, 0], bm[:, 0], 0)
    reward, done, info = sim.step(acts)
    buf.ptr = i
    buf.add(obs, mm[:, 0], bm[:, 0], a0, logp, value, reward[:, 0],
            done.float(), torch.zeros(N, dtype=torch.bool, device=dev))
sync()
with torch.no_grad():
    last_val = learner(sim.observe(), 0)[2]
# update 需要 opt
opt = torch.optim.Adam(learner.parameters(), lr=ppo_cfg.lr)
# 预热一次
try:
    ppo_update(learner, opt, buf, last_val, ppo_cfg, 0.05)
    sync()
    t0 = time.perf_counter()
    stats = ppo_update(learner, opt, buf, last_val, ppo_cfg, 0.05)
    sync()
    t_upd = (time.perf_counter() - t0) * 1000
except Exception as ex:
    print(f"ppo_update FAIL {type(ex).__name__}: {str(ex)[:150]}")
    t_upd = float("nan")
per_tick_upd = t_upd / ppo_cfg.rollout_steps
print(f"\nppo_update 一次: {t_upd:7.2f} ms (摊每 tick {per_tick_upd:.2f} ms)")
print(f"collect({t_tick:.1f}) + update摊({per_tick_upd:.1f}) = "
      f"每 tick {t_tick + per_tick_upd:.1f} ms → "
      f"{N/(t_tick+per_tick_upd)*1e3/1e4:.2f}万 env-steps/s")
