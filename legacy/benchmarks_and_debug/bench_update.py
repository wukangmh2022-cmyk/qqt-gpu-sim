"""ppo_update 加速组合实测：baseline vs autocast vs compile(learner) vs 组合。

目标：update 6134ms（N=4096）→ 10x。
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

dev = pick_device()
assert dev.startswith("npu")
N = 4096
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.10)
ppo_cfg = PPOConfig()

def sync():
    torch.npu.synchronize()

# learner + sim + 真实 buffer
obs_shape = cfg.obs_shape
learner = ActorCritic(obs_shape, arch="cnn").to(dev).eval()
sim = BatchedSim(cfg, N, device=dev, seed=0)
sim.reset_all()
buf = RolloutBuffer(ppo_cfg.rollout_steps, N, obs_shape, dev,
                    obs_dtype=torch.float16 if cfg.obs_fp16 else torch.float32)
mv = torch.randint(0, 5, (N, 2), device=dev)
acts = torch.stack([mv, torch.ones(N, 2, dtype=torch.long, device=dev)], dim=-1)
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

def time_update(learner_net, opt, autocast=False, epochs=None):
    c = PPOConfig()
    if epochs is not None:
        c.epochs = epochs
    try:
        ppo_update(learner_net, opt, buf, last_val, c, 0.05,
                   autocast=autocast)
        sync()
        t0 = time.perf_counter()
        ppo_update(learner_net, opt, buf, last_val, c, 0.05,
                   autocast=autocast)
        sync()
        return (time.perf_counter() - t0) * 1000
    except Exception as ex:
        print(f"  FAIL {type(ex).__name__}: {str(ex)[:120]}")
        return float("nan")

print("== ppo_update 组合对比（N=4096, rollout=128）==")
# baseline
opt = torch.optim.Adam(learner.parameters(), lr=ppo_cfg.lr)
t = time_update(learner, opt)
print(f"baseline (fp32 eager, ep=4): {t:8.1f} ms")

# autocast
opt = torch.optim.Adam(learner.parameters(), lr=ppo_cfg.lr)
t = time_update(learner, opt, autocast=True)
print(f"autocast fp16      (ep=4): {t:8.1f} ms")

# compile learner
try:
    torch._dynamo.reset()
    learner_c = torch.compile(learner, backend="npu", dynamic=False)
    opt = torch.optim.Adam(learner_c.parameters(), lr=ppo_cfg.lr)
    t = time_update(learner_c, opt)
    print(f"compile(AutoFuse)  (ep=4): {t:8.1f} ms")
    # compile + autocast
    opt = torch.optim.Adam(learner_c.parameters(), lr=ppo_cfg.lr)
    t = time_update(learner_c, opt, autocast=True)
    print(f"compile+autocast   (ep=4): {t:8.1f} ms")
    # compile + epochs=2
    opt = torch.optim.Adam(learner_c.parameters(), lr=ppo_cfg.lr)
    t = time_update(learner_c, opt, epochs=2)
    print(f"compile+autocast   (ep=2): {t:8.1f} ms")
except Exception as ex:
    print(f"compile FAIL {type(ex).__name__}: {str(ex)[:150]}")

# epochs 减半（无 compile）
opt = torch.optim.Adam(learner.parameters(), lr=ppo_cfg.lr)
t = time_update(learner, opt, epochs=2)
print(f"eager epochs=2     (ep=2): {t:8.1f} ms")
opt = torch.optim.Adam(learner.parameters(), lr=ppo_cfg.lr)
t = time_update(learner, opt, autocast=True, epochs=2)
print(f"autocast epochs=2  (ep=2): {t:8.1f} ms")
