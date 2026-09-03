"""CNN update 热点 op profile + MLP 架构对比（910B 小卷积效率 vs GEMM）。"""
import sys, time, torch, torch.profiler as prof
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
obs_shape = cfg.obs_shape

def sync():
    torch.npu.synchronize()

def build_buf(learner):
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
    return buf, last_val

def time_update(learner_net):
    opt = torch.optim.Adam(learner_net.parameters(), lr=ppo_cfg.lr)
    try:
        ppo_update(learner_net, opt, buf, last_val, ppo_cfg, 0.05)
        sync()
        t0 = time.perf_counter()
        ppo_update(learner_net, opt, buf, last_val, ppo_cfg, 0.05)
        sync()
        return (time.perf_counter() - t0) * 1000
    except Exception as ex:
        print(f"  FAIL {type(ex).__name__}: {str(ex)[:120]}")
        return float("nan")

print("== MLP vs CNN update ==")
learner_cnn = ActorCritic(obs_shape, arch="cnn").to(dev).eval()
buf, last_val = build_buf(learner_cnn)
t_cnn = time_update(learner_cnn)
print(f"CNN update: {t_cnn:8.1f} ms")
learner_mlp = ActorCritic(obs_shape, arch="mlp").to(dev).eval()
t_mlp = time_update(learner_mlp)
print(f"MLP update: {t_mlp:8.1f} ms  x{t_cnn/t_mlp:.2f}")

print("\n== CNN update 热点 op（CPU self Top20）==")
opt = torch.optim.Adam(learner_cnn.parameters(), lr=ppo_cfg.lr)
with prof.profile(activities=[prof.ProfilerActivity.CPU]) as p:
    ppo_update(learner_cnn, opt, buf, last_val, ppo_cfg, 0.05)
    sync()
kavg = p.key_averages()
tot = sum(e.self_cpu_time_total for e in kavg)
print(f"总 CPU self: {tot/1e3:.0f} ms, 事件数: {sum(e.count for e in kavg)}")
rows = sorted(kavg, key=lambda e: -e.self_cpu_time_total)[:20]
for e in rows:
    print(f"  {e.self_cpu_time_total/1e3:8.1f} ms x{e.count:6d}  {e.key}")
