"""服务器 CPU 多 seed 快速 eval：duel_cnn vs astar/hunter（N=512 并行）。
对照本地 MacBook reliable3 的 0.463 —— 验证 CPU 架构/RNG 是否影响 astar 行为。"""
import sys, time; sys.path.insert(0, ".")
import torch
torch.compile = lambda fn, **kw: fn
import sim.torch_sim as ts
ts._HAS_TRITON = False
from sim.config import SimConfig
from sim.factory import make_sim
from sim.bots import make_bot
from train.model import ActorCritic

bot_kind = sys.argv[1] if len(sys.argv) > 1 else "astar"
ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "ckpt/duel_cnn.pt"
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 256
N = int(sys.argv[4]) if len(sys.argv) > 4 else 512
seeds = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1, 2, 3]

cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800, open_fraction=1.0,
                open_growth_bombs=8, open_growth_blast=6, open_growth_speed=1.68)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = ActorCritic(tuple(ck["obs_shape"]), arch="cnn", n_players=2)
model.load_state_dict(ck["model"]); model.eval()
tot = [0, 0, 0]
t0 = time.time()
for seed in seeds:
    torch.manual_seed(seed); random_seed = seed
    import random as _r; _r.seed(random_seed)
    sim = make_sim(cfg, N, backend="torch", device="cpu", seed=seed)
    bot = make_bot(sim, bot_kind)
    win = draw = loss = 0; guard = 0
    while (win + draw + loss) < episodes and guard < 6000:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        with torch.no_grad():
            a0, _, _ = model.act(obs, mm[:, 0], bm[:, 0], 0)
        a1 = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        rew, done, info = sim.step(torch.stack([a0, a1], 1), auto_reset=True)
        if bool(done.any()):
            w0 = info["winner"][:, 0]
            win += int((done & w0).sum()); loss += int((done & info["winner"][:, 1]).sum())
            draw += int((done & ~w0 & ~info["winner"][:, 1]).sum())
        guard += 1
    tot[0] += win; tot[1] += draw; tot[2] += loss
    print(f"  seed={seed} {win}W/{draw}D/{loss}L = {win/max(1,win+draw+loss):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
n = tot[0] + tot[1] + tot[2]
print(f"CPUFAST-RESULT open80 {bot_kind}: {tot[0]}W/{tot[1]}D/{tot[2]}L = "
      f"{tot[0]/max(1,n):.3f} ({n} games, {len(seeds)} seeds)", flush=True)
