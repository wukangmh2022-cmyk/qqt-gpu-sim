"""NPU 上评估：open80(80%成长) x astar/hunter —— 对比 CPU eval（验证 NPU astar
行为是否与 CPU 一致；训练内部 wr 0.694 vs CPU eval 0.463 的差异来源）。

用法：python3 eval_npu.py <bot> <ckpt> [episodes] [N]
2026-08-15: 修复 auto_reset bug（False 时 done 不复位 → 结果不可信，同 eval_cnn_bots）；
            ckpt 参数化；N 默认 512（训练占 NPU 时降内存）。
"""
import sys, time
sys.path.insert(0, ".")
import torch
torch.compile = lambda fn, **kw: fn
import sim.torch_sim as ts
ts._HAS_TRITON = False
from sim.config import SimConfig
from sim.factory import make_sim
from sim.bots import make_bot
from train.model import ActorCritic

bot_kind = sys.argv[1] if len(sys.argv) > 1 else "astar"
ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "ckpt/cnn_course.pt"
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 512
N = int(sys.argv[4]) if len(sys.argv) > 4 else 512

torch.manual_seed(0)
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800, open_fraction=1.0,
                open_growth_bombs=8, open_growth_blast=6, open_growth_speed=1.68)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = ActorCritic(tuple(ck["obs_shape"]), arch="cnn", n_players=2)
model.load_state_dict(ck["model"]); model.eval().to("npu:0")
sim = make_sim(cfg, N, backend="torch", device="npu:0", seed=0)
bot = make_bot(sim, bot_kind)
win = draw = loss = 0; guard = 0; t0 = time.time()
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
    n = win + draw + loss
    if n % 256 == 0 and n > 0:
        print(f"  tick={guard} 局数={n} NPU胜率={win/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
n = win + draw + loss
print(f"NPU-RESULT open80 {bot_kind}: {win}W/{draw}D/{loss}L = {win/max(1,n):.3f} ({n} games)", flush=True)
