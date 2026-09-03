"""DCU：MLP policy forward/evaluate 的 inductor compile 原型。

目标：旧 profile 里 learner.act @5632 batch 曾 113ms/tick（异常偏高）。
forward 是纯张量算子（无 RNG）→ 可整段 compile；sample/log_prob 用设备
RNG 会 graph-break，保持 eager。pid 固定（学习者恒 0）→ 视角权重重排
被烤进 kernel。对拍 compiled vs eager。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from train.model import ActorCritic
from sim.config import SimConfig

torch.manual_seed(0)
cfg = SimConfig()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5632
dev = "cuda"
net = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(dev)
obs = torch.randn(N, *cfg.obs_shape, dtype=torch.float16, device=dev)
mmask = torch.randint(0, 2, (N, 5), device=dev).bool()
bmask = torch.randint(0, 2, (N, 2), device=dev).bool()

# 学习者恒 pid=0：编译一个 pid 专用的 forward（inv_cols 重排烤进图）
fwd = torch.compile(net.forward, backend="inductor", dynamic=False)

def bench(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000

# eager forward
ml, bl, val = net.forward(obs, 0)
torch.cuda.synchronize()
te = bench(lambda: net.forward(obs, 0))
# compiled forward
try:
    mc, bc, vc = fwd(obs, 0)
    torch.cuda.synchronize()
    md = max((ml - mc).abs().max().item(), (bl - bc).abs().max().item(),
             (val - vc).abs().max().item())
    tc = bench(lambda: fwd(obs, 0))
    print(f"N={N} forward: eager {te:.2f}ms  inductor {tc:.2f}ms  "
          f"speedup {te/tc:.2f}x  maxdiff {md:.2e}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

# act 整段（sample 部分 eager）对照：compile 前后端到端 act 耗时
def act_eager():
    return net.act(obs, mmask, bmask, 0)
ta = bench(act_eager)
dm, db, val2 = net.dists(obs, mmask, bmask, 0)
am, ab = dm.sample(), db.sample()
t_s = time.perf_counter()
logp = dm.log_prob(am) + db.log_prob(ab)
print(f"N={N} act(eager fwd+sample) {ta:.2f}ms")
