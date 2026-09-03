#!/usr/bin/env python3
"""评估 LSTM 1B 训练的当前 ckpt 对固定规则 bot 的胜率。

默认对手 hunter（astar 纯进攻 + 吃道具），可用 --opponents 换 greedy/astar。

在 CPU 上跑（训练进程 191501 占满 NPU 61.6/64GB，无法再放第二个模型），
N=128 小 sim，通常几十秒完成，不影响训练。

用法：
  python3 eval_lstm_vs_hunter.py [ckpt 路径] [对手名] [局数]
  例：python3 eval_lstm_vs_hunter.py ckpt/lstm_1b.pt hunter 128
"""

import os
import sys
import time

# 必须在 import torch 之前：禁止 torch_npu 自动加载（评估在 CPU 上跑，
# torchair 会把算子派发到被训练进程占用的 NPU 0 上 → aivec 异常抢卡）。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

# CPU 评估：让 torch.compile 变 no-op。服务器 torch_sim._danger_c 用
# torch.compile(backend='npu') 分档编译 danger_map —— CPU 张量进 NPU 图会
# device 不匹配（danger 落 npu:0）并和训练进程抢卡。no-op 后走未编译的
# 纯 torch danger_map，位级一致，CPU 安全。
torch.compile = lambda fn, **kw: fn

sys.path.insert(0, ".")

import sim.torch_sim as _torch_sim
# CPU 评估禁用 triton kernel：triton-ascend 的 kernel 只在 NPU 上执行，
# CPU 张量的 host 指针被当 NPU 地址 → aivec 矢量核异常（error mask
# 0x6500020bd00028c，见 sim/dev.py docstring / HANDOFF_20260810）。
_torch_sim._HAS_TRITON = False

from sim.config import SimConfig
from sim.factory import make_sim
from sim.bots import make_bot
from train.model import ActorCritic
from train.ppo import PPOConfig, SelfPlayRunner


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/lstm_1b.pt"
    kind = sys.argv[2] if len(sys.argv) > 2 else "hunter"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    if kind not in ("random", "greedy", "astar", "hunter"):
        raise SystemExit(f"未知 bot 类型: {kind}")

    device = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_shape = tuple(ck["obs_shape"])
    ck_np = ck.get("n_players")
    assert ck_np == 2, f"期望 1v1 ckpt，实际 n_players={ck_np}"
    learner = ActorCritic(obs_shape, arch="lstm", n_players=ck_np).to(device)
    learner.load_state_dict(ck["model"])
    learner.eval()
    step = ck.get("global_step", "?")
    print(f"[eval] ckpt={ckpt_path} step={step} arch=lstm obs={obs_shape} "
          f"device={device}", flush=True)

    cfg = SimConfig()                      # 与训练一致：map_mode=open 1v1
    sim = make_sim(cfg, 128, backend="torch", device="cpu", seed=0)
    opp = make_bot(sim, kind)
    print(f"[eval] 对手 {kind}  episodes={episodes}", flush=True)

    pcfg = PPOConfig(rollout_steps=128, bptt_window=8)
    runner = SelfPlayRunner(sim, learner, [opp], pcfg, 1.0)
    runner.clear_stats()
    guard = 0
    t0 = time.time()
    while runner.ep_stats["count"] < episodes and guard < 16:
        runner.collect()
        guard += 1
        n = runner.ep_stats["count"]
        print(f"  [collect {guard}] 局数={n} 胜率={runner.win_rate():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    s = runner.ep_stats
    wr = runner.win_rate()
    print(f"\n=== 结果（{s['count']} 局, {time.time()-t0:.0f}s）===")
    print(f"vs {kind}: win={s['win']} draw={s['draw']} loss={s['loss']} "
          f"胜率={wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
