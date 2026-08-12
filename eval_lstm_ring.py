#!/usr/bin/env python3
"""LSTM 泛化性测试：在**训练没见过的地图**上评估模型胜率。

设计（用户定）：环岛/特殊设计地图只用于测泛化，不进训练课程。训练中立柱
数量超过 ~10 块（wall_density ≥ 0.35，s3 起）后，定期在这里测一次，看模型
对没见过地图形态的泛化能力。

CPU 上跑（训练进程占满 NPU），N=128，几十秒完成。

用法：
  python3 eval_lstm_ring.py [ckpt] [map] [对手] [局数]
  map ∈ {ring, corridor, open, pillar}
    ring     环岛（中间永久墙山体 + 环带可炸墙）—— 纯泛化
    corridor 顶墙 + 左右可炸墙
    open     纯空场（训练见过的）
    pillar   open + 随机立柱 0.5（训练 s4/s5 见过的）
  对手 ∈ {random, greedy, astar, hunter}
  例：python3 eval_lstm_ring.py ckpt/lstm_course.pt ring hunter 128
"""

import os
import sys
import time

# 必须在 import torch 之前：禁止 torch_npu 自动加载（CPU 评估，防派发到
# 被训练占用的 NPU）。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

# CPU 评估：torch.compile no-op（服务器 _danger_c 用 backend='npu' 分档编译，
# CPU 张量进 NPU 图会 device 不匹配 + 抢卡）。
torch.compile = lambda fn, **kw: fn

sys.path.insert(0, ".")

import sim.torch_sim as _torch_sim
# 禁用 triton kernel（triton-ascend 只在 NPU 执行，CPU 张量会被当 NPU 地址）。
_torch_sim._HAS_TRITON = False

from sim.config import SimConfig
from sim.factory import make_sim
from sim.bots import make_bot
from train.model import ActorCritic
from train.ppo import PPOConfig, SelfPlayRunner

MAPS = {
    "ring":      SimConfig(map_mode="corridor", open_fraction=0.0,
                           ring_fraction=1.0),
    "corridor":  SimConfig(map_mode="corridor", open_fraction=0.0),
    "open":      SimConfig(map_mode="open"),
    "pillar":    SimConfig(map_mode="open", wall_density=0.5),
}


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/lstm_course.pt"
    map_name = sys.argv[2] if len(sys.argv) > 2 else "ring"
    kind = sys.argv[3] if len(sys.argv) > 3 else "hunter"
    episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    if map_name not in MAPS:
        raise SystemExit(f"未知地图: {map_name}（可选 {list(MAPS)}）")
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
    print(f"[eval] ckpt={ckpt_path} step={step} arch=lstm "
          f"map={map_name} 对手={kind} episodes={episodes}", flush=True)

    cfg = MAPS[map_name]
    sim = make_sim(cfg, 128, backend="torch", device="cpu", seed=0)
    opp = make_bot(sim, kind)
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
    print(f"\n=== 泛化结果（{s['count']} 局, {time.time()-t0:.0f}s）===")
    print(f"map={map_name} vs {kind}: win={s['win']} draw={s['draw']} "
          f"loss={s['loss']} 胜率={wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
