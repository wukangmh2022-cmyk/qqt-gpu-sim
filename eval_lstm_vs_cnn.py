#!/usr/bin/env python3
"""LSTM（lstm_course.pt）vs CNN（duel_cnn.pt）1v1 胜率测试。

本地 MacBook 上跑（CPU），N=128，默认 256 局。地图可选：
  open / corridor / pillar / ring（默认 open —— 两者训练都见过的纯空场）。

用法：python3 eval_lstm_vs_cnn.py [地图] [局数]
"""

import os
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

torch.compile = lambda fn, **kw: fn

sys.path.insert(0, ".")

import sim.torch_sim as _torch_sim
_torch_sim._HAS_TRITON = False

from sim.config import SimConfig
from sim.factory import make_sim
from train.model import ActorCritic
from train.ppo import PPOConfig, SelfPlayRunner

MAPS = {
    "open":      SimConfig(map_mode="open"),
    "corridor":  SimConfig(map_mode="corridor", open_fraction=0.0),
    "cnn":       SimConfig(map_mode="corridor", open_fraction=0.34,
                           ring_fraction=0.33),   # duel_cnn 的训练分布
    "pillar":    SimConfig(map_mode="open", wall_density=0.5),
    "ring":      SimConfig(map_mode="corridor", open_fraction=0.0,
                           ring_fraction=1.0),
}


def main() -> None:
    map_name = sys.argv[1] if len(sys.argv) > 1 else "open"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    if map_name not in MAPS:
        raise SystemExit(f"未知地图: {map_name}（可选 {list(MAPS)}）")

    device = torch.device("cpu")
    ck_l = torch.load("ckpt/lstm_course_min.pt", map_location="cpu", weights_only=False)
    ck_c = torch.load("ckpt/duel_cnn_min.pt", map_location="cpu", weights_only=False)
    lstm_step = ck_l.get("global_step", "?")
    cnn_step = ck_c.get("global_step", "?")

    learner = ActorCritic(tuple(ck_l["obs_shape"]), arch="lstm",
                          n_players=2).to(device)
    learner.load_state_dict(ck_l["model"])
    learner.eval()

    cnn = ActorCritic(tuple(ck_c["obs_shape"]), arch="cnn",
                      n_players=2).to(device)
    cnn.load_state_dict(ck_c["model"])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    print(f"[eval] LSTM(lstm_course step={lstm_step}) vs "
          f"CNN(duel_cnn step={cnn_step} elo={ck_c.get('elo')}) "
          f"map={map_name} episodes={episodes} device=cpu", flush=True)

    cfg = MAPS[map_name]
    sim = make_sim(cfg, 128, backend="torch", device="cpu", seed=0)
    pcfg = PPOConfig(rollout_steps=128, bptt_window=8)
    runner = SelfPlayRunner(sim, learner, [cnn], pcfg, 1.0)
    runner.clear_stats()
    guard = 0
    t0 = time.time()
    while runner.ep_stats["count"] < episodes and guard < 24:
        runner.collect()
        guard += 1
        n = runner.ep_stats["count"]
        print(f"  [collect {guard}] 局数={n} "
              f"LSTM胜率={runner.win_rate():.3f} ({time.time()-t0:.0f}s)", flush=True)
    s = runner.ep_stats
    wr = runner.win_rate()
    print(f"\n=== 结果（{s['count']} 局, {time.time()-t0:.0f}s, map={map_name}）===")
    print(f"LSTM vs CNN: win={s['win']} draw={s['draw']} loss={s['loss']} "
          f"LSTM胜率={wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
