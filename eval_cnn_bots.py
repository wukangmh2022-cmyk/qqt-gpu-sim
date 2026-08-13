#!/usr/bin/env python3
"""CNN（duel_cnn.pt）泛化能力本地摸底：7 种地图属性 × 4 种 bot。

目的：回答"现有 CNN 需要练习哪些泛化能力、对打寻路 AI（astar/hunter）
打到多少"——据此设计下一步训练课程（地图/敌人渐进）。

口径与 play/duel.py --map-mode open 完全一致（corridor + open_fraction=1.0
+ open_growth_pct → 8/6/1.68 @80%）；其余地图按 SimConfig 字面语义。

用法：python3 eval_cnn_bots.py [地图] [bot] [局数] [seed]
  地图：open80 / open40 / pure-open / corridor / cnn-mix / pillar / ring
  bot ：astar / hunter / greedy / random
  seed：make_sim 的初始 seed（默认 0）。astar 的模式随机走全局 RNG，单 seed
  方差大，结论应跑多 seed 合并（见 eval_cnn_bots.sh）。
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
from sim.bots import make_bot
from train.model import ActorCritic

# 7 种地图属性：launcher/duel 口径 + 训练分布 + 泛化变体
MAPS = {
    # duel.py open 默认：corridor + open_fraction=1.0 + 80% 成长上限（8/6/1.68）
    "open80":    SimConfig(map_mode="corridor", max_steps=1800, open_fraction=1.0,
                           open_growth_bombs=8, open_growth_blast=6,
                           open_growth_speed=1.68),
    # 40% 成长上限（4/3/0.84）—— 成长初值变化
    "open40":    SimConfig(map_mode="corridor", max_steps=1800, open_fraction=1.0,
                           open_growth_bombs=4, open_growth_blast=3,
                           open_growth_speed=0.84),
    # 纯 open 固定能力无成长（3/3/1.0，无宝箱）—— CNN 可能的老训练环境
    "pure-open": SimConfig(map_mode="open", max_steps=1800),
    # 纯走廊（顶墙 4 行，无 open 区）
    "corridor":  SimConfig(map_mode="corridor", open_fraction=0.0),
    # duel_cnn 训练分布（corridor + 34% open + 33% ring）
    "cnn-mix":   SimConfig(map_mode="corridor", open_fraction=0.34,
                           ring_fraction=0.33),
    # pillars 图案空地
    "pillar":    SimConfig(map_mode="open", wall_density=0.5),
    # 环岛（中心山体 + 环带宝箱）
    "ring":      SimConfig(map_mode="corridor", open_fraction=0.0,
                           ring_fraction=1.0),
}

BOTS = ("astar", "hunter", "greedy", "random")


def main():
    map_name = sys.argv[1] if len(sys.argv) > 1 else "open80"
    bot_kind = sys.argv[2] if len(sys.argv) > 2 else "astar"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    cfg = MAPS[map_name]

    ck = torch.load("ckpt/duel_cnn_min.pt", map_location="cpu", weights_only=False)
    cnn = ActorCritic(tuple(ck["obs_shape"]), arch="cnn", n_players=2)
    cnn.load_state_dict(ck["model"])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)
    cnn_step = ck.get("global_step", "?")
    print(f"[eval] CNN(duel_cnn step={cnn_step}) vs {bot_kind} "
          f"map={map_name} episodes={episodes} seed={seed}", flush=True)

    sim = make_sim(cfg, 128, backend="torch", device="cpu", seed=seed)
    bot = make_bot(sim, bot_kind)

    win = draw = loss = 0
    guard = 0
    t0 = time.time()
    while (win + draw + loss) < episodes and guard < 6000:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        with torch.no_grad():
            a0, _, _ = cnn.act(obs, mm[:, 0], bm[:, 0], 0)
        a1 = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        rew, done, info = sim.step(torch.stack([a0, a1], dim=1), auto_reset=False)
        if bool(done.any()):
            w0 = info["winner"][:, 0]
            win += int((done & w0).sum())
            loss += int((done & info["winner"][:, 1]).sum())
            draw += int((done & ~w0 & ~info["winner"][:, 1]).sum())
        guard += 1
        n = win + draw + loss
        if n % 128 == 0 and n > 0:
            print(f"  tick={guard} 局数={n} CNN胜率={win / n:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    n = win + draw + loss
    wr = win / max(1, n)
    print(f"=== CNN(duel_cnn) vs {bot_kind} @ {map_name}: "
          f"{win}胜/{draw}平/{loss}负 = {wr:.3f}（{n} 局, "
          f"{time.time() - t0:.0f}s, tick={guard}）===", flush=True)


if __name__ == "__main__":
    main()
