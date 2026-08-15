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

# 7 种地图属性：**与 cnn_curriculum 训练 cfg 逐字段一致**（speed=3.0/max_steps=1800
# 为训练口径 —— 教训：旧版漏 speed（默认 3.6）导致评估环境≠训练环境，复测全失真；
# launcher/duel 也是 speed=3.0，故本表同时是 launcher 口径）。
MAPS = {
    # s2b-open80 训练 cfg：corridor + open_fraction=1.0 + 80% 成长上限（8/6/1.68）
    "open80":    SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                           open_fraction=1.0, open_growth_bombs=8,
                           open_growth_blast=6, open_growth_speed=1.68),
    # s2a-open40 训练 cfg：40% 成长（3/3/0.84，SimConfig 默认 open_growth_*）
    "open40":    SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                           open_fraction=1.0),
    # s5-pure-open 训练 cfg：纯 open 固定能力无成长（map_mode=open，speed=3.0）
    "pure-open": SimConfig(map_mode="open", speed=3.0, max_steps=1800,
                           open_fraction=0.0),
    # s3a-corridor 训练 cfg：纯走廊硬形态（顶墙4/通道5/边缘连续段 0.45）
    "corridor":  SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                           open_fraction=0.0, top_wall_rows=4, corridor_width=5,
                           wall_density=0.45),
    # s1-cnn-mix 训练 cfg：corridor + open_fraction=0.5（无 ring）
    "cnn-mix":   SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                           open_fraction=0.5),
    # s6-pillar 训练 cfg：map_mode=open + pillars 图案
    "pillar":    SimConfig(map_mode="open", speed=3.0, max_steps=1800,
                           open_fraction=0.0, wall_density=0.5),
    # 环岛（训练外泛化测试，用户指定）：corridor + ring_fraction=1.0
    "ring":      SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                           open_fraction=0.0,
                           ring_fraction=1.0),
}

BOTS = ("astar", "hunter", "greedy", "random")


def main():
    map_name = sys.argv[1] if len(sys.argv) > 1 else "open80"
    bot_kind = sys.argv[2] if len(sys.argv) > 2 else "astar"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    ckpt_path = sys.argv[5] if len(sys.argv) > 5 else "ckpt/duel_cnn_min.pt"
    cfg = MAPS[map_name]

    # 关键：astar/hunter 的 mode_ticker 走 torch 全局 RNG（aggressive/flee 随机），
    # 不固定会让每进程 bot 行为随机漂移 → 同 seed 不同局数结果差异巨大
    # （实测 open80 astar seed0: 128局 0.846 vs 512局 0.254）。固定 RNG 后
    # 每 seed 确定，多 seed 合并才是真实水平。
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        rew, done, info = sim.step(torch.stack([a0, a1], dim=1), auto_reset=True)
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
    # 纯数字行（locale 无关，避免中文提取失败）：RESULT <win> <draw> <loss> <wr>
    print(f"RESULT {win} {draw} {loss} {wr:.3f} games={n}", flush=True)


if __name__ == "__main__":
    main()
