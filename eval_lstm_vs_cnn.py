#!/usr/bin/env python3
"""LSTM（lstm_course.pt）vs CNN（duel_cnn.pt）1v1 胜率测试（v2，修正视角）。

v2 修正：**对手网络必须用 pid=0 视角 + 观测重排**（训练时 learner 恒为
player 0，只有 pid=0 视角被优化；物理 P1 的模型用 _swap_player_channels
把自己搬到通道 0 再 pid=0，见 play/duel.py）。上一版用 SelfPlayRunner 会让
CNN 走 _opponent_actions 的 pid=1 乱视角（"把对手泡数当自己"）→ 假完胜。
本版自写对打循环：P0=LSTM(local 特征)，P1=CNN(重排+pid=0)，胜负按
info['winner'] 统计。

本地 MacBook CPU，N=128。用法：python3 eval_lstm_vs_cnn.py [地图] [局数]
地图：open / corridor / cnn（duel_cnn 训练分布）/ pillar / ring
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
from sim.obs import local_view_features
from train.model import ActorCritic

MAPS = {
    # "open" 与 play/duel.py --map-mode open 完全同一环境：map_mode=corridor +
    # open_fraction=1.0 + 默认 80% 成长上限（泡8/威6/速1.68）+ 1800 tick。
    # 注意：SimConfig(map_mode="open") 是"纯 open 固定能力无成长"分支（3/3/1.0），
    # 和 launcher 不是一个环境，之前用它测出 100% 是假象（见 git log）。
    "open":      SimConfig(map_mode="corridor", max_steps=1800, open_fraction=1.0,
                           open_growth_bombs=8, open_growth_blast=6,
                           open_growth_speed=1.68),
    "corridor":  SimConfig(map_mode="corridor", open_fraction=0.0),
    "cnn":       SimConfig(map_mode="corridor", open_fraction=0.34,
                           ring_fraction=0.33),   # duel_cnn 的训练分布
    "pillar":    SimConfig(map_mode="open", wall_density=0.5),
    "ring":      SimConfig(map_mode="corridor", open_fraction=0.0,
                           ring_fraction=1.0),
}


def _swap_player_channels(obs: torch.Tensor) -> torch.Tensor:
    """物理 P1 的模型"自己 = 通道 0"：per-player 通道 0↔1 互换（同 play/duel.py）。"""
    p = 2
    base = 2 * p + 3
    c = obs.shape[1]
    idx = list(range(c))
    segs = [range(0, p), range(p, 2 * p)]
    if c > base:
        segs += [range(base + 1, base + 1 + p),
                 range(base + 1 + p, base + 1 + 2 * p),
                 range(base + 1 + 2 * p, base + 1 + 3 * p)]
    for seg in segs:
        seg = list(seg)
        idx[seg[0]], idx[seg[1]] = idx[seg[1]], idx[seg[0]]
    return obs[:, idx]


def main() -> None:
    map_name = sys.argv[1] if len(sys.argv) > 1 else "open"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    if map_name not in MAPS:
        raise SystemExit(f"未知地图: {map_name}（可选 {list(MAPS)}）")

    device = torch.device("cpu")
    ck_l = torch.load("ckpt/lstm_1b_min.pt", map_location="cpu", weights_only=False)
    ck_c = torch.load("ckpt/duel_cnn_min.pt", map_location="cpu", weights_only=False)
    lstm_step = ck_l.get("global_step", "?")
    cnn_step = ck_c.get("global_step", "?")

    learner = ActorCritic(tuple(ck_l["obs_shape"]), arch="lstm",
                          n_players=2).to(device)
    learner.load_state_dict(ck_l["model"])
    learner.eval()
    for p in learner.parameters():
        p.requires_grad_(False)

    cnn = ActorCritic(tuple(ck_c["obs_shape"]), arch="cnn",
                      n_players=2).to(device)
    cnn.load_state_dict(ck_c["model"])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    print(f"[eval] LSTM(lstm_1b step={lstm_step}) vs "
          f"CNN(duel_cnn step={cnn_step} elo={ck_c.get('elo')}) "
          f"map={map_name} episodes={episodes} device=cpu", flush=True)

    cfg = MAPS[map_name]
    sim = make_sim(cfg, 128, backend="torch", device="cpu", seed=0)
    hidden = None
    win = draw = loss = 0
    t0 = time.time()
    guard = 0
    while (win + draw + loss) < episodes and guard < 3000:
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        # P0 = LSTM：局部特征（only_p0 省一半计算），hidden 沿 tick 传递
        lf = local_view_features(sim.cfg, obs, sim.pos, sim.alive, sim.t,
                                 sim.fuse, sim.hp, only_p0=True)
        feats = (lf[0][:, 0], lf[1][:, 0], lf[2][:, 0])
        with torch.no_grad():
            a0, _, _, hidden = learner.act(feats, mm[:, 0], bm[:, 0], 0, hidden)
        # P1 = CNN：观测重排（自己→通道0）+ pid=0 视角（训练优化视角）
        with torch.no_grad():
            a1, _, _ = cnn.act(_swap_player_channels(obs),
                               mm[:, 1], bm[:, 1], 0)
        rew, done, info = sim.step(torch.stack([a0, a1], dim=1),
                                   auto_reset=False)
        if bool(done.any()):
            w0 = info["winner"][:, 0]            # (N,) 该局 P0（LSTM）赢
            win += int((done & w0).sum())
            loss += int((done & info["winner"][:, 1]).sum())
            draw += int((done & ~w0 & ~info["winner"][:, 1]).sum())
            hidden = None                        # 对局结束，LSTM 记忆清零
        guard += 1
        if (win + draw + loss) % 64 == 0 or guard % 100 == 0:
            n = win + draw + loss
            wr = win / max(1, n)
            print(f"  [tick {guard}] 局数={n} LSTM胜率={wr:.3f} "
                  f"({win}胜/{draw}平/{loss}负, {time.time()-t0:.0f}s)", flush=True)
    n = win + draw + loss
    wr = win / max(1, n)
    print(f"\n=== 结果（{n} 局, {time.time()-t0:.0f}s, map={map_name}）===")
    print(f"LSTM vs CNN: win={win} draw={draw} loss={loss} "
          f"LSTM胜率={wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
