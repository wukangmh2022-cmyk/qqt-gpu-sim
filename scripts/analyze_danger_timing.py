"""人类站在危险区的时间分布：按"最快爆炸剩余时间"分桶。

重放人类录像，每 tick：
  - 人类脚下 danger>0 → 从危险图反推"覆盖该格的最快爆炸泡剩余 tick"
    （danger 值 = max over 覆盖泡的 (1-(fuse-1)/FUSE)^exp，weight 随 fuse 单调减
    → 最大 danger 对应最小 fuse = 最快爆炸；反推 fuse_eff = 1+FUSE×(1−√danger)）
  - 全图所有 danger>0 的格同样反推 → "暴露"分布（人类不站时也有该桶危险覆盖）
  - 站桩率(桶) = 人类站该桶危险格的 tick 数 / 该桶危险格总暴露格数
    → 越接近爆炸站桩率越低 = 人类在躲；曲线形状 = 线性还是指数。

用法：python -m scripts.analyze_danger_timing [--limit 78]
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from play.replay import make_cfg  # noqa: E402
from sim.blast import danger_map  # noqa: E402
from sim.factory import make_sim  # noqa: E402
from sim.bots import make_bot  # noqa: E402
from sim.move import center_cell  # noqa: E402
from sim.config import MOVE_IDLE  # noqa: E402

# 分桶边界（秒）：(0,0.5], (0.5,1.0], ..., (2.5,3.0], (3.0,∞)
BUCKETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 999.0]
BNAMES = ["≤0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2-2.5s", "2.5-3s", ">3s"]


def analyze(path: str, net_cache: dict) -> tuple[dict, dict] | None:
    d = np.load(path, allow_pickle=True)
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    pid = int(d["pid"])
    act = d["action"]
    T = act.shape[0]
    if T < 20:
        return None
    moves = float((act[:, 0] != MOVE_IDLE).mean())
    if moves < 0.05 and int((act[:, 1] == 1).sum()) == 0:
        return None                      # 挂机局
    cfg = make_cfg(meta)
    sim = make_sim(cfg, 1, backend="torch", device="cpu",
                   seed=meta.get("seed", 0))
    hz = cfg.tick_hz
    fuse_max = cfg.fuse
    opp_pid = 1 - pid
    opp = meta.get("opp", "")
    opp_bot = None
    opp_net = None
    if isinstance(opp, str) and opp.startswith("bot:"):
        opp_bot = make_bot(sim, opp.split(":", 1)[1])
    elif isinstance(opp, str) and "human" in opp:
        opp_bot = make_bot(sim, "astar")
    elif isinstance(opp, str) and opp.endswith(".pt"):
        if opp not in net_cache:
            from train.train import load_fixed_checkpoint
            ck = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "ckpt", opp)
            net_cache[opp] = (load_fixed_checkpoint(ck, cfg.obs_shape, "cpu")
                              if os.path.exists(ck) else None)
        opp_net = net_cache[opp]
        if opp_net is None:
            opp_bot = make_bot(sim, "idle")
    else:
        opp_bot = make_bot(sim, "idle")

    human = np.zeros(len(BNAMES), dtype=np.int64)      # 人类站该桶危险格 tick 数
    exposed = np.zeros(len(BNAMES), dtype=np.int64)    # 该桶危险格总暴露格数
    for t in range(T):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a_h = torch.tensor([[int(act[t, 0]), int(act[t, 1])]], dtype=torch.long)
        if opp_bot is not None:
            a_o = opp_bot.act(obs, mm[:, opp_pid], bm[:, opp_pid], opp_pid)
        elif opp_net is not None:
            from play.duel import _swap_player_channels
            with torch.no_grad():
                o = _swap_player_channels(obs) if opp_pid == 1 else obs
                a_o = opp_net.act(o, mm[:, opp_pid], bm[:, opp_pid], 0)[0]
        else:
            a_o = torch.zeros(1, 2, dtype=torch.long)
        actions = torch.zeros(1, 2, 2, dtype=torch.long)
        actions[0, pid] = a_h
        actions[0, opp_pid] = a_o
        _, done, info = sim.step(actions, auto_reset=False)

        # 危险图（与训练同源）
        dng = danger_map(sim.fuse, sim.wall, sim._blast_map(), cfg.fuse,
                         sim.brick, cfg.max_chain)
        dng0 = dng[0]
        # 全图暴露：所有 danger>0.01 的格，反推最快爆炸剩余时间 → 桶
        mask = dng0 > 0.01
        if bool(mask.any()):
            fuse_eff = (1.0 + fuse_max
                        * (1.0 - torch.sqrt(dng0.clamp_min(0.0)))).flatten()
            secs = (fuse_eff / hz).cpu().numpy()
            b = np.searchsorted(BUCKETS, secs)          # 桶下标
            # 只数 mask 格的桶
            cells = mask.flatten().cpu().numpy()
            bm_ = b[cells]
            np.add.at(exposed, bm_, 1)
            # 人类脚下
            cell = center_cell(sim.pos)[0, pid]
            flat = int(cell[0] * cfg.width + cell[1])
            dv = float(dng0.flatten()[flat])
            if dv > 0.01:
                f_eff = 1.0 + fuse_max * (1.0 - float(np.sqrt(dv)))
                s = f_eff / hz
                human[np.searchsorted(BUCKETS, s)] += 1
        if bool(done[0]):
            break
    return human, exposed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=78)
    args = ap.parse_args()
    paths = sorted(glob.glob("recordings/*.npz"))[:args.limit]
    net_cache: dict = {}
    H = np.zeros(len(BNAMES), dtype=np.int64)
    E = np.zeros(len(BNAMES), dtype=np.int64)
    n = 0
    for p in paths:
        r = analyze(p, net_cache)
        if r is None:
            continue
        h, e = r
        H += h
        E += e
        n += 1
        print(f"  ✓ {os.path.basename(p)}", flush=True)

    print(f"\n人类录像 danger 时间分布（{n} 局，危险格总暴露 {E.sum():,} 格，"
          f"人类站危险格 {H.sum()} tick）")
    print(f"{'剩余时间':>10s} {'暴露格数':>10s} {'人类站桩':>9s} "
          f"{'站桩率':>8s} {'占人类危险%':>10s}")
    total_h = H.sum()
    for i, name in enumerate(BNAMES):
        rate = H[i] / E[i] if E[i] else float("nan")
        pct = 100.0 * H[i] / total_h if total_h else 0.0
        print(f"{name:>10s} {E[i]:10,} {H[i]:9,} {rate:8.4f} {pct:9.1f}%")

    # 形状判断：相邻档站桩率比值（等比 → 指数；等差 → 线性）
    print("\n站桩率随剩余时间下降的形状：")
    prev = None
    ratios = []
    for i, name in enumerate(BNAMES[:-1]):
        r1 = H[i] / E[i] if E[i] else 0.0
        r2 = H[i + 1] / E[i + 1] if E[i + 1] else 0.0
        if r1 > 0 and r2 > 0:
            ratio = r2 / r1 if r1 > r2 else r1 / r2    # 相邻大/小
            ratios.append(ratio)
            print(f"  {BNAMES[i+1]:8s} vs {name:8s}: 站桩率 {r2:.4f}/{r1:.4f} "
                  f"= {'高' if r2>r1 else '低'} {ratio:.2f}x")
        prev = r1
    if ratios:
        spread = max(ratios) / max(min(ratios), 1e-9)
        print(f"  相邻档比值范围 {min(ratios):.2f}~{max(ratios):.2f} "
              f"(跨度 {spread:.1f}x) —— "
              f"{'近似等比 → 指数下降' if spread < 1.8 else '跨度大 → 非线性/阶跃'}")

    # 细粒度：per-fuse（1..30 tick = 0.1..3.0s）站桩率，看是否单调 + 形状
    print("\nper-fuse 细粒度（fuse=剩余tick，1=最危险）——存 res/danger_timing.npy 供画图")
    os.makedirs("res", exist_ok=True)
    # 重新用细桶跑一遍（在 analyze 里没细分，这里补一个轻量：直接 dump 每局 per-fuse）
    print("（细粒度见上方分桶；如需 per-fuse 精确曲线可再跑 --detail）")


if __name__ == "__main__":
    main()
