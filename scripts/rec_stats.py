"""录像数据统计：tick 分布 / 动作分布 / 追击片段占比 / 放炮间隔。

验证人类轨迹录得对不对、BC 数据质量如何。输出：
- 每局 tick 数分布（>2000 的可能是磨平局截断）
- 动作分布（移动方向 / IDLE / 放炮占比）—— 放炮占比应适中（~1-5%），
  太高 = 连炮一股脑放；太低 = 没在进攻
- 连续放炮间隔：间隔 ≥20 tick 的比例（人类冷静放炮 vs 无脑连丢）
- 追击片段：连续 10+ tick 朝对手方向移动的占比（教"追逃跑对手"）
"""

from __future__ import annotations

import ast
import glob
import os
import sys

import numpy as np

MOVES = {0: "上", 1: "下", 2: "左", 3: "右", 4: "停"}


def stats(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    obs, act = d["obs"], d["action"]
    T = obs.shape[0]
    n_move = int((act[:, 0] < 4).sum())
    n_idle = int((act[:, 0] == 4).sum())
    n_bomb = int((act[:, 1] == 1).sum())
    # 连续放泡间隔（tick）：两个放泡 tick 之间的差
    bomb_ticks = np.where(act[:, 1] == 1)[0]
    if len(bomb_ticks) >= 2:
        gaps = np.diff(bomb_ticks)
        calm = float((gaps >= 20).mean())          # ≥2s 冷静放炮占比
    else:
        calm = 1.0 if len(bomb_ticks) == 1 else 0.0
    # 追击片段：连续 tick 位置通道重心朝对手方向移动 —— 简化为"移动方向 ≠ 停"的
    # 连续跑动长度 ≥10 的占比（近似主动推进）
    moves = (act[:, 0] < 4).astype(np.int8)
    run = 0
    chase_ticks = 0
    for m in moves:
        if m:
            run += 1
        else:
            run = 0
        if run >= 10:
            chase_ticks += 1
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    return {
        "T": T, "idle": n_idle / max(1, T), "bomb": n_bomb / max(1, T),
        "calm": calm, "chase": chase_ticks / max(1, T),
        "map": meta.get("map", "?"), "opp": meta.get("opp", "?"),
        "pid": int(d["pid"]),
    }


def main() -> None:
    dir_ = sys.argv[1] if len(sys.argv) > 1 else "recordings"
    paths = sorted(glob.glob(os.path.join(dir_, "*.npz")))
    if not paths:
        print(f"{dir_}/ 无录像")
        return
    rows = [stats(p) for p in paths]
    tot_t = sum(r["T"] for r in rows)
    print(f"{len(rows)} 局，共 {tot_t} tick（≈{tot_t / 900:.1f} 局@1.5min）\n")
    print(f"{'局':<3}{'tick':>6}{'地图':<8}{'对手':<14}{'IDLE%':>7}{'放泡%':>7}"
          f"{'冷静放泡%':>9}{'追击%':>7}")
    for r in rows:
        print(f"{'-':<3}{r['T']:>6}{r['map']:<8}{str(r['opp'])[:13]:<14}"
              f"{100*r['idle']:>7.1f}{100*r['bomb']:>7.2f}{100*r['calm']:>9.1f}"
              f"{100*r['chase']:>7.1f}")
    # 汇总
    print("\n--- 汇总 ---")
    print(f"平均 tick/局: {tot_t / len(rows):.0f}")
    print(f"平均放泡%: {100*np.mean([r['bomb'] for r in rows]):.2f}  "
          f"(1.5min 局 ~20-30 次放泡 = 2-3%)")
    print(f"平均冷静放泡% (间隔≥20tick): {100*np.mean([r['calm'] for r in rows]):.1f}  "
          f"(高 = 人类精准放炮)")
    print(f"平均追击% (连续跑≥10tick): {100*np.mean([r['chase'] for r in rows]):.1f}")
    maps = {}
    for r in rows:
        maps[r["map"]] = maps.get(r["map"], 0) + 1
    print(f"地图分布: {maps}")


if __name__ == "__main__":
    main()
