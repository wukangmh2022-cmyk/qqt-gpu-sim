"""用人类录像验证奖励因子：分布方差 / 正交性 / 当前数值合理性。

方法：重放每局人类录像（人类侧 replay 录的动作，对手侧按 meta 决策，sim 真实
推进），逐 tick 拆解全部奖励因子的**原始输入值**（不带系数），得到每局每因子的
时间序列。然后：
  A. 每因子跨局分布：均值/σ/中位数/极差/非零占比（"这个信号天然多稀疏、多抖"）
  B. 因子间相关矩阵（正交性）：高相关 = 信号重叠/冗余，梯度被重复推
  C. 当前数值合理性：当前系数 × 单局典型触发 → 单局贡献，与 hit（±1.2）对比

因子定义（人类侧 pid）：
  dmg       本 tick 人类掉血点数（hit 的负项输入）
  dealt     本 tick 人类对对手造成伤害点数（hit 的正项输入）
  danger    人类脚下危险图原始值（0~1，×danger_penalty 才是惩罚）
  cover     放炮覆盖敌人人数（×place_cover_reward）
  chain     放炮连锁到的泡数 × 时间因子（×place_chain_reward）
  dist      近身定位 gain（×place_dist_reward）
  chainblst 爆炸时刻跨 owner 连锁的泡数 × 点火标志（×chain_blast_bonus）
  brick     人类踩到宝箱（0/1，×brick_reward）
  combo     combo 数 × 间隔因子（×combo_reward）
  step      恒 1（×step_penalty）
  term      ±1（死亡终局胜/负）、超时血量差比例（×win_bonus 体系）

用法：python -m scripts.analyze_reward_factors [--limit 78] [--out res/reward_factors.txt]
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
from play.replay import make_cfg, load_rec  # noqa: E402
from sim.blast import danger_map  # noqa: E402
from sim.factory import make_sim  # noqa: E402
from sim.bots import make_bot  # noqa: E402
from sim.move import center_cell  # noqa: E402
from sim.obs import can_place  # noqa: E402
from sim.config import MOVE_IDLE  # noqa: E402

FACTORS = ["dmg", "dealt", "danger", "cover", "chain", "dist",
           "chainblst", "brick", "combo", "step", "term_win", "term_timeout"]


def factorize_place(sim, cfg, placed, alive0):
    """复刻 _place_predict_reward 的三段（只读），返回 (cover, chain, dist) 各 (1,P)。"""
    n, p = placed.shape
    w = cfg.width
    live = sim.fuse > 0
    blast_map = sim._blast_map()
    fuse_frac = (sim.fuse.float() / float(cfg.fuse)).clamp(0.0, 1.0)
    weight = (cfg.chain_time_factor
              + (1.0 - cfg.chain_time_factor) * (1.0 - fuse_frac))
    cell = center_cell(sim.pos)
    flat_cell = cell[..., 0] * w + cell[..., 1]
    placed_map = torch.zeros(n, w * cfg.height, dtype=torch.bool,
                             device=sim.pos.device)
    placed_map.scatter_(1, flat_cell, placed)
    placed_map = placed_map.view(n, cfg.height, w)
    cover = torch.zeros(n, p, device=sim.pos.device)
    dist = torch.zeros(n, p, device=sim.pos.device)
    chain = torch.zeros(n, p, device=sim.pos.device)
    cooldown_ok = sim.since_bomb >= cfg.place_dist_cooldown
    cell_f = cell.float()
    for me in range(p):
        seed = torch.zeros(n, cfg.height * w, dtype=torch.bool,
                           device=sim.pos.device)
        seed.scatter_(1, flat_cell[:, me].unsqueeze(1),
                      placed[:, me].view(n, 1))
        seed = seed.view(n, cfg.height, w)
        cov = sim._rays_safe(seed) if hasattr(sim, "_rays_safe") else None
        # 直接用 rays（与 _place_predict_reward 同源）
        from sim.blast import rays
        cov = rays(seed, sim.wall, live, blast_map, sim.brick)
        cov_flat = cov.view(n, -1)
        for o in range(p):
            if o == me:
                continue
            under = cov_flat.gather(1, flat_cell[:, o].unsqueeze(1)).squeeze(1)
            cover[:, me] += (under & alive0[:, o]).float()
            if cfg.place_dist_reward > 0:
                dd = (cell_f[:, me] - cell_f[:, o]).norm(dim=-1)
                ok = placed[:, me] & (~under) & (dd < cfg.place_dist_radius) \
                    & cooldown_ok[:, me] & alive0[:, o]
                gain = 1.0 - (dd / cfg.place_dist_radius)
                dist[:, me] += (ok * gain.clamp(min=0.0)).float()
        chained = cov & live & (sim.owner >= 0) & ~placed_map
        chain[:, me] = (weight * chained.float()).flatten(1).sum(dim=1)
    return cover * alive0.float(), chain * alive0.float(), dist * alive0.float()


def analyze_one(path: str, net_cache: dict) -> dict | None:
    d = np.load(path, allow_pickle=True)
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    pid = int(d["pid"])
    if "pid" not in d or pid not in (0, 1):
        return None
    act = d["action"]                       # (T,2) 人类动作
    T = act.shape[0]
    if T < 20:
        return None
    # 过滤挂机局：人类几乎不动也不放炮（冒烟/挂机测试产物，无分析价值）
    moves = float((act[:, 0] != MOVE_IDLE).mean())
    bombs_n = int((act[:, 1] == 1).sum())
    if moves < 0.05 and bombs_n == 0:
        return None
    cfg = make_cfg(meta)
    sim = make_sim(cfg, 1, backend="torch", device="cpu",
                   seed=meta.get("seed", 0))
    opp_pid = 1 - pid
    # 对手（同 ReplaySim）
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
            if os.path.exists(ck):
                try:
                    net_cache[opp] = load_fixed_checkpoint(
                        ck, cfg.obs_shape, "cpu")
                except Exception:
                    net_cache[opp] = None
            else:
                net_cache[opp] = None
        opp_net = net_cache[opp]
        if opp_net is None:                  # ckpt 不在本地/加载失败 → idle
            opp_bot = make_bot(sim, "idle")
    else:
        opp_bot = make_bot(sim, "idle")

    cols = {k: [] for k in FACTORS}
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

        # ---- step 前快照 ----
        hp0 = sim.hp[0].clone()
        owner0 = sim.owner[0].clone()
        fuse0 = sim.fuse[0].clone()
        crate0 = sim.crate[0].clone() if hasattr(sim, "crate") else None
        cell_pid = center_cell(sim.pos)[0, pid]
        flat_pid = int(cell_pid[0] * cfg.width + cell_pid[1])
        # 人类放炮成功判定（can_place 只读）
        al0 = sim.alive[0].clone()
        placed = torch.zeros(1, 2, dtype=torch.bool)
        if bool(a_h[0, 1]) and bool(al0[pid]):
            canp = can_place(cfg, sim.fuse, sim.owner, sim.pos,
                             sim.brick if hasattr(sim, "brick") else None,
                             sim.bombs_cap)
            placed[0, pid] = canp[0, pid]
        alive0f = al0.unsqueeze(0)

        # auto_reset=False：死亡不重置（录像一局跑完就停，防死亡重置循环污染因子）
        reward, done, info = sim.step(actions, auto_reset=False)

        # ---- 因子分解（人类侧）----
        dmg = (hp0.to(torch.int32) - sim.hp[0].to(torch.int32)).clamp(min=0)
        cols["dmg"].append(float(dmg[pid]))
        cols["dealt"].append(float(dmg[opp_pid]))
        # danger：step 后状态算人类脚下（与 step 内同源）
        dng = danger_map(sim.fuse, sim.wall, sim._blast_map(), cfg.fuse,
                         sim.brick, cfg.max_chain)
        cols["danger"].append(float(dng[0].flatten()[flat_pid]))
        # place 三项：人类放炮成功 tick
        if bool(placed[0, pid]):
            cover, chain, dist_ = factorize_place(sim, cfg, placed, alive0f)
            cols["cover"].append(float(cover[0, pid]))
            cols["chain"].append(float(chain[0, pid]))
            cols["dist"].append(float(dist_[0, pid]))
        else:
            cols["cover"].append(0.0); cols["chain"].append(0.0); cols["dist"].append(0.0)
        # chainblast：跨 owner 连锁 × 点火（用 step 前快照 + info trig）
        trig = info["trig"][0]
        nat = trig & (fuse0 == 0)
        chained = trig & ~nat
        fired = bool((nat & (owner0 == pid)).sum())
        cross = int((chained & (owner0 != pid)).sum())
        cols["chainblst"].append(float(cross * fired))
        # brick：人类**step 后**脚下 crate 消失（step 前快照有、step 后没有；
        # 人类移动后才踩到 → 必须用 step 后位置判定，step 前位置会漏）
        if crate0 is not None:
            cell_after = center_cell(sim.pos)[0, pid]
            flat_after = int(cell_after[0] * cfg.width + cell_after[1])
            gone = bool(crate0.flatten()[flat_after]) and \
                not bool(sim.crate[0].flatten()[flat_after])
            cols["brick"].append(float(gone))
        else:
            cols["brick"].append(0.0)
        # combo：dealt>0 且本 tick 没掉血 → combo 因子（近似：只要 dealt>0 计 1 级）
        if float(dmg[pid]) == 0 and float(dmg[opp_pid]) > 0:
            cols["combo"].append(1.0)
        else:
            cols["combo"].append(0.0)
        cols["step"].append(1.0)
        # 终局（只记一次：auto_reset=False 时死亡后 done 恒 True，不重复累计）
        if bool(done[0]):
            na = int(info["n_alive"][0])
            if na == 1:
                cols["term_win"].append(1.0 if bool(sim.alive[0, pid]) else -1.0)
            else:
                cols["term_win"].append(0.0)
            hp = sim.hp[0].float()
            diff = hp[pid] - hp[opp_pid]
            cols["term_timeout"].append(float(diff) / cfg.max_hp
                                        if na == 2 else 0.0)
            # 死亡后 sim 停在 done 态，后续 tick 的 done 仍 True → 立即停录
            break
        else:
            cols["term_win"].append(0.0)
            cols["term_timeout"].append(0.0)

    return {k: np.array(v, dtype=np.float64) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=78)
    ap.add_argument("--out", default="res/reward_factors.txt")
    args = ap.parse_args()

    paths = sorted(glob.glob("recordings/*.npz"))[:args.limit]
    net_cache: dict = {}
    per_game = {}                       # factor -> 每局累计
    all_ticks = {k: [] for k in FACTORS}
    n_games = 0
    for p in paths:
        res = analyze_one(p, net_cache)
        if res is None:
            continue
        n_games += 1
        for k in FACTORS:
            per_game.setdefault(k, []).append(float(res[k].sum()))
            all_ticks[k].extend(res[k].tolist())
        print(f"  ✓ {os.path.basename(p)} ({len(res['step'])}tick)", flush=True)

    # ---- A. 跨局分布 ----
    lines = []
    lines.append(f"人类录像因子分布（{n_games} 局，总 tick {len(all_ticks['step'])}）\n")
    lines.append(f"{'因子':10s} {'单局均值':>9s} {'σ':>8s} {'中位':>8s} "
                 f"{'极差':>9s} {'非零%':>7s} {'单局典型累计':>11s}")
    for k in FACTORS:
        g = np.array(per_game[k])
        t = np.array(all_ticks[k])
        nz = 100.0 * (t != 0).mean()
        lines.append(
            f"{k:10s} {g.mean():9.3f} {g.std():8.3f} {np.median(g):8.3f} "
            f"{g.max()-g.min():9.3f} {nz:6.1f}% {np.median(g):11.3f}")

    # ---- B. 正交性：因子相关矩阵（tick 粒度，只对非平凡因子） ----
    mat = ["dmg", "dealt", "danger", "cover", "chain", "dist",
           "chainblst", "brick", "combo"]
    X = np.stack([np.array(all_ticks[k]) for k in mat])     # (K, T)
    # Pearson（去均值除 σ，零 σ 列归零）
    Xc = X - X.mean(axis=1, keepdims=True)
    s = Xc.std(axis=1, keepdims=True)
    Xn = np.divide(Xc, s, out=np.zeros_like(Xc), where=s > 1e-9)
    R = (Xn @ Xn.T) / max(1, Xn.shape[1] - 1)
    lines.append("\n因子相关矩阵（|r|≥0.3 高亮，正交=低相关）")
    lines.append("      " + " ".join(f"{m[:4]:>6s}" for m in mat))
    for i, m in enumerate(mat):
        row = " ".join(f"{R[i,j]:6.2f}" if abs(R[i,j]) >= 0.3 else f"{R[i,j]:6.2f}"
                       for j in range(len(mat)))
        lines.append(f"{m[:4]:>6s} " + row)
    hi = [(mat[i], mat[j], R[i, j]) for i in range(len(mat))
          for j in range(i + 1, len(mat)) if abs(R[i, j]) >= 0.3]
    hi.sort(key=lambda x: -abs(x[2]))
    if hi:
        lines.append("高相关对：")
        for a_, b_, r in hi[:8]:
            lines.append(f"  {a_:10s} × {b_:10s}  r={r:+.2f}")

    # ---- C. 当前数值合理性 ----
    coeff = {"dmg": 1.5, "dealt": 1.5, "danger": 0.015, "cover": 0.05,
             "chain": 0.20, "dist": 0.0, "chainblst": 0.0, "brick": 0.05,
             "combo": 0.10, "step": 0.001, "term_win": 10.0, "term_timeout": 1.6}
    lines.append("\n当前数值 → 单局典型贡献（= 系数 × 单局中位累计，与 hit ±1.5 比）")
    hit_ref = 1.5
    for k in FACTORS:
        contrib = coeff[k] * np.median(per_game[k])
        mark = ""
        if k in ("danger", "brick", "chain") and abs(contrib) > 2 * hit_ref:
            mark = "  ← 过强"
        if k in ("dist", "combo", "chainblst") and abs(contrib) < 0.05 * hit_ref:
            mark = "  ← 死信号"
        lines.append(f"  {k:10s} {coeff[k]:7.3f} × {np.median(per_game[k]):9.3f} "
                     f"= {contrib:8.3f}  ({contrib/hit_ref:5.1f}x hit){mark}")

    # ---- D. 稠密 vs 核心汇总（用户：稠密累计会不会超过核心）----
    # 口径：稠密 = 每 tick/高频塑形（danger/brick/step/chain 累计，绝对值）；
    # 核心正 = 稀疏主奖励（造成伤害 dealt + 击杀固定 term_win 的**胜局**贡献）；
    # 核心负 = 掉血 dmg（惩罚，真实信号）。term_win 在这批录像几乎全负
    # （人类 vs 模型输多）→ 击杀胜局单独算：用 term_win>0 的局中位 × 胜局占比。
    dense = {k: abs(coeff[k] * np.median(per_game[k])) for k in
             ("danger", "brick", "step", "chain")}     # 每 tick/高频
    tw = np.array(per_game["term_win"])
    win_games = tw[tw > 0]
    win_frac = len(win_games) / max(1, len(tw))
    tw_win_med = float(np.median(win_games)) if len(win_games) else 0.0
    dealt_med = float(np.median(per_game["dealt"]))
    core_pos = {"dealt": coeff["dealt"] * dealt_med,
                f"term_win(击杀{win_frac:.0%}局)":
                    coeff["term_win"] * tw_win_med * win_frac}
    core_neg = {"dmg": coeff["dmg"] * np.median(per_game["dmg"])}
    lines.append("\n稠密（每 tick/高频）vs 核心（稀疏）单局典型贡献：")
    d_sum = sum(dense.values())
    cp_sum = sum(core_pos.values())
    cn_sum = abs(sum(core_neg.values()))
    for k, v in dense.items():
        lines.append(f"  稠密 {k:10s} {v:8.3f}")
    lines.append(f"  稠密合计            {d_sum:8.3f}  ({d_sum/hit_ref:4.1f}x hit)")
    for k, v in core_pos.items():
        lines.append(f"  核心正 {k:16s} {v:8.3f}")
    lines.append(f"  核心正合计          {cp_sum:8.3f}  ({cp_sum/hit_ref:4.1f}x hit)")
    for k, v in core_neg.items():
        lines.append(f"  核心负 {k:16s} {v:8.3f}")
    lines.append(f"  核心负合计          {cn_sum:8.3f}  ({cn_sum/hit_ref:4.1f}x hit)")
    lines.append(f"\n  稠密/核心正 = {d_sum/max(cp_sum,1e-9):.2f} —— "
                 f"{'✅ 稠密塑形 < 核心奖励' if d_sum < cp_sum else '⚠️ 稠密塑形压过核心奖励'}")

    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
