#!/usr/bin/env python3
"""训练监控报告生成器。

用法:
  python monitor/build_report.py            # 只用本地已有数据重建 CSV + 图
  python monitor/build_report.py --pull     # 先从 rank0 拉全量日志再重建（需 /tmp/ndrun）

产物（monitor/ 下）:
  train_curve.csv    逐 iter 曲线（iter,dt,sps,loss,kill,alpha,ep_len,gs）
  gate_history.csv   门禁评估与晋级时间线（time,stage,winrate,games,kind）
  report.png         四联图：loss / kill+ep_len / sps / 门禁胜率时间线
"""
import os, re, sys, subprocess, csv
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MON = os.path.join(ROOT, "monitor")
LOG = os.path.join(MON, "train_r0_full.log")
GATE = os.path.join(MON, "gate_history.log")

ITER_RE = re.compile(
    r"iter (\d+)/\d+ [\d.]+s \([\d.]+s avg\) ([\d,]+) sps \(avg [\d,]+\)"
    r" loss=([-\d.]+) rew=([-\d.]+) explore=([\d.]+) kill=([\d.]+) "
    r"α=([\d.]+) gs=([\d,]+) ep_len=([\d.naN]+)")
GATE_RE = re.compile(r"\[([\d\- :]+)\] \[Gate\] (Stage \d+) vs Baseline winrate=([\d.]+)% \((\d+)/(\d+)\)")
PROMO_RE = re.compile(r"\[([\d\- :]+)\] PROMOTION -> (Stage \d+)")

def pull():
    """从 rank0 拉全量训练日志与门禁历史（/tmp/ndrun 封装已存在时）。"""
    ndrun = "/tmp/ndrun/cmd_0"
    if not os.path.exists(ndrun):
        print("SSH 封装 /tmp/ndrun 不存在，跳过拉取，使用本地已有数据"); return
    r0 = "/root/private_data/train_r0.log"
    res = "/root/private_data/qqt-gpu-sim/multicard_result.txt"
    subprocess.run([ndrun, f"cat {r0}"], stdout=open(LOG + ".tmp", "w"),
                   stderr=subprocess.DEVNULL, timeout=300)
    subprocess.run([ndrun, f"cat {res}"], stdout=open(GATE + ".tmp", "w"),
                   stderr=subprocess.DEVNULL, timeout=120)
    for t in (LOG + ".tmp", GATE + ".tmp"):
        ok = os.path.exists(t) and os.path.getsize(t) > 0
        dst = LOG if "train_r0" in t else GATE
        if ok:
            os.replace(t, dst); print(f"拉取成功: {dst} ({os.path.getsize(dst)} bytes)")
        else:
            os.remove(t)
            print(f"拉取失败（SSH 不可达？），保留旧数据: {dst}")

def parse_iters():
    rows = []
    if not os.path.exists(LOG): return rows
    for line in open(LOG, errors="ignore"):
        m = ITER_RE.search(line)
        if m:
            it, sps, loss, rew, exp, kill, alpha, gs, epl = m.groups()
            rows.append([int(it), float(sps.replace(",", "")), float(loss), float(rew),
                         float(exp), float(kill), float(alpha), int(gs.replace(",", "")),
                         float(epl)])
    return rows

def parse_gates():
    rows = []  # (time, kind, stage, winrate, games)
    if not os.path.exists(GATE): return rows
    for line in open(GATE, errors="ignore"):
        m = GATE_RE.search(line)
        if m:
            t, st, wr, w, tot = m.groups()
            rows.append([t.strip(), "gate", st, float(wr), int(tot)])
            continue
        m = PROMO_RE.search(line)
        if m:
            rows.append([m.group(1).strip(), "promotion", m.group(2), None, None])
    return rows

def main():
    if "--pull" in sys.argv: pull()
    os.makedirs(MON, exist_ok=True)
    iters = parse_iters(); gates = parse_gates()

    with open(os.path.join(MON, "train_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "sps", "loss", "rew", "explore", "kill", "alpha", "gs", "ep_len"])
        w.writerows(iters)
    with open(os.path.join(MON, "gate_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "kind", "stage", "winrate_pct", "games"])
        w.writerows(gates)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    its = [r[0] for r in iters]
    if iters:
        ax = axes[0][0]
        ax.plot(its, [r[2] for r in iters], lw=0.8); ax.set_title("loss"); ax.set_xlabel("iter")
        ax = axes[0][1]
        ax.plot(its, [r[5] for r in iters], lw=0.8, label="kill(击杀率)")
        ax2 = ax.twinx(); ax2.plot(its, [r[8] for r in iters], color="tab:orange", lw=0.8, label="ep_len")
        ax.set_title("kill / ep_len"); ax.set_xlabel("iter"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        ax = axes[1][0]
        ax.plot(its, [r[1] for r in iters], lw=0.8); ax.set_title("sps"); ax.set_xlabel("iter")
    ax = axes[1][1]
    gt = [g for g in gates if g[1] == "gate"]
    if gt:
        xs = list(range(len(gt)))
        ax.plot(xs, [g[3] for g in gt], marker="o", ms=3, lw=1)
        for i, g in enumerate(gates):
            if g[1] == "promotion":
                ax.axvline(len([x for x in gt if x[0] <= g[0]]), color="red", ls="--", lw=0.8)
                ax.text(len([x for x in gt if x[0] <= g[0]]), 45, g[2], rotation=90, fontsize=7, color="red")
        ax.set_xticks(xs[::max(1, len(xs)//8)])
        ax.set_xticklabels([f"{g[0][5:16]} {g[2]}" for g in gt][::max(1, len(gt)//8)], rotation=30, fontsize=7)
        ax.axhline(75, color="gray", ls=":", lw=0.7); ax.set_ylim(0, 100)
    ax.set_title("Gate winrate vs 冻结基线（红虚线=晋级）")
    fig.suptitle(f"48 卡自博弈训练监控  生成于 {datetime.now():%m-%d %H:%M}  (iter {its[-1] if iters else '-'})")
    fig.tight_layout()
    out = os.path.join(MON, "report.png")
    fig.savefig(out, dpi=110)
    print(f"报告已生成: {out}  (iters={len(iters)}, gates={len(gt)})")

if __name__ == "__main__":
    main()
