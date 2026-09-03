#!/usr/bin/env python3
"""
monitor/watchdog.py - 训练时模型监控与策略退化/死锁拦截守护进程

核心功能：
1. 监听 ckpt/ 与 ckpt_local/ 目录下的新 Checkpoint (params_it*.pkl)
2. 自动增量导出为 Web ONNX (deploy/export_jax_onnx.py)
3. 自动触发 128 局极速 WebJS Headless 真实评测 (scripts/eval_headless_parallel.js，4 并发 ~3.1 分钟)
4. 运行策略崩溃/死锁判定规则引擎 (Anomaly Engine)
5. 持久化指标至 monitor/history.json，并在触发严重退化时生成 monitor/ALERT.json
"""
import os
import sys
import glob
import json
import time
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITOR_DIR = os.path.join(ROOT, "monitor")
CKPT_DIRS = [os.path.join(ROOT, "ckpt"), os.path.join(ROOT, "ckpt_local")]
MODELS_DIR = os.path.join(ROOT, "web", "models")
HISTORY_FILE = os.path.join(MONITOR_DIR, "history.json")
ALERT_FILE = os.path.join(MONITOR_DIR, "ALERT.json")
EVAL_SCRIPT = os.path.join(ROOT, "scripts", "eval_headless_parallel.js")
EXPORT_SCRIPT = os.path.join(ROOT, "deploy", "export_jax_onnx.py")
PYTHON_BIN = os.path.join(ROOT, ".venv", "bin", "python") if os.path.exists(os.path.join(ROOT, ".venv", "bin", "python")) else sys.executable

# 判定红线规则 (基于 128 局 4 场景 Headless 基准)
HEALTH_RULES = {
    # 规则 1: 空场景对战 Hunter 严重死锁/退化
    "open_hunter_min_win_rate": 10.0,    # 胜率低于 10% 报警
    "open_hunter_max_timeout": 35.0,     # 超时率超过 35% 判定消极死锁
    "open_hunter_min_bombs": 25.0,       # 单局平均放炮低于 25 判定极度发呆
    "open_hunter_max_idle_ratio": 60.0,  # 发呆静止移动比例超过 60% 报警 (CleanRL)
    "open_hunter_min_entropy": 0.60,     # 动作经验熵低于 0.60 判定单一套路化 (CleanRL)
    # 规则 2: 自杀失控
    "max_suicide_rate": 12.0,            # 任意场景自杀率 > 12% 判定身法失控
    # 规则 3: 基础杀伤力崩塌
    "open_idle_min_win_rate": 95.0,      # 打空旷静止木桩必须 ≥ 95%
    "open_idle_max_ticks": 450.0,        # 打静止木桩平均局长不得 > 450 ticks
    # 规则 4: 复杂图破砖与控图寻路失能 (RLlib)
    "full_idle_min_win_rate": 18.0,      # 复杂图打木桩破砖寻路胜率不得 < 18%
    "full_idle_min_explored": 15.0,      # 241 复杂图领地控图率不得 < 15% (RLlib Territory)
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(history):
    os.makedirs(MONITOR_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def export_onnx(ckpt_path):
    """自动将 JAX pkl 转换为 Web ONNX 模型"""
    cmd = [PYTHON_BIN, EXPORT_SCRIPT, ckpt_path]
    print(f"[Watchdog] 正在导出 ONNX: {os.path.basename(ckpt_path)} ...")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Watchdog] 导出 ONNX 失败: {res.stderr}")
        return False
    return True

def run_eval(model_name, games=32, workers=4):
    """运行多核心 WebJS Headless 对战评测"""
    cmd = [
        "node", EVAL_SCRIPT,
        "--model", model_name,
        "--games", str(games),
        "--workers", str(workers)
    ]
    print(f"[Watchdog] 启动 128 局真实评测: {model_name} (workers={workers}, games={games})...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[Watchdog] 评测失败: {res.stderr}")
        return None

    # 解析标准 Markdown 表格输出
    # | 对手 / 地图场景 | 胜 | 负 | 同归 | 超时 | 胜率(%) | 自杀 | 炮/局 | 命中/局 | 平均局长 |
    reports = {}
    lines = res.stdout.splitlines()
    table_started = False
    for line in lines:
        if "| 对手 / 地图场景 |" in line or "| 对手 / 地图 |" in line:
            table_started = True
            continue
        if table_started and line.startswith("|") and not ":---" in line:
            parts = [p.strip().replace("**", "").replace("%", "") for p in line.split("|")[1:-1]]
            if len(parts) >= 10:
                dom_name = parts[0]
                dom_id = (
                    "open_hunter" if "AI Hunter" in dom_name and "空" in dom_name
                    else "full_hunter" if "AI Hunter" in dom_name
                    else "open_idle" if "空" in dom_name
                    else "full_idle"
                )
                try:
                    reports[dom_id] = {
                        "name": dom_name,
                        "wins": int(parts[1]),
                        "losses": int(parts[2]),
                        "mutuals": int(parts[3]),
                        "timeouts": int(parts[4]),
                        "winRate": float(parts[5]),
                        "suicides": int(parts[6]),
                        "avgBombs": float(parts[7]),
                        "avgHits": float(parts[8]),
                        "avgTicks": float(parts[9]),
                        "avgExplored": float(parts[10]) if len(parts) > 10 else 0.0,
                        "avgIdle": float(parts[11]) if len(parts) > 11 else 0.0,
                        "avgEntropy": float(parts[12]) if len(parts) > 12 else 0.0,
                    }
                except ValueError as e:
                    pass

    import re, math
    m = re.search(r"it(\d+)", model_name)
    iteration = int(m.group(1)) if m else 0

    def compute_elo(wr_pct, anchor_elo=1500):
        wr = max(1.0, min(99.0, wr_pct)) / 100.0
        return round(anchor_elo + 400.0 * math.log10(wr / (1.0 - wr)))

    oh_wr = reports.get("open_hunter", {}).get("winRate", 25.0)
    fh_wr = reports.get("full_hunter", {}).get("winRate", 50.0)
    elo_oh = compute_elo(oh_wr, 1500)
    elo_fh = compute_elo(fh_wr, 1500)
    composite_elo = round(0.4 * elo_oh + 0.6 * elo_fh)

    # 检查是否存在伴生元数据 meta.json
    meta_info = None
    for d in CKPT_DIRS:
        meta_cand = os.path.join(d, f"{model_name}.meta.json")
        if os.path.exists(meta_cand):
            try:
                with open(meta_cand, "r", encoding="utf-8") as f_meta:
                    meta_info = json.load(f_meta)
                break
            except Exception:
                pass

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelName": model_name,
        "iteration": iteration,
        "meta": meta_info,
        "elo": {
            "anchor": 1500,
            "vsOpenHunter": elo_oh,
            "vsFullHunter": elo_fh,
            "composite": composite_elo
        },
        "evalDurationSeconds": round(dt, 2),
        "totalGames": games * 4,
        "domains": reports
    }

def evaluate_health(report):
    """根据硬核规则引擎判定健康状态 (融合 CleanRL 优化诊断与 RLlib 博弈死锁指标)"""
    issues = []
    doms = report.get("domains", {})

    # 1. 检查 open_hunter
    oh = doms.get("open_hunter")
    if oh:
        timeout_rate = (oh["timeouts"] / max(1, oh["wins"] + oh["losses"] + oh["timeouts"] + oh["mutuals"])) * 100
        suicide_rate = (oh["suicides"] / max(1, oh["wins"] + oh["losses"] + oh["timeouts"] + oh["mutuals"])) * 100
        if oh["winRate"] < HEALTH_RULES["open_hunter_min_win_rate"] and timeout_rate > HEALTH_RULES["open_hunter_max_timeout"]:
            issues.append(f"【策略消极死锁】空场景 vs Hunter 胜率仅 {oh['winRate']}%，超时率达 {timeout_rate:.1f}%")
        if oh["avgBombs"] < HEALTH_RULES["open_hunter_min_bombs"]:
            issues.append(f"【发呆单一】空场景放炮量断崖下跌至局均 {oh['avgBombs']} 颗")
        if oh.get("avgIdle", 0) > HEALTH_RULES["open_hunter_max_idle_ratio"]:
            issues.append(f"【CleanRL 发呆站桩】空场景发呆静止移动比例达 {oh['avgIdle']}%")
        if 0 < oh.get("avgEntropy", 1.0) < HEALTH_RULES["open_hunter_min_entropy"]:
            issues.append(f"【CleanRL 动作熵崩溃】动作经验熵降至 {oh['avgEntropy']}，过早单一套路化")
        if suicide_rate > HEALTH_RULES["max_suicide_rate"]:
            issues.append(f"【自杀失控】空场景自杀率激增至 {suicide_rate:.1f}%")

    # 2. 检查 open_idle (基础木桩测试)
    oi = doms.get("open_idle")
    if oi:
        if oi["winRate"] < HEALTH_RULES["open_idle_min_win_rate"]:
            issues.append(f"【基础击杀退化】空场景木桩胜率异常跌落至 {oi['winRate']}%")
        if oi["avgTicks"] > HEALTH_RULES["open_idle_max_ticks"]:
            issues.append(f"【动作套路发呆】击杀静止目标耗时异常变长 ({oi['avgTicks']} ticks)")

    # 3. 检查 full_idle (迷宫破砖寻路与领地探索测试)
    fi = doms.get("full_idle")
    if fi:
        if fi["winRate"] < HEALTH_RULES["full_idle_min_win_rate"]:
            issues.append(f"【长程破砖寻路丧失】241 复杂图破砖寻路胜率跌至 {fi['winRate']}%")
        if 0 < fi.get("avgExplored", 100) < HEALTH_RULES["full_idle_min_explored"]:
            issues.append(f"【RLlib 控图瘫痪】241 复杂图领地控图率仅 {fi['avgExplored']}%，畏缩在角落")

    if not issues:
        status = "HEALTHY"
        severity = "info"
    elif len(issues) == 1 and "消极" not in issues[0]:
        status = "WARNING"
        severity = "warning"
    else:
        status = "DEGRADED"
        severity = "critical"

    return {
        "status": status,
        "severity": severity,
        "issues": issues,
        "checkedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def record_alert(report, health):
    """产生结构化告警文件供人或 Agent 消费"""
    alert = {
        "alertTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": report["modelName"],
        "status": health["status"],
        "severity": health["severity"],
        "issues": health["issues"],
        "metrics": report["domains"],
        "suggestedActions": [
            "1. 立即暂停训练，防止策略在消极/自杀样本中持续漂移崩溃",
            "2. 回滚至上一个健康检查点 (如 it68_scheme1)",
            "3. 调优超参数：加大动作空间合法掩码熵正则系数 (ent_coef 由 0.01 提升至 0.02~0.03)",
            "4. 检查是否出现 Top25% advantage filter 导致的过度保守化"
        ]
    }
    with open(ALERT_FILE, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, ensure_ascii=False)
    print(f"\n🚨 [Watchdog 报警] 检测到策略异常: {health['status']}")
    for iss in health["issues"]:
        print(f"   ⚠️  {iss}")
    print(f"   详细告警已写入: {ALERT_FILE}\n")

def process_checkpoint(ckpt_path, games=32, workers=4):
    """处理单个 Checkpoint 的全流程"""
    stem = os.path.basename(ckpt_path).replace(".pkl", "")
    onnx_file = os.path.join(MODELS_DIR, f"{stem}.onnx")

    # 1. 确保 ONNX 模型已生成
    if not os.path.exists(onnx_file) or os.path.getmtime(ckpt_path) > os.path.getmtime(onnx_file):
        ok = export_onnx(ckpt_path)
        if not ok:
            return None

    # 2. 运行快速评测
    report = run_eval(stem, games=games, workers=workers)
    if not report:
        return None

    # 3. 运行退化判定
    health = evaluate_health(report)
    report["health"] = health

    # 4. 战力巅峰与里程碑判定 (Peak Elo Milestone)
    history = load_history()
    try:
        import monitor.retention as ret
        is_mile, mile_reason = ret.check_and_mark_milestone(report, history)
        if is_mile:
            print(f"[Watchdog] 🌟 恭喜！{stem} 判定为里程碑: {mile_reason}")
    except Exception as e:
        report["isMilestone"] = False

    # 替换或追加
    idx = next((i for i, h in enumerate(history) if h.get("modelName") == stem), -1)
    if idx >= 0:
        history[idx] = report
    else:
        history.append(report)
    save_history(history)

    # 4.5 自动执行本地轻量滚动清理 (保留最新 6 个非里程碑档，保护所有里程碑与黄金档)
    try:
        import monitor.retention as ret
        ret.prune_local_storage(keep_last_n=6, dry_run=False)
    except Exception as e:
        pass

    # 5. 告警触发
    if health["status"] in ["WARNING", "DEGRADED"]:
        record_alert(report, health)
    else:
        if os.path.exists(ALERT_FILE):
            try:
                # 恢复健康，移除严重告警
                with open(ALERT_FILE, "r", encoding="utf-8") as f:
                    old_alert = json.load(f)
                if old_alert.get("severity") != "critical":
                    os.remove(ALERT_FILE)
            except Exception:
                pass

    print(f"[Watchdog] 完成对 {stem} 的验收，状态: {health['status']}")
    return report

def get_discovered_ckpts():
    """扫描所有已知 checkpoint 文件"""
    ckpts = []
    for d in CKPT_DIRS:
        if os.path.exists(d):
            ckpts.extend(glob.glob(os.path.join(d, "params_it*.pkl")))
    # 过滤掉 ema 冗余档
    ckpts = [p for p in ckpts if "_ema" not in os.path.basename(p)]
    # 按修改时间排序
    ckpts.sort(key=lambda p: os.path.getmtime(p))
    return ckpts

def main_loop(poll_interval=15):
    """主轮询监听循环"""
    print("==========================================================================")
    print("👀 强化学习训练时监控守护进程 (Watchdog Daemon) 已启动")
    print(f"   监听目录: {CKPT_DIRS}")
    print(f"   轮询间隔: {poll_interval}s | 单次验收: 4 场景各 32 局 (总计 128 局)")
    print(f"   健康规则: 死锁/超时率拦截、自杀率拦截、破砖寻路失能拦截")
    print("==========================================================================")

    seen = set(h.get("modelName") for h in load_history() if "modelName" in h)

    while True:
        ckpts = get_discovered_ckpts()
        for p in ckpts:
            stem = os.path.basename(p).replace(".pkl", "")
            if stem not in seen:
                print(f"\n🔔 [新检查点发现] 检测到未评测的快照: {stem}")
                res = process_checkpoint(p, games=32, workers=4)
                if res:
                    seen.add(stem)

        time.sleep(poll_interval)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval-one":
        target = sys.argv[2]
        process_checkpoint(target, games=32, workers=4)
    else:
        main_loop()
