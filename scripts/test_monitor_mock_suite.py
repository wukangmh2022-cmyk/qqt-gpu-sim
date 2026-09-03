#!/usr/bin/env python3
"""
scripts/test_monitor_mock_suite.py - 训练监控与博弈分析套件全量 Mock 自动化验证脚本
"""
import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import monitor.heartbeat as hb
import monitor.cycle_detector as cd
import monitor.retention as ret

def test_cycle_detector():
    print("\n[TEST 1/5] 验证 cycle_detector.py (Hodge 分解与闭环检测)...")
    
    # 场景 A: 纯线性递增健康模型 (A > B > C)
    healthy_models = ["it100", "it80", "it60"]
    healthy_matches = [
        {"model0": "it100", "model1": "it80", "m0_wins": 25, "total_games": 32}, # 78.1%
        {"model0": "it80", "model1": "it60", "m0_wins": 24, "total_games": 32},  # 75.0%
        {"model0": "it100", "model1": "it60", "m0_wins": 30, "total_games": 32}, # 93.8%
    ]
    res_healthy = cd.analyze_league_cycles(healthy_matches, healthy_models)
    assert not res_healthy["has_cycles"], "错误：健康线性模型不应判定为存在闭环"
    assert res_healthy["hodge"]["cyclic_ratio"] < 5.0, f"错误：线性模型涡流比应接近0，实际={res_healthy['hodge']['cyclic_ratio']}"
    print(f"  ✓ 场景 A (线性递增) 通过: has_cycles=False, cyclic_ratio={res_healthy['hodge']['cyclic_ratio']}% (状态: {res_healthy['hodge']['status']})")

    # 场景 B: 石头剪刀布循环克制 (A > B > C > A)
    cycle_models = ["strat_A", "strat_B", "strat_C"]
    cycle_matches = [
        {"model0": "strat_A", "model1": "strat_B", "m0_wins": 26, "total_games": 32}, # A 胜 B
        {"model0": "strat_B", "model1": "strat_C", "m0_wins": 25, "total_games": 32}, # B 胜 C
        {"model0": "strat_C", "model1": "strat_A", "m0_wins": 27, "total_games": 32}, # C 胜 A
    ]
    res_cycle = cd.analyze_league_cycles(cycle_matches, cycle_models)
    assert res_cycle["has_cycles"], "错误：石头剪刀布应检测出闭环"
    assert res_cycle["cycle_count"] == 1, f"错误：应检测到1个3-环，实际={res_cycle['cycle_count']}"
    assert res_cycle["hodge"]["cyclic_ratio"] > 80.0, f"错误：纯循环涡流比应极高，实际={res_cycle['hodge']['cyclic_ratio']}"
    print(f"  ✓ 场景 B (剪刀石头布) 通过: has_cycles=True, 闭环链路={res_cycle['cycles'][0]['summary']}, 涡流比={res_cycle['hodge']['cyclic_ratio']}%")

def test_heartbeat_logic():
    print("\n[TEST 2/5] 验证 heartbeat.py (心跳状态机与疑似假死/掉线探测)...")
    
    # 测试 SSH 探测
    ssh_ok, ssh_msg = hb.test_ssh_connectivity()
    print(f"  ✓ SSH 端口探测测试: reachable={ssh_ok}, info={ssh_msg}")

    # 全量心跳运行
    state = hb.check_heartbeat(enable_remote_ping=False)
    assert "status" in state and "idleSeconds" in state, "心跳结构字段不完整"
    print(f"  ✓ 当前系统心跳状态: status={state['status']}, idleSeconds={state['idleSeconds']}s, message={state['message']}")

def test_retention_milestone():
    print("\n[TEST 3/5] 验证 retention.py (Peak Elo 巅峰里程碑与滚动淘汰)...")

    mock_history = [
        {"modelName": "it50", "elo": {"composite": 1400}, "health": {"status": "HEALTHY"}},
        {"modelName": "it68", "elo": {"composite": 1437}, "health": {"status": "HEALTHY"}},
    ]

    # 测试用例 1: 创历史新高 1480 Elo (健康) -> 必须标记为里程碑
    report_new_peak = {
        "modelName": "it80",
        "iteration": 80,
        "elo": {"composite": 1480},
        "health": {"status": "HEALTHY"}
    }
    is_m, reason = ret.check_and_mark_milestone(report_new_peak, mock_history)
    assert is_m, "错误：突破历史峰值应当判定为 Milestone"
    assert "最高战力巅峰" in reason, f"原因描述不符合预期: {reason}"
    print(f"  ✓ 战力新高测试通过: isMilestone={is_m}, reason={reason}")

    # 测试用例 2: Elo 仅 1420 (未突破历史 1437) -> 不能标记为里程碑
    report_normal = {
        "modelName": "it85",
        "iteration": 85,
        "elo": {"composite": 1420},
        "health": {"status": "HEALTHY"}
    }
    is_m2, reason2 = ret.check_and_mark_milestone(report_normal, mock_history)
    assert not is_m2, "错误：未突破历史峰值不应被标记为 Milestone"
    print(f"  ✓ 常规迭代测试通过: isMilestone={is_m2}, reason={reason2}")

    # 测试用例 3: Elo 高达 1600 但策略退化 (DEGRADED) -> 绝不能标记为里程碑 (门禁拦截)
    report_degraded = {
        "modelName": "it90_buggy",
        "iteration": 90,
        "elo": {"composite": 1600},
        "health": {"status": "DEGRADED"}
    }
    is_m3, reason3 = ret.check_and_mark_milestone(report_degraded, mock_history)
    assert not is_m3, "错误：退化死锁模型绝对不可判定为 Milestone"
    print(f"  ✓ 门禁拦截测试通过: isMilestone={is_m3}, reason={reason3}")

    # 测试用例 4: 本地清理 Dry-Run (白名单保护)
    prune_res = ret.prune_local_storage(keep_last_n=5, dry_run=True)
    assert prune_res["protectedCount"] >= 2, "白名单保护数异常"
    print(f"  ✓ 磁盘保护测试通过: 受保护里程碑数={prune_res['protectedCount']}, 待修剪老档数={prune_res['deletedCount']}")

def test_live_server_endpoints():
    print("\n[TEST 4/5] 验证 monitor/server.py 在线 HTTP API 接口...")
    base_url = "http://localhost:8088"
    
    endpoints = [
        ("/api/status", ["status", "heartbeat", "health", "availableCheckpoints"]),
        ("/api/train_telemetry", None), # 列表
        ("/api/history", None),         # 列表
        ("/api/league", ["has_cycles", "hodge", "matrix"]),
    ]

    for ep, required_keys in endpoints:
        req = urllib.request.Request(f"{base_url}{ep}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200, f"接口 {ep} 返回状态码 {resp.status}"
            data = json.loads(resp.read().decode('utf-8'))
            if required_keys:
                for k in required_keys:
                    assert k in data, f"接口 {ep} 缺少关键字段 {k}"
            print(f"  ✓ 接口 {ep} 响应正常 (HTTP 200 OK)")

def test_eval_headless_quick():
    print("\n[TEST 5/5] 验证 eval_headless_parallel.js (双模型换边真机对决快速打通)...")
    import subprocess
    cmd = [
        "node",
        "scripts/eval_headless_parallel.js",
        "--model", "params_it00000068_hlgauss_top25foractor_patch3_k32",
        "--opp-model", "params_it00000068_hlgauss_top25foractor_patch3_k32",
        "--games", "1",
        "--workers", "1"
    ]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"Headless 评测失败: {res.stderr or res.stdout}"
    assert "评测汇总成果" in res.stdout, "输出未包含评测汇总表格"
    print("  ✓ WebJS Headless Model-vs-Model 换边对决运行完全通过！")

if __name__ == "__main__":
    print("==========================================================================")
    print("🚀 QQT-RL 训练监控、博弈闭环分析与生命周期管理自动化验证测试")
    print("==========================================================================")
    test_cycle_detector()
    test_heartbeat_logic()
    test_retention_milestone()
    test_live_server_endpoints()
    test_eval_headless_quick()
    print("\n🎉 ALL 5 TEST SUITES PASSED! 全部组件逻辑验证完全正确！\n")
