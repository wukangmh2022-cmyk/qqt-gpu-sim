#!/usr/bin/env python3
"""
monitor/cycle_detector.py - 自博弈非传递性与胜率循环 (Rock-Paper-Scissors / Hodge Decomposition) 检测器

基于博弈论与 DeepMind "Spinning Top" 定理：
1. 建立跨代 Checkpoint 两两对战矩阵 Payoff Matrix M
2. 有向 3-环检测 (Directed 3-Cycle Detection: A 胜 B, B 胜 C, C 胜 A)
3. 严谨 Hodge 正交分解 (Helmholtz-Hodge Decomposition):
   M = M_transitive (Elo 实力梯度) + M_cyclic (纯旋转涡流)
   Cyclic Ratio = ||M_cyclic||^2 / ||M||^2
"""
import json
import os
import numpy as np

def build_payoff_matrix(matches, model_names):
    """
    根据历史对决记录构建 N x N 反对称净胜率矩阵 M:
    M[i][j] = P(i beats j) - 0.5
    """
    n = len(model_names)
    idx_map = {name: i for i, name in enumerate(model_names)}
    wins = np.zeros((n, n), dtype=np.float64)
    totals = np.zeros((n, n), dtype=np.float64)

    for m in matches:
        m0, m1 = m.get("model0"), m.get("model1")
        if m0 in idx_map and m1 in idx_map:
            i, j = idx_map[m0], idx_map[m1]
            w = m.get("m0_wins", 0)
            t = m.get("total_games", 0)
            if t > 0:
                wins[i, j] += w
                totals[i, j] += t
                # 换边累加
                wins[j, i] += (t - w - m.get("draws", 0))
                totals[j, i] += t

    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if totals[i, j] > 0:
                p = wins[i, j] / totals[i, j]
                M[i, j] = p - 0.5
                M[j, i] = -(p - 0.5)
    return M

def detect_directed_3_cycles(M, model_names, threshold=0.03):
    """
    检测所有长度为 3 的循环克制三元组 (A -> B -> C -> A)
    threshold 为显著性阈值 (如 0.03 代表胜率需超过 53% 才建立克制边)
    """
    n = len(model_names)
    cycles = []

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if M[i, j] > threshold:  # i 胜 j
                for k in range(n):
                    if k == i or k == j: continue
                    if M[j, k] > threshold and M[k, i] > threshold:  # j 胜 k 且 k 胜 i
                        # 确保按字典序去重
                        cycle_tuple = tuple(sorted([model_names[i], model_names[j], model_names[k]]))
                        cycle_detail = {
                            "cycle": [model_names[i], model_names[j], model_names[k]],
                            "summary": f"{model_names[i]} ➔ {model_names[j]} ➔ {model_names[k]} ➔ {model_names[i]}",
                            "edges": [
                                {"from": model_names[i], "to": model_names[j], "winRate": round((M[i, j] + 0.5) * 100, 1)},
                                {"from": model_names[j], "to": model_names[k], "winRate": round((M[j, k] + 0.5) * 100, 1)},
                                {"from": model_names[k], "to": model_names[i], "winRate": round((M[k, i] + 0.5) * 100, 1)},
                            ]
                        }
                        if not any(c["cycle_key"] == cycle_tuple for c in cycles):
                            cycles.append({"cycle_key": cycle_tuple, **cycle_detail})
    return cycles

def compute_hodge_decomposition(M):
    """
    对反对称博弈矩阵执行 Hodge 正交分解：
    M = M_transitive + M_cyclic
    其中 M_transitive[i, j] = s[i] - s[j]，s 为一维潜力势能 (即 Elo 标量评分)
    """
    n = M.shape[0]
    if n < 3:
        return {
            "cyclic_ratio": 0.0,
            "transitive_ratio": 1.0,
            "status": "HEALTHY",
            "potentials": [0.0] * n
        }

    # s_i = (1 / n) * sum_j M_ij
    s = np.mean(M, axis=1)

    # 构造传递性梯度矩阵 M_trans
    M_trans = np.zeros_like(M)
    for i in range(n):
        for j in range(n):
            M_trans[i, j] = s[i] - s[j]

    # 构造旋度/环流矩阵 M_cyclic
    M_cyclic = M - M_trans

    norm_total = np.sum(M ** 2)
    norm_cyclic = np.sum(M_cyclic ** 2)

    if norm_total < 1e-8:
        cyclic_ratio = 0.0
    else:
        cyclic_ratio = float(norm_cyclic / norm_total)

    transitive_ratio = max(0.0, 1.0 - cyclic_ratio)

    if cyclic_ratio > 0.35:
        status = "CRITICAL_CYCLE"
    elif cyclic_ratio > 0.15:
        status = "WARNING_CYCLE"
    else:
        status = "HEALTHY"

    return {
        "cyclic_ratio": round(cyclic_ratio * 100, 1),
        "transitive_ratio": round(transitive_ratio * 100, 1),
        "status": status,
        "potentials": [round(float(v), 4) for v in s]
    }

def analyze_league_cycles(matches, model_names):
    """全量联赛循环分析入口"""
    if len(model_names) < 3:
        return {
            "has_cycles": False,
            "cycle_count": 0,
            "cycles": [],
            "hodge": {"cyclic_ratio": 0.0, "status": "HEALTHY"},
            "matrix": []
        }

    M = build_payoff_matrix(matches, model_names)
    cycles = detect_directed_3_cycles(M, model_names)
    hodge = compute_hodge_decomposition(M)

    # 转换为前端展示格式的矩阵
    matrix_display = []
    for i, m0 in enumerate(model_names):
        row = {"model": m0, "scores": {}}
        for j, m1 in enumerate(model_names):
            if i == j:
                row["scores"][m1] = 50.0
            else:
                row["scores"][m1] = round((M[i, j] + 0.5) * 100, 1)
        matrix_display.append(row)

    return {
        "has_cycles": len(cycles) > 0 or hodge["cyclic_ratio"] > 25.0,
        "cycle_count": len(cycles),
        "cycles": cycles,
        "hodge": hodge,
        "matrix": matrix_display
    }

if __name__ == "__main__":
    # 测试构造石头剪刀布闭环样例: A 胜 B, B 胜 C, C 胜 A
    models = ["model_A", "model_B", "model_C"]
    test_matches = [
        {"model0": "model_A", "model1": "model_B", "m0_wins": 70, "total_games": 100}, # A 胜 B (70%)
        {"model0": "model_B", "model1": "model_C", "m0_wins": 65, "total_games": 100}, # B 胜 C (65%)
        {"model0": "model_C", "model1": "model_A", "m0_wins": 75, "total_games": 100}, # C 胜 A (75%)
    ]
    res = analyze_league_cycles(test_matches, models)
    print(json.dumps(res, indent=2, ensure_ascii=False))
