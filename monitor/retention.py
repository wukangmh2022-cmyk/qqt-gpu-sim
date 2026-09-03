#!/usr/bin/env python3
"""
monitor/retention.py - 检查点生命周期管理、滚动清理与巅峰里程碑 (Milestone) 判定

规则协议 (3-Tier Retention Policy):
1. Peak Elo 里程碑 (永久锁定):
   - 经 128 局评测后，刷新历史最高 Universal Elo (且未触发退化告警) 的 Checkpoint
   - 永不被滚动清理淘汰，并自动加入跨代联赛对抗池
2. 容灾恢复滚动档 (云端/本地):
   - 仅保留最新的 3 个带优化器全量恢复档 (94MB)，多余的自动 prune
3. 评估候选轻量档:
   - 滚动保留最新的 5 个轻量参数档 (29MB)；其余非里程碑老档自动清理，防止磁盘爆炸
"""
import os
import re
import glob
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "ckpt")
WEB_MODELS_DIR = os.path.join(ROOT, "web", "models")
HISTORY_FILE = os.path.join(ROOT, "monitor", "history.json")

# 黄金固定保护白名单 (永远不删)
GOLDEN_PROTECTED = [
    "params_it00000068_hlgauss_top25foractor_patch3_k32",
    "params_it00000068_scheme1_actor_top25_critic_all_patch3_k32"
]

def check_and_mark_milestone(report, history):
    """
    判定是否为战力巅峰里程碑 (Peak Elo Milestone)
    准则：
    1. 策略必须是 HEALTHY (发呆率、自杀率健康，无死锁)
    2. Elo 必须严格高于当前历史所有已记录模型的最高分 (净战力突破)
    """
    health = report.get("health", {})
    if health.get("status") not in ["HEALTHY", "info"]:
        return False, "未通过健康门禁，存在死锁或退化"

    curr_elo = report.get("elo", {}).get("composite", 0)
    if curr_elo <= 0:
        return False, "无有效 Elo 评分"

    max_hist_elo = 0
    for h in history:
        # 跳过当前正在判定的报告
        if h.get("modelName") == report.get("modelName"):
            continue
        h_elo = h.get("elo", {}).get("composite", 0)
        if h_elo > max_hist_elo:
            max_hist_elo = h_elo

    if curr_elo > max_hist_elo:
        reason = f"🏆 创历史最高战力巅峰 (Elo: {curr_elo} > 历史峰值: {max_hist_elo})"
        report["isMilestone"] = True
        report["milestoneReason"] = reason
        return True, reason

    # 辅准则：关键代数大节点 (如 1000, 5000, 10000 iters 且健康)
    it = report.get("iteration", 0)
    if it > 0 and (it % 1000 == 0):
        reason = f"🚩 阶段性千轮大关里程碑 (Iter {it})"
        report["isMilestone"] = True
        report["milestoneReason"] = reason
        return True, reason

    report["isMilestone"] = False
    return False, "常规迭代快照"

def prune_local_storage(keep_last_n=5, dry_run=False):
    """
    对本地 ckpt/ 与 web/models/ 执行滚动垃圾回收
    - 保护白名单与所有 isMilestone 标记的文件
    - 仅保留最近的 keep_last_n 个非里程碑快照
    """
    # 1. 收集受保护的快照集合
    protected_models = set(GOLDEN_PROTECTED)
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            for h in history:
                if h.get("isMilestone"):
                    protected_models.add(h.get("modelName"))
        except Exception:
            pass

    # 2. 扫描本地 ckpt/*.pkl
    pkl_files = glob.glob(os.path.join(CKPT_DIR, "params_it*.pkl"))
    # 按 mtime 排序 (最新排在最后)
    pkl_files.sort(key=lambda p: os.path.getmtime(p))

    deleted_pkl = []
    retained_pkl = []

    # 筛选出非受保护的文件
    unprotected = [p for p in pkl_files if os.path.splitext(os.path.basename(p))[0] not in protected_models]

    if len(unprotected) > keep_last_n:
        to_delete = unprotected[:-keep_last_n]
        retained_pkl = unprotected[-keep_last_n:]
        for p in to_delete:
            stem = os.path.splitext(os.path.basename(p))[0]
            if not dry_run:
                try:
                    os.remove(p)
                    deleted_pkl.append(stem)
                    # 同时删除 web/models/ 下对应的 onnx 和 json
                    for ext in [".onnx", ".json"]:
                        wpath = os.path.join(WEB_MODELS_DIR, stem + ext)
                        if os.path.exists(wpath):
                            os.remove(wpath)
                except Exception as e:
                    print(f"[Prune Error] 删除 {stem} 失败: {e}")
            else:
                deleted_pkl.append(stem)

    return {
        "deletedCount": len(deleted_pkl),
        "deletedModels": deleted_pkl,
        "protectedCount": len(protected_models),
        "dryRun": dry_run
    }

def get_cloud_cleanup_command(cloud_ckpt_dir="/root/private_data/qqt-gpu-sim/ckpt", keep_n=3):
    """生成远端集群自动滚动删除老旧 checkpoint 的 shell 单行命令"""
    cmd = (
        f"cd {cloud_ckpt_dir} && "
        f"ls -t ckpt_*_r*.pkl 2>/dev/null | tail -n +{keep_n + 1} | xargs -I {{}} rm -f {{}} && "
        f"ls -t params_it*.pkl 2>/dev/null | tail -n +8 | xargs -I {{}} rm -f {{}}"
    )
    return cmd

if __name__ == "__main__":
    res = prune_local_storage(keep_last_n=5, dry_run=True)
    print("Prune Dry-Run Result:", json.dumps(res, indent=2, ensure_ascii=False))
    print("\nCloud Clean Command:\n", get_cloud_cleanup_command())
