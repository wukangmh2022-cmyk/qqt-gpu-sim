#!/usr/bin/env python3
"""
30分钟定时巡检脚本：
1. 检查全部 12 节点（24 卡）进程健康度。
2. 拉取 Rank 0 最新日志，解析关键指标（iter, sps, loss, mean_reward, kill_rate, entropy 等）。
3. 评估 5Hz 宏观步调整下的策略演进曲线（是否出现预期的短期回撤与稳步回升）。
"""
import sys
import re
import subprocess

def run_ssh(port, host, password, cmd):
    ssh_cmd = [
        "/opt/homebrew/bin/sshpass", "-p", password,
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10",
        "-p", str(port), f"root@{host}", cmd
    ]
    try:
        res = subprocess.run(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def parse_nodes(nodes_file):
    nodes = []
    pending_port = pending_host = None
    with open(nodes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'^ssh\s+-p\s+(\d+)\s+root@(\S+)', line)
            if m:
                pending_port, pending_host = m.group(1), m.group(2)
            elif pending_port and pending_host:
                nodes.append((pending_port, pending_host, line))
                pending_port = pending_host = None
    return nodes

def check_cluster():
    nodes = parse_nodes("deploy_10node/nodes_current.txt")
    print(f"=== [1/3] 集群存活状态校验（共 {len(nodes)} 台）===")
    active_count = 0
    for idx, (port, host, pw) in enumerate(nodes):
        pid = run_ssh(port, host, pw, "pgrep -f 'jax_bomb.train_real'")
        if pid and any(p.isdigit() for p in pid.split()):
            clean_pid = pid.split()[0]
            print(f"  ✓ Rank {idx} (port {port}): 在线 [PID {clean_pid}]")
            active_count += 1
        else:
            print(f"  ✗ Rank {idx} (port {port}): 异常离线或未启动 [输出: {pid}]")
    
    print(f"\n在线卡数: {active_count * 2} / {len(nodes) * 2} 卡")
    
    # 检查 Rank 0 日志
    print("\n=== [2/3] Rank 0 训练遥测 ===")
    r0_port, r0_host, r0_pw = nodes[0]
    log_tail = run_ssh(r0_port, r0_host, r0_pw, "tail -n 25 /root/train_r0.log 2>/dev/null")
    print(log_tail)

    # 4. 自动拉取最新 checkpoint 并执行本地 Headless 真实对局基准评测 (180 局)
    print("\n=== [4/4] 本地 Headless 真实对局基准评测 (180 局) ===")
    try:
        res = subprocess.run(["bash", "scripts/pull_and_eval_latest.sh"], timeout=600)
    except Exception as e:
        print(f"评测执行异常: {e}")

if __name__ == "__main__":
    check_cluster()

