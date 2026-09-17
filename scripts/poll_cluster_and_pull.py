#!/usr/bin/env python3
import subprocess, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODES_FILE = ROOT / "deploy_10node" / "nodes_current.txt"

def load_nodes():
    nodes = []
    lines = NODES_FILE.read_text(encoding="utf-8").splitlines()
    pending_port = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^ssh\s+-p\s+(\d+)\s+root@", line)
        if m:
            pending_port = int(m.group(1))
        elif pending_port:
            nodes.append((len(nodes), pending_port, line))
            pending_port = None
    return nodes

def run_ssh(port, pwd, cmd):
    c = [
        "/opt/homebrew/bin/sshpass", "-p", pwd,
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=8",
        "-p", str(port),
        "root@ssh.zzai.scnet.cn",
        cmd
    ]
    try:
        res = subprocess.run(c, capture_output=True, text=True, timeout=12)
        return res.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

nodes = load_nodes()
nw = len(nodes)
print(f"=== 集群 {nw} 节点 {nw * 2} 卡当前运行状态 ===")
for rank, port, pwd in nodes:
    line = run_ssh(port, pwd, f"tail -n 1 /root/private_data/train_r{rank}.log 2>/dev/null || true")
    print(f"[Rank {rank:02d} | port {port}]: {line}")

if nodes:
    r0_port, r0_pwd = nodes[0][1], nodes[0][2]
    print(f"\n=== Rank 0 (port {r0_port}) 检查点产出情况 ===")
    ckpt_out = run_ssh(r0_port, r0_pwd, "ls -lht /root/private_data/qqt-gpu-sim_r0/ckpt_local/params_it*.meta.json 2>/dev/null | head -10")
    print(ckpt_out if ckpt_out else "暂无新检查点")
