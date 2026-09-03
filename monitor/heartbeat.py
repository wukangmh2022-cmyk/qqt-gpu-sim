#!/usr/bin/env python3
"""
monitor/heartbeat.py - 集群连接与训练心跳（疑似掉线/假死/掉卡）检测模块
"""
import os
import re
import time
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITOR_DIR = os.path.join(ROOT, "monitor")
LOCAL_CKPT_DIR = os.path.join(ROOT, "ckpt")
TRAIN_LOG_FILE = os.path.join(MONITOR_DIR, "train_r0_full.log")
MULTICARD_RESULT = os.path.join(ROOT, "multicard_result.txt")

# 默认集群配置 (可通过环境变量覆盖)
NODE0_HOST = os.environ.get("CLUSTER_HOST", "ssh.zzai.scnet.cn")
NODE0_PORT = os.environ.get("CLUSTER_PORT", "11187")
NODE0_PW = os.environ.get("CLUSTER_PW", "YTD5MAZHDEY5JRE")

# 掉线与假死判定阈值 (正常每轮约 10~15 秒)
HEARTBEAT_WARN_SECONDS = 120   # 超过 2 分钟无新日志 -> 报警疑似卡顿
HEARTBEAT_DEAD_SECONDS = 300   # 超过 5 分钟无新日志 -> 判定掉线或假死

ITER_TIME_RE = re.compile(r"\[([\d\- :]+)\] iter \d+")

def get_last_log_info():
    """读取本地最新日志条目的时间戳与内容"""
    target = None
    for p in [TRAIN_LOG_FILE, MULTICARD_RESULT]:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            target = p
            break
    if not target:
        return None, None

    last_line = ""
    try:
        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
    except Exception:
        return None, None

    if not last_line:
        return None, None

    # 提取时间戳
    m = re.search(r"\[([\d\- :]+)\]", last_line)
    if m:
        try:
            dt = datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S")
            return dt, last_line
        except Exception:
            pass

    # 兜底：取文件的修改时间 mtime
    mtime = datetime.fromtimestamp(os.path.getmtime(target))
    return mtime, last_line

def test_ssh_connectivity(timeout=4):
    """测试 SSH 端口连通性"""
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-p", str(NODE0_PORT),
        f"root@{NODE0_HOST}",
        "echo alive"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # BatchMode 会因为密码认证返回 255，但如果包含 Permission denied 说明网络是通的！
        # 如果是 Connection refused 或 Operation timed out 则说明网络断开！
        if res.returncode == 0:
            return True, "认证成功并连通"
        err = (res.stderr or res.stdout).lower()
        if "permission denied" in err:
            return True, "SSH 端口通畅 (等待密码/公钥认证)"
        if "timed out" in err or "connection refused" in err or "no route" in err:
            return False, "SSH 网络不可达或端口已关闭"
        return False, f"SSH 连接异常: {err.splitlines()[0] if err else '未知错误'}"
    except subprocess.TimeoutExpired:
        return False, "SSH 连接超时 (>3s)"
    except Exception as e:
        return False, str(e)

def try_pull_remote():
    """尝试通过 expect 拉取远端日志与检查点 (尽力而为，不阻断主流程)"""
    exp_script = f"""
set timeout 15
spawn scp -o StrictHostKeyChecking=accept-new -P {NODE0_PORT} root@{NODE0_HOST}:/root/private_data/qqt-gpu-sim/multicard_result.txt {MONITOR_DIR}/multicard_result.txt
expect {{
  "password:" {{ send "{NODE0_PW}\\r"; exp_continue }}
  eof
}}
"""
    try:
        res = subprocess.run(["expect", "-c", exp_script], capture_output=True, text=True, timeout=20)
        return "Permission denied" not in res.stdout
    except Exception:
        return False

def check_heartbeat(enable_remote_ping=True):
    """
    全量心跳检查：
    1. 检查本地是否有活跃训练进程
    2. 检查日志最新产出时间（心跳倒计时）
    3. 检查远端 SSH 网络连通性
    """
    now = datetime.now()
    last_dt, last_line = get_last_log_info()

    # 1. 检查本地 GPU / JAX 训练进程
    local_running = False
    try:
        ps = subprocess.run(["pgrep", "-f", "multicard_train.py"], capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            local_running = True
    except Exception:
        pass

    # 2. 计算心跳静默时间 (Idle Seconds)
    idle_seconds = int((now - last_dt).total_seconds()) if last_dt else 999999
    last_heartbeat_str = last_dt.strftime("%Y-%m-%d %H:%M:%S") if last_dt else "未知"

    # 3. 远端连通性测试
    ssh_ok = False
    ssh_msg = "未启用网络探测"
    if enable_remote_ping:
        ssh_ok, ssh_msg = test_ssh_connectivity()

    # 4. 综合健康判定 (State Machine)
    if local_running:
        mode = "local"
        if idle_seconds <= HEARTBEAT_WARN_SECONDS:
            status = "RUNNING"
            color = "green"
            msg = f"本地训练中 (心跳正常: {idle_seconds}s 前更新)"
        elif idle_seconds <= HEARTBEAT_DEAD_SECONDS:
            status = "HANG_SUSPECTED"
            color = "yellow"
            msg = f"本地训练疑似卡死 (已 {idle_seconds}s 无新 iter 输出)"
        else:
            status = "STOPPED"
            color = "gray"
            msg = f"本地训练已停止 (静止 {idle_seconds}s)"
    else:
        mode = "remote"
        if not ssh_ok:
            status = "DISCONNECTED"
            color = "red"
            msg = f"远端集群网络不可达 / 掉线 ({ssh_msg})"
        else:
            if idle_seconds <= HEARTBEAT_WARN_SECONDS:
                status = "RUNNING"
                color = "green"
                msg = f"集群训练运行中 (心跳正常: {idle_seconds}s 前)"
            elif idle_seconds <= HEARTBEAT_DEAD_SECONDS:
                status = "HANG_SUSPECTED"
                color = "yellow"
                msg = f"集群疑似假死/卡顿 (已 {idle_seconds}s 无更新，可能掉卡或死锁)"
            else:
                status = "IDLE_OR_FINISHED"
                color = "gray"
                msg = f"集群无活动训练任务 (最后记录: {idle_seconds}s 前)"

    return {
        "status": status,
        "mode": mode,
        "color": color,
        "message": msg,
        "idleSeconds": idle_seconds,
        "lastHeartbeat": last_heartbeat_str,
        "lastLine": (last_line[:120] + "…") if last_line and len(last_line) > 120 else (last_line or ""),
        "sshReachable": ssh_ok,
        "sshDetail": ssh_msg,
        "checkedAt": now.strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    import json
    hb = check_heartbeat()
    print(json.dumps(hb, indent=2, ensure_ascii=False))
