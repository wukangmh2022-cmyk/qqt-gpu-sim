#!/usr/bin/env python3
"""
scripts/auto_watch_and_finish.py
自动监控远端 DCU 8 卡训练进度，训练完成后自动：
1. 拷贝远端 ckpt/、ckpt_local/ 与 multicard_result.txt 回本地
2. 校验本地检查点完整性
3. 执行远端关机 (poweroff) 节省算力开销
4. 本地转换 ONNX 并启动 Headless 盲测与性能回归评估
"""

import os
import sys
import time
import subprocess
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REMOTE_EXEC = os.path.join(ROOT, "scripts", "remote_exec.py")
REMOTE_SYNC = os.path.join(ROOT, "scripts", "remote_sync.py")
PYTHON_EXE = os.path.join(ROOT, ".venv", "bin", "python")


def run_cmd(cmd, check=True):
    print(f"==> [CMD] {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and p.returncode != 0:
        print(f"Error ({p.returncode}): {p.stderr}", flush=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def check_remote_progress():
    rc, out, _ = run_cmd(f"python3 {REMOTE_EXEC} 'tail -n 10 /root/qqt-gpu-sim/multicard_result.txt'", check=False)
    if rc != 0:
        return None, None, False
    
    # Check if process is still running
    rc_ps, out_ps, _ = run_cmd(f"python3 {REMOTE_EXEC} 'ps aux | grep multicard_train | grep -v grep'", check=False)
    is_running = (rc_ps == 0 and "multicard_train" in out_ps)

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    last_line = lines[-1] if lines else ""
    # Find the most recent iter line for display
    iter_lines = [l for l in lines if "iter " in l and "/" in l]
    display_line = iter_lines[-1] if iter_lines else last_line

    is_finished = False
    matches = re.findall(r"iter\s+(\d+)/(\d+)", out)
    if matches:
        cur_it, tot_it = int(matches[-1][0]), int(matches[-1][1])
        if cur_it >= tot_it and not is_running:
            is_finished = True
    if "RUN finish" in out:
        is_finished = True
        
    return display_line, is_running, is_finished


def download_artifacts():
    print("\n📦 开始将远端检查点与训练日志拉取回本地...", flush=True)
    # Pull ckpt_local
    run_cmd(f"python3 {REMOTE_SYNC} --pull /root/qqt-gpu-sim/ckpt_local {ROOT}")
    # Pull ckpt
    run_cmd(f"python3 {REMOTE_SYNC} --pull /root/qqt-gpu-sim/ckpt {ROOT}")
    # Pull multicard_result.txt
    run_cmd(f"python3 {REMOTE_SYNC} --pull /root/qqt-gpu-sim/multicard_result.txt {ROOT}")
    print("✅ 远端文件拉取完成！", flush=True)


def verify_local_artifacts():
    ckpt_local_dir = os.path.join(ROOT, "ckpt_local")
    if not os.path.isdir(ckpt_local_dir):
        print(f"❌ 警告: 本地 {ckpt_local_dir} 不存在！", flush=True)
        return False
    pkls = [f for f in os.listdir(ckpt_local_dir) if f.endswith(".pkl")]
    if not pkls:
        print("❌ 警告: ckpt_local 下未发现任何 .pkl 权重文件！", flush=True)
        return False
    
    print(f"✅ 校验通过，发现 {len(pkls)} 个权重快照: {sorted(pkls)}", flush=True)
    for p in sorted(pkls):
        path = os.path.join(ckpt_local_dir, p)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   - {p}: {size_mb:.2f} MB", flush=True)
    return True


def shutdown_remote():
    print("\n🔌 准备关闭远端机器 (poweroff) 以切断计费...", flush=True)
    rc, out, _ = run_cmd(f"python3 {REMOTE_EXEC} 'sync; poweroff'", check=False)
    print(f"关机指令已发送: {out}", flush=True)
    time.sleep(5)
    # Verify disconnection
    rc_check, _, _ = run_cmd(f"python3 {REMOTE_EXEC} 'uptime'", check=False)
    if rc_check != 0:
        print("✅ 远端机器已成功断联/关机！", flush=True)
    else:
        print("⚠️ 提示: 远端机器尚未断开，正在关机流程中...", flush=True)


def run_local_evaluation():
    print("\n🎯 开始执行本地评估...", flush=True)
    # 1. 导出 ONNX 模型
    export_cmd = f"PYTHONPATH={ROOT} {PYTHON_EXE} deploy/export_jax_onnx.py"
    print(f"导出 ONNX: {export_cmd}", flush=True)
    run_cmd(export_cmd)

    # 2. 找到最新的 ONNX 模型
    models_dir = os.path.join(ROOT, "web", "models")
    onnx_files = sorted([f for f in os.listdir(models_dir) if f.startswith("params_it00000068") and f.endswith(".onnx")])
    if not onnx_files:
        onnx_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".onnx")], key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    
    if onnx_files:
        latest_model = onnx_files[-1].replace(".onnx", "")
        print(f"\n🎮 启动 Headless 盲测对战 (模型: {latest_model})...", flush=True)
        # 对战: vs Hunter
        cmd_hunter = f"node scripts/headless_test.js --models {latest_model} --opp hunter --maps 2 --ep 3"
        print(f"执行对战测试: {cmd_hunter}", flush=True)
        rc, out, _ = run_cmd(cmd_hunter)
        print(out)
    else:
        print("⚠️ 未找到生成的 ONNX 模型，跳过 Headless 对战。", flush=True)


def main():
    print("=======================================================", flush=True)
    print("🚀 QQT 训练自动收尾守护进程启动", flush=True)
    print("   目标: 监控 570M 训练 -> 拷贝检查点 -> 关机远端 -> 本地评测", flush=True)
    print("=======================================================", flush=True)

    while True:
        last_line, is_running, is_finished = check_remote_progress()
        now_str = time.strftime("%H:%M:%S")
        if last_line:
            print(f"[{now_str}] 状态: {'运行中' if is_running else '已停止'} | {last_line}", flush=True)
        else:
            print(f"[{now_str}] 无法获取远端状态，5 秒后重试...", flush=True)

        if is_finished or (last_line and not is_running and "iter" in last_line):
            # Check if iter reached target
            m = re.search(r"iter\s+(\d+)/(\d+)", last_line or "")
            if m and int(m.group(1)) >= int(m.group(2)):
                print("\n🎉 远端 570M 训练已全部跑完！", flush=True)
                break
            elif not is_running:
                print("\n⚠️ 远端进程已退出，开始进入收尾流程！", flush=True)
                break

        time.sleep(25)

    # 1. 拷贝远端文件回本地
    download_artifacts()

    # 2. 校验文件完整性
    if not verify_local_artifacts():
        print("❌ 检查点校验失败，为保护数据安全，暂不关机！请人工排查！", flush=True)
        sys.exit(1)

    # 3. 远端关机
    shutdown_remote()

    # 4. 本地评估
    run_local_evaluation()
    print("\n🏁 全流程自动化任务圆满完成！", flush=True)


if __name__ == "__main__":
    main()
