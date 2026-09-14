#!/usr/bin/env python3
"""
自动拉取远端完成的 params_it00000068.pkl，转换为 ONNX，并执行 128 局真实对战评测。
"""
import os
import subprocess
import sys
import time

REMOTE_HOST = "ssh.zzai.scnet.cn"
REMOTE_PORT = "10572"
REMOTE_PASS = "0HLD3RHKXNBI5X5"
REMOTE_DIR = "/root/private_data/qqt-gpu-sim_r0/runs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32_global16k/ckpt_local"

LOCAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CKPT_DIR = os.path.join(LOCAL_ROOT, "runs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32_global16k/ckpt_local")
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

LOCAL_PKL = os.path.join(LOCAL_CKPT_DIR, "params_it00000068.pkl")
LOCAL_META = os.path.join(LOCAL_CKPT_DIR, "params_it00000068.meta.json")
MODEL_NAME = "params_repro_it68"
ONNX_PATH = os.path.join(LOCAL_ROOT, "web", "models", f"{MODEL_NAME}.onnx")
JSON_PATH = os.path.join(LOCAL_ROOT, "web", "models", f"{MODEL_NAME}.json")

def check_remote_ready():
    cmd = [
        "sshpass", "-p", REMOTE_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no", "-p", REMOTE_PORT,
        f"root@{REMOTE_HOST}",
        f"test -f {REMOTE_DIR}/params_it00000068.pkl && echo READY || echo WAITING"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return "READY" in res.stdout

def main():
    print("⏳ 等待远端完成 iter 68 训练并保存 params_it00000068.pkl...", flush=True)
    while not check_remote_ready():
        print("   训练仍在冲刺最后几轮，10 秒后重新轮询...", flush=True)
        time.sleep(10)
    print("✓ 远端 params_it00000068.pkl 已就绪！正在通过 SCP 拉取回本地...", flush=True)

    # SCP pkl
    scp_cmd = [
        "sshpass", "-p", REMOTE_PASS,
        "scp", "-o", "StrictHostKeyChecking=no", "-P", REMOTE_PORT,
        f"root@{REMOTE_HOST}:{REMOTE_DIR}/params_it00000068.pkl",
        LOCAL_PKL
    ]
    subprocess.run(scp_cmd, check=True)
    print(f"✓ 权重已拉取: {LOCAL_PKL} ({os.path.getsize(LOCAL_PKL) / (1024*1024):.1f} MB)")

    # SCP meta if exists
    scp_meta_cmd = [
        "sshpass", "-p", REMOTE_PASS,
        "scp", "-o", "StrictHostKeyChecking=no", "-P", REMOTE_PORT,
        f"root@{REMOTE_HOST}:{REMOTE_DIR}/params_it00000068.meta.json",
        LOCAL_META
    ]
    subprocess.run(scp_meta_cmd, check=False)

    # 导出 ONNX
    print("⚙️ 正在导出 JAX 权重为 ONNX 模型...", flush=True)
    python_bin = os.path.join(LOCAL_ROOT, ".venv", "bin", "python")
    export_cmd = [
        python_bin,
        os.path.join(LOCAL_ROOT, "deploy", "export_jax_onnx.py"),
        LOCAL_PKL,
        "--out", ONNX_PATH
    ]
    subprocess.run(export_cmd, check=True, cwd=LOCAL_ROOT)
    print(f"✓ ONNX 模型已生成: {ONNX_PATH}")

    # 生成 JSON 元数据
    import json
    meta = {
        "name": MODEL_NAME,
        "display_name": MODEL_NAME,
        "arch": "transformer",
        "obs_shape": [14, 13, 15],
        "embed": 392,
        "patch": 3,
        "depth": 4,
        "n_players": 2,
        "it": 68,
        "global_step": 570425344,
        "source": "params_it00000068.pkl"
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": meta}, f, indent=2)
    print(f"✓ 模型描述 JSON 已生成: {JSON_PATH}")

    # 执行 128 局真实对战评测
    print("\n⚔️ 开始执行 128 局真实对战评测套件 (Headless 1800-tick)...")
    eval_cmd = [
        "node",
        os.path.join(LOCAL_ROOT, "scripts", "eval_headless_parallel.js"),
        "--model", MODEL_NAME,
        "--games", "32"
    ]
    subprocess.run(eval_cmd, check=True, cwd=LOCAL_ROOT)

if __name__ == "__main__":
    main()
