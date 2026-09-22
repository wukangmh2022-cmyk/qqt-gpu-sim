#!/usr/bin/env bash
# ==============================================================================
# 每 30 分钟自动化流水线：
# 1. 从超算集群拉取最新的 params_it*_ema.pkl
# 2. 导出为 Web JSON 与 ONNX 模型
# 3. 运行 Headless 真实对战评测（对决 时空A*、FleeBot、BFS猎人）
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

R0_PORT="11791"
R0_HOST="ssh.zzai.scnet.cn"
R0_PASS="UV2HHLJPHT4VYSN"
REMOTE_CKPT_DIR="/root/private_data/stage5_64g_runs/ckpt_local"

echo "=== [1/3] 检查远程集群最新模型快照 ==="
REMOTE_LATEST=$(/opt/homebrew/bin/sshpass -p "$R0_PASS" ssh \
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 \
  -p "$R0_PORT" "root@$R0_HOST" \
  "ls -1t $REMOTE_CKPT_DIR/*_ema.pkl 2>/dev/null | head -1" | tr -d '\r\n')

if [ -z "$REMOTE_LATEST" ]; then
  echo "未检测到远程 EMA 检查点！"
  exit 1
fi

BN=$(basename "$REMOTE_LATEST")
STEM="${BN%.pkl}"
echo "  远程最新快照: $BN"

# 拉取到本地 ckpt/
mkdir -p ckpt
if [ ! -f "ckpt/$BN" ]; then
  echo "  正在拉取 $BN 至本地 ckpt/ ..."
  /opt/homebrew/bin/sshpass -p "$R0_PASS" scp \
    -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -P "$R0_PORT" "root@$R0_HOST:$REMOTE_LATEST" "ckpt/$BN"
  echo "  ✓ 拉取完成: ckpt/$BN"
else
  echo "  ✓ 本地已有最新快照: ckpt/$BN"
fi

echo "=== [2/3] 导出 ONNX 与 Web JSON 权重 ==="
if [ ! -f "web/models/${STEM}.onnx" ]; then
  .venv/bin/python deploy/export_jax_ckpt.py "ckpt/$BN"
  .venv/bin/python deploy/export_jax_onnx.py "ckpt/$BN"
  echo "  ✓ 权重转换导出就绪: web/models/${STEM}.onnx"
else
  echo "  ✓ 模型已处于导出就绪状态: web/models/${STEM}.onnx"
fi

echo "=== [3/3] 运行 Headless 真实对战基准评测 (全场景 6 领域对决，共 180 局) ==="
node scripts/eval_headless_parallel.js --model "$STEM" --domain benchmark --games 30 --workers 6
