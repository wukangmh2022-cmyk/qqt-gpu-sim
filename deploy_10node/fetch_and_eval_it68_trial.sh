#!/usr/bin/env bash
# ==============================================================================
# 从 Rank 0 拉取 trial 跑出的最新 ckpt_local，导出 ONNX 并执行 Headless 评测
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"

# 提取 Rank 0 节点信息
RANK0_PORT=""; RANK0_HOST=""; RANK0_PASS=""
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d '\r' | sed 's/^[ \t]*//;s/[ \t]*$//')"
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue

  if [[ "$line" =~ ^ssh\ -p\ ([0-9]+)\ root@([^ ]+) ]]; then
    RANK0_PORT="${BASH_REMATCH[1]}"
    RANK0_HOST="${BASH_REMATCH[2]}"
  elif [ -n "$RANK0_PORT" ] && [ -n "$RANK0_HOST" ]; then
    RANK0_PASS="$line"
    break
  fi
done < "$NODES_FILE"

echo "=== Rank 0 节点: $RANK0_HOST:$RANK0_PORT ==="

EXPECT_SCP="/tmp/it68_trial_work/scp_0"
EXPECT_CMD="/tmp/it68_trial_work/cmd_0"

# 1. 检查远端是否已产出 params 快照
RUN_DIR="/root/private_data/qqt-gpu-sim_r0/runs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32_global16k"
OUT=$("$EXPECT_CMD" "ls -t $RUN_DIR/ckpt_local/params_it*.pkl 2>/dev/null | head -1" | tr -d '\r' | tail -1)

if [ -z "$OUT" ] || [[ "$OUT" == *"No such file"* ]] || [[ "$OUT" != *".pkl"* ]]; then
  echo "⚠️ 远端尚未产出轻量快照，请稍候再拉取！"
  exit 1
fi

REMOTE_PKL="$OUT"
PKL_NAME="$(basename "$REMOTE_PKL")"
LOCAL_DIR="$ROOT/runs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32_global16k/ckpt_local"
mkdir -p "$LOCAL_DIR"

echo "📦 发现远端快照: $REMOTE_PKL -> 拉取回本地 $LOCAL_DIR/$PKL_NAME"
"$EXPECT_SCP" "$REMOTE_PKL" "$LOCAL_DIR/$PKL_NAME" >/dev/null 2>&1

echo "✅ 成功拉取权重快照！大小: $(ls -lh "$LOCAL_DIR/$PKL_NAME" | awk '{print $5}')"

# 2. 导出 ONNX
echo "🔄 导出为 Web ONNX 模型..."
ONNX_STEM="${PKL_NAME%.pkl}"
PYTHONPATH="$ROOT" "$ROOT/.venv/bin/python" "$ROOT/deploy/export_jax_onnx.py" "$LOCAL_DIR/$PKL_NAME"

# 3. 运行 Headless 评测
echo "🎮 启动 Headless 盲测对战 AI Hunter (8 局速测)..."
node "$ROOT/scripts/eval_headless_parallel.js" --model "$ONNX_STEM" --domain open_hunter --games 8
node "$ROOT/scripts/eval_headless_parallel.js" --model "$ONNX_STEM" --domain full_hunter --games 8

