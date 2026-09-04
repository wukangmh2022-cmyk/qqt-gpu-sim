#!/bin/bash
# 启动浏览器版服务器（web/）
#
# 用法：
#   bash scripts/serve_web.sh [端口] [--export]        # 默认 8080
#
# 默认行为：纯启动 Web 服务器，不执行模型导出，不修改 web/models 及 index.json。
# 仅当显式传入 --export 参数时，才执行增量模型导出。

set -e
cd "$(dirname "$0")/.."
PORT="8080"
DO_EXPORT=false

for arg in "$@"; do
  if [[ "$arg" == "--export" ]]; then
    DO_EXPORT=true
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    PORT="$arg"
  fi
done

if [[ "$DO_EXPORT" == true ]]; then
  echo "== 增量导出 ckpt → web/models（新档自检，已最新跳过）=="
  .venv/bin/python deploy/export_ckpt.py --incremental --verify || echo "（torch 导出跳过/失败，继续开服）"
  echo "== transformer：JSON（纯 JS 兜底）=="
  .venv/bin/python deploy/export_jax_ckpt.py --incremental --verify || echo "（JAX JSON 导出跳过/失败，继续开服）"
  echo "== transformer：ONNX（WebGPU/WASM 推理）=="
  .venv/bin/python deploy/export_jax_onnx.py --incremental --verify || echo "（JAX ONNX 导出跳过/失败，继续开服）"
fi

echo "== 开服 http://localhost:${PORT} =="
exec .venv/bin/python -m http.server -d web "$PORT"
