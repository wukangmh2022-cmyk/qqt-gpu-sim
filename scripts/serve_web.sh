#!/bin/bash
# 启动浏览器版启动器（web/）：先增量导出最新模型（torch mlp/cnn + JAX transformer
# 的 JSON 与 ONNX 双格式），再开 http.server。
#
# 用法：
#   bash scripts/serve_web.sh [端口]        # 默认 8080
#
# 每次启动自动检测 ckpt/ 比 web/models 下产物新的档并转化（带前向自检），
# 已是最新的跳过 —— 训练完新模型后直接重跑本脚本即可刷新页面可选项。

set -e
cd "$(dirname "$0")/.."
PORT="${1:-8080}"

echo "== 增量导出 ckpt → web/models（新档自检，已最新跳过）=="
# 导出为尽力而为：无新档/失败都不阻断开服（export 脚本"无可导出"时返回 1，
# set -e 下会直接杀掉整个脚本 —— 2026-08-30 修复）
.venv/bin/python deploy/export_ckpt.py --incremental --verify || echo "（torch 导出跳过/失败，继续开服）"
echo "== transformer：JSON（纯 JS 兜底）=="
.venv/bin/python deploy/export_jax_ckpt.py --incremental --verify || echo "（JAX JSON 导出跳过/失败，继续开服）"
echo "== transformer：ONNX（WebGPU/WASM 推理）=="
.venv/bin/python deploy/export_jax_onnx.py --incremental --verify || echo "（JAX ONNX 导出跳过/失败，继续开服）"

echo "== 开服 http://localhost:${PORT} =="
exec .venv/bin/python -m http.server -d web "$PORT"
