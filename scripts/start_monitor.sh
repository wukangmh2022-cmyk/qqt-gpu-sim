#!/usr/bin/env bash
# scripts/start_monitor.sh - 一键拉起强化学习监控与控制面
# 包含 Localhost:8088 仪表盘 + 训练时 Watchdog 守护进程

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${1:-8088}"
PYTHON="${ROOT}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "=========================================================================="
echo "⚡ 正在启动 QQT-RL 训练监控与策略拦截系统"
echo "   工作目录: $ROOT"
echo "   监控端口: http://localhost:${PORT}"
echo "   核心底座: 4 并发 128 局真实 WebJS 极速验收 (3.1 min)"
echo "   诊断体系: CleanRL 优化熵 + RLlib 博弈死锁/控图率 + 自动告警拦截"
echo "=========================================================================="

mkdir -p "$ROOT/monitor" "$ROOT/ckpt" "$ROOT/ckpt_local"

# 1. 检查并拉起后台 Watchdog 守护进程 (避免重复运行)
if pgrep -f "monitor/watchdog.py" > /dev/null; then
  echo "👀 检测到已有 Watchdog 守护进程在后台运行。"
else
  echo "🚀 启动后台 Watchdog 守护进程..."
  nohup "$PYTHON" monitor/watchdog.py > monitor/watchdog.log 2>&1 &
  echo "   Watchdog 日志写入: monitor/watchdog.log (PID: $!)"
fi

# 2. 前台拉起 Localhost 控制面 Web 服务
echo "🌐 启动 Localhost 控制台: http://localhost:${PORT}"
echo "   按 Ctrl+C 可停止 Web 控制服务。"
echo "=========================================================================="
exec "$PYTHON" monitor/server.py "$PORT"
