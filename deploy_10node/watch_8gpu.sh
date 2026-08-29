#!/bin/bash
# 监控单机 8 卡训练：每 60s 拉 train_8gpu.log 状态；进程在但日志 5 分钟
# 无更新 → 标 [疑似卡死]（pmap 同步屏障下某卡掉线 = 全体卡死，需重启）。
# 用法: bash watch_8gpu.sh nodes_8gpu.txt
set -u
NODES_FILE="${1:?用法: watch_8gpu.sh nodes_8gpu.txt}"
WORK=/tmp/ndrun8
[ -d "$WORK" ] || { echo "先跑 launch_8gpu.sh 生成封装"; exit 1; }
read -r N_PORT N_HOST N_PASS < "$NODES_FILE"

while true; do
  clear 2>/dev/null || true
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  单机 8 卡训练状态 (${N_HOST}:${N_PORT}) ==="
  out=$("$WORK/cmd" "L=/root/private_data/train_8gpu.log; ps aux | grep -c '[j]ax_bomb.train_real'; stat -c %Y \$L 2>/dev/null || echo 0; date +%s; df -B1G /root/private_data 2>/dev/null | tail -1 | awk '{print \$4}'; grep -E 'iter [0-9]+/' \$L 2>/dev/null | tail -1" 2>/dev/null | tail -5)
  live=$(echo "$out" | sed -n 1p)
  mtime=$(echo "$out" | sed -n 2p)
  now=$(echo "$out" | sed -n 3p)
  diskg=$(echo "$out" | sed -n 4p)
  last=$(echo "$out" | sed -n 5p)
  if [ "$mtime" = "0" ]; then
    st="[无日志]"
  elif [ "$live" = "0" ]; then
    st="[已退出]"
  elif [ $(( now - mtime )) -gt 300 ]; then
    st="[疑似卡死 $(( now - mtime ))s无更新]"
  else
    st="[运行中]"
  fi
  line=$(echo "$last" | grep -oE 'iter [0-9]+/[0-9]+.*' | head -c 80)
  diskw=""
  [ -n "$diskg" ] && [ "$diskg" -lt 10 ] 2>/dev/null && diskw=" ⚠磁盘<${diskg}G"
  echo "  $st$diskw  $line"
  echo "=== 卡死处理：kill 进程 → 重跑 launch_8gpu.sh（自动接续断点）==="
  sleep 60
done
