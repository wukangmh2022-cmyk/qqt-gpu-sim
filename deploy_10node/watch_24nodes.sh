#!/bin/bash
# 监控 24 台训练：每 60s 拉各节点 train_r<i>.log 状态；进程在但日志 5 分钟
# 无更新 → 标 [疑似卡死]（RCCL 同步屏障下某台掉线=全体卡死，需降级重启）。
# 用法: bash watch_24nodes.sh nodes_24x2.txt
# 注意: 重新 launch 前先停掉本脚本（launch 会 rm -rf /tmp/ndrun）。
set -u
NODES_FILE="${1:?用法: watch_24nodes.sh nodes_24x2.txt}"
WORK=/tmp/ndrun
[ -d "$WORK" ] || { echo "先跑 launch_24nodes.sh 生成封装"; exit 1; }
N_PORT=(); N_HOST=(); N_PASS=()
while read -r p h pw; do
  [ -z "$p" ] && continue
  N_PORT+=("$p"); N_HOST+=("$h"); N_PASS+=("$pw")
done < "$NODES_FILE"
NW=${#N_PORT[@]}

while true; do
  clear 2>/dev/null || true
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $NW 台训练状态 ==="
  for i in $(seq 0 $((NW-1))); do
    # 进程检测匹配实际命令行 python3 -m jax_bomb.train_real（不能用 multicast_train）
    out=$("$WORK/cmd_$i" "L=/root/private_data/train_r$i.log; ps aux | grep -c '[j]ax_bomb.train_real'; stat -c %Y \$L 2>/dev/null || echo 0; date +%s; df -B1G /root/private_data 2>/dev/null | tail -1 | awk '{print \$4}'; grep -E 'iter [0-9]+/' \$L 2>/dev/null | tail -1" 2>/dev/null | tail -5)
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
    echo "  rank $i ${N_HOST[$i]} $st$diskw  $line"
  done
  echo "=== 掉线降级：全部 pkill → 用存活节点重写 nodes_24x2.txt → launch_24nodes.sh（自动接续断点）==="
  sleep 60
done
