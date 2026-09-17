#!/usr/bin/env bash
# 监控 11 节点 22 卡破局长训实时状态
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES_FILE="${1:-$SCRIPT_DIR/nodes_current.txt}"
WORK="/tmp/league_breakthrough_work"

N_PORT=(); N_HOST=(); N_PASS=()
PENDING_PORT=""; PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d "\r" | sed "s/^[ \t]*//;s/[ \t]*$//")"
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue
  if [[ "$line" =~ ^ssh\ -p\ ([0-9]+)\ root@([^ ]+) ]]; then
    PENDING_PORT="${BASH_REMATCH[1]}"
    PENDING_HOST="${BASH_REMATCH[2]}"
  elif [ -n "$PENDING_PORT" ] && [ -n "$PENDING_HOST" ]; then
    N_PORT+=("$PENDING_PORT")
    N_HOST+=("$PENDING_HOST")
    N_PASS+=("$line")
    PENDING_PORT=""; PENDING_HOST=""
  fi
done < "$NODES_FILE"
NW=${#N_PORT[@]}

echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $NW 节点 $((NW*2)) 卡破局训练状态 ==="
for i in $(seq 0 $((NW-1))); do
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
  echo "  rank $i ${N_HOST[$i]}:$st$diskw  $line"
done
