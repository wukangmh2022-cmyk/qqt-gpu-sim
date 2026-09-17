#!/bin/bash
# 把 rank0 的参数快照（params_it*.pkl / *.meta.json）从 notebook 拉回本地
# 用法: bash pull_ckpt_local.sh [nodes.txt] [目标目录]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODES_FILE="${1:-$ROOT/deploy_10node/nodes_current.txt}"
DST="${2:-$ROOT/ckpt_local}"

PORT=""
HOST=""
PW=""

while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d '\r' | sed 's/^[ \t]*//;s/[ \t]*$//')"
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue
  if [[ "$line" =~ ^ssh\ -p\ ([0-9]+)\ root@([^ ]+) ]]; then
    PORT="${BASH_REMATCH[1]}"
    HOST="${BASH_REMATCH[2]}"
  elif [ -n "$PORT" ] && [ -n "$HOST" ]; then
    PW="$line"
    break
  fi
done < "$NODES_FILE"

if [ -z "$PORT" ] || [ -z "$PW" ]; then
  echo "错误: 未能从 $NODES_FILE 解析出 Rank 0 节点信息"
  exit 1
fi

mkdir -p "$DST"
/opt/homebrew/bin/sshpass -p "$PW" scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" "root@$HOST:/root/private_data/qqt-gpu-sim_r0/ckpt_local/params_*" "$DST/" 2>/dev/null || true
/opt/homebrew/bin/sshpass -p "$PW" scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$PORT" "root@$HOST:/root/private_data/qqt-gpu-sim_r0/ckpt/params_*" "$DST/" 2>/dev/null || true

if ls "$DST"/params_*.pkl >/dev/null 2>&1; then
  echo "=== 本机 $DST 最新快照： ==="
  ls -lh "$DST"/params_*.pkl 2>/dev/null | tail -5
else
  echo "当前暂无新快照生成"
fi
