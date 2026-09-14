#!/bin/bash
# 把 rank0 的参数快照（params_it*.pkl / *.meta.json）从 notebook 拉回本地
# 用法: bash pull_ckpt_local.sh nodes.txt [目标目录]
set -u
NODES_FILE="${1:?用法: pull_ckpt_local.sh nodes.txt [dir]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${2:-$ROOT/ckpt_local}"

PORT=""
HOST=""
PW=""
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  line="${line#"${line%%[![:space:]]*}"}"
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue
  if [ -n "$PORT" ] && [ -n "$HOST" ] && [ -z "$PW" ]; then
    PW="$line"
    break
  fi
  if [[ "$line" =~ ^ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+root@([^[:space:]]+)$ ]]; then
    PORT="${BASH_REMATCH[1]}"
    HOST="${BASH_REMATCH[2]}"
    continue
  fi
  read -r p h pw extra <<< "$line"
  if [ -n "${p:-}" ] && [ -n "${h:-}" ] && [ -n "${pw:-}" ]; then
    PORT="$p"; HOST="$h"; PW="$pw"
    break
  fi
done < "$NODES_FILE"

[ -n "$PORT" ] && [ -n "$HOST" ] && [ -n "$PW" ] || { echo "无法从 $NODES_FILE 解析 rank0 信息"; exit 1; }

mkdir -p "$DST"
echo "=== 从 rank0 ${HOST}:${PORT} 拉取快照 → $DST ==="
expect -c "
set timeout 300
spawn scp -o StrictHostKeyChecking=accept-new -P $PORT root@$HOST:/root/private_data/qqt-gpu-sim_r0/ckpt_local/params_* $DST/
expect {
  \"password:\" { send \"$PW\r\"; exp_continue }
  \"yes/no\" { send \"yes\r\"; exp_continue }
  \"*No such file*\" { puts \"NO_SNAPSHOTS\" }
  eof { }
  timeout { puts \"PULL_TIMEOUT\" }
}" 2>&1 | grep -v "assword" | tail -3
if ls "$DST"/params_*.pkl >/dev/null 2>&1; then
  echo "=== 本机 $DST 最新快照： ==="
  ls -la "$DST"/params_*.pkl 2>/dev/null | tail -5
else
  echo "（尚无快照——检查点正在生成中）"
fi
