#!/usr/bin/env bash
# 新 DCU 环境 scp 上传/下载 helper。用法：
#   scripts/dcu2_scp.sh <local> <remote-path>   # 上传
#   scripts/dcu2_scp.sh <remote> <local-path> -d # 下载（第三个参数 -d）
set -euo pipefail
_ENV_F="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$_ENV_F" ] && source "$_ENV_F"
HOST="${DCU2_USER:-root}@${DCU2_HOST:-ssh.zzai.scnet.cn}"
PORT="${DCU2_PORT:-10717}"
PASS="${DCU2_PASS:-}"
if [ -z "$PASS" ]; then echo "DCU2_PASS empty" >&2; exit 1; fi

SRC="$1"; DST="$2"; MODE="${3:-up}"
if [ "$MODE" = "up" ]; then
  REMOTE_PATH="scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no -P $PORT $SRC $HOST:$DST"
else
  REMOTE_PATH="scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no -P $PORT $HOST:$SRC $DST"
fi
expect -c "
set timeout 1200
spawn $REMOTE_PATH
expect {
    \"*assword:*\" { send \"$PASS\\r\"; exp_continue }
    \"Permission denied*\" { puts \"AUTH FAILED\"; exit 1 }
    eof
}"
