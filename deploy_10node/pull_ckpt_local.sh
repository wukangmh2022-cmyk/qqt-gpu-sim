#!/bin/bash
# 把 rank0 的 30 分钟参数快照（params_it*.pkl）从 notebook 拉回本地
# 用法: bash pull_ckpt_local.sh nodes.txt [目标目录]
set -u
NODES_FILE="${1:?用法: pull_ckpt_local.sh nodes.txt [dir]}"
DST="${2:-/Users/a1-6/Documents/llm-train/qqt-gpu-sim/ckpt_local}"
read -r PORT HOST PW < "$NODES_FILE"
mkdir -p "$DST"
echo "=== 从 rank0 ${HOST}:${PORT} 拉 params_it*.pkl → $DST ==="
expect -c "
set timeout 300
spawn scp -o StrictHostKeyChecking=accept-new -P $PORT root@$HOST:/root/private_data/qqt-gpu-sim/ckpt_local/params_*.pkl $DST/
expect {
  \"password:\" { send \"$PW\r\"; exp_continue }
  \"yes/no\" { send \"yes\r\"; exp_continue }
  \"*No such file*\" { puts \"NO_SNAPSHOTS\" }
  eof { }
  timeout { puts \"PULL_TIMEOUT\" }
}" 2>&1 | grep -v "assword" | tail -3
if ls "$DST"/params_*.pkl >/dev/null 2>&1; then
  echo "=== 本机 $DST 最新快照： ==="
  ls -la "$DST"/params_*.pkl 2>/dev/null | tail -3
else
  echo "（尚无快照——训练未满 30 分钟、rank0 未写，或 ckpt_local/ 路径不对）"
fi
