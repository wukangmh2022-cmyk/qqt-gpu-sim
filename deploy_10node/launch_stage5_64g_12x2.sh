#!/usr/bin/env bash
# ==============================================================================
# 第五阶段：12 节点 x 2 卡 64GB DCU 12 小时 5Hz 宗师全图长训
# Warm-Start 继承 params_it00000831_ema.pkl
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/stage5_64g_12x2gpu_12h.toml"

# 1. 读取节点列表
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
echo "=== 已提取全部 $NW 台在线双卡节点用于 $((NW * 2)) 卡 Stage 5 (64GB 5Hz 全图长训) ==="

WORK="/tmp/stage5_64g_work"
rm -rf "$WORK" && mkdir -p "$WORK"

for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOcmd
#!/bin/bash
exec /opt/homebrew/bin/sshpass -p "${N_PASS[$i]}" ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -p ${N_PORT[$i]} root@${N_HOST[$i]} "\$@"
EOcmd
  cat > "$WORK/scp_$i" <<EOscp
#!/bin/bash
exec /opt/homebrew/bin/sshpass -p "${N_PASS[$i]}" scp -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P ${N_PORT[$i]} "\$1" "root@${N_HOST[$i]}:\$2"
EOscp
  chmod +x "$WORK/cmd_$i" "$WORK/scp_$i"
done

echo "=== [1/4] 清理全部 $NW 台节点历史残留进程 ==="
for i in $(seq 0 $((NW-1))); do
  "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; rm -f /root/train_r*.log; true" >/dev/null 2>&1 || true
  sleep 0.2
done
echo "  ✓ $NW 台节点残留进程清理完毕"

echo "=== [2/4] 初始化持久化存储目录 ==="
"$WORK/cmd_0" "mkdir -p /root/private_data/stage5_64g_runs/ckpt_local" >/dev/null 2>&1 || true
MASTER_IP=$("$WORK/cmd_0" "hostname -I | awk '{print \$1}'" | tr -d '\r\n')
echo "  ✓ Coordinator 节点: $MASTER_IP:29500"

echo "=== [3/4] 分发各节点启动脚本 (WORLD_SIZE=$NW) ==="
for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/start_rank_$i.sh" <<EOSTR
#!/bin/bash
export WORLD_SIZE=$NW
export RANK=$i
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=29500

source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_LIBRARY_PATH=/opt/mpi/lib:\${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/mpi/lib/libmpi.so
export HP_DOMAIN_RAND=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export PYTHONPATH="/root/qqt-gpu-sim:\${PYTHONPATH:-}"
export CKPT_LOCAL_DIR="/root/private_data/stage5_64g_runs/ckpt_local"
export CKPT_LOCAL_EVERY=15

cd /root/qqt-gpu-sim
nohup python3 -u -m jax_bomb.train_real --config $CONFIG_REL </dev/null > /root/train_r$i.log 2>&1 &
PID=\$!
echo "RANK_${i}_STARTED_PID_\${PID}"
EOSTR
  chmod +x "$WORK/start_rank_$i.sh"
  "$WORK/scp_$i" "$WORK/start_rank_$i.sh" "/root/start_rank.sh" >/dev/null 2>&1
  sleep 0.2
done
echo "  ✓ 12 台节点 start_rank.sh 分发就绪"

echo "=== [4/4] 按序启动分布式训练集群 ==="
echo "  -> 启动 Rank 0 (Coordinator)..."
"$WORK/cmd_0" "bash /root/start_rank.sh"
echo "  -> 等待 5 秒确保 Coordinator 绑定 29500 端口..."
sleep 5

echo "  -> 启动 Rank 1 ~ Rank $((NW-1))..."
for i in $(seq 1 $((NW-1))); do
  "$WORK/cmd_$i" "bash /root/start_rank.sh" &
  sleep 0.3
done
wait

echo "=== 全部 $NW 台双卡节点已拉起！等待 10 秒校验各节点进程状态... ==="
sleep 10

for i in $(seq 0 $((NW-1))); do
  status=$("$WORK/cmd_$i" "ps aux | grep train_real | grep -v grep | awk '{print \$2}'" 2>/dev/null | tr -d '\r\n' || echo "")
  if [ -n "$status" ]; then
    echo "  ✓ Rank $i 在线 (PID: $status)"
  else
    echo "  ✗ Rank $i 未检测到进程！"
  fi
done
