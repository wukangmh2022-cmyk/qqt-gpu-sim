#!/usr/bin/env bash
# ==============================================================================
# 第五阶段：12 节点 x 2 卡 64GB DCU 12 小时 5Hz 宗师全图长训
# Warm-Start 继承 params_it00000831_ema.pkl
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/stage5_64g_8x2gpu_12h.toml"

# 1. 读取节点
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
echo "=== 已提取全部 $NW 台在线双卡节点用于 $((NW * 2)) 卡 Stage 5 (5Hz 宗师全图长训) ==="

WORK="/tmp/stage5_64g_8x2_work"
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

echo "=== [1/4] 清理全部 $NW 台节点历史训练残留进程与旧日志 ==="
for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; rm -f /root/private_data/train_r$i.log; true" >/dev/null 2>&1 ) &
done
wait
"$WORK/cmd_0" "rm -f /root/private_data/train_r*.log" >/dev/null 2>&1 || true
echo "  ✓ $NW 台节点清理完毕"

echo "=== [2/4] 打包并同步最新代码、配置、wheels 与 it831_ema 底模至集群共享存储 ==="
rm -rf /tmp/jaxbomb_64g_8x2 && mkdir -p /tmp/jaxbomb_64g_8x2/scripts /tmp/jaxbomb_64g_8x2/configs /tmp/jaxbomb_64g_8x2/ckpt /tmp/jaxbomb_64g_8x2/tests /tmp/jaxbomb_64g_8x2/wheels
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_64g_8x2/
cp -r "$ROOT/tests" /tmp/jaxbomb_64g_8x2/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_64g_8x2/cluster_selfcheck.py 2>/dev/null || true
cp "$ROOT/levels.json" /tmp/jaxbomb_64g_8x2/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_64g_8x2/configs/
cp "$ROOT/ckpt/params_it00000831_ema.pkl" /tmp/jaxbomb_64g_8x2/ckpt/ 2>/dev/null || cp "$ROOT/ckpt_local/params_it00000831_ema.pkl" /tmp/jaxbomb_64g_8x2/ckpt/
cp /tmp/dcu_wheels/*.whl /tmp/jaxbomb_64g_8x2/wheels/ 2>/dev/null || true

cat > /tmp/jaxbomb_64g_8x2/start_rank.sh <<'EOSTR'
#!/bin/bash
RANK=$1
WORLD_SIZE=$2
MASTER=$3
CONFIG=$4

export WORLD_SIZE=$WORLD_SIZE
export RANK=$RANK
export MASTER_ADDR=$MASTER
export MASTER_PORT=29500

source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1)
export HP_DOMAIN_RAND=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export PYTHONPATH="/root/private_data/qqt-gpu-sim_r$RANK:${PYTHONPATH:-}"

cd /root/private_data/qqt-gpu-sim_r$RANK
nohup python3 -u -m jax_bomb.train_real --config $CONFIG </dev/null > /root/private_data/train_r$RANK.log 2>&1 &
echo "STARTED_RANK_$RANK"
EOSTR
chmod +x /tmp/jaxbomb_64g_8x2/start_rank.sh

find /tmp/jaxbomb_64g_8x2 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_64g_8x2 && tar czf /tmp/jaxbomb_64g_8x2.tgz jax_bomb tests levels.json configs ckpt start_rank.sh wheels)
LOCAL_MD5=$(md5 -q /tmp/jaxbomb_64g_8x2.tgz 2>/dev/null || md5sum /tmp/jaxbomb_64g_8x2.tgz | awk '{print $1}')
echo "  ✓ 本地归档构建完成 MD5: $LOCAL_MD5"

echo "=== [3/4] 上传代码包至 Rank 0 并解压至各 Rank 工作区 ==="
"$WORK/scp_0" /tmp/jaxbomb_64g_8x2.tgz /root/private_data/jaxbomb_stage5.tgz
echo "  ✓ 代码包已推送至 Rank 0"

for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "rm -rf /root/private_data/qqt-gpu-sim_r$i && mkdir -p /root/private_data/qqt-gpu-sim_r$i && tar xzf /root/private_data/jaxbomb_stage5.tgz -C /root/private_data/qqt-gpu-sim_r$i" >/dev/null 2>&1 ) &
done
wait
echo "  ✓ 全集群各 Rank 工作区已就位"

MASTER_IP=$("$WORK/cmd_0" "hostname -I | awk '{print \$1}'" | tr -d '\r\n')
echo "=== [4/4] 启动分布式训练集群 (MASTER=$MASTER_IP, WORLD_SIZE=$NW) ==="
for i in $(seq 0 $((NW-1))); do
  "$WORK/cmd_$i" "bash /root/private_data/qqt-gpu-sim_r$i/start_rank.sh $i $NW $MASTER_IP $CONFIG_REL"
done

echo "=== 全部 $NW 台双卡节点已在后台拉起！==="
echo "可使用监控脚本或查看 Rank 0 日志："
echo "  bash deploy_10node/monitor_cluster.sh"
