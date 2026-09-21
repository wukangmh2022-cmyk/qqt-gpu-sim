#!/usr/bin/env bash
# ==============================================================================
# 第五阶段：超算互联网 6 节点 x 4 卡 (24 卡 DCU) 8 小时 5Hz 宗师全图长训
# Warm-Start 继承 params_it00000831_ema.pkl
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="${1:-$SCRIPT_DIR/nodes_current.txt}"
CONFIG_REL="configs/stage5_8h_6x4gpu_5hz.toml"

if [ ! -f "$NODES_FILE" ]; then
  echo "错误: 节点文件不存在: $NODES_FILE"
  exit 1
fi

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
echo "=== 已提取全部 $NW 台 4 卡节点用于 $((NW * 4)) 卡 Stage 5 (8h 5Hz 全图长训) ==="

if [ "$NW" -ne 6 ]; then
  echo "注意: 当前节点数 $NW != 6，请确认是否为 6 节点配置（当前将以 WORLD_SIZE=$NW 继续启动）。"
fi

WORK="/tmp/scnet_6x4_work"
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

echo "=== [2/4] 打包最新代码、配置、241 地图与 it831_ema 底模 ==="
rm -rf /tmp/jaxbomb_6x4 && mkdir -p /tmp/jaxbomb_6x4/scripts /tmp/jaxbomb_6x4/configs /tmp/jaxbomb_6x4/ckpt /tmp/jaxbomb_6x4/tests /tmp/jaxbomb_6x4/wheels
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_6x4/
cp -r "$ROOT/tests" /tmp/jaxbomb_6x4/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_6x4/cluster_selfcheck.py 2>/dev/null || true
cp "$ROOT/levels.json" /tmp/jaxbomb_6x4/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_6x4/configs/
cp "$ROOT/ckpt/params_it00000831_ema.pkl" /tmp/jaxbomb_6x4/ckpt/ 2>/dev/null || cp "$ROOT/ckpt_local/params_it00000831_ema.pkl" /tmp/jaxbomb_6x4/ckpt/
cp /tmp/dcu_wheels/*.whl /tmp/jaxbomb_6x4/wheels/ 2>/dev/null || true

cat > /tmp/jaxbomb_6x4/start_rank.sh <<'EOSTR'
#!/bin/bash
RANK=$1
WORLD_SIZE=$2
MASTER=$3
CONFIG=$4

export WORLD_SIZE=$WORLD_SIZE
export RANK=$RANK
export MASTER_ADDR=$MASTER
export MASTER_PORT=29500

source /opt/dtk/env.sh 2>/dev/null || true
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
chmod +x /tmp/jaxbomb_6x4/start_rank.sh

find /tmp/jaxbomb_6x4 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_6x4 && tar czf /tmp/jaxbomb_6x4.tgz jax_bomb tests levels.json configs ckpt start_rank.sh wheels)
LOCAL_MD5=$(md5 -q /tmp/jaxbomb_6x4.tgz 2>/dev/null || md5sum /tmp/jaxbomb_6x4.tgz | awk '{print $1}')
echo "  ✓ 本地归档构建完成 MD5: $LOCAL_MD5"

echo "=== [3/4] 上传代码包至各节点工作区 ==="
for i in $(seq 0 $((NW-1))); do
  echo "  --> 正在推送代码包到节点 $i (${N_HOST[$i]}:${N_PORT[$i]})..."
  "$WORK/scp_$i" /tmp/jaxbomb_6x4.tgz /root/private_data/jaxbomb_6x4.tgz
  "$WORK/cmd_$i" "rm -rf /root/private_data/qqt-gpu-sim_r$i && mkdir -p /root/private_data/qqt-gpu-sim_r$i && tar xzf /root/private_data/jaxbomb_6x4.tgz -C /root/private_data/qqt-gpu-sim_r$i"
done
echo "  ✓ 全集群各 Rank 工作区已就位"

MASTER_IP=$("$WORK/cmd_0" "hostname -I | awk '{print \$1}'" | tr -d '\r\n')
echo "=== [4/4] 启动分布式训练集群 (MASTER=$MASTER_IP, WORLD_SIZE=$NW) ==="
for i in $(seq 0 $((NW-1))); do
  "$WORK/cmd_$i" "bash /root/private_data/qqt-gpu-sim_r$i/start_rank.sh $i $NW $MASTER_IP $CONFIG_REL"
done

echo "=== 全部 $NW 台 4 卡节点已在后台拉起！==="
echo "请查看 Rank 0 日志验证启动状态："
echo "  bash deploy_10node/watch_6x4nodes.sh"
