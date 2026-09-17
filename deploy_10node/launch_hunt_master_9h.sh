#!/usr/bin/env bash
# ==============================================================================
# DCU 进阶进攻 9 小时超长训 - 动静结合宗师追猎版 (Warm-Start 继承 it800_ema)
# 核心特征：
# 1. 初始血量域随机化 (HP_DOMAIN_RAND=1: 40% 1-HP生死局, 30% 2-3 HP, 30% 5 HP, 15% 非对称)
# 2. 高智商游走型 Bot (Idle Bot 30%, Roam Bot 30%, Smart Kite Bot 40% 防自灭)
# 3. 对称席位注入 (P0 / P1 交替陪练) + 优势度与回报屏蔽 (mask_bot_advantages 杜绝规则样本污染)
# 由 configs/long_9h_hunt_master.toml 全权驱动
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/long_9h_hunt_master.toml"

# 1. 读取节点
N_PORT=(); N_HOST=(); N_PASS=()
PENDING_PORT=""; PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d '\r' | sed 's/^[ \t]*//;s/[ \t]*$//')"
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
echo "=== 已提取全部 $NW 台在线双卡节点用于 $((NW * 2)) 卡动静结合宗师追猎 9 小时长训 ==="

WORK="/tmp/hunt_master_9h_work"
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

echo "=== [1/4] 清理全部 $NW 台节点历史训练残留进程 ==="
for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; true" >/dev/null 2>&1 ) &
done
wait
echo "  ✓ $NW 台节点清理完毕"

echo "=== [2/4] 打包并同步最新代码、配置、wheels 与 it800_ema 底模到全部 $NW 节点 ==="
rm -rf /tmp/jaxbomb_master_9h && mkdir -p /tmp/jaxbomb_master_9h/scripts /tmp/jaxbomb_master_9h/web/assets/maps /tmp/jaxbomb_master_9h/configs /tmp/jaxbomb_master_9h/ckpt /tmp/jaxbomb_master_9h/tests /tmp/jaxbomb_master_9h/wheels
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_master_9h/
cp -r "$ROOT/tests" /tmp/jaxbomb_master_9h/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_master_9h/cluster_selfcheck.py
cp "$ROOT/levels.json" /tmp/jaxbomb_master_9h/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_master_9h/configs/
cp "$ROOT/ckpt/params_it00000800_ema.pkl" /tmp/jaxbomb_master_9h/ckpt/
cp /tmp/dcu_wheels/*.whl /tmp/jaxbomb_master_9h/wheels/

cat > /tmp/jaxbomb_master_9h/start_rank.sh <<'EOSTR'
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

cd /root/private_data/qqt-gpu-sim_r$RANK
nohup python3 -u -m jax_bomb.train_real --config $CONFIG </dev/null > /root/private_data/train_r$RANK.log 2>&1 &
echo "STARTED_RANK_$RANK"
EOSTR
chmod +x /tmp/jaxbomb_master_9h/start_rank.sh

find /tmp/jaxbomb_master_9h -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_master_9h && tar czf /tmp/jaxbomb_master_9h.tgz jax_bomb tests levels.json configs ckpt cluster_selfcheck.py start_rank.sh wheels)

for i in $(seq 0 $((NW-1))); do
  (
    "$WORK/scp_$i" /tmp/jaxbomb_master_9h.tgz /root/private_data/ >/dev/null 2>&1
    "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim_r$i && mkdir -p qqt-gpu-sim_r$i && tar xzf jaxbomb_master_9h.tgz -C qqt-gpu-sim_r$i && pip install --quiet --no-index --find-links=/root/private_data/qqt-gpu-sim_r$i/wheels --no-deps /root/private_data/qqt-gpu-sim_r$i/wheels/*.whl >/dev/null 2>&1 || true" >/dev/null 2>&1
    echo "  ✓ [rank $i] 最新血量域随机化、高智商游走防自灭代码与 it800_ema 底模部署完成"
  ) &
done
wait

echo "=== [3/4] 集群自检与获取 Rank0 内网协调 IP ==="
for i in $(seq 0 $((NW-1))); do
  (
    sleep 0.$((i % 4))
    res=$("$WORK/cmd_$i" "source /opt/dtk/env.sh 2>/dev/null; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1); cd /root/private_data/qqt-gpu-sim_r$i && python3 cluster_selfcheck.py 2>/dev/null || true")
    echo "  [rank $i] $res"
  ) &
done
wait

MASTER=$("$WORK/cmd_0" "hostname -I" | tr -d '\r' | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1 || true)
if [ -z "$MASTER" ]; then
  MASTER="172.31.203.105"
fi
echo "  ✓ Rank 0 协调 IP: $MASTER"

echo "=== [4/4] 启动 $NW 节点 $((NW * 2)) 卡新训练 (MASTER: ${MASTER}:29500, NW=$NW) ==="
# Rank 0 先行启动 coordinator
"$WORK/cmd_0" "bash /root/private_data/qqt-gpu-sim_r0/start_rank.sh 0 $NW $MASTER $CONFIG_REL"

sleep 3

# Rank 1..$((NW-1))
for i in $(seq 1 $((NW-1))); do
  echo "  --> 启动 Rank $i ..."
  "$WORK/cmd_$i" "bash /root/private_data/qqt-gpu-sim_r$i/start_rank.sh $i $NW $MASTER $CONFIG_REL" &
  sleep 0.4
done
wait

echo "=== 全部 $NW 个 Rank ($((NW * 2)) 张 DCU 卡) 已全量点火，9 小时动静结合宗师追猎长训正式启动！ ==="
