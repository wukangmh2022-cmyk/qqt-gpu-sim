#!/bin/bash
# 15 机 × 2 卡（共 30 张 DCU）有损同步（Local SGD）训练编排
# 目标：使用当前最新 14 通道观测态训练 570M 步模型（34 iter × 16.77M steps/iter ≈ 570.28M steps）
#
# 用法:  bash launch_15nodes.sh [nodes_current.txt] [--deploy] [--resume]
#   --deploy 先并发推送代码+wheels 并跑 setup（新开节点必用）
#   --resume 接续各节点断点（默认 --fresh 从头训练）
set -u

NODES_FILE="${1:-nodes_current.txt}"
if [ "${NODES_FILE:0:2}" = "--" ]; then
  set -- "nodes_current.txt" "$@"
  NODES_FILE="$1"
fi
shift || true

DO_DEPLOY=0; FRESH="--fresh"
for a in "$@"; do
  [ "$a" = "--deploy" ] && DO_DEPLOY=1
  { [ "$a" = "--resume" ] || [ "$a" = "--no-fresh" ]; } && FRESH=""
done

if [ ! -f "$NODES_FILE" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  if [ -f "$SCRIPT_DIR/$NODES_FILE" ]; then
    NODES_FILE="$SCRIPT_DIR/$NODES_FILE"
  else
    echo "找不到节点文件: $NODES_FILE"
    exit 1
  fi
fi

# 570M 步训练参数配置：
# 30 卡 (15 机 × 2 卡): NUM_ENVS=32760 (1092 envs/卡), NUM_STEPS=256
# 每轮 global steps = 2 × 32760 × 256 = 16,773,120
# 34 轮 = 570,286,080 global steps (~570M)
ITERS="${ITERS:-34}"
NUM_ENVS="${NUM_ENVS:-32760}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32760}"
PATCH="${PATCH:-4}"
ADV_TOP_FRAC="${ADV_TOP_FRAC:-0.25}"
EMA_DECAY="${EMA_DECAY:-0.999}"
LSGD_K="${LSGD_K:-256}"; LSGD_MODE="${LSGD_MODE:-param}"
CKPT_EVERY="${CKPT_EVERY:-10}"
CKPT_LOCAL_EVERY="${CKPT_LOCAL_EVERY:-5}"

LEVELS_FILE="${LEVELS_FILE:-}"
LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-}"
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.0}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-0}"
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.0}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-0}"
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.0}"
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
CURRICULUM_JSON="${CURRICULUM_JSON:-web/assets/maps/curriculum.json}"

N_PORT=(); N_HOST=(); N_PASS=()
PENDING_PORT=""; PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'
'}"
  line="${line#"${line%%[![:space:]]*}"}"
  [ -z "$line" ] && continue
  [ "${line:0:1}" = "#" ] && continue
  if [ -n "$PENDING_PORT" ]; then
    N_PORT+=("$PENDING_PORT"); N_HOST+=("$PENDING_HOST"); N_PASS+=("$line")
    PENDING_PORT=""; PENDING_HOST=""
    continue
  fi
  if [[ "$line" =~ ^ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+root@([^[:space:]]+)$ ]]; then
    PENDING_PORT="${BASH_REMATCH[1]}"; PENDING_HOST="${BASH_REMATCH[2]}"
    continue
  fi
  read -r p h pw extra <<< "$line"
  [ -n "${p:-}" ] && [ -n "${h:-}" ] && [ -n "${pw:-}" ] || { echo "节点行格式不对: $line"; exit 1; }
  N_PORT+=("$p"); N_HOST+=("$h"); N_PASS+=("$pw")
done < "$NODES_FILE"
[ -z "$PENDING_PORT" ] || { echo "ssh 行后缺密码行: ${PENDING_HOST}:${PENDING_PORT}"; exit 1; }

NW=${#N_PORT[@]}
echo "=== 共 $NW 台节点（rank0 = ${N_HOST[0]}:${N_PORT[0]}）==="
[ "$NW" -lt 2 ] && { echo "至少 2 台节点"; exit 1; }

WORK=/tmp/ndrun; rm -rf "$WORK"; mkdir -p "$WORK"
for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOF
#!/usr/bin/expect -f
set timeout 300
set cmd [lindex \$argv 0]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p ${N_PORT[$i]} root@${N_HOST[$i]} \$cmd
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "TIMEOUT_$i" }
}
EOF
  cat > "$WORK/scp_$i" <<EOF
#!/usr/bin/expect -f
set timeout 300
set src [lindex \$argv 0]
set dst [lindex \$argv 1]
spawn scp -o StrictHostKeyChecking=accept-new -P ${N_PORT[$i]} \$src root@${N_HOST[$i]}:\$dst
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "SCP_TIMEOUT_$i" }
}
EOF
  chmod +x "$WORK/cmd_$i" "$WORK/scp_$i"
done

if [ "$DO_DEPLOY" = "1" ]; then
  echo "=== [1/5] 并发部署基础包+wheels+setup ($NW 台节点) ==="
  PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_24node.tar.gz"
  [ -f "$PKG" ] || PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || PKG="/tmp/dcu_deploy/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || { echo "找不到基础部署包"; exit 1; }
  PKG_NAME="$(basename "$PKG")"

  for i in $(seq 0 $((NW-1))); do
    (
      echo "  [rank $i] scp $PKG_NAME..."
      "$WORK/scp_$i" "$PKG" /root/private_data/ >/dev/null 2>&1
      echo "  [rank $i] 解压并运行 setup_notebook.sh..."
      "$WORK/cmd_$i" "cd /root/private_data && tar xzf $PKG_NAME; if [ -d dcu_deploy ]; then cd dcu_deploy; fi; bash setup_notebook.sh" > /tmp/deploy_$i.log 2>&1
      echo "  ✓ [rank $i] 基础环境就绪"
    ) &
  done
  wait
  echo "=== 基础环境部署完成 ==="
fi

echo "=== [2/5] 打包并推最新代码到全部节点 ==="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf /tmp/jaxbomb_cluster && mkdir -p /tmp/jaxbomb_cluster/scripts /tmp/jaxbomb_cluster/web/assets/maps
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_cluster/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_cluster/cluster_selfcheck.py
cp "$ROOT/scripts/start_rank.sh" /tmp/jaxbomb_cluster/start_rank.sh
chmod +x /tmp/jaxbomb_cluster/start_rank.sh
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_cluster/levels.json
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_cluster/web/assets/maps/levels.json
cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_cluster/web/assets/maps/curriculum.json 2>/dev/null || true
cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_cluster/curriculum.json 2>/dev/null || true
for s in quick_check_levels.py quick_check_bush.py quick_check_crate_semantics.py \
         quick_check_obs_move.py quick_check_js_jax_move.py quick_check_anti_tunnel.py; do
  cp "$ROOT/scripts/$s" /tmp/jaxbomb_cluster/scripts/ 2>/dev/null || true
done
find /tmp/jaxbomb_cluster -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_cluster && tar czf /tmp/jaxbomb_cluster.tgz jax_bomb levels.json curriculum.json web scripts cluster_selfcheck.py start_rank.sh)

for i in $(seq 0 $((NW-1))); do
  (
    "$WORK/scp_$i" /tmp/jaxbomb_cluster.tgz /root/private_data/ >/dev/null 2>&1
    if [ -n "$FRESH" ]; then
      CLEAN_DIR="rm -rf qqt-gpu-sim_r$i && mkdir -p qqt-gpu-sim_r$i"
    else
      CLEAN_DIR="mkdir -p qqt-gpu-sim_r$i"
    fi
    "$WORK/cmd_$i" "cd /root/private_data && $CLEAN_DIR && tar xzf jaxbomb_cluster.tgz -C qqt-gpu-sim_r$i && ls qqt-gpu-sim_r$i/jax_bomb/jax_env.py >/dev/null" > /tmp/code_$i.log 2>&1
    echo "  ✓ [rank $i] 代码同步成功"
  ) &
done
wait
echo "=== 代码同步完成 ==="

echo "=== [3/5] 并发环境自检 ==="
ALL_OK=1
for i in $(seq 0 $((NW-1))); do
  (
    out=$("$WORK/cmd_$i" "source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); cd /root/private_data/qqt-gpu-sim_r$i 2>/dev/null || exit 9; python3 cluster_selfcheck.py" 2>/dev/null | grep SELFCHECK_OK | tail -1)
    echo "$out" > /tmp/selfcheck_$i.out
  ) &
done
wait

for i in $(seq 0 $((NW-1))); do
  out=$(cat /tmp/selfcheck_$i.out 2>/dev/null)
  if [ -z "$out" ]; then
    echo "  ✗ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: 自检失败（请检查 /tmp/deploy_$i.log）"
    ALL_OK=0
  else
    ndev=$(echo "$out" | grep -oE 'devices= ?[0-9]+' | grep -oE '[0-9]+')
    if [ "$ndev" = "2" ]; then
      echo "  ✓ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: $out"
    else
      echo "  ✗ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: 卡数 $ndev ≠ 2"
      ALL_OK=0
    fi
  fi
done
[ "$ALL_OK" = "1" ] || { echo "=== 有节点自检失败，中止启动 ==="; exit 1; }

echo "=== [4/5] 获取各节点内网 IP ==="
N_IP=()
for i in $(seq 0 $((NW-1))); do
  ip=$("$WORK/cmd_$i" "hostname -I" | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
  N_IP+=("$ip")
  echo "  rank $i ${N_HOST[$i]}:${N_PORT[$i]} -> $ip"
done
MASTER="${N_IP[0]:-127.0.0.1}"
[ "$MASTER" = "127.0.0.1" ] && { echo "rank0 IP 获取失败"; exit 1; }

echo "=== [5/5] 启动分布式训练 ==="
echo "  MASTER_ADDR   : ${MASTER}:29500"
echo "  WORLD_SIZE    : ${NW} 节点 × 2 卡 = $((NW * 2)) 张 DCU"
echo "  TOTAL ITERS   : ${ITERS} (每轮 16.77M steps, 累计 570.28M 步)"
echo "  NUM_ENVS      : ${NUM_ENVS} (每卡 $((NUM_ENVS / (NW * 2))) envs)"
echo "  OBS CHANNELS  : 14 channels (含 pushable 箱体与全部属性)"
echo "  ARCH / PATCH  : Transformer depth=4 embed=392 patch=${PATCH}"
echo "  LSGD          : K=${LSGD_K} MODE=${LSGD_MODE}"
echo "  CHECKPOINTS   : 每 ${CKPT_EVERY} 轮存盘，本地每 ${CKPT_LOCAL_EVERY} 轮"

echo "  清理各节点历史遗留训练进程..."
for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; true" >/dev/null 2>&1 ) &
done
wait
sleep 2

echo "  拉起 Coordinator (rank 0)..."
"$WORK/cmd_0" "cd /root/private_data/qqt-gpu-sim_r0 && export WORLD_SIZE=$NW RANK=0 MASTER_ADDR=$MASTER MASTER_PORT=29500 CKPT_DIR=ckpt CKPT_EVERY=$CKPT_EVERY CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=$CKPT_LOCAL_EVERY ITERS=$ITERS NUM_ENVS=$NUM_ENVS NUM_STEPS=$NUM_STEPS MINIBATCH=$MINIBATCH PATCH=$PATCH ADV_TOP_FRAC=$ADV_TOP_FRAC EMA_DECAY=$EMA_DECAY LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE FRESH_FLAG=$FRESH && nohup bash start_rank.sh </dev/null > /root/private_data/train_r0.log 2>&1 & echo RANK_0_STARTED" | tail -1
sleep 3

echo "  并发拉起 Worker (rank 1~$((NW-1)))..."
for i in $(seq 1 $((NW-1))); do
  (
    "$WORK/cmd_$i" "cd /root/private_data/qqt-gpu-sim_r$i && export WORLD_SIZE=$NW RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500 CKPT_DIR=ckpt CKPT_EVERY=$CKPT_EVERY CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=$CKPT_LOCAL_EVERY ITERS=$ITERS NUM_ENVS=$NUM_ENVS NUM_STEPS=$NUM_STEPS MINIBATCH=$MINIBATCH PATCH=$PATCH ADV_TOP_FRAC=$ADV_TOP_FRAC EMA_DECAY=$EMA_DECAY LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE FRESH_FLAG=$FRESH && nohup bash start_rank.sh </dev/null > /root/private_data/train_r$i.log 2>&1 & echo RANK_${i}_STARTED" | tail -1
  ) &
done
wait

echo "=== 等待 30s 做健康检查 ==="
sleep 30
ALL_OK=1
for i in $(seq 0 $((NW-1))); do
  last=$("$WORK/cmd_$i" "tail -4 /root/private_data/train_r$i.log 2>/dev/null || echo NOLOG" | tail -4 | grep -v "spawn ssh\|password:" | tail -2 | tr '\n' ' ' | head -c 160)
  if [ -z "$last" ] || echo "$last" | grep -q NOLOG; then
    echo "  ✗ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: 日志为空"
    ALL_OK=0
  elif echo "$last" | grep -qiE "Traceback|Error|Exception|exited with"; then
    echo "  ✗ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: 报错 -> $last"
    ALL_OK=0
  else
    echo "  ✓ rank $i ${N_HOST[$i]}:${N_PORT[$i]}: $last"
  fi
done

if [ "$ALL_OK" = "1" ]; then
  echo ""
  echo "=========================================================="
  echo "🎉 15 机 × 2 卡集群已成功拉起！"
  echo "监控命令: bash watch_15nodes.sh $NODES_FILE"
  echo "日志位置: 各机器 /root/private_data/train_r<i>.log"
  echo "拉取断点: bash pull_ckpt_local.sh $NODES_FILE"
  echo "=========================================================="
else
  echo "=== 部分节点启动有报错，请看上方详情 ==="
  exit 1
fi
