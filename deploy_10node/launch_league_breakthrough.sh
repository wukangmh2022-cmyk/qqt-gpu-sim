#!/usr/bin/env bash
# ==============================================================================
# DCU 联赛攻防全开破局长训 - 动静结合宗师版 (Warm-Start 继承 aggr419_ema)
# 核心特征：
# 1. 底模继承：params_aggr_it00000419_ema.pkl (0% 超时、开阔地极速肉搏战意之王)
# 2. 彻底拆除消极避战枷锁：mutual_hit_penalty=0.0, double_death_penalty=0.0, win_bonus=10.0
# 3. 多元联赛对手池：15% 纯静止木桩 [4, 0] + 15% 时空 A* 竞技猎人 + 10% 修复版拉扯 FleeBot + 60% 自对弈
# 4. 开阔地权重提升至 20% (empty=0.20)，血量域随机化全覆盖 (HP_DOMAIN_RAND=1)
# 由 configs/long_12h_league_breakthrough.toml 全权驱动
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/long_12h_league_breakthrough.toml"

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
echo "=== 已提取全部 $NW 台在线双卡节点用于 $((NW * 2)) 卡联赛攻防全开破局长训 ==="

WORK="/tmp/league_breakthrough_work"
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

echo "=== [2/4] 打包并同步最新代码、配置、wheels 与 aggr419_ema 底模至集群共享存储 ==="
rm -rf /tmp/jaxbomb_league && mkdir -p /tmp/jaxbomb_league/scripts /tmp/jaxbomb_league/web/assets/maps /tmp/jaxbomb_league/configs /tmp/jaxbomb_league/ckpt /tmp/jaxbomb_league/tests /tmp/jaxbomb_league/wheels
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_league/
cp -r "$ROOT/tests" /tmp/jaxbomb_league/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_league/cluster_selfcheck.py
cp "$ROOT/levels.json" /tmp/jaxbomb_league/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_league/configs/
cp "$ROOT/ckpt/params_aggr_it00000419_ema.pkl" /tmp/jaxbomb_league/ckpt/
cp /tmp/dcu_wheels/*.whl /tmp/jaxbomb_league/wheels/

cat > /tmp/jaxbomb_league/start_rank.sh <<'EOSTR'
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

export PYTHONPATH="/root/private_data/qqt-gpu-sim_r$RANK:${PYTHONPATH:-}"

cd /root/private_data/qqt-gpu-sim_r$RANK
nohup python3 -u -m jax_bomb.train_real --config $CONFIG </dev/null > /root/private_data/train_r$RANK.log 2>&1 &
echo "STARTED_RANK_$RANK"
EOSTR
chmod +x /tmp/jaxbomb_league/start_rank.sh

find /tmp/jaxbomb_league -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_league && tar czf /tmp/jaxbomb_league.tgz jax_bomb tests levels.json configs ckpt cluster_selfcheck.py start_rank.sh wheels)
LOCAL_MD5=$(md5 -q /tmp/jaxbomb_league.tgz 2>/dev/null || md5sum /tmp/jaxbomb_league.tgz | awk '{print $1}')
echo "  ✓ 本地构建部署包完成 (MD5: $LOCAL_MD5, 体积 $(du -sh /tmp/jaxbomb_league.tgz | awk '{print $1}'))"

echo "  --> 向全部 $NW 台节点并发传输部署包并就地解压..."
DEPLOY_FAIL=0
for i in $(seq 0 $((NW-1))); do
  (
    "$WORK/scp_$i" /tmp/jaxbomb_league.tgz /root/private_data/jaxbomb_master.tgz >/dev/null 2>&1
    "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim_r$i && mkdir -p qqt-gpu-sim_r$i && tar xzf jaxbomb_master.tgz -C qqt-gpu-sim_r$i && ls /root/private_data/qqt-gpu-sim_r$i/jax_bomb/jax_env.py >/dev/null && echo RANK_${i}_READY"
  ) || DEPLOY_FAIL=1 &
  if (( (i + 1) % 4 == 0 )); then
    wait
  fi
done
wait

if [ "$DEPLOY_FAIL" -ne 0 ]; then
  echo "❌ 节点分发解压失败！" >&2
  exit 1
fi
echo "  ✓ 全部 $NW 台节点最新代码、底模与配置就绪"

echo "=== [3/4] 集群自检与获取 Rank0 内网协调 IP ==="
CHECK_FAIL=0
for i in $(seq 0 $((NW-1))); do
  (
    res=$("$WORK/cmd_$i" "source /opt/dtk/env.sh 2>/dev/null; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1); cd /root/private_data/qqt-gpu-sim_r$i && python3 cluster_selfcheck.py 2>/dev/null || true" | tr -d "\r\n")
    echo "  [rank $i] $res"
    if ! echo "$res" | grep -q "SELFCHECK_OK"; then
      echo "  ❌ [rank $i] 自检失败！" >&2
      exit 1
    fi
  ) || CHECK_FAIL=1 &
done
wait

if [ "$CHECK_FAIL" -ne 0 ]; then
  echo "❌ 集群中有节点自检未通过，已紧急阻断启动！请检查日志！" >&2
  exit 1
fi

MASTER=$("$WORK/cmd_0" "hostname -I" | tr -d "\r" | grep -oE "172\.31\.[0-9]+\.[0-9]+" | head -1 || true)
if [ -z "$MASTER" ]; then
  echo "❌ 无法获取 Rank 0 内网协调 IP！已阻断！" >&2
  exit 1
fi
echo "  ✓ Rank 0 协调 IP: $MASTER"

echo "=== [4/4] 启动 $NW 节点 $((NW * 2)) 卡新训练 (MASTER: ${MASTER}:29500, NW=$NW) ==="
echo "  --> 启动 Rank 0 (Coordinator) ..."
"$WORK/cmd_0" "bash /root/private_data/qqt-gpu-sim_r0/start_rank.sh 0 $NW $MASTER $CONFIG_REL"

sleep 5

for i in $(seq 1 $((NW-1))); do
  echo "  --> 启动 Rank $i ..."
  "$WORK/cmd_$i" "bash /root/private_data/qqt-gpu-sim_r$i/start_rank.sh $i $NW $MASTER $CONFIG_REL" &
done
wait

echo "=== 全部 $NW 个 Rank ($((NW * 2)) 张 DCU 卡) 已点火，等待 20s 验证初态... ==="
sleep 20
for i in $(seq 0 $((NW-1))); do
  res=$("$WORK/cmd_$i" "tail -2 /root/private_data/train_r$i.log 2>/dev/null || echo NO_LOG" | tr '\r\n' ' ' | head -c 120)
  echo "  [rank $i] $res"
done

