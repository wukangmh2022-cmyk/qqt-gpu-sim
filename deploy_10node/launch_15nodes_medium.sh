#!/usr/bin/env bash
# ==============================================================================
# 15 节点 30 卡 DCU 3 小时中型长训启动脚本 (Warm-Start 继承 it68 战力)
# 由 configs/medium_3h_15nodes_patch3_k32_warmstart.toml 全权驱动
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/medium_3h_15nodes_patch3_k32_warmstart.toml"

NW=15  # 全部 15 台机器共 30 卡

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
    [ "${#N_PORT[@]}" -ge "$NW" ] && break
  fi
done < "$NODES_FILE"

echo "=== 已提取全部 $NW 台节点用于 30 卡 3 小时中型长训 ==="

WORK="/tmp/medium_15nodes_work"
rm -rf "$WORK" && mkdir -p "$WORK"

for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOcmd
#!/usr/bin/expect -f
set timeout 45
set cmd [lindex \$argv 0]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p ${N_PORT[$i]} root@${N_HOST[$i]} \$cmd
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "TIMEOUT_$i" }
}
EOcmd
  cat > "$WORK/scp_$i" <<EOscp
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
EOscp
  chmod +x "$WORK/cmd_$i" "$WORK/scp_$i"
done

echo "=== [1/4] 清理全部 15 台节点历史训练残留进程 ==="
for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; true" >/dev/null 2>&1 ) &
done
wait
echo "  ✓ 15 台节点清理完毕"

echo "=== [2/4] 打包并同步最新代码、配置与底模到全部 15 节点 ==="
rm -rf /tmp/jaxbomb_medium && mkdir -p /tmp/jaxbomb_medium/scripts /tmp/jaxbomb_medium/web/assets/maps /tmp/jaxbomb_medium/configs /tmp/jaxbomb_medium/ckpt
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_medium/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_medium/cluster_selfcheck.py
cp "$ROOT/levels.json" /tmp/jaxbomb_medium/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_medium/configs/
cp "$ROOT/ckpt/params_it00000068.pkl" /tmp/jaxbomb_medium/ckpt/
find /tmp/jaxbomb_medium -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_medium && tar czf /tmp/jaxbomb_medium.tgz jax_bomb levels.json configs ckpt cluster_selfcheck.py)

for i in $(seq 0 $((NW-1))); do
  (
    "$WORK/scp_$i" /tmp/jaxbomb_medium.tgz /root/private_data/ >/dev/null 2>&1
    "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim_r$i && mkdir -p qqt-gpu-sim_r$i && tar xzf jaxbomb_medium.tgz -C qqt-gpu-sim_r$i" >/dev/null 2>&1
    echo "  ✓ [rank $i] 代码与预训练底模部署完成"
  ) &
done
wait

echo "=== [3/4] 获取 Rank0 内网协调 IP ==="
N_IP=()
for i in $(seq 0 $((NW-1))); do
  ip=$("$WORK/cmd_$i" "hostname -I" | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
  N_IP+=("$ip")
  echo "  rank $i ${N_HOST[$i]}:${N_PORT[$i]} -> $ip"
done
MASTER="${N_IP[0]:-127.0.0.1}"
[ "$MASTER" = "127.0.0.1" ] && { echo "rank0 IP 获取失败"; exit 1; }

echo "=== [4/4] 启动 15 节点 30 卡中型长训 (MASTER: ${MASTER}:29500, NW=$NW) ==="
# Rank 0
"$WORK/cmd_0" "cd /root/private_data/qqt-gpu-sim_r0 && export WORLD_SIZE=$NW RANK=0 MASTER_ADDR=$MASTER MASTER_PORT=29500 && source /opt/dtk/env.sh 2>/dev/null && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1) && nohup python3 -u -m jax_bomb.train_real --config $CONFIG_REL </dev/null > /root/private_data/train_r0.log 2>&1 & echo RANK_0_LAUNCHED" | tail -1

# Rank 1..14
for i in $(seq 1 $((NW-1))); do
  (
    "$WORK/cmd_$i" "cd /root/private_data/qqt-gpu-sim_r$i && export WORLD_SIZE=$NW RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500 && source /opt/dtk/env.sh 2>/dev/null && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1) && nohup python3 -u -m jax_bomb.train_real --config $CONFIG_REL </dev/null > /root/private_data/train_r$i.log 2>&1 & echo RANK_${i}_LAUNCHED" | tail -1
  ) &
done
wait

echo "=== 全部 $NW 个 Rank (30 张 DCU 卡) 已全量点火，3 小时中型训练正式开始！ ==="
