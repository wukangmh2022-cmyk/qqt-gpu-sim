#!/usr/bin/env bash
# ==============================================================================
# 标杆模型 10-15 分钟严格验证脚本 (8 节点 16 卡 DCU)
# 严格由 configs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml 驱动
# 彻底移除冷启动 EMA，使用 Patch=3, K=32, 14 通道
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODES_FILE="$SCRIPT_DIR/nodes_current.txt"
CONFIG_REL="configs/repro_it68_scheme1_actor_top25_critic_all_patch3_k32.toml"

NW=8  # 标杆 TOML world_size=8 (前 8 台节点共 16 卡)

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

echo "=== 已提取前 $NW 台节点用于标杆模型重跑验证 ==="

WORK="/tmp/it68_trial_work"
rm -rf "$WORK" && mkdir -p "$WORK"

for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOcmd
#!/usr/bin/expect -f
set timeout 30
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

echo "=== [1/4] 清理前 8 台节点历史训练进程 ==="
for i in $(seq 0 $((NW-1))); do
  ( "$WORK/cmd_$i" "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; true" >/dev/null 2>&1 ) &
done
wait

echo "=== [2/4] 打包并分发最新代码与 TOML 到前 $NW 节点 ==="
rm -rf /tmp/jaxbomb_it68 && mkdir -p /tmp/jaxbomb_it68/scripts /tmp/jaxbomb_it68/web/assets/maps /tmp/jaxbomb_it68/configs
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_it68/
cp "$ROOT/scripts/cluster_selfcheck.py" /tmp/jaxbomb_it68/cluster_selfcheck.py
cp "$ROOT/levels.json" /tmp/jaxbomb_it68/levels.json
cp -r "$ROOT/configs"/* /tmp/jaxbomb_it68/configs/
find /tmp/jaxbomb_it68 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_it68 && tar czf /tmp/jaxbomb_it68.tgz jax_bomb levels.json configs cluster_selfcheck.py)

for i in $(seq 0 $((NW-1))); do
  (
    "$WORK/scp_$i" /tmp/jaxbomb_it68.tgz /root/private_data/ >/dev/null 2>&1
    "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim_r$i && mkdir -p qqt-gpu-sim_r$i && tar xzf jaxbomb_it68.tgz -C qqt-gpu-sim_r$i" >/dev/null 2>&1
    echo "  ✓ [rank $i] 代码与配置同步完成"
  ) &
done
wait

echo "=== [3/4] 获取 Rank0 内网 IP ==="
N_IP=()
for i in $(seq 0 $((NW-1))); do
  ip=$("$WORK/cmd_$i" "hostname -I" | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
  N_IP+=("$ip")
  echo "  rank $i ${N_HOST[$i]}:${N_PORT[$i]} -> $ip"
done
MASTER="${N_IP[0]:-127.0.0.1}"
[ "$MASTER" = "127.0.0.1" ] && { echo "rank0 IP 获取失败"; exit 1; }

echo "=== [4/4] 启动标杆 TOML 训练 (MASTER: ${MASTER}:29500, NW=$NW) ==="
# Rank 0
"$WORK/cmd_0" "cd /root/private_data/qqt-gpu-sim_r0 && export WORLD_SIZE=$NW RANK=0 MASTER_ADDR=$MASTER MASTER_PORT=29500 && source /opt/dtk/env.sh 2>/dev/null && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1) && nohup python3 -u -m jax_bomb.train_real --config $CONFIG_REL </dev/null > /root/private_data/train_r0.log 2>&1 & echo RANK_0_LAUNCHED" | tail -1

# Rank 1..7
for i in $(seq 1 $((NW-1))); do
  (
    "$WORK/cmd_$i" "cd /root/private_data/qqt-gpu-sim_r$i && export WORLD_SIZE=$NW RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500 && source /opt/dtk/env.sh 2>/dev/null && unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy && export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1) && nohup python3 -u -m jax_bomb.train_real --config $CONFIG_REL </dev/null > /root/private_data/train_r$i.log 2>&1 & echo RANK_${i}_LAUNCHED" | tail -1
  ) &
done
wait

echo "=== 全部 $NW 个 Rank 已拉起，开始执行 10-15 分钟验证跑 ==="
