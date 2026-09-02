#!/bin/bash
set -e
MASTER_IP="172.31.204.214"
MASTER_PORT=29540
echo "=== 快速拉起 10 节点训练 (MASTER=$MASTER_IP:$MASTER_PORT) ==="

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "pkill -9 -f python3 2>/dev/null; fuser -k 29540/tcp 2>/dev/null; rm -rf /root/private_data/train_r*.log /root/private_data/qqt-gpu-sim/ckpt/* /root/private_data/qqt-gpu-sim/ckpt_local/* 2>/dev/null" >/dev/null 2>&1 || true
  sleep 0.3
done

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim; export LC_ALL=C.UTF-8 LANG=C.UTF-8; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=10 RANK=$i MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT; export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5; nohup python3 -u -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 --iters 15500 --lsgd-k 32 --lsgd-mode grad --levels levels.json --level-weights \"empty=0.05,功夫=0.1,比武=0.15\" --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 --explore-reward-coef 0.01 --explore-reward-anneal-steps 30000000000 --brick-reward-coef 0.05 --reward-anneal-k 1.2 --fresh > /root/private_data/train_r$i.log 2>&1 & echo RANK\${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 4
  sleep 0.5
done
echo "=== 10 节点已全部直接拉起 ==="
