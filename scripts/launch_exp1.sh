#!/bin/bash
set -e
MASTER_IP="172.31.25.109"
echo "=== 重启训练：严格对齐原版参数 (empty=0.05,功夫=0.1,比武=0.15, 全图均匀无课程) ==="

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard_train 2>/dev/null; rm -rf /root/private_data/train_r*.log /root/private_data/qqt-gpu-sim/ckpt/* /root/private_data/qqt-gpu-sim/ckpt_local/* 2>/dev/null && echo RANK\${i}_CLEANED" | tail -1
done

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=10 RANK=$i MASTER_ADDR=$MASTER_IP MASTER_PORT=29500; export LSGD_K=256 LSGD_MODE=param CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=30; nohup python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 --iters 15500 --lsgd-k 256 --lsgd-mode param --levels levels.json --level-weights \"empty=0.05,功夫=0.1,比武=0.15\" --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 --explore-reward-coef 0.01 --explore-reward-anneal-steps 30000000000 --brick-reward-coef 0.05 --reward-anneal-k 1.2 --fresh > /root/private_data/train_r$i.log 2>&1 & echo RANK\${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 4
done
echo "=== 10 个节点全部拉起 ==="
