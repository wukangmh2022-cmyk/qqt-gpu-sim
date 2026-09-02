#!/bin/bash
# ==============================================================================
# 实验 C 启动脚本模板：Param Mode K=16 + Adam 状态同步 (--lsgd-sync-state)
# ==============================================================================
cd /root/private_data/qqt-gpu-sim || exit 9
source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)
export WORLD_SIZE=23 RANK=$1 MASTER_ADDR=__MASTER__ MASTER_PORT=29500
export LSGD_K=16 LSGD_MODE=param CKPT_DIR=ckpt CKPT_EVERY=15
export CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5

rm -rf /root/private_data/qqt-gpu-sim/ckpt_local/*
rm -rf /root/private_data/qqt-gpu-sim/ckpt/*

nohup python3 -u -m jax_bomb.multicard_train \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01 \
  --crate-reward-coef 0.15 --crate-reward-anneal-steps 30000000000 \
  --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 \
  --iters 500 --lsgd-k 16 --lsgd-mode param --lsgd-sync-state \
  --fresh > /root/private_data/train_r$1.log 2>&1 &
echo "RANK${1}_STARTED_PARAM_K16_SYNC"
