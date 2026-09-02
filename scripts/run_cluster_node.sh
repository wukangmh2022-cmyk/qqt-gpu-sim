#!/bin/bash
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)
source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
cd /root/private_data/qqt-gpu-sim
mkdir -p ckpt

export WORLD_SIZE=24
export RANK=$1
export MASTER_ADDR=172.31.54.127
export MASTER_PORT=29500

exec python3 -m jax_bomb.multicard_train \
  --arch transformer --embed 392 --depth 4 --patch 3 --heads 4 --ff-factor 4 \
  --adv-top-frac 1.0 --ema-decay 0.999 \
  --num-envs 24576 --num-steps 256 --minibatch 24576 --epochs 1 \
  --lsgd-k 256 --lsgd-mode param --lsgd-bf16 \
  --iters 10000 \
  --curriculum-json curriculum.json \
  --curriculum-min-iters 20 \
  --curriculum-eval-every 10 \
  --curriculum-eval-steps 1800 \
  --ckpt-dir ckpt \
  --ckpt-every 30 \
  --ckpt-local-dir ckpt \
  --ckpt-local-every 15
