#!/bin/bash
# 910B 真实训练启动（NPU 设备，resume duel_cnn 300M ckpt）
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
exec python3 -m train.train \
  --resume ckpt/duel_cnn.pt \
  --num-envs 2048 \
  --device npu:0 \
  --total-steps 302000000 \
  --single-stage \
  > train_run2.log 2>&1
