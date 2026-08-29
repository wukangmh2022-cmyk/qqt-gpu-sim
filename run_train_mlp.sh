#!/bin/bash
# 910B MLP 训练（无 resume，从零，测真实训练 SPS）
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
exec python3 -m train.train \
  --num-envs 2048 \
  --arch mlp \
  --device npu:0 \
  --total-steps 3000000 \
  > train_mlp.log 2>&1
