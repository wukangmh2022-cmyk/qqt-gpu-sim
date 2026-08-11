#!/bin/bash
# 910B LSTM 架构（BombermanNet：局部7×7 CNN + 相对坐标 + 全局状态 + LSTM）
# 从零训练到 1B 步。s1 阶段：1v1 + 纯规则 bot 对手（greedy + astar 混入）——
# LSTM 的神经网络对手路径（build_opponents 池快照 / _opponent_actions 收共享
# obs）尚未支持，用 bot 启蒙；课程化之后再加网络对手。
# bptt_window=8：truncated BPTT（910B 实测反向 -78%）；N=4096 是 LSTM 显存上限。
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
exec python3 -m train.train \
  --arch lstm \
  --num-envs 4096 \
  --device npu:0 \
  --single-stage \
  --map-mode open \
  --rollout-steps 128 \
  --minibatches 1 \
  --bptt-window 8 \
  --max-mem-frac 0.85 \
  --bot-opponents greedy \
  --bot-opp-prob 1.0 \
  --warmup-steps 1000000000 \
  --total-steps 1000000000 \
  --oversample-dying 1 \
  --snapshot-every 20 \
  --time-budget 86400 \
  --ckpt ckpt/lstm_1b.pt \
  --log-csv ckpt/train_lstm_1b.csv \
  > train_lstm_1b.log 2>&1
