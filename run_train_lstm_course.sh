#!/bin/bash
# LSTM 双维度课程训练（2026-08-12）：敌人与地图循序渐进 + 天梯自我对弈。
#   课程表见 train/curriculum.py::lstm_curriculum —— 每阶段排好对手（bot）
#   与地图（open/corridor/ring + wall_density 随机立柱克制递增），胜率达标
#   或跑满对局数自动晋级；s5-ladder 进入天梯（池子 ELO 就近采样为主）。
# -u 关掉 stdout 块缓冲（之前日志卡在首行误导诊断）。
# resume 用 --resume ckpt/lstm_course.pt 接力（cstate 恢复课程阶段）。
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
exec python3 -u -m train.train \
  --arch lstm \
  --num-envs 4096 \
  --device npu:0 \
  --rollout-steps 128 \
  --minibatches 1 \
  --bptt-window 8 \
  --max-mem-frac 0.85 \
  --total-steps 2000000000 \
  --snapshot-every 20 \
  --time-budget 86400 \
  --ckpt ckpt/lstm_course.pt \
  --log-csv ckpt/train_lstm_course.csv \
  > train_lstm_course.log 2>&1
