#!/bin/bash
# LSTM 双维度课程训练（2026-08-12，v3）：敌人与地图循序渐进 + 天梯自我对弈。
#   课程表 train/curriculum.py::lstm_curriculum —— 每阶段排好对手（bot）与地图
#   （open 随机立柱渐进 / corridor 顶墙 2→4 + 通道 7→5 + 边缘连续段渐进），
#   胜率达标或跑满对局数自动晋级；s5-ladder 进入天梯（池子 ELO 就近采样）。
#   ckpt 存在则自动 resume 接力（保留课程进度），否则从零开始。
# -u 关掉 stdout 块缓冲；环岛不进训练（eval_lstm_ring.py 做泛化测试）。
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
RESUME=""
[ -f ckpt/lstm_course.pt ] && RESUME="--resume ckpt/lstm_course.pt"
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
  $RESUME \
  --ckpt ckpt/lstm_course.pt \
  --log-csv ckpt/train_lstm_course.csv \
  > train_lstm_course.log 2>&1
