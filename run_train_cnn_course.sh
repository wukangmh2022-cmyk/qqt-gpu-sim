#!/bin/bash
# CNN 泛化专项课程训练（2026-08-12）：resume duel_cnn 300M，敌人/地图渐进，
# 把对打寻路 AI（astar/hunter）胜率从本地摸底值拉到 90%+。
#   课程表 train/curriculum.py::cnn_curriculum ——
#     s1-cnn-mix 保底(100%) → s2-open-empty(空场80%成长 astar 78%) →
#     s3-corridor(纯走廊 astar 75%/hunter 45%) → s4-pure-open(固定能力 2%) →
#     s5-pillar(立柱 0%) → s6-mix-ladder(混合+天梯收尾)。
#   环岛不进训练（用户指定留泛化测试）。
# ckpt 存在则自动 resume 接力（保留课程进度），否则从 duel_cnn 300M 起步。
# -u 关掉 stdout 块缓冲。
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
RESUME=""
if [ -f ckpt/cnn_course.pt ]; then
  RESUME="--resume ckpt/cnn_course.pt"
elif [ -f ckpt/duel_cnn.pt ]; then
  RESUME="--resume ckpt/duel_cnn.pt"
fi
exec python3 -u -m train.train \
  --arch cnn \
  --num-envs 2048 \
  --device npu:0 \
  --rollout-steps 128 \
  --minibatches 2 \
  --max-mem-frac 0.85 \
  --total-steps 2000000000 \
  --snapshot-every 20 \
  --time-budget 86400 \
  --map-mode corridor \
  --cnn-course \
  --bot-opp-prob 0.5 \
  $RESUME \
  --ckpt ckpt/cnn_course.pt \
  --log-csv ckpt/train_cnn_course.csv \
  > train_cnn_course.log 2>&1
