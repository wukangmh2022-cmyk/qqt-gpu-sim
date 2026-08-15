#!/bin/bash
# CNN 对打寻路 AI 专精课程训练（2026-08-15 重写）：resume duel_cnn 300M，
# 把对打 astar/hunter 胜率拉到 90%+（launcher open80 环境）。
#   课程表 train/curriculum.py::cnn_curriculum ——
#     s1-open80-astar → s2-open80-hunter → s3-open80-both（混合验收）。
#   2026-08-15 诊断：旧版末尾 s7 天梯用自博弈快照当对手，wr 虚高（对弱自己
#   0.93 但对纯 astar 只剩 0.45）。新版全程纯规则 bot 对手、无天梯，内部 wr
#   即对目标 bot 的真实胜率（bot_prob=1.0 由 stage 指定）。
# **只从 duel_cnn 300M 基线起步**：旧 cnn_course.pt 已证明被天梯带偏，不再 resume。
# 手动续跑请先备份旧 ckpt 或删除，脚本不自动 resume cnn_course.pt。
# -u 关掉 stdout 块缓冲。
source ~/Ascend/ascend-toolkit/set_env.sh
cd ~/qqt-gpu-sim
RESUME=""
if [ -f ckpt/duel_cnn.pt ]; then
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
  --bot-opp-prob 1.0 \
  $RESUME \
  --ckpt ckpt/cnn_course.pt \
  --log-csv ckpt/train_cnn_course.csv \
  > train_cnn_course.log 2>&1
