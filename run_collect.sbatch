#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 01:00:00
#SBATCH -o collect_%j.log
#SBATCH -J qqt-collect
# 离线蒸馏全量收集（torch 环境）：纯空场 50% + copy 变换 50%，满级无宝箱，
# 双视角 obs7/logits/masks。teacher = duel_nobc_11b_live（最强：115亿步）。
source ~/das_torch/env.sh
cd ~/qqt-gpu-sim
python deploy/collect_distill.py \
  --teacher ckpt/duel_nobc_11b_live.pt \
  --out distill_data \
  --ticks 600 \
  --num-envs 2048 \
  --seed 0 \
  --opp-nets duel_nobc_3B duel_nobc_8b_live duel_cnn cnn_course_latest_min duel_5x3 duel_course_453M \
  --opp-bots random greedy astar 2>&1 | tail -25
