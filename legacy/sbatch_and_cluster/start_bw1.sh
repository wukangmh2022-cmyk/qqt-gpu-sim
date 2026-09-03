#!/bin/bash
# BW-1 单卡训练：open 空场 + 纯自博弈 MLP（实测最优路线，SCNet hx1hdnormal 分区）
#
# 2026-08-16 用户定：用空场景训练，不要规则 bot（astar/hunter Dijkstra），
# 直接和 MLP 对战（纯自博弈，对手 = 模型池里的自己）。
# open 场景已强制纯空（sim/torch_sim.py 删掉 wall_density 随机立柱）。
# 配额实测：每作业 1 张 DCU 卡 + 最多 16 CPU 核 → 本脚本完全满足。
# 运行前确保 ~/das_torch/env.sh 存在（DTK 26.04 + DAS torch 2.5.1 venv）。
#
# 提交：sbatch start_bw1.sh

#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o train_bw1_%j.log
#SBATCH -J qqt-bw1

set -e
source ~/das_torch/env.sh
cd ~/qqt-gpu-sim

# corridor + open_fraction=1.0 走混合分支 open 子类（宝箱/成长/回收，51k SPS）
# 太慢；--map-mode open 是纯 open 分支（无宝箱无成长，sim 最轻，实测 277k SPS）。
# 无 --bot-opponents/--fixed-bots/--fixed-ckpt → 纯自博弈（对手 = 池快照 MLP）。
exec python -m train.train \
  --backend torch --device cuda --arch mlp --single-stage \
  --map-mode open \
  --num-envs 32768 --total-steps 1_600_000_000 \
  --rollout-steps 128 --minibatches 4 \
  --warmup-steps 0 \
  --explore-anneal --suicide-penalty 2.0 --timeout-draw \
  --combo-reward 0.05 --combo-gap-factor 0.9 \
  --ckpt private_data/duel_open_bw1.pt \
  --log-csv private_data/train_open_bw1.csv \
  --time-budget 43000
