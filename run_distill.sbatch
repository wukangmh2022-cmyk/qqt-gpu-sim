#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 01:00:00
#SBATCH -o distill_%j.log
#SBATCH -J qqt-distill
# 离线蒸馏：student(mlp_bf16 768) KL 到 teacher 分布，warm-start 权重落盘 ckpt/student.pt。
# 数据 = collect_distill 全量（纯 50% + 变换 50%，9 对手，~1000万+ 帧，超上限自动抽稀）。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
mkdir -p ckpt
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --distill-data "distill_data/*.npz" \
  --distill-iters 400 --distill-batch 8192 --distill-max-frames 2000000 \
  --save ckpt/student.pt \
  2>&1 | tee distill_train.log
