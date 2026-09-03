#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:10:00
#SBATCH -o distill_smoke2_%j.log
#SBATCH -J qqt-distill-smoke
# 蒸馏更新路径 smoke：真实 npz（greedy，纯收集不抽稀）→ 5 轮 KL → --save 验证。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --distill-data "distill_data/greedy.npz" \
  --distill-iters 5 --distill-batch 1024 --distill-max-frames 200000 \
  --save /tmp/student_smoke.pt \
  2>&1 | tee distill_smoke2.log
