#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:10:00
#SBATCH -o ppo_smoke2_%j.log
#SBATCH -J qqt-ppo-smoke2
# NaN 修复验证：--load ckpt/student.pt（正常权重）→ 3 iter PPO → --save，
# 检查落盘权重无 NaN（熵 0*(-inf) 修复）。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --load ckpt/student.pt \
  --num-envs 128 --num-steps 32 \
  --minibatch 256 --epochs 1 \
  --iters 3 --lr 3e-4 \
  --save ckpt/ppo_fix_smoke.pt \
  2>&1 | tee ppo_smoke2.log
