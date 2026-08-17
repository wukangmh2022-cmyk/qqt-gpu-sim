#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 03:00:00
#SBATCH -o ppo_%j.log
#SBATCH -J qqt-ppo
# 正式自对弈 PPO：warm-start 自 ckpt/student.pt（蒸馏产物），2层768 bf16，
# wall+legal_mask+chain_cap=8。结束保存 ckpt/final.pt。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --load ckpt/student.pt \
  --num-envs 4096 --num-steps 256 \
  --minibatch 8192 --epochs 1 \
  --iters 15000 --lr 3e-4 \
  --save ckpt/final.pt \
  2>&1 | tee ppo_train.log
