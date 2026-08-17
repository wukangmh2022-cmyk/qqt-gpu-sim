#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:10:00
#SBATCH -o ppo_smoke_%j.log
#SBATCH -J qqt-ppo-smoke
# PPO 续跑 smoke：--load ckpt/student.pt（真实蒸馏产物）→ 2 iter 自对弈 → --save 验证。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --load ckpt/student.pt \
  --num-envs 128 --num-steps 32 \
  --minibatch 256 --epochs 1 \
  --iters 2 --lr 3e-4 \
  --save ckpt/ppo_smoke.pt \
  2>&1 | tee ppo_smoke.log
