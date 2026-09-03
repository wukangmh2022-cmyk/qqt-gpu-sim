#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 03:00:00
#SBATCH -o jax17_%j.log
#SBATCH -J qqt-jax17
# 正式自对弈 PPO 训练（环境已对齐：wall+legal_mask+chain_cap=8，纯空场）
# 2层768 bf16（150万参数），~3.1M SPS → 3 小时 ≈ 340 亿步
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --num-envs 4096 --num-steps 256 \
  --minibatch 8192 --epochs 1 \
  --iters 15000 --lr 3e-4 \
  2>&1 | tee jax17_train.log
