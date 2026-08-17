#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax16_%j.log
#SBATCH -J qqt-jax16
# 同节点 A/B：带 mask（生产） vs --no-mask（旧版行为），量化 wall+mask 开销
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== A. 带 mask：mlp_bf16 2层768 mb=8192/1 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -5
echo "=== B. --no-mask：mlp_bf16 2层768 mb=8192/1 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 --no-mask 2>&1 | tail -5
echo "=== C. 带 mask：mlp_bf16 2层256 mb=8192/1 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 256 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -5
echo "=== D. --no-mask：mlp_bf16 2层256 mb=8192/1 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 256 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 --no-mask 2>&1 | tail -5
