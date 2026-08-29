#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax9_%j.log
#SBATCH -J qqt-jax9
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp 2层256 (4096, 化繁为简正式版规则) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --hidden 256 --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -6
echo "=== mlp 2层256 (8192 envs) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --hidden 256 --num-envs 8192 --num-steps 256 --iters 2 2>&1 | tail -5
