#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:50:00
#SBATCH -o jax7_%j.log
#SBATCH -J qqt-jax7
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp4 (4096, 5-Linear 768 ≈321万参数, danger 短路) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -5
echo "=== mlp4 (8192 envs) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 8192 --num-steps 256 --iters 2 2>&1 | tail -4
