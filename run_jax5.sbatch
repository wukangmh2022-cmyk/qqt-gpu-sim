#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:50:00
#SBATCH -o jax5_%j.log
#SBATCH -J qqt-jax5
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== cnn bf16 (4096) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch cnn --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -4
echo "=== transformer bf16 (512) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch transformer --num-envs 512 --num-steps 256 --iters 2 2>&1 | tail -4
echo "=== transformer bf16 (2048) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch transformer --num-envs 2048 --num-steps 128 --iters 2 2>&1 | tail -4
echo "=== mlp control (4096) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --num-envs 4096 --num-steps 256 --iters 2 2>&1 | tail -3
