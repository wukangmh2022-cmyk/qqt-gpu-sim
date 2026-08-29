#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:50:00
#SBATCH -o jax6_%j.log
#SBATCH -J qqt-jax6
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp4 (4096, hidden=768, 正式版规则) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -5
echo "=== mlp4 (4096, steps 512) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 512 --iters 2 2>&1 | tail -4
echo "=== mlp control (4096, 旧 2 层) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --num-envs 4096 --num-steps 256 --iters 2 2>&1 | tail -3
