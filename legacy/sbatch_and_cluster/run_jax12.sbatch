#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax12_%j.log
#SBATCH -J qqt-jax12
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp4 大参数 (4096, mb=2048, epochs=2, 化繁为简正式版) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -6
echo "=== mlp4 大参数 (4096, mb=8192, epochs=2) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 256 --minibatch 8192 --iters 3 2>&1 | tail -6
echo "=== mlp4 大参数 (4096, mb=8192, epochs=1) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
