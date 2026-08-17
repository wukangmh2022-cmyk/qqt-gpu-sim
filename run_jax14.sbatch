#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax14_%j.log
#SBATCH -J qqt-jax14
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp fp32 2层768 (150万, mb=8192, epochs=1) 对照 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
echo "=== mlp bf16 2层768 (150万, mb=8192, epochs=1) 量化 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
echo "=== mlp fp32 2层256 (37万, mb=8192, epochs=1) 对照 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --hidden 256 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
echo "=== mlp bf16 2层256 (37万, mb=8192, epochs=1) 量化 ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 256 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
