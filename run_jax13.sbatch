#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax13_%j.log
#SBATCH -J qqt-jax13
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== mlp 2层768 (150万, mb=8192, epochs=1) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
echo "=== mlp4 5层512 (166万, mb=8192, epochs=1) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --hidden 512 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -6
echo "=== mlp4 5层512 (166万, mb=8192, epochs=2) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp4 --hidden 512 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 2 --iters 3 2>&1 | tail -6
