#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:50:00
#SBATCH -o jax3_%j.log
#SBATCH -J qqt-jax3
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== probe: conv vs matmul vs full fwd/bwd ==="
~/miniconda3/bin/python probe_jax_cnn.py 2>&1 | tail -16
echo "=== transformer (4096, embed=192 depth=4) — fixed tuple bug ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch transformer --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -4
echo "=== mlp (4096) control ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -4
