#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:50:00
#SBATCH -o jax4_%j.log
#SBATCH -J qqt-jax4
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== probe: transformer attention 169 vs 25 tokens ==="
~/miniconda3/bin/python probe_jax_tf.py 2>&1 | tail -10
echo "=== cnn GAP (4096) — 修掉 flatten 大 FC ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch cnn --num-envs 4096 --num-steps 256 --iters 3 2>&1 | tail -4
echo "=== transformer small (512 envs, 64 steps) ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch transformer --num-envs 512 --num-steps 64 --iters 2 2>&1 | tail -5
