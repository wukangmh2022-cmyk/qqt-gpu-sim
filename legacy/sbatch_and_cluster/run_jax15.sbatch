#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax15_%j.log
#SBATCH -J qqt-jax15
# wall+legal_mask 逻辑验证 + 速度对比（对照 run_jax14 的 1.56M/1.85M 基线）
source ~/jax_env.sh
cd ~/qqt-gpu-sim
echo "=== 0. wall+mask 逻辑验证 ==="
~/miniconda3/bin/python verify_wall_mask.py 2>&1 | tail -30
echo "=== 1. mlp bf16 2层768 (150万, mb=8192, epochs=1) 带 wall+mask ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -5
echo "=== 2. mlp bf16 2层256 (37万, mb=8192, epochs=1) 带 wall+mask ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 256 --num-envs 4096 --num-steps 256 --minibatch 8192 --epochs 1 --iters 3 2>&1 | tail -5
echo "=== 3. mlp bf16 2层768 mb=2048 epochs=2（原 1.85M 同口径） ==="
~/miniconda3/bin/python -m jax_bomb.jax_train --arch mlp_bf16 --hidden 768 --num-envs 4096 --num-steps 256 --minibatch 2048 --epochs 2 --iters 3 2>&1 | tail -5
