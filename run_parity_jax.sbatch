#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:20:00
#SBATCH -o parity_jax_%j.log
#SBATCH -J qqt-parity-jax
# 对拍 jax 侧轨迹导出（torch 侧在 das_torch 环境跑 run_parity_torch.sh，本地 --compare 比对）
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python verify_torch_jax_parity.py --side jax --ticks 300 --seed 0 --out ~/parity_jax.jsonl 2>&1 | tail -3
