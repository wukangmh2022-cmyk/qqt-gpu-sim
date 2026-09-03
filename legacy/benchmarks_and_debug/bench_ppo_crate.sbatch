#!/bin/bash
#SBATCH -J qqt-bench-crate
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -o bench_crate_%j.out
#SBATCH -e bench_crate_%j.err
# 新版（宝箱/地图生成）端到端 sps 计时：同生产 PPO 配置，随机权重起步，
# 只跑 8 个计时迭代（脚本内 warmup 2 次编译），不存 ckpt。
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python -m jax_bomb.jax_train \
  --arch mlp_bf16 --hidden 768 \
  --num-envs 4096 --num-steps 256 \
  --minibatch 8192 --epochs 1 \
  --iters 8 --lr 3e-4 \
  2>&1 | tee bench_crate.log
