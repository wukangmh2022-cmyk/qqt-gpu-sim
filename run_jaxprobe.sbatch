#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=55G
#SBATCH -t 00:40:00
#SBATCH -o jax_probe_%j.log
#SBATCH -J qqt-jaxprobe
source ~/jax_env.sh
cd ~/qqt-gpu-sim
~/miniconda3/bin/python probe_jax_cnn.py 2>&1 | tail -16
