#!/bin/bash
# 对拍 torch 侧轨迹导出（das_torch 环境，服务器直接跑；jax 侧用 run_parity_jax.sbatch）
source ~/das_torch/env.sh
cd /root/qqt-gpu-sim 2>/dev/null || cd ~/qqt-gpu-sim
python verify_torch_jax_parity.py --side torch --ticks 300 --seed 0 --out /root/parity_torch.jsonl 2>&1 | tail -3
echo "--- 本机（有 torch+jax 的环境）比对 ---"
python verify_torch_jax_parity.py --compare /root/parity_torch.jsonl /root/parity_jax.jsonl 2>&1 | tail -3
