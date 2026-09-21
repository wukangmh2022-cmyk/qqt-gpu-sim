#!/usr/bin/env bash
# ==============================================================================
# SCNet 6 节点 x 4 卡 (24 卡 DCU) Stage 5 (8h 5Hz) 统一启动脚本
# ==============================================================================
set -euo pipefail

ROOT_DIR="/public/home/acjeb36fmf/qqt-gpu-sim"
cd "$ROOT_DIR"

# 1. 激活环境与动态库
source /opt/dtk/env.sh 2>/dev/null || source /public/software/compiler/rocm/dtk-26.04/env.sh 2>/dev/null || true
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1 || true)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HP_DOMAIN_RAND=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

PYTHON_BIN="/public/home/acjeb36fmf/env_jax/bin/python"

# 2. 判断当前是否已经是分布式 Rank 实例（平台注入了 RANK / WORLD_SIZE）
if [ -n "${RANK:-}" ] && [ -n "${WORLD_SIZE:-}" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    echo "=== [分布式实例模式] Rank: $RANK / $WORLD_SIZE, Master: ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} ==="
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    exec $PYTHON_BIN -u -m jax_bomb.train_real --config configs/stage5_8h_6x4gpu_5hz.toml
fi

# 3. 判断是否处于 Slurm 派发环境中（SLURM_PROCID 存在）
if [ -n "${SLURM_PROCID:-}" ] && [ -n "${SLURM_NPROCS:-}" ] && [ "$SLURM_NPROCS" -gt 1 ]; then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=${SLURM_NNODES:-$SLURM_NPROCS}
    MASTER_HOST=$(scontrol show hostnames "${SLURM_NODELIST:-localhost}" 2>/dev/null | head -n 1 || echo "localhost")
    export MASTER_ADDR=$(getent hosts "$MASTER_HOST" 2>/dev/null | awk '{print $1}')
    [ -z "$MASTER_ADDR" ] && export MASTER_ADDR="$MASTER_HOST"
    export MASTER_PORT=29500
    echo "=== [Slurm 实例模式] Rank: $RANK / $WORLD_SIZE, Master: $MASTER_ADDR:$MASTER_PORT ==="
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    exec $PYTHON_BIN -u -m jax_bomb.train_real --config configs/stage5_8h_6x4gpu_5hz.toml
fi

# 4. 如果处于 Slurm 主节点分配但未分发（主控负责 srun 分发至全部节点）
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_NNODES:-}" ] && [ "$SLURM_NNODES" -gt 1 ]; then
    echo "=== [Slurm 主控分发模式] 节点数: $SLURM_NNODES, 节点列表: $SLURM_NODELIST ==="
    exec srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash "$ROOT_DIR/run_cluster.sh"
fi

# 5. 单机 4 卡回退
echo "=== [单机模式] 启动单节点 4 卡 ==="
export RANK=0
export WORLD_SIZE=1
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
exec $PYTHON_BIN -u -m jax_bomb.train_real --config configs/stage5_8h_6x4gpu_5hz.toml
