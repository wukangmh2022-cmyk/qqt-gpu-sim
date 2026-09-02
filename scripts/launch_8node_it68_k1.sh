#!/usr/bin/env bash
# 8 节点（平台注入 WORLD_SIZE/RANK/MASTER_*）复跑 Iteration 68。
# 代码与单机 8 卡复跑完全相同；默认 LSGD_K=1，只有跨机同步周期不同。
# 保持单机 8 卡基线的全局 batch（16384 env/iter）；16 卡时即 1024 env/replica，
# 避免因卡数翻倍而把优化器有效 batch 也翻倍，确保 Iter68 效果可比较。

REPO="${QQT_REPO:-}"
if [[ -z "$REPO" ]]; then
  for cand in /root/private_data/qqt-gpu-sim /root/qqt-gpu-sim \
              /mnt/data/qqt-gpu-sim /public/home/*/qqt-gpu-sim \
              /home/*/qqt-gpu-sim; do
    if [[ -f "$cand/jax_bomb/multicard_train.py" ]]; then REPO="$cand"; break; fi
  done
fi
if [[ -z "$REPO" || ! -f "$REPO/jax_bomb/multicard_train.py" ]]; then
  echo "ERROR: 找不到 qqt-gpu-sim；请设置 QQT_REPO=/实际路径" >&2
  exit 2
fi
cd "$REPO"

source /opt/dtk/env.sh 2>/dev/null || true
set -euo pipefail
ulimit -c 0 || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* \
           /public/software/mpi/*/lib/libmpi.so*; do
  if [[ -f "$mpi" ]]; then export LD_PRELOAD="$mpi"; break; fi
done

export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 保持与单机 8 卡任务相同的全局 rollout/batch 负载。
NLOCAL=$(python3 -c 'import jax; print(jax.local_device_count())')
NUM_ENVS="${NUM_ENVS:-16384}"
MINIBATCH="${MINIBATCH:-$NUM_ENVS}"
LSGD_K="${LSGD_K:-1}"
LSGD_MODE="${LSGD_MODE:-param}"
RUN_DIR="${RUN_DIR:-$REPO/runs/repro_it68_8node_k${LSGD_K}}"
mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/ckpt_local" "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train_r${RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "repo=$REPO world=$WORLD_SIZE rank=$RANK master=$MASTER_ADDR:$MASTER_PORT"
echo "local_devices=$NLOCAL num_envs=$NUM_ENVS minibatch=$MINIBATCH lsgd_k=$LSGD_K"
echo "obs=14ch arch=transformer embed=392 depth=4 patch=4"

exec python3 -u -m jax_bomb.train_real \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --num-envs "$NUM_ENVS" --num-steps 256 --minibatch "$MINIBATCH" --epochs 2 \
  --iters 68 --lr 3e-4 \
  --gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01 \
  --levels levels.json --level-weights "empty=0.05,功夫=0.1,比武=0.15" \
  --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 \
  --explore-reward-coef 0.0 --brick-reward-coef 0.0 --reward-anneal-k 0.0 \
  --lsgd-k "$LSGD_K" --lsgd-mode "$LSGD_MODE" \
  --checkpoint --ckpt-dir "$RUN_DIR/ckpt" --ckpt-every 15 \
  --ckpt-local-dir "$RUN_DIR/ckpt_local" --ckpt-local-every 5 \
  --fresh 2>&1 | tee "$LOG"
