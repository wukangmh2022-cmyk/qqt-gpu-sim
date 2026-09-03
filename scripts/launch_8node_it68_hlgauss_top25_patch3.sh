#!/usr/bin/env bash
# Config-driven launcher. Experiment parameters live in a commented TOML file;
# WORLD_SIZE/RANK/MASTER_* remain platform-injected topology variables.
REPO="${QQT_REPO:-}"
if [[ -z "$REPO" ]]; then
  for cand in /root/private_data/qqt-gpu-sim /root/qqt-gpu-sim \
              /mnt/data/qqt-gpu-sim /public/home/*/qqt-gpu-sim \
              /home/*/qqt-gpu-sim; do
    if [[ -f "$cand/jax_bomb/multicard_train.py" ]]; then REPO="$cand"; break; fi
  done
fi
if [[ -z "$REPO" || ! -f "$REPO/jax_bomb/multicard_train.py" ]]; then
  echo "ERROR: 找不到 qqt-gpu-sim；请设置 QQT_REPO=/实际路径" >&2; exit 2
fi
cd "$REPO"
source /opt/dtk/env.sh 2>/dev/null || true
set -euo pipefail
ulimit -c 0 || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* /public/software/mpi/*/lib/libmpi.so*; do
  if [[ -f "$mpi" ]]; then export LD_PRELOAD="$mpi"; break; fi
done
export WORLD_SIZE="${WORLD_SIZE:-1}" RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29520}"
CONFIG="${TRAIN_CONFIG:-$REPO/configs/repro_it68_asym_timeout_k64_failed.toml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: 找不到 TRAIN_CONFIG=$CONFIG" >&2; exit 2
fi
read -r CFG_NAME CFG_ENVS CFG_MB CFG_K < <(python3 - "$CONFIG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    c = tomllib.load(f)
print(c.get("run", {}).get("name", "experiment"),
      c.get("rollout", {}).get("num_envs", 0),
      c.get("rollout", {}).get("minibatch", 0),
      c.get("distributed", {}).get("lsgd_k", 0))
PY
)
RUN_DIR="${RUN_DIR:-$REPO/runs/$CFG_NAME}"
export CKPT_DIR="${CKPT_DIR:-$RUN_DIR/ckpt}"
export CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-$RUN_DIR/ckpt_local}"
mkdir -p "$RUN_DIR/ckpt" "$RUN_DIR/ckpt_local" "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train_r${RANK}_$(date +%Y%m%d_%H%M%S).log"
NLOCAL=$(python3 -c 'import jax; print(jax.local_device_count())')
echo "repo=$REPO world=$WORLD_SIZE rank=$RANK master=$MASTER_ADDR:$MASTER_PORT"
echo "config=$CONFIG name=$CFG_NAME local_devices=$NLOCAL num_envs=$CFG_ENVS minibatch=$CFG_MB lsgd_k=$CFG_K"
exec python3 -u -m jax_bomb.train_real --config "$CONFIG" 2>&1 | tee "$LOG"
