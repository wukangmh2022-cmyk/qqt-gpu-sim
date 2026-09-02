#!/usr/bin/env bash
# 单机 8 卡复刻 ViTModel_68
#
# 目标：用当前正式 14 通道协议复现旧 Iteration 68 的训练动力学，
# 而不是继续 Exp-2 的 it120。保留旧的 8 卡 pmap、K=0、纯胜负+crate
# bootstrap 配置；仅观测协议使用当前 14ch（含 pushable）。
# = 2×16384×256×68 = 570,425,344 global environment steps。
#
# 用法（机器开机后在仓库根目录执行）：
#   bash scripts/launch_single_8gpu_it68.sh
#
# 可选覆盖：
#   REPRO_ITERS=8       # 先做短程冒烟
#   REPRO_NUM_ENVS=8192 # 显存紧张时降低每轮环境数（默认 16384）
#   REPRO_FRESH=0       # 使用本次 run 目录已有 checkpoint 接续
#   REPRO_RUN_DIR=...   # 指定输出目录

REPO="${QQT_REPO:-}"
if [[ -z "$REPO" ]]; then
  for cand in \
    /root/private_data/qqt-gpu-sim \
    /root/qqt-gpu-sim \
    /mnt/data/qqt-gpu-sim \
    /public/home/*/qqt-gpu-sim \
    /home/*/qqt-gpu-sim; do
    if [[ -f "$cand/jax_bomb/multicard_train.py" ]]; then
      REPO="$cand"
      break
    fi
  done
fi
if [[ -z "$REPO" || ! -f "$REPO/jax_bomb/multicard_train.py" ]]; then
  echo "ERROR: 找不到 qqt-gpu-sim；请设置 QQT_REPO=/实际路径" >&2
  exit 2
fi
cd "$REPO"

# 单机训练必须强制走 WORLD_SIZE=1；不要继承之前多机任务的 rendezvous。
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset FTP_PROXY ftp_proxy NO_PROXY no_proxy
source /opt/dtk/env.sh 2>/dev/null || true
# DTK env 在此节点的 rocm_smi 探测会返回非零；必须在严格错误模式外
# 完成 source，否则脚本会在 GPU 自检前退出。后续训练恢复严格模式。
set -euo pipefail
# DCU/JAX 编译失败时平台可能生成约 20GB 的 core 文件；训练本身不需要
# core，关闭它避免再次把 notebook 挂载盘写满。关闭预分配也给编译器留余量。
ulimit -c 0 || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export JAX_TRACEBACK_FILTERING=off
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* \
           /public/software/mpi/*/lib/libmpi.so*; do
  if [[ -f "$mpi" ]]; then
    export LD_PRELOAD="$mpi"
    break
  fi
done

echo "=== repo=$REPO ==="
echo "=== checking dependencies and exactly 8 local devices ==="
python3 - <<'PY'
import jax
print("jax=", jax.__version__)
print("devices=", jax.devices())
if jax.local_device_count() != 8:
    raise SystemExit(f"需要单机 8 卡，实际看到 {jax.local_device_count()} 张")
PY

if ! grep -q -- "N_OBS_CH = 14" jax_bomb/jax_env.py; then
  echo "ERROR: 当前代码不是 14 通道版本；请上传最新代码" >&2
  exit 3
fi
if ! python3 -c 'import optax, chex' >/dev/null 2>&1; then
  echo "ERROR: 缺少 optax/chex 依赖" >&2
  exit 4
fi

REPRO_ITERS="${REPRO_ITERS:-68}"
REPRO_NUM_ENVS="${REPRO_NUM_ENVS:-16384}"
REPRO_RUN_DIR="${REPRO_RUN_DIR:-$REPO/runs/repro_it68_14ch}"
CKPT_DIR="$REPRO_RUN_DIR/ckpt"
CKPT_LOCAL_DIR="$REPRO_RUN_DIR/ckpt_local"
LOG_DIR="$REPRO_RUN_DIR/logs"
mkdir -p "$CKPT_DIR" "$CKPT_LOCAL_DIR" "$LOG_DIR"

FRESH_FLAG=""
if [[ "${REPRO_FRESH:-1}" == "1" ]]; then
  FRESH_FLAG="--fresh"
fi

echo "=== Iteration 68 reproduction ==="
echo "iters=$REPRO_ITERS  global_steps_per_iter=8,388,608"
echo "obs=14ch  devices=8  lsgd_k=0  run_dir=$REPRO_RUN_DIR"
echo "注意：这是当前 14 通道协议；本次从头初始化，不接续旧 checkpoint。"

exec python3 -u -m jax_bomb.train_real \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --num-envs "$REPRO_NUM_ENVS" --num-steps 256 --minibatch "$REPRO_NUM_ENVS" --epochs 2 \
  --iters "$REPRO_ITERS" --lr 3e-4 \
  --gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01 \
  --levels levels.json \
  --level-weights "empty=0.05,功夫=0.1,比武=0.15" \
  --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 \
  --explore-reward-coef 0.0 --brick-reward-coef 0.0 \
  --reward-anneal-k 0.0 \
  --lsgd-k 0 --lsgd-mode param \
  --checkpoint \
  --ckpt-dir "$CKPT_DIR" --ckpt-every 15 \
  --ckpt-local-dir "$CKPT_LOCAL_DIR" --ckpt-local-every 5 \
  $FRESH_FLAG 2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
