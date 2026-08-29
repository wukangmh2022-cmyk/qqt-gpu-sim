#!/bin/sh
# R0：双机 8 卡 JAX ViT 训练
# 平台启动脚本会用 /bin/sh 执行，因此本文件只使用 POSIX sh 语法。
# 先启动 R0；它会在 jax.distributed.initialize() 等待 R1。
set -e

# 平台注入 WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT。
# R0 只在平台没有注入时使用默认值；MASTER_ADDR 不猜、不写死。
WORLD_SIZE="${WORLD_SIZE:-2}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-23456}"
export WORLD_SIZE RANK MASTER_ADDR MASTER_PORT

if [ "$WORLD_SIZE" != "2" ]; then
  echo "ERROR: WORLD_SIZE=$WORLD_SIZE，R0 必须属于双实例任务" >&2
  exit 1
fi
if [ "$RANK" != "0" ]; then
  echo "ERROR: 当前任务 RANK=$RANK，不是 R0" >&2
  exit 1
fi
if [ -z "$MASTER_ADDR" ]; then
  echo "ERROR: 平台没有注入 MASTER_ADDR" >&2
  exit 1
fi

# DTK/MPI 必须在任何 JAX import 前设置。
. /opt/dtk/env.sh 2>/dev/null || true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset FTP_PROXY ftp_proxy NO_PROXY no_proxy

MPI_LIB=""
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* /public/software/mpi/*/lib/libmpi.so*; do
  if [ -f "$mpi" ]; then
    MPI_LIB="$mpi"
    export LD_PRELOAD="$mpi"
    break
  fi
done

echo "========== R0 START =========="
echo "hostname=$(hostname 2>/dev/null || true)"
echo "local_ip=$(hostname -I 2>/dev/null || true)"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "MPI_LIB=${MPI_LIB:-NOT_FOUND}"
getent hosts "$MASTER_ADDR" 2>&1 || true

# 本轮上传包和解压目录。
UPLOAD="/root/private_data/qqt_upload"
REPO="/root/qqt-gpu-sim"
if [ ! -f "$UPLOAD/jaxbomb.tgz" ]; then
  echo "ERROR: 找不到 $UPLOAD/jaxbomb.tgz" >&2
  exit 1
fi
mkdir -p "$REPO"
rm -rf "$REPO/jax_bomb" "$REPO/web" "$REPO/scripts" "$REPO/levels.json"
tar xzf "$UPLOAD/jaxbomb.tgz" -C "$REPO"
cd "$REPO"

grep -R -q "N_OBS_CH = 14" jax_bomb
grep -R -q "push_t" jax_bomb
grep -R -q "brick-reward-coef" jax_bomb
grep -R -q "reward-anneal-k" jax_bomb
echo "CODE_CHECK=PASS"

# optax 必须用同一个 python3 安装和验证；失败时立即退出，不进入训练。
if python3 -c 'import optax, chex' >/dev/null 2>&1; then
  echo "DEPENDENCIES=EXIST"
else
  echo "DEPENDENCIES=INSTALL"
  python3 -m pip install --no-index --find-links="$UPLOAD/wheels" --no-deps --force-reinstall \
    optax chex dm-tree toolz wrapt etils typing_extensions absl-py attrs
fi
python3 -c 'import optax, chex; print("optax=" + optax.__version__); print("chex=" + chex.__version__)'
echo "DEPENDENCY_CHECK=PASS"

# 这里确认 ROCm/JAX 仍然可见；MPI 已在本进程环境中预加载。
python3 -c 'import jax; print("jax=" + jax.__version__); print("local_device_count=" + str(jax.local_device_count())); print("devices=" + str(jax.devices())); assert jax.local_device_count() == 8'
echo "DEVICE_CHECK=PASS"

mkdir -p "$REPO/ckpt" "$REPO/ckpt_local" "$REPO/logs"
LOG="$REPO/logs/train_rank0.log"
echo "READY: entering distributed training; R0 waits for R1"
echo "LOG=$LOG"

python3 -m jax_bomb.train_real \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 --iters 15500 \
  --levels "$REPO/levels.json" \
  --level-weights "empty=0.05,功夫=0.1,比武=0.15" \
  --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 \
  --explore-reward-coef 0.01 --explore-reward-anneal-steps 30000000000 \
  --brick-reward-coef 0.05 --reward-anneal-k 1.2 \
  --lsgd-k 0 --lsgd-mode param --fresh 2>&1 | tee "$LOG"
