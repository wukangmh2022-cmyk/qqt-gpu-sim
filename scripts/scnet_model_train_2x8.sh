#!/bin/bash
# QQ堂 JAX ViT 260B 训练启动脚本：2 个训练任务 × 每任务 8 卡
#
# 用法：把本文件全文分别粘贴到两台机器的「训练任务启动脚本」里。
# 两个任务必须使用同一份代码上传目录、同一组平台分布式变量：
#   WORLD_SIZE=2, RANK=0/1, MASTER_ADDR, MASTER_PORT
#
# 启动顺序：先启动 rank0，再启动 rank1。rank0 会在
# jax.distributed.initialize() 处等待 rank1；rank1 晚几秒/几分钟启动都可以，
# 两边 rendezvous 成功后才会一起开始第一轮训练。不要让两个任务各自 WORLD_SIZE=1。
#
# 网页上传方式：把 upload_2x8gpu/qqt_upload/ 文件夹上传到两台机器相同的
# 数据盘位置（每个账号各有一份）。本脚本会自动找到 qqt_upload/jaxbomb.tgz，
# 解压到该账号的数据盘下的 qqt-gpu-sim；已经解压过则复用现有代码。
# 也支持直接上传/挂载一个名为 qqt-gpu-sim 的代码目录。
#
# 本轮配置：ViT 392/4/4/4/4，14 通道观测（ch13=可推箱），260B，
# 空场景 5% + 功夫 10% + 比武 15%，其余地图均分；课程关闭；
# crate 0.5、explore 0.01、brick 0.05，30B bootstrap 退火，动态 k=1.2。
set -e

# ── 1. 自动定位/解压代码 ───────────────────────────────────────────────
REPO="${QQT_REPO:-}"
if [ -n "$REPO" ] && [ ! -d "$REPO" ]; then
  echo "ERROR: QQT_REPO=$REPO 不存在" >&2
  exit 1
fi
if [ -z "$REPO" ]; then
  for cand in \
      /root/private_data/qqt-gpu-sim \
      /root/private_data/*/qqt-gpu-sim \
      /mnt/data/qqt-gpu-sim \
      /mnt/data/*/qqt-gpu-sim \
      /mnt/qqt-gpu-sim \
      /public/home/*/qqt-gpu-sim \
      /home/*/qqt-gpu-sim \
      /root/qqt-gpu-sim; do
    if [ -f "$cand/jax_bomb/jax_env.py" ]; then REPO="$cand"; break; fi
  done
fi
# 上传的是 qqt_upload 文件夹时，解压其内的 jaxbomb.tgz。
if [ -z "$REPO" ]; then
  for pkg in \
      /root/private_data/qqt_upload/jaxbomb.tgz \
      /root/private_data/*/qqt_upload/jaxbomb.tgz \
      /mnt/data/qqt_upload/jaxbomb.tgz \
      /mnt/data/*/qqt_upload/jaxbomb.tgz \
      /mnt/qqt_upload/jaxbomb.tgz \
      /public/home/*/qqt_upload/jaxbomb.tgz \
      /home/*/qqt_upload/jaxbomb.tgz; do
    if [ -f "$pkg" ]; then
      BASE="$(dirname "$(dirname "$pkg")")"
      REPO="$BASE/qqt-gpu-sim"
      mkdir -p "$REPO"
      tar xzf "$pkg" -C "$REPO"
      echo "=== 从 $pkg 解压代码到 $REPO ==="
      break
    fi
  done
fi
if [ -z "$REPO" ] || [ ! -f "$REPO/jax_bomb/jax_env.py" ]; then
  echo "ERROR: 找不到代码。请上传 upload_2x8gpu/qqt_upload/ 整个文件夹，" >&2
  echo "       或设置 QQT_REPO=/你的实际/qqt-gpu-sim 路径。" >&2
  exit 1
fi
cd "$REPO"
echo "=== REPO=$REPO ==="

# 代码版本校验：14 通道/推箱/动态奖励必须存在
if ! grep -R -q "N_OBS_CH = 14" jax_bomb 2>/dev/null \
   || ! grep -R -q "push_t" jax_bomb 2>/dev/null \
   || ! grep -R -q "brick-reward-coef" jax_bomb 2>/dev/null \
   || ! grep -R -q "reward-anneal-k" jax_bomb 2>/dev/null; then
  echo "ERROR: 代码缺少 14 通道/推箱/动态奖励实现：上传的不是本轮最终代码。" >&2
  exit 1
fi
echo "=== 代码版本 OK（14ch + push-box + reward annealing） ==="

# ── 2. DTK/通信环境 ────────────────────────────────────────────────────
source /opt/dtk/env.sh 2>/dev/null || true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* \
           /public/software/mpi/*/lib/libmpi.so*; do
  if [ -f "$mpi" ]; then export LD_PRELOAD="$mpi"; break; fi
done

# ── 3. 分布式变量：优先平台注入，worker 变量仅作兜底 ─────────────────────
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# 某些平台只注入 worker0/worker1，不注入 PyTorch 变量；从 worker 信息补齐。
if [ -n "${worker0:-}" ] && [ -n "${worker1:-}" ]; then
  if [ "$WORLD_SIZE" = "1" ]; then WORLD_SIZE=2; fi
  if [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    MASTER_ADDR="$worker0"
  fi
  if [ "${RANK:-0}" = "0" ]; then
    host="$(hostname 2>/dev/null || true)"
    case "$host" in
      *worker-1*|*worker1*) RANK=1 ;;
      *worker-0*|*worker0*) RANK=0 ;;
    esac
  fi
fi
if [ "$WORLD_SIZE" != "2" ]; then
  echo "ERROR: WORLD_SIZE=$WORLD_SIZE，必须是 2。请把两个任务放在同一个双实例分布式任务中，" >&2
  echo "       或确认平台给启动脚本注入 WORLD_SIZE=2/RANK=0,1。" >&2
  exit 1
fi
if [ "$RANK" != "0" ] && [ "$RANK" != "1" ]; then
  echo "ERROR: RANK=$RANK，必须是 0 或 1。" >&2
  exit 1
fi
if [ -z "$MASTER_ADDR" ]; then
  echo "ERROR: MASTER_ADDR 为空。必须使用平台注入的 rank0 内网地址，不能填 127.0.0.1。" >&2
  exit 1
fi
echo "=== WORLD_SIZE=$WORLD_SIZE RANK=$RANK MASTER=$MASTER_ADDR:$MASTER_PORT ==="
echo "=== 两个任务通过 JAX rendezvous 同步；rank0 先起、rank1 后起即可 ==="

# ── 4. 依赖和 8 卡自检 ──────────────────────────────────────────────────
# 上传目录带 wheels 时，仅在版本不符时安装，不改动预装 JAX/NumPy。
for wdir in \
    "$(dirname "$(dirname "$REPO")")/qqt_upload/wheels" \
    /root/private_data/qqt_upload/wheels \
    /mnt/data/qqt_upload/wheels; do
  if [ -d "$wdir" ]; then WHEELS="$wdir"; break; fi
done
if ! python3 -c 'import optax, chex; assert optax.__version__ == "0.2.8"; assert chex.__version__ == "0.1.92"' 2>/dev/null; then
  if [ -n "${WHEELS:-}" ]; then
    pip install --no-index --find-links="$WHEELS" --no-deps --force-reinstall \
      optax chex dm-tree toolz wrapt etils typing_extensions absl-py attrs
  else
    echo "ERROR: optax/chex 版本不匹配，且找不到上传的 wheels/。" >&2
    exit 1
  fi
fi
python3 - <<'PY'
import jax, optax, chex
n = jax.local_device_count()
print("jax", jax.__version__, "optax", optax.__version__, "chex", chex.__version__)
print("local_device_count:", n)
if n != 8:
    raise SystemExit(f"本任务应看到 8 张卡，实际 {n}")
PY

# ── 5. 训练参数 ─────────────────────────────────────────────────────────
# 双机 8 卡：32768 全局 env，16.78M 步/iter；15500 iter ≈ 260B。
# 平台两个任务分别执行同一命令，WORLD_SIZE/RANK 决定各自副本。
ITERS="${ITERS:-15500}"
NUM_ENVS="${NUM_ENVS:-32768}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32768}"
LSGD_K="${LSGD_K:-0}"
LSGD_MODE="${LSGD_MODE:-param}"
LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-empty=0.05,功夫=0.1,比武=0.15}"
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.5}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-30000000000}"
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.01}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-30000000000}"
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.05}"
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"

# 两个任务共享网络权重，但各账号各自保存本地 ckpt；本轮默认从头训。
# 如平台重启后需要接续：在启动任务环境中设置 FRESH_FLAG=""。
FRESH_FLAG="${FRESH_FLAG:---fresh}"
CKPT_DIR="${CKPT_DIR:-$REPO/ckpt}"
CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-$REPO/ckpt_local}"
CKPT_EVERY="${CKPT_EVERY:-30}"
CKPT_LOCAL_EVERY="${CKPT_LOCAL_EVERY:-30}"
mkdir -p "$CKPT_DIR" "$CKPT_LOCAL_DIR" "$REPO/logs"
LOG="$REPO/logs/multicard_$(date +%Y%m%d_%H%M%S)_r${RANK}.log"

# ── 6. 启动：rank0 先起会在 rendezvous 阻塞，rank1 后起自动汇合 ──────────
set -o pipefail
printf '=== TRAIN rank=%s/%s iters=%s envs=%s steps=%s mb=%s ===\n' \
  "$RANK" "$WORLD_SIZE" "$ITERS" "$NUM_ENVS" "$NUM_STEPS" "$MINIBATCH"
printf '=== weights=%s crate=%s/%s explore=%s/%s brick=%s anneal_k=%s fresh=%s ===\n' \
  "$LEVEL_WEIGHTS" "$CRATE_REWARD_COEF" "$CRATE_REWARD_ANNEAL" \
  "$EXPLORE_REWARD_COEF" "$EXPLORE_REWARD_ANNEAL" "$BRICK_REWARD_COEF" \
  "$REWARD_ANNEAL_K" "$FRESH_FLAG"

python3 -m jax_bomb.train_real \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --num-envs "$NUM_ENVS" --num-steps "$NUM_STEPS" \
  --minibatch "$MINIBATCH" --epochs 2 --iters "$ITERS" \
  --levels "$REPO/levels.json" --level-weights "$LEVEL_WEIGHTS" \
  --crate-reward-coef "$CRATE_REWARD_COEF" \
  --crate-reward-anneal-steps "$CRATE_REWARD_ANNEAL" \
  --explore-reward-coef "$EXPLORE_REWARD_COEF" \
  --explore-reward-anneal-steps "$EXPLORE_REWARD_ANNEAL" \
  --brick-reward-coef "$BRICK_REWARD_COEF" \
  --reward-anneal-k "$REWARD_ANNEAL_K" \
  --lsgd-k "$LSGD_K" --lsgd-mode "$LSGD_MODE" \
  $FRESH_FLAG 2>&1 | tee "$LOG"

echo "=== 训练结束 rank=$RANK；日志=$LOG ==="
