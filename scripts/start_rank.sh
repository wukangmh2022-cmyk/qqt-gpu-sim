#!/bin/bash
source /opt/dtk/env.sh 2>/dev/null || true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* /public/software/mpi/*/lib/libmpi.so*; do
  if [ -f "$mpi" ]; then
    export LD_PRELOAD="$mpi"
    break
  fi
done

PATCH="${PATCH:-4}"
ADV_TOP_FRAC="${ADV_TOP_FRAC:-0.25}"
EMA_DECAY="${EMA_DECAY:-0.999}"
NUM_ENVS="${NUM_ENVS:-32760}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32760}"
ITERS="${ITERS:-34}"
LSGD_K="${LSGD_K:-256}"
LSGD_MODE="${LSGD_MODE:-param}"
FRESH_FLAG="${FRESH_FLAG---fresh}"
CKPT_DIR="${CKPT_DIR:-ckpt}"
CKPT_EVERY="${CKPT_EVERY:-10}"
CKPT_LOCAL_DIR="${CKPT_LOCAL_DIR:-ckpt_local}"
CKPT_LOCAL_EVERY="${CKPT_LOCAL_EVERY:-5}"

exec python3 -u -m jax_bomb.train_real \
  --arch transformer \
  --embed 392 \
  --depth 4 \
  --patch $PATCH \
  --heads 4 \
  --ff-factor 4 \
  --adv-top-frac $ADV_TOP_FRAC \
  --ema-decay $EMA_DECAY \
  --num-envs $NUM_ENVS \
  --num-steps $NUM_STEPS \
  --minibatch $MINIBATCH \
  --epochs 2 \
  --iters $ITERS \
  --lsgd-k $LSGD_K \
  --lsgd-mode $LSGD_MODE \
  --checkpoint \
  --ckpt-dir "$CKPT_DIR" \
  --ckpt-every "$CKPT_EVERY" \
  --ckpt-local-dir "$CKPT_LOCAL_DIR" \
  --ckpt-local-every "$CKPT_LOCAL_EVERY" \
  --curriculum-json web/assets/maps/curriculum.json \
  $FRESH_FLAG
