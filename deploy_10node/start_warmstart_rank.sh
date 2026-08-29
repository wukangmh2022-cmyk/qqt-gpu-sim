#!/bin/bash
# Runs one already-validated warm-start rank from its isolated run directory.

# DTK's environment script references optional variables, so load it before
# enabling nounset for the rest of this wrapper.
source /opt/dtk/env.sh 2>/dev/null || true
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$RUN_DIR/run_manifest.env"
source "$RUN_DIR/rank.env"
source "$RUN_DIR/runtime.env"

[ "${RUN_STATE:?missing RUN_STATE}" = "RUNNING" ] || {
  echo "RUN_STATE must be RUNNING" >&2
  exit 1
}
[ "${STOP_AT_EPOCH:?missing STOP_AT_EPOCH}" -gt "$(date +%s)" ] || {
  echo "stop deadline has already passed" >&2
  exit 1
}

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD="$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)"
export WORLD_SIZE RANK MASTER_ADDR MASTER_PORT
export LSGD_K LSGD_MODE
export CKPT_DIR="$RUN_DIR/checkpoint_state"
export CKPT_EVERY=30
export CKPT_LOCAL_DIR="$RUN_DIR/checkpoint_local"
export CKPT_LOCAL_EVERY=30

if [ -s "$RUN_DIR/train.pid" ] && kill -0 "$(cat "$RUN_DIR/train.pid")" 2>/dev/null; then
  echo "a training process is already live for this isolated run" >&2
  exit 1
fi

cd "$RUN_DIR"
export PYTHONPATH="$RUN_DIR/pydeps:${PYTHONPATH:-}"
python3 -m jax_bomb.train_real \
  --arch "$ARCH" --embed "$EMBED" --depth "$DEPTH" --patch "$PATCH" \
  --heads "$HEADS" --ff-factor "$FF_FACTOR" \
  --num-envs "$NUM_ENVS" --num-steps "$NUM_STEPS" --minibatch "$MINIBATCH" \
  --epochs "$EPOCHS" --iters "$ITERS" --lsgd-k "$LSGD_K" --lsgd-mode "$LSGD_MODE" \
  --levels levels.json --level-weights "$LEVEL_WEIGHTS" \
  --crate-reward-coef "$CRATE_REWARD_COEF" \
  --crate-reward-anneal-steps "$CRATE_REWARD_ANNEAL" \
  --explore-reward-coef "$EXPLORE_REWARD_COEF" \
  --explore-reward-anneal-steps "$EXPLORE_REWARD_ANNEAL" \
  --brick-reward-coef "$BRICK_REWARD_COEF" \
  --reward-anneal-k "$REWARD_ANNEAL_K" \
  --reward-anneal-step-offset "$REWARD_ANNEAL_STEP_OFFSET" \
  > "$RUN_DIR/train.log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$RUN_DIR/train.pid"

(
  delay=$((STOP_AT_EPOCH - $(date +%s)))
  if [ "$delay" -gt 0 ]; then
    sleep "$delay"
  fi
  if kill -0 "$train_pid" 2>/dev/null; then
    printf '[timer] deadline reached; SIGTERM pid=%s\n' "$train_pid" >> "$RUN_DIR/train.log"
    kill -TERM "$train_pid"
  fi
) > "$RUN_DIR/timer.log" 2>&1 &
timer_pid=$!
printf '%s\n' "$timer_pid" > "$RUN_DIR/timer.pid"

set +e
wait "$train_pid"
status=$?
set -e
kill "$timer_pid" 2>/dev/null || true
wait "$timer_pid" 2>/dev/null || true
exit "$status"
