#!/bin/bash
# Safe isolated 10-node x 2-device parameter warm-start launcher.
#
# The stage phase creates a unique run directory and verifies every rank without
# starting training. The start phase accepts only that immutable local metadata,
# revalidates the staged run, then starts its ranks with a fresh 48-hour deadline.
# Neither phase deletes, overwrites, or signals an existing remote training run.
set -euo pipefail

usage() {
  printf '%s\n' "usage: $0 <nodes.txt> --stage-only [--deploy]"
  printf '%s\n' "       $0 <nodes.txt> --start <deploy_10node/.run_*.env>"
}

[ "$#" -ge 2 ] || { usage >&2; exit 2; }
NODES_FILE="$1"
shift
MODE=""
DO_DEPLOY=0
META_FILE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage-only)
      [ -z "$MODE" ] || { usage >&2; exit 2; }
      MODE="stage"
      ;;
    --start)
      [ -z "$MODE" ] && [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      MODE="start"
      META_FILE="$2"
      shift
      ;;
    --deploy)
      DO_DEPLOY=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done
[ -n "$MODE" ] || { usage >&2; exit 2; }
[ -f "$NODES_FILE" ] || { printf 'ERROR: nodes file not found: %s\n' "$NODES_FILE" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORLD_SIZE=10
MASTER_PORT=29503
SOURCE_GLOBAL_STEPS=7457472512
RUN_SECONDS="${RUN_SECONDS:-172800}"
ITERS="${ITERS:-15500}"
NUM_ENVS="${NUM_ENVS:-32768}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32768}"
EPOCHS=2
ARCH=transformer
EMBED=392
DEPTH=4
PATCH=4
HEADS=4
FF_FACTOR=4
LSGD_K="${LSGD_K:-256}"
LSGD_MODE="${LSGD_MODE:-param}"
LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-empty=0.05,功夫=0.1,比武=0.15}"
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.5}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-30000000000}"
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.01}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-30000000000}"
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.05}"
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
REWARD_ANNEAL_STEP_OFFSET="$SOURCE_GLOBAL_STEPS"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [ "$MODE" = "stage" ]; then
  [ -f "$ROOT/ckpt_local/params_it00000889.pkl" ] || fail "missing params_it00000889.pkl"
  [ -f "$ROOT/web/assets/maps/levels.json" ] || fail "missing levels.json"
  [ -f "$ROOT/deploy_10node/bootstrap_warmstart_889.py" ] || fail "missing warm-start helper"
  [ -f "$ROOT/deploy_10node/start_warmstart_rank.sh" ] || fail "missing rank launcher"
  [ -d "$ROOT/jax_bomb" ] || fail "missing jax_bomb package"
  if rg -q '^\s*bush\[i\]\s*=\s*bush_raw\s*$' "$ROOT/jax_bomb/levels.py"; then
    fail "jax_bomb/levels.py references undefined bush_raw"
  fi
else
  [ -f "$META_FILE" ] || fail "run metadata not found: $META_FILE"
  # This file is created by this launcher with mode 600 and contains no password.
  # shellcheck disable=SC1090
  source "$META_FILE"
  [ "${MANIFEST_VERSION:-}" = "1" ] || fail "unrecognized run metadata"
  [ "${WORLD_SIZE:-}" = "10" ] || fail "metadata topology is not 10 nodes"
  [ "${RUN_SECONDS:-}" -gt 0 ] || fail "metadata has invalid duration"
  [ "${REWARD_ANNEAL_STEP_OFFSET:-}" = "$SOURCE_GLOBAL_STEPS" ] || \
    fail "metadata does not preserve params_it00000889 fixed anneal progress"
fi

command -v expect >/dev/null || fail "expect is required for password SSH"

N_PORT=()
N_HOST=()
N_PASS=()
PENDING_PORT=""
PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  if [ -n "$PENDING_PORT" ]; then
    N_PORT+=("$PENDING_PORT")
    N_HOST+=("$PENDING_HOST")
    N_PASS+=("$line")
    PENDING_PORT=""
    PENDING_HOST=""
    continue
  fi
  if [[ "$line" =~ ^ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+root@([^[:space:]]+)$ ]]; then
    PENDING_PORT="${BASH_REMATCH[1]}"
    PENDING_HOST="${BASH_REMATCH[2]}"
    continue
  fi
  read -r port host password extra <<< "$line"
  [ -n "${port:-}" ] && [ -n "${host:-}" ] && [ -n "${password:-}" ] || fail "invalid node row"
  [ -z "${extra:-}" ] || fail "node rows must be: port host password"
  N_PORT+=("$port")
  N_HOST+=("$host")
  N_PASS+=("$password")
done < "$NODES_FILE"
[ -z "$PENDING_PORT" ] || fail "missing password after SSH command"
[ "${#N_PORT[@]}" -eq "$WORLD_SIZE" ] || fail "expected 10 nodes, got ${#N_PORT[@]}"

endpoint_digest() {
  local rank
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    printf '%s:%s\n' "${N_HOST[$rank]}" "${N_PORT[$rank]}"
  done | shasum -a 256 | awk '{print $1}'
}
ENDPOINTS_SHA256="$(endpoint_digest)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/qqt_warmstart.XXXXXX")"
chmod 700 "$WORK"
STARTED_RANKS=()
LOCKED_RANKS=()
cleanup() {
  if [ "${CLEANUP_STARTED_RANKS:-0}" = '1' ] && [ "${#STARTED_RANKS[@]}" -gt 0 ]; then
    stop_started_ranks || true
  fi
  if [ "${CLEANUP_RUNNING_STATE:-0}" = '1' ]; then
    for cleanup_rank in $(seq 0 $((WORLD_SIZE - 1))); do
      remote_cmd "$cleanup_rank" "cd '$REMOTE_ROOT'; rm -f RUNNING; touch START_FAILED" >/dev/null 2>&1 || true
    done
  fi
  if [ "${CLEANUP_LOCKS:-0}" = '1' ] && [ "${#LOCKED_RANKS[@]}" -gt 0 ]; then
    release_locks || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  printf '%s' "${N_PASS[$rank]}" > "$WORK/pass_$rank"
  chmod 600 "$WORK/pass_$rank"
  printf '%s\n' '#!/usr/bin/expect -f' 'set timeout 180' \
    'set command [lindex $argv 0]' \
    "set handle [open [file join [file dirname [info script]] pass_$rank] r]" \
    'set password [read $handle]' 'close $handle' \
    "spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p ${N_PORT[$rank]} root@${N_HOST[$rank]} \$command" \
    'expect {' \
    '  "password:" { send -- "$password\r"; exp_continue }' \
    '  "yes/no" { send -- "yes\r"; exp_continue }' \
    '  eof {}' \
    '  timeout { puts stderr "SSH_TIMEOUT"; exit 124 }' \
    '}' \
    'set result [wait]' \
    'exit [lindex $result 3]' > "$WORK/ssh_$rank"
  printf '%s\n' '#!/usr/bin/expect -f' 'set timeout 600' \
    'set source [lindex $argv 0]' 'set destination [lindex $argv 1]' \
    "set handle [open [file join [file dirname [info script]] pass_$rank] r]" \
    'set password [read $handle]' 'close $handle' \
    "spawn scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -P ${N_PORT[$rank]} \$source root@${N_HOST[$rank]}:\$destination" \
    'expect {' \
    '  "password:" { send -- "$password\r"; exp_continue }' \
    '  "yes/no" { send -- "yes\r"; exp_continue }' \
    '  eof {}' \
    '  timeout { puts stderr "SCP_TIMEOUT"; exit 124 }' \
    '}' \
    'set result [wait]' \
    'exit [lindex $result 3]' > "$WORK/scp_$rank"
  chmod 700 "$WORK/ssh_$rank" "$WORK/scp_$rank"
done

remote_cmd() {
  local rank="$1"
  shift
  "$WORK/ssh_$rank" "$*"
}

remote_copy() {
  local rank="$1"
  "$WORK/scp_$rank" "$2" "$3"
}

preflight() {
  local rank check reason
  local -a pids=()
  mkdir -p "$WORK/preflight"
  printf '%s\n' '=== preflight: SSH, Python imports, exactly 2 local devices ==='
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    (
      remote_cmd "$rank" "source /opt/dtk/env.sh 2>/dev/null || true; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); python3 -c 'import jax; print(\"JAX_PREFLIGHT\", jax.local_device_count())'; python3 -c 'import optax' 2>/dev/null || python3 -m pip install --user --quiet --no-deps optax==0.2.8; python3 -c 'import jax, optax; print(\"PREFLIGHT\", jax.local_device_count(), optax.__version__)'" \
        > "$WORK/preflight/$rank.out" 2>&1
    ) &
    pids[$rank]=$!
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    wait "${pids[$rank]}" || true
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    check="$(tr '\n' ' ' < "$WORK/preflight/$rank.out")"
    if ! printf '%s' "$check" | grep -q 'PREFLIGHT 2'; then
      case "$check" in
        *'timed out during banner exchange'*) reason='SSH banner timeout' ;;
        *'Connection timed out'*) reason='SSH connect timeout' ;;
        *'Permission denied'*) reason='SSH authentication failed' ;;
        *) reason='JAX/DTK preflight failed' ;;
      esac
      detail="$(printf '%s' "$check" | tr -cd '[:print:] ' | tail -c 320)"
      fail "rank $rank preflight failed: $reason${detail:+; detail=$detail}"
    fi
    printf '  rank %s: devices=2\n' "$rank"
  done
}

write_manifest() {
  local path="$1"
  {
    printf 'MANIFEST_VERSION=%q\n' '1'
    printf 'RUN_TAG=%q\n' "$RUN_TAG"
    printf 'REMOTE_ROOT=%q\n' "$REMOTE_ROOT"
    printf 'WORLD_SIZE=%q\n' "$WORLD_SIZE"
    printf 'MASTER_ADDR=%q\n' "$MASTER_ADDR"
    printf 'MASTER_PORT=%q\n' "$MASTER_PORT"
    printf 'PAYLOAD_SHA256=%q\n' "$PAYLOAD_SHA256"
    printf 'ENDPOINTS_SHA256=%q\n' "$ENDPOINTS_SHA256"
    printf 'RUN_SECONDS=%q\n' "$RUN_SECONDS"
    printf 'ITERS=%q\n' "$ITERS"
    printf 'ARCH=%q\n' "$ARCH"
    printf 'EMBED=%q\n' "$EMBED"
    printf 'DEPTH=%q\n' "$DEPTH"
    printf 'PATCH=%q\n' "$PATCH"
    printf 'HEADS=%q\n' "$HEADS"
    printf 'FF_FACTOR=%q\n' "$FF_FACTOR"
    printf 'NUM_ENVS=%q\n' "$NUM_ENVS"
    printf 'NUM_STEPS=%q\n' "$NUM_STEPS"
    printf 'MINIBATCH=%q\n' "$MINIBATCH"
    printf 'EPOCHS=%q\n' "$EPOCHS"
    printf 'LSGD_K=%q\n' "$LSGD_K"
    printf 'LSGD_MODE=%q\n' "$LSGD_MODE"
    printf 'LEVEL_WEIGHTS=%q\n' "$LEVEL_WEIGHTS"
    printf 'CRATE_REWARD_COEF=%q\n' "$CRATE_REWARD_COEF"
    printf 'CRATE_REWARD_ANNEAL=%q\n' "$CRATE_REWARD_ANNEAL"
    printf 'EXPLORE_REWARD_COEF=%q\n' "$EXPLORE_REWARD_COEF"
    printf 'EXPLORE_REWARD_ANNEAL=%q\n' "$EXPLORE_REWARD_ANNEAL"
    printf 'BRICK_REWARD_COEF=%q\n' "$BRICK_REWARD_COEF"
    printf 'REWARD_ANNEAL_K=%q\n' "$REWARD_ANNEAL_K"
    printf 'REWARD_ANNEAL_STEP_OFFSET=%q\n' "$REWARD_ANNEAL_STEP_OFFSET"
    printf 'WARMSTART_SOURCE=%q\n' 'params_it00000889.pkl'
    printf 'WARMSTART_SOURCE_ITERATION=%q\n' '889'
    printf 'WARMSTART_STATE=%q\n' 'fresh_optimizer_environment_rng'
  } > "$path"
}

write_metadata() {
  local path="$1"
  write_manifest "$path"
  printf 'LOCAL_METADATA=%q\n' "$path" >> "$path"
  chmod 600 "$path"
}

stage_run() {
  local rank ip result digest check
  local -a ips=() pids=() digests=()
  RUN_TAG="warmstart_889_48h_$(date +%Y%m%d_%H%M%S)"
  REMOTE_ROOT="/root/private_data/qqt-runs/$RUN_TAG"

  preflight
  if [ "$DO_DEPLOY" = '1' ]; then
    printf '%s\n' '=== optional environment setup check ==='
    for rank in $(seq 0 $((WORLD_SIZE - 1))); do
      remote_cmd "$rank" 'test -f /opt/dtk/env.sh' >/dev/null || \
        fail "rank $rank is missing /opt/dtk/env.sh; environment setup is required"
    done
  fi

  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    ip="$(remote_cmd "$rank" "hostname -I" 2>/dev/null | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1 || true)"
    [ -n "$ip" ] || fail "rank $rank has no 172.31.x.x coordinator address"
    ips+=("$ip")
    printf '  rank %s %s -> %s\n' "$rank" "${N_HOST[$rank]}" "$ip"
  done
  MASTER_ADDR="${ips[0]}"

  printf '%s\n' '=== preflight: workers can reach rank0 rendezvous endpoint ==='
  probe_pid="$(remote_cmd 0 "nohup python3 -m http.server $MASTER_PORT --bind $MASTER_ADDR >/tmp/qqt_probe_$RUN_TAG.log 2>&1 & echo \$!" 2>/dev/null | tail -1)"
  [ -n "$probe_pid" ] || fail 'cannot start temporary rank0 reachability probe'
  for rank in $(seq 1 $((WORLD_SIZE - 1))); do
    remote_cmd "$rank" "python3 -c 'import socket; s=socket.create_connection((\"$MASTER_ADDR\", $MASTER_PORT), 5); s.close(); print(\"TCP_OK\")'" \
      2>/dev/null | grep -q TCP_OK || {
        remote_cmd 0 "kill $probe_pid 2>/dev/null || true" >/dev/null 2>&1 || true
        fail "rank $rank cannot reach rank0 coordinator"
      }
  done
  remote_cmd 0 "kill $probe_pid 2>/dev/null || true" >/dev/null 2>&1 || true

  STAGE="$WORK/stage"
  mkdir -p "$STAGE"
  cp -R "$ROOT/jax_bomb" "$STAGE/jax_bomb"
  cp "$ROOT/web/assets/maps/levels.json" "$STAGE/levels.json"
  cp "$ROOT/deploy_10node/bootstrap_warmstart_889.py" "$STAGE/bootstrap_warmstart_889.py"
  cp "$ROOT/deploy_10node/start_warmstart_rank.sh" "$STAGE/start_warmstart_rank.sh"
  mkdir -p "$STAGE/pydeps"
  cp -R "$ROOT/.venv/lib/python3.12/site-packages/optax" "$STAGE/pydeps/optax"
  cp -R "$ROOT/.venv/lib/python3.12/site-packages/absl" "$STAGE/pydeps/absl"
  cp "$ROOT/ckpt_local/params_it00000889.pkl" "$STAGE/params_it00000889.pkl"
  (
    cd "$STAGE"
    tar czf "$WORK/run_payload.tgz" jax_bomb levels.json bootstrap_warmstart_889.py \
      start_warmstart_rank.sh pydeps params_it00000889.pkl
  )
  PAYLOAD_SHA256="$(shasum -a 256 "$WORK/run_payload.tgz" | awk '{print $1}')"
  write_manifest "$WORK/run_manifest.env"
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    {
      printf 'WORLD_SIZE=%q\n' "$WORLD_SIZE"
      printf 'RANK=%q\n' "$rank"
      printf 'MASTER_ADDR=%q\n' "$MASTER_ADDR"
      printf 'MASTER_PORT=%q\n' "$MASTER_PORT"
    } > "$WORK/rank_$rank.env"
  done

  printf '%s\n' '=== create isolated run directories and upload immutable payload ==='
  mkdir -p "$WORK/upload"
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    (
      remote_cmd "$rank" "umask 077; test ! -e '$REMOTE_ROOT'; mkdir -p '$REMOTE_ROOT'" &&
      remote_copy "$rank" "$WORK/run_payload.tgz" "$REMOTE_ROOT/run_payload.tgz" &&
      remote_copy "$rank" "$WORK/run_manifest.env" "$REMOTE_ROOT/run_manifest.env" &&
      remote_copy "$rank" "$WORK/rank_$rank.env" "$REMOTE_ROOT/rank.env" &&
      remote_cmd "$rank" "cd '$REMOTE_ROOT'; test \"\$(shasum -a 256 run_payload.tgz | awk '{print \$1}')\" = '$PAYLOAD_SHA256'; tar xzf run_payload.tgz; rm -f run_payload.tgz; chmod 700 bootstrap_warmstart_889.py start_warmstart_rank.sh; printf '%s\\n' 'PAYLOAD_STAGED' > PAYLOAD_STAGED" &&
      printf '%s\n' UPLOAD_OK
    ) > "$WORK/upload/$rank.out" 2>&1 &
    pids[$rank]=$!
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    wait "${pids[$rank]}" || true
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    check="$(tr '\n' ' ' < "$WORK/upload/$rank.out")"
    printf '%s' "$check" | grep -q UPLOAD_OK || fail "rank $rank upload failed"
    printf '  rank %s: payload verified\n' "$rank"
  done

  printf '%s\n' '=== initialize rank-local full warm-start checkpoints ==='
  mkdir -p "$WORK/bootstrap"
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    (
      remote_cmd "$rank" "cd '$REMOTE_ROOT'; source /opt/dtk/env.sh 2>/dev/null || true; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export PYTHONPATH='$REMOTE_ROOT/pydeps' && python3 bootstrap_warmstart_889.py --params params_it00000889.pkl --ckpt-dir checkpoint_state --rank $rank --world-size $WORLD_SIZE --levels levels.json --level-weights '$LEVEL_WEIGHTS' --crate-reward-coef $CRATE_REWARD_COEF --crate-reward-anneal-steps $CRATE_REWARD_ANNEAL --explore-reward-coef $EXPLORE_REWARD_COEF --explore-reward-anneal-steps $EXPLORE_REWARD_ANNEAL --brick-reward-coef $BRICK_REWARD_COEF --reward-anneal-k $REWARD_ANNEAL_K --reward-anneal-step-offset $REWARD_ANNEAL_STEP_OFFSET --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE" \
        > "$WORK/bootstrap/$rank.out" 2>&1
    ) &
    pids[$rank]=$!
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    wait "${pids[$rank]}" || true
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    result="$(tr '\n' ' ' < "$WORK/bootstrap/$rank.out")"
    printf '%s' "$result" | grep -q "WARMSTART_OK rank=$rank" || \
      fail "rank $rank warm-start checkpoint creation failed; detail=$(printf '%s' "$result" | tail -c 500)"
    remote_cmd "$rank" "python3 -c 'import json; p=json.load(open(\"$REMOTE_ROOT/checkpoint_state/warmstart_r$rank.json\")); assert p[\"source_global_steps\"] == $REWARD_ANNEAL_STEP_OFFSET; assert p[\"checkpoint_iteration\"] == 0'" >/dev/null || \
      fail "rank $rank warm-start descriptor validation failed"
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    remote_cmd "$rank" "test -s '$REMOTE_ROOT/checkpoint_state/ckpt_00000000_r$rank.pkl'; touch '$REMOTE_ROOT/STAGED'" >/dev/null || \
      fail "rank $rank staged checkpoint verification failed"
  done

  META_LOCAL="$ROOT/deploy_10node/.run_${RUN_TAG}.env"
  write_metadata "$META_LOCAL"
  printf '%s\n' '=== staging completed; no training process was started ==='
  printf 'run: %s\nremote root: %s\nmetadata: %s\n' "$RUN_TAG" "$REMOTE_ROOT" "$META_LOCAL"
}

release_locks() {
  local rank
  for rank in "${LOCKED_RANKS[@]}"; do
    remote_cmd "$rank" "rmdir '$REMOTE_ROOT/.start_lock' 2>/dev/null || true" >/dev/null 2>&1 || true
  done
}

stop_started_ranks() {
  local rank
  for rank in "${STARTED_RANKS[@]}"; do
    remote_cmd "$rank" "cd '$REMOTE_ROOT'; for f in train.pid timer.pid; do [ -s \"\$f\" ] && kill -TERM \"\$(cat \"\$f\")\" 2>/dev/null || true; done" >/dev/null 2>&1 || true
  done
}

start_run() {
  local rank digest check STOP_AT
  local -a digests=()
  printf '%s\n' "=== validate staged run: $RUN_TAG ==="
  preflight
  [ "$(endpoint_digest)" = "$ENDPOINTS_SHA256" ] || fail "nodes file endpoints do not match staged metadata"

  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    remote_cmd "$rank" "set -eu; test -f '$REMOTE_ROOT/STAGED'; test -s '$REMOTE_ROOT/checkpoint_state/ckpt_00000000_r$rank.pkl'; . '$REMOTE_ROOT/run_manifest.env'; [ \"\$RUN_TAG\" = '$RUN_TAG' ]; [ \"\$PAYLOAD_SHA256\" = '$PAYLOAD_SHA256' ]; python3 -c 'import json; p=json.load(open(\"$REMOTE_ROOT/checkpoint_state/warmstart_r$rank.json\")); assert p[\"source_global_steps\"] == $REWARD_ANNEAL_STEP_OFFSET; assert p[\"checkpoint_iteration\"] == 0'" >/dev/null || \
      fail "rank $rank staged run validation failed"
  done

  CLEANUP_LOCKS=1
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    remote_cmd "$rank" "cd '$REMOTE_ROOT'; test ! -f RUNNING; if [ -s train.pid ] && kill -0 \"\$(cat train.pid)\" 2>/dev/null; then exit 1; fi; mkdir .start_lock" >/dev/null || {
      release_locks
      fail "rank $rank cannot acquire its isolated-run start lock"
    }
    LOCKED_RANKS+=("$rank")
  done

  STOP_AT=$(( $(date +%s) + RUN_SECONDS ))
  {
    printf 'RUN_STATE=%q\n' 'RUNNING'
    printf 'STOP_AT_EPOCH=%q\n' "$STOP_AT"
  } > "$WORK/runtime.env"
  CLEANUP_RUNNING_STATE=1
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    remote_copy "$rank" "$WORK/runtime.env" "$REMOTE_ROOT/runtime.env" || {
      stop_started_ranks
      fail "rank $rank did not accept runtime metadata"
    }
  done
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    remote_cmd "$rank" "touch '$REMOTE_ROOT/RUNNING'" >/dev/null || {
      stop_started_ranks
      fail "rank $rank could not enter RUNNING state"
    }
  done

  printf '%s\n' "=== start rank 0, then ranks 1-9; deadline $(date -r "$STOP_AT" '+%Y-%m-%d %H:%M:%S') ==="
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    if ! remote_cmd "$rank" "nohup '$REMOTE_ROOT/start_warmstart_rank.sh' > '$REMOTE_ROOT/launcher.log' 2>&1 & echo RANK${rank}_STARTED" | grep -q "RANK${rank}_STARTED"; then
      stop_started_ranks
      fail "rank $rank did not accept isolated-run launch"
    fi
    STARTED_RANKS+=("$rank")
    CLEANUP_STARTED_RANKS=1
    [ "$rank" = '0' ] && sleep 5
  done

  sleep 45
  printf '%s\n' '=== post-launch health check ==='
  for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    check="$(remote_cmd "$rank" "cd '$REMOTE_ROOT'; pid=\$(cat train.pid 2>/dev/null || true); if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then echo LIVE; else echo EXITED; fi; tail -8 train.log 2>/dev/null" 2>&1 || true)"
    printf '%s\n' "--- rank $rank ---"
    printf '%s\n' "$check" | tail -9
    if ! printf '%s\n' "$check" | grep -q '^LIVE'; then
      stop_started_ranks
      fail "rank $rank training process is not live"
    fi
    if printf '%s\n' "$check" | grep -qiE 'Traceback|ERROR|Exception|WARMSTART_ERROR'; then
      stop_started_ranks
      fail "rank $rank reported an immediate error"
    fi
  done

  CLEANUP_STARTED_RANKS=0
  CLEANUP_RUNNING_STATE=0
  CLEANUP_LOCKS=0
  printf '%s\n' '=== all 10 ranks started ==='
  printf 'run: %s\nremote root: %s\ndeadline epoch: %s\n' "$RUN_TAG" "$REMOTE_ROOT" "$STOP_AT"
  printf 'checkpoints: %s/checkpoint_state and rank0 %s/checkpoint_local\n' "$REMOTE_ROOT" "$REMOTE_ROOT"
}

if [ "$MODE" = 'stage' ]; then
  stage_run
else
  start_run
fi
