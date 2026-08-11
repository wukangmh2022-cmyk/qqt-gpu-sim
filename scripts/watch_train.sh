#!/bin/bash
# 从根目录 .env 读取 DCU 连接（.env 不入库，见 .gitignore）
_ENV_F="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$_ENV_F" ] && source "$_ENV_F"
# DCU 训练健康监控（常驻，异常自动恢复）
# 用法：nohup bash scripts/watch_train.sh > /tmp/watch_train.log 2>&1 &
# 每 CHECK_INTERVAL 秒检查一次：进程存活 + csv 推进 + 日志无 error。
# 进程死了且快照可用 → 自动 resume 重启（与 relaunch_1023m.sh 同命令）。
# 连续 2 次重启 60s 内又死 → 停止并报告，需要人工介入。
# 2026-08-10 建立：训练 01:34 崩溃后无人值守 6h 才被发现，立此监控。

HOST="root@${DCU_HOST:-ssh.zzai.scnet.cn}"
PORT="${DCU_PORT:-10630}"
PASS="${DCU_PASS:-YOUR_PASSWORD}"
CSV="/root/ckpt/train_1023m_finetune.csv"
LOG="/root/train_1023m_finetune.log"
CKPT="/root/private_data/duel_course.pt"
CHECK_INTERVAL=1200          # 20 分钟
RESTART_WAIT=60              # 重启后确认存活窗口
MAX_CONSEC_FAIL=2            # 连续重启失败上限

# 与 relaunch_1023m.sh 一致的重启命令
RELAUNCH_CMD='cd /root && source /opt/dtk-26.04/env.sh >/dev/null 2>&1 && nohup python -m train.train --backend torch --device cuda --arch mlp --single-stage --map-mode corridor --open-fraction 0.5 --ring-fraction 0 --hazard-fraction 0 --num-envs 5632 --total-steps 1_200_000_000 --rollout-steps 128 --minibatches 4 --warmup-steps 150_000_000 --fixed-opp-prob 0.4 --bot-opponents greedy,astar,hunter --fixed-bots astar,hunter --explore-anneal --bc-data recordings/ --bc-coef 0.3 --bc-batch 256 --bc-every 1 --fixed-ckpt rw8=private_data/duel_rw8.pt --fixed-ckpt 5x2=private_data/duel_5x2.pt --fixed-ckpt 5x3=private_data/duel_5x3.pt --fixed-ckpt cnn=private_data/duel_cnn.pt --snapshot-every 20 --time-budget 43200 --seed 0 --oversample-dying 3 --resume private_data/duel_course.pt --ckpt private_data/duel_course.pt --log-csv ckpt/train_1023m_finetune.csv --combo-reward 0.10 --timeout-draw --lr-final 1e-4 > train_1023m_finetune.log 2>&1 < /dev/null &'

ssh_run() { sshpass -p "$PASS" ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=15 "$HOST" -p "$PORT" "$1" 2>&1; }

last_step=""
consec_fail=0
log_ts() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log_ts "监控启动，每 ${CHECK_INTERVAL}s 检查一次"
while true; do
  sleep "$CHECK_INTERVAL"

  alive=$(ssh_run 'ps aux | grep "train.train" | grep -v grep | wc -l' | tr -d ' ')
  if [ -z "$alive" ]; then
    log_ts "SSH 连接失败（跳过本轮）"
    continue
  fi

  errs=$(ssh_run "grep -cE '\[error\]|Traceback|RuntimeError' $LOG" | tr -d ' ')
  step=$(ssh_run "tail -1 $CSV | cut -d, -f1" | tr -d ' ')
  sps=$(ssh_run "tail -1 $CSV | cut -d, -f11" | tr -d ' ')

  if [ "$alive" != "0" ] && [ "${errs:-0}" = "0" ]; then
    # 健康
    progress=""
    if [ -n "$last_step" ] && [ -n "$step" ] && [ "$step" = "$last_step" ]; then
      progress=" ⚠CSV未推进"
    fi
    log_ts "OK alive=$alive step=$step sps=$sps$progress"
    last_step="$step"
    consec_fail=0
    continue
  fi

  # 异常：进程死 或 日志有 error
  err_txt=$(ssh_run "grep -E '\[error\]|RuntimeError' $LOG | tail -3")
  ckpt_time=$(ssh_run "stat -c %y $CKPT 2>/dev/null" )
  log_ts "⚠ 异常：alive=$alive errs=${errs:-?} step=$step | 最近error: $err_txt | 快照: $ckpt_time"

  # 尝试恢复：进程死了才重启（进程活着但有 error 日志时不自动动，可能是历史 error）
  if [ "$alive" = "0" ]; then
    log_ts "进程已死，尝试 resume 重启…"
    ssh_run "$RELAUNCH_CMD"
    sleep "$RESTART_WAIT"
    alive2=$(ssh_run 'ps aux | grep "train.train" | grep -v grep | wc -l' | tr -d ' ')
    if [ -n "$alive2" ] && [ "$alive2" != "0" ]; then
      log_ts "重启成功（进程数=$alive2），已恢复。"
      consec_fail=0
    else
      consec_fail=$((consec_fail + 1))
      log_ts "重启后 $RESTART_WAIT s 内未存活（第 ${consec_fail} 次失败）"
      if [ "$consec_fail" -ge "$MAX_CONSEC_FAIL" ]; then
        log_ts "连续 $MAX_CONSEC_FAIL 次重启失败，停止自动恢复，需要人工介入！"
        log_ts "最后 30 行日志："
        ssh_run "tail -30 $LOG"
        exit 1
      fi
    fi
  else
    log_ts "进程活着但有 error 日志（可能历史），不自动动，下轮复查。"
  fi
  last_step="$step"
done
