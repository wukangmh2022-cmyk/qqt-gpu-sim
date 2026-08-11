#!/bin/bash
# 从根目录 .env 读取 DCU 连接（.env 不入库，见 .gitignore）
_ENV_F="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$_ENV_F" ] && source "$_ENV_F"
# 每 15 分钟定时：DCU 训练健康监控 + 空闲时挖 learner.act 的 DCU 开销
# 由 CronCreate 调度（跨会话持久）。合并 watch_train.sh 的恢复逻辑 + 交接 §8 待办 #2 的 profile。
# 逻辑：健康检查（alive/CSV 推进/error/sps）→ 异常时自动 resume 重启（连续 2 次失败停止等人工）
#       → 训练不在跑（空闲）且 learner.act profile 未做过 → 服务器跑 prof_env 挖 HIP kernel 分布。
# 2026-08-10 建立。状态在 /tmp（重启即清）；健康日志 scripts/cron_dcu_watch.log；profile 输出 scripts/profiles/。

HOST="root@${DCU_HOST:-ssh.zzai.scnet.cn}"
PORT="${DCU_PORT:-10630}"
PASS="${DCU_PASS:-YOUR_PASSWORD}"
# 2026-08-11 08:3x 起：no-BC 3B 续跑 3.5B（brick 0.15、退火 k=0.6、+CNN 敌人、hunter 降频）
CSV="/root/ckpt/train_nobc2.5B.csv"
LOG="/root/train_nobc2.5B.log"
CKPT="/root/private_data/duel_nobc2.5B.pt"
TOTAL_STEPS=3500000000        # --total-steps 上限，达到后不自动重启

BASE="/Users/pippo/operater-dev/qqt-gpu-sim"
STATE_LOG="$BASE/scripts/cron_dcu_watch.log"     # 每轮健康记录（追加）
PROF_OUT_DIR="$BASE/scripts/profiles"            # learner.act profile 输出目录
LOCK="/tmp/dcu_cron.lock"
LAST_STEP_F="/tmp/dcu_cron_last_step"
FAILS_F="/tmp/dcu_cron_fails"
PROFILED_F="/tmp/dcu_cron_learner_act_profiled"
MILESTONE=1200000000         # 1.2B 里程碑：到点把 ckpt 拷回本地备份（用户要求，一次性）
LOCAL_CKPT_DIR="$BASE/ckpt"  # 启动器(launcher.py)只扫 ckpt/ 目录，备份必须放这里
BACKUP_NAME="duel_nobc_1.2B.pt"
RESTART_WAIT=60
MAX_CONSEC_FAIL=2

# 与 relaunch_nobc3.5b.sh 一致（no-BC 3B 续跑 3.5B），fallback 直接调服务器脚本防配置漂移
RELAUNCH_CMD='cd /root && bash /root/relaunch_nobc3.5b.sh'

ssh_run() { sshpass -p "$PASS" ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=15 "$HOST" -p "$PORT" "$1" 2>&1; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATE_LOG"; }

# ---- 服务器空闲时：挖 learner.act 的 DCU 开销（交接 §8 待办 #2，只做一次）----
run_profile() {
  if [ -f "$PROFILED_F" ]; then
    log "learner.act profile 已做过（rm $PROFILED_F 可重跑），跳过"
    return
  fi
  a=$(ssh_run 'ps aux | grep "train.train" | grep -v grep | wc -l' | tr -d ' ')
  if [ -n "$a" ] && [ "$a" != "0" ]; then
    log "检测到训练进程仍在，跳过 profile"
    return
  fi
  mkdir -p "$PROF_OUT_DIR"
  ts=$(date '+%Y%m%d_%H%M%S')
  out="$PROF_OUT_DIR/learner_act_dcu_${ts}.txt"
  log "服务器空闲：跑 learner.act DCU kernel profile（prof_env --n 20000 --opp none --profile）→ $out"
  ssh_run 'cd /root && source /opt/dtk-26.04/env.sh >/dev/null 2>&1 && PYTHONPATH=/root python scripts/prof_env.py --n 20000 --ticks 30 --profile --device cuda --opp none' > "$out" 2>&1
  if [ -s "$out" ]; then
    touch "$PROFILED_F"
    log "profile 完成：$out（$(wc -l < "$out" | tr -d ' ') 行）。分析重点：分段耗时 + HIP/CUDA kernel top 表"
  else
    log "profile 输出为空（可能出错），见 $out"
  fi
}

# ---- 单例锁（mkdir 原子，macOS 无 flock）----
mkdir -p "$PROF_OUT_DIR"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[$(date '+%m-%d %H:%M:%S')] 上一轮还在跑，跳过" >> "$STATE_LOG"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ---- 健康检查 ----
alive=$(ssh_run 'ps aux | grep "train.train" | grep -v grep | wc -l' | tr -d ' ')
# SSH 错误文本（如 "Permission denied ..."）不是数字 → 按连接失败处理，防污染日志/弄崩算术
if ! [[ "$alive" =~ ^[0-9]+$ ]]; then
  log "SSH 响应异常（非数字：${alive}），疑似限流/认证失败，跳过本轮"
  exit 0
fi
errs=$(ssh_run "grep -cE '\[error\]|Traceback|RuntimeError' $LOG" | tr -d ' ')
step=$(ssh_run "tail -1 $CSV | cut -d, -f1" | tr -d ' ')
sps=$(ssh_run "tail -1 $CSV | cut -d, -f11" | tr -d ' ')

delta=""
if [ -f "$LAST_STEP_F" ]; then
  prev=$(cat "$LAST_STEP_F" 2>/dev/null)
  if [ -n "$prev" ] && [ -n "$step" ] \
     && [[ "$prev" =~ ^[0-9]+$ ]] && [[ "$step" =~ ^[0-9]+$ ]]; then
    if [ "$prev" -le "$step" ] 2>/dev/null; then
      delta=" +$((step - prev))步"
    else
      # step 回退 = tail-1 读到旧进程之后的新进程行（resume 点更早）→ 重启信号
      delta=" ⚠回退$((prev - step))步(疑似重启/resume)"
    fi
  fi
fi
[ -n "$step" ] && echo "$step" > "$LAST_STEP_F"

# 重启信号单独醒目记一笔（不进 OK 行，防被淹没）
if [ -n "$step" ] && [ -f "$LAST_STEP_F" ] && [ -n "$delta" ] \
   && [[ "$delta" == *"回退"* ]]; then
  log "⚠ step 回退检测：上次=$prev 本次=$step，疑似训练重启/被外部接管，需人工确认"
fi

# 1.2B 里程碑备份：到点把 ckpt 拷回本地（一次性，失败下轮重试）
if [ -n "$step" ] && [[ "$step" =~ ^[0-9]+$ ]] \
   && [ "$step" -ge "$MILESTONE" ] && [ ! -f "$LOCAL_CKPT_DIR/$BACKUP_NAME" ]; then
  mkdir -p "$LOCAL_CKPT_DIR"
  log "step=$step 达 1.2B 里程碑：拷贝 ckpt → $LOCAL_CKPT_DIR/$BACKUP_NAME"
  sshpass -p "$PASS" scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
    -P "$PORT" "$HOST:/root/private_data/duel_course.pt" \
    "$LOCAL_CKPT_DIR/$BACKUP_NAME.part" 2>/dev/null
  if [ -f "$LOCAL_CKPT_DIR/$BACKUP_NAME.part" ]; then
    lsz=$(stat -f %z "$LOCAL_CKPT_DIR/$BACKUP_NAME.part" 2>/dev/null)
    ssz=$(ssh_run "stat -c %s /root/private_data/duel_course.pt" | tr -d ' ')
    if [ -n "$lsz" ] && [ -n "$ssz" ] && [[ "$ssz" =~ ^[0-9]+$ ]] \
       && [ "$lsz" -eq "$ssz" ] 2>/dev/null; then
      mv "$LOCAL_CKPT_DIR/$BACKUP_NAME.part" "$LOCAL_CKPT_DIR/$BACKUP_NAME"
      log "备份完成：$BACKUP_NAME（$lsz bytes，与服务器大小一致）"
    else
      rm -f "$LOCAL_CKPT_DIR/$BACKUP_NAME.part"
      log "⚠ 备份大小不一致（本地=$lsz 服务器=$ssz），已删 .part，下轮重试"
    fi
  else
    log "⚠ 备份 scp 失败，下轮重试"
  fi
fi

if [ "$alive" != "0" ] && [ "${errs:-0}" = "0" ]; then
  log "OK alive=$alive step=$step sps=$sps${delta}"
  echo 0 > "$FAILS_F" 2>/dev/null
  exit 0
fi

# 异常
err_txt=$(ssh_run "grep -E '\[error\]|RuntimeError' $LOG | tail -3")
log "⚠ 异常 alive=$alive errs=${errs:-?} step=$step${delta} | 最近error: $err_txt"

if [ "$alive" = "0" ]; then
  # 自然结束（达到 total-steps / time-budget）→ 不重启，等人工收尾；顺便挖 learner.act
  if [ -n "$step" ] && [ "$step" -ge "$TOTAL_STEPS" ]; then
    log "训练已达 total-steps 上限（$TOTAL_STEPS），不自动重启，等人工处理（可 BC 收尾）。"
    run_profile
    exit 0
  fi
  fails=$(cat "$FAILS_F" 2>/dev/null || echo 0)
  if [ "$fails" -ge "$MAX_CONSEC_FAIL" ]; then
    log "连续 $MAX_CONSEC_FAIL 次重启失败，停止自动恢复，需要人工介入！"
    run_profile
    exit 0
  fi
  log "进程已死（第 $((fails + 1)) 次尝试），resume 重启…"
  # 优先复用最近一次训练的完整命令行（保留用户对 --num-envs 等配置的改动），
  # 无历史进程时才用下方 RELAUNCH_CMD fallback
  cur=$(ssh_run 'ps -eo args | grep "python -m train.train" | grep -v grep | tail -1' | tr -d '\r')
  if [[ "$cur" == python* ]]; then
    ssh_run "cd /root && source /opt/dtk-26.04/env.sh >/dev/null 2>&1 && nohup $cur > \"$LOG\" 2>&1 < /dev/null &"
  else
    ssh_run "$RELAUNCH_CMD"
  fi
  sleep "$RESTART_WAIT"
  alive2=$(ssh_run 'ps aux | grep "train.train" | grep -v grep | wc -l' | tr -d ' ')
  if [ -n "$alive2" ] && [ "$alive2" != "0" ]; then
    log "重启成功（进程数=$alive2），已恢复。"
    echo 0 > "$FAILS_F" 2>/dev/null
  else
    fails=$((fails + 1)); echo "$fails" > "$FAILS_F"
    log "重启后 $RESTART_WAIT s 内未存活（第 $fails 次失败），下轮再试或转 profile。"
  fi
else
  log "进程活着但有 error 日志（可能历史），不自动动，下轮复查。"
fi
