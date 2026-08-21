#!/bin/bash
# 双机并行测试轮询：DCU1（mlp 8192×512）+ DCU2（ViT 分测）
# 后台启动：nohup bash scripts/poll_dual.sh > /tmp/poll_dual.log 2>&1 &
# 每 60s 检查一次两台进度 + hwmon 功耗，输出到 /tmp/poll_dual.txt
_ENV_F="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$_ENV_F" ] && source "$_ENV_F"
OUT=/tmp/poll_dual.txt
KEY=~/.ssh/dcu_key

for round in $(seq 1 20); do
  {
    echo "=== $(date '+%H:%M:%S') round $round ==="
    echo "--- DCU1 mlp512 ---"
    ssh -i $KEY -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o ServerAliveInterval=10 \
      -p 65032 actts28ojm@zzeshell.scnet.cn \
      'tail -2 ~/dcu1_mlp512.txt 2>/dev/null; P=$(ls /sys/class/drm/card0/device/hwmon/hwmon*/power1_average 2>/dev/null | head -1); [ -n "$P" ] && echo "PWR1: $(cat $P)"' 2>/dev/null | grep -v signature
    echo "--- DCU2 vit_split ---"
    /Users/a1-6/Documents/llm-train/qqt-gpu-sim/scripts/dcu2.sh \
      'tail -3 /root/vit_split.txt 2>/dev/null; echo "PWR2: $(cat /sys/class/drm/card8/device/hwmon/hwmon13/power1_average 2>/dev/null)"' 2>/dev/null | grep -vE "password|spawn"
  } > "$OUT" 2>&1
  sleep 60
done
