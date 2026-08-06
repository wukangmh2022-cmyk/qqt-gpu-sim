#!/usr/bin/env bash
# ============================================================
# CNN 接力：把被 TIME_BUDGET 截停在 90.1M 的 duel_cnn.pt 续训到 300M。
# 实验条件与 dcu_launch_rw8_cnn.sh 的 CNN 阶段**完全一致**（不许变变量）：
#   ARCH=cnn、GAE 默认 0.95、OVERSAMPLE=3（rw8 遗留）、地图混合
#   open 0.34 / ring 0.33 / corridor 0.33、5632 env。
# CSV 用追加模式（train.py 已 open(...,"a")），90.1M 历史保留；
# TRAIN_DONE.flag 是上一轮的哨兵，先删掉避免误读。
# ============================================================
# env.sh 引用了未设置变量：必须在 set -u **之前** source，否则 127 静默退出。
source /opt/dtk-26.04/env.sh >/dev/null 2>&1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
set -u
cd ~

rm -f /root/private_data/TRAIN_DONE.flag

echo "==== [cnn] resume duel_cnn.pt -> 300M ===="
export NUM_ENVS=5632 ROLLOUT=128 MINIBATCHES=4
export TOTAL_STEPS=300000000
export ARCH=cnn GAE_LAMBDA= OVERSAMPLE=3
export OPEN_FRACTION=0.34 RING_FRACTION=0.33
export RESUME=/root/private_data/duel_cnn.pt
export CKPT=/root/private_data/duel_cnn.pt
export LOG_CSV=/root/private_data/train_cnn.csv
export TIME_BUDGET=27000 SNAPSHOT_EVERY=20 SEED=0 SKIP_SMOKE=1
scripts/dcu_train.sh

echo ALL_DONE > /root/private_data/TRAIN_DONE.flag
