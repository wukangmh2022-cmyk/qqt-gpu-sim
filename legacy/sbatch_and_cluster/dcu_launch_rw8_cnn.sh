#!/usr/bin/env bash
# ============================================================
# 新一轮双实验（顺序跑，单卡 64GB 不能并发 2×5632）：
#   1) rw8：MLP 从 duel_rw7.pt（300.48M 步, 14 通道同布局）直接续训
#      +120M 步 → total=420.48M；GAE λ=0.9（A/B 臂）+ 濒死过采样 3。
#   2) CNN：从头训练 300M 步做纯对照（arch 不同，rw7 权重不可用）；
#      GAE λ 用默认 0.95（CNN 自己单独 A/B）。
# 地图混合沿用 rw7 实测比例：open 0.34 / ring 0.33 / corridor 0.33。
# 每段日志独立；全部完成写 TRAIN_DONE.flag。
# ============================================================
# env.sh 内部引用了未设置的环境变量：必须在 set -u 生效**之前** source，
# 否则在 set -u 下 source 会以 127 静默退出。dcu_train.sh 自己不 source，
# 由调用方负责 —— 这里先 source 再开 set -u。
source /opt/dtk-26.04/env.sh >/dev/null 2>&1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
set -u
cd ~

echo "==== [rw8] MLP resume from duel_rw7.pt ===="
export NUM_ENVS=5632 ROLLOUT=128 MINIBATCHES=4
export TOTAL_STEPS=420482560            # 300.48M + 120M
export ARCH=mlp GAE_LAMBDA=0.9 OVERSAMPLE=3
export OPEN_FRACTION=0.34 RING_FRACTION=0.33
export RESUME=/root/private_data/duel_rw7.pt
export CKPT=/root/private_data/duel_rw8.pt
export LOG_CSV=/root/private_data/train_rw8.csv
export TIME_BUDGET=7200 SNAPSHOT_EVERY=20 SEED=0 SKIP_SMOKE=1
scripts/dcu_train.sh

echo "==== [cnn] from scratch ===="
export TOTAL_STEPS=300000000
export ARCH=cnn GAE_LAMBDA=             # 空 → 脚本不带 flag → 默认 0.95（CNN 自己的 A/B）
export RESUME=
export CKPT=/root/private_data/duel_cnn.pt
export LOG_CSV=/root/private_data/train_cnn.csv
export TIME_BUDGET=10800
scripts/dcu_train.sh

echo ALL_DONE > /root/private_data/TRAIN_DONE.flag
