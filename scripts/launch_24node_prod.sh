#!/bin/bash
# ==============================================================================
# 24 节点 (48 DCU 卡) 多机强化学习生产训练启动脚本
# ==============================================================================
set -e

MASTER_IP="172.31.129.179"
MASTER_PORT=29600
WORLD_SIZE=24

echo "=========================================================================="
echo "🚀 正在拉起 24 节点 (48 卡 DCU) 生产级自博弈 PPO 强化学习训练"
echo "   - 同步模式: LSGD Grad Mode (零漂移梯度同步, K=32)"
echo "   - 奖励配置: 初代纯净配置 (explore=0, brick=0, crate=0.5 500M退火)"
echo "   - 动作先验: Bomb Head 偏置 b=-2.2 (8.04% 健康先验)"
echo "   - 总环境数: 78,624 并行环境 (每卡 1638 envs, 总吞吐预计 ~80万 SPS)"
echo "=========================================================================="

rm -rf /tmp/jaxbomb_24node && mkdir -p /tmp/jaxbomb_24node
cp -r jax_bomb levels.json scripts /tmp/jaxbomb_24node/
find /tmp/jaxbomb_24node -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_24node && tar czf /tmp/jaxbomb_24node.tgz jax_bomb levels.json scripts)

# 1. 安全杀死各节点遗留进程（精确匹配防误杀 Jupyter）
echo "🧹 清理 24 节点历史训练进程与端口..."
for i in $(seq 0 23); do
  /tmp/ndrun/cmd_$i "pkill -9 -f jax_bomb 2>/dev/null; pkill -9 -f train_real 2>/dev/null; pkill -9 -f eval 2>/dev/null; fuser -k 29550/tcp 29560/tcp 29570/tcp 29580/tcp 29590/tcp 29600/tcp 2>/dev/null || true; sleep 0.5; rm -rf /root/private_data/train_r*.log /root/private_data/qqt-gpu-sim 2>/dev/null || true; mkdir -p /root/private_data/qqt-gpu-sim" >/dev/null 2>&1 || true
done
sleep 1

# 2. 分发代码包
echo "📦 分发代码至 24 节点..."
for i in $(seq 0 23); do
  /tmp/ndrun/scp_$i /tmp/jaxbomb_24node.tgz /root/private_data/ >/dev/null 2>&1 || true
  /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim && tar xzf /root/private_data/jaxbomb_24node.tgz && echo RANK\${i}_SYNC_OK" | tail -1
done

# 3. 顺序拉起 24 节点训练
echo "🔥 顺序拉起 24 节点 JAX 分布式训练进程..."
for i in $(seq 0 23); do
  /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim; export LC_ALL=C.UTF-8 LANG=C.UTF-8; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=$WORLD_SIZE RANK=$i MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT; export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5; nohup python3 -u -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs 78624 --num-steps 256 --minibatch 78624 --epochs 2 --iters 30000 --lsgd-k 32 --lsgd-mode grad --levels levels.json --level-weights \"empty=0.05,功夫=0.1,比武=0.15\" --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 --explore-reward-coef 0.0 --brick-reward-coef 0.0 --reward-anneal-k 1.2 --fresh > /root/private_data/train_r$i.log 2>&1 & echo RANK\${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 4
  sleep 0.5
done

echo "=========================================================================="
echo "✅ 24 节点 (48 卡 DCU) 生产训练集群已全量拉起！"
echo "=========================================================================="
