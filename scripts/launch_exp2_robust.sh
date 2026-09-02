#!/bin/bash
set -e
MASTER_IP="172.31.129.179"
MASTER_PORT=29588
echo "=== 稳健拉起 Exp-2 (纯净初代奖励: explore=0, brick=0, crate=0.5 30B退火) ==="

rm -rf /tmp/jaxbomb_10node && mkdir -p /tmp/jaxbomb_10node
cp -r jax_bomb levels.json scripts /tmp/jaxbomb_10node/
find /tmp/jaxbomb_10node -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_10node && tar czf /tmp/jaxbomb_10node.tgz jax_bomb levels.json scripts)

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "pkill -9 -f jax_bomb 2>/dev/null; pkill -9 -f train_real 2>/dev/null; pkill -9 -f eval 2>/dev/null; fuser -k 29550/tcp 29560/tcp 29570/tcp 2>/dev/null || true; sleep 1; rm -rf /root/private_data/train_r*.log /root/private_data/qqt-gpu-sim 2>/dev/null || true; mkdir -p /root/private_data/qqt-gpu-sim" >/dev/null 2>&1 || true
done
sleep 1

for i in $(seq 0 9); do
  /tmp/ndrun/scp_$i /tmp/jaxbomb_10node.tgz /root/private_data/ >/dev/null 2>&1 || true
  /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim && tar xzf /root/private_data/jaxbomb_10node.tgz && echo RANK\${i}_SYNC_OK" | tail -1
done

echo "=== 代码已分发完毕，开始顺序拉起训练进程 ==="

for i in $(seq 0 9); do
    /tmp/ndrun/cmd_$i "cd /root/private_data/qqt-gpu-sim; export LC_ALL=C.UTF-8 LANG=C.UTF-8; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=10 RANK=$i MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT; export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5; nohup python3 -u -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 --iters 15500 --lsgd-k 32 --lsgd-mode grad --levels levels.json --level-weights \"empty=0.05,功夫=0.1,比武=0.15\" --crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 --explore-reward-coef 0.0 --brick-reward-coef 0.0 --reward-anneal-k 1.2 --fresh > /root/private_data/train_r$i.log 2>&1 & echo RANK\${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 4
  sleep 0.8
done

echo "=== Exp-2 纯净奖励 10 节点集群全部拉起完毕！ ==="
