#!/bin/bash
set -e

MASTER=$(/tmp/ndrun/cmd_0 "hostname -I" 2>/dev/null | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
echo "Master IP = $MASTER"

ACTIVE_NODES=(0 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23)
WORLD_SIZE=${#ACTIVE_NODES[@]}
echo "WORLD_SIZE = $WORLD_SIZE"

cat << 'EOF' > /tmp/train_cmd_grad.sh
#!/bin/bash
cd /root/private_data/qqt-gpu-sim || exit 9
source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)
export WORLD_SIZE=__WORLD_SIZE__ RANK=$1 MASTER_ADDR=__MASTER__ MASTER_PORT=29500
export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=15
export CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5

rm -rf /root/private_data/qqt-gpu-sim/ckpt_local/*
rm -rf /root/private_data/qqt-gpu-sim/ckpt/*

nohup python3 -u -m jax_bomb.multicard_train \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01 \
  --crate-reward-coef 0.15 --crate-reward-anneal-steps 30000000000 \
  --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 \
  --iters 500 --lsgd-k 32 --lsgd-mode grad \
  --fresh > /root/private_data/train_r$1.log 2>&1 &
echo "RANK${1}_STARTED_GRAD_K32"
EOF

sed -i '' "s/__MASTER__/$MASTER/g" /tmp/train_cmd_grad.sh
sed -i '' "s/__WORLD_SIZE__/$WORLD_SIZE/g" /tmp/train_cmd_grad.sh
chmod +x /tmp/train_cmd_grad.sh

echo "=== 杀停并推送 ==="
for node in "${ACTIVE_NODES[@]}"; do
  /tmp/ndrun/cmd_$node "pkill -9 -f 'python3.*jax_bomb' 2>/dev/null; pkill -9 -f multicard 2>/dev/null; pkill -9 -f train_real 2>/dev/null; truncate -s 0 /root/private_data/train_r*.log" 2>/dev/null | tail -1 &
done
wait

for node in "${ACTIVE_NODES[@]}"; do
  /tmp/ndrun/scp_$node /tmp/train_cmd_grad.sh /root/private_data/train_cmd.sh >/dev/null 2>&1 &
done
wait
echo "Delivery complete."

echo "=== 按 rank 0..22 顺序拉起 ==="
rank=0
for node in "${ACTIVE_NODES[@]}"; do
  /tmp/ndrun/cmd_$node "bash /root/private_data/train_cmd.sh $rank" 2>/dev/null | tail -1 &
  rank=$((rank + 1))
done
wait
echo "=== 全部 23 节点（46 卡）Grad Mode 已同步启动！ ==="
