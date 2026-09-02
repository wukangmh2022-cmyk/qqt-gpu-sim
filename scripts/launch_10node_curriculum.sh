#!/bin/bash
set -e

MASTER=$(/tmp/ndrun/cmd_0 "hostname -I" 2>/dev/null | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
echo "Master IP = $MASTER"

WORLD_SIZE=10
echo "WORLD_SIZE = $WORLD_SIZE"

cat << 'EOF' > /tmp/train_cmd_10node.sh
#!/bin/bash
cd /root/private_data/qqt-gpu-sim || exit 9
source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)
export WORLD_SIZE=10 RANK=$1 MASTER_ADDR=__MASTER__ MASTER_PORT=29500
export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=15
export CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5

rm -rf /root/private_data/qqt-gpu-sim/ckpt_local/*
rm -rf /root/private_data/qqt-gpu-sim/ckpt/*

nohup python3 -u -m jax_bomb.multicard_train \
  --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
  --gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01 \
  --crate-reward-coef 0.15 --crate-reward-anneal-steps 30000000000 \
  --explore-reward-coef 0.02 --explore-reward-anneal-steps 30000000000 \
  --brick-reward-coef 0.10 \
  --curriculum-json curriculum.json \
  --num-envs 20480 --num-steps 256 --minibatch 20480 --epochs 2 \
  --iters 15500 --lsgd-k 32 --lsgd-mode grad \
  --fresh > /root/private_data/train_r$1.log 2>&1 &
echo "RANK${1}_STARTED_CURRICULUM"
EOF

sed -i '' "s/__MASTER__/$MASTER/g" /tmp/train_cmd_10node.sh
chmod +x /tmp/train_cmd_10node.sh

echo "=== 杀停旧训练并同步代码与课程文件到 10 台节点 ==="
for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "pkill -9 -f 'python3.*jax_bomb' 2>/dev/null; pkill -9 -f multicard 2>/dev/null; pkill -9 -f train_real 2>/dev/null; truncate -s 0 /root/private_data/train_r*.log" 2>/dev/null | tail -1 &
done
wait

for i in $(seq 0 9); do
  /tmp/ndrun/scp_$i /Users/a1-6/Documents/llm-train/qqt-gpu-sim/curriculum.json /root/private_data/qqt-gpu-sim/curriculum.json >/dev/null 2>&1 &
  /tmp/ndrun/scp_$i /Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json /root/private_data/qqt-gpu-sim/levels.json >/dev/null 2>&1 &
  /tmp/ndrun/scp_$i /tmp/train_cmd_10node.sh /root/private_data/train_cmd.sh >/dev/null 2>&1 &
done
wait
echo "Files synchronized."

echo "=== 顺序拉起 10 台节点（课程 + 探索 + 破砖奖励）==="
for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "bash /root/private_data/train_cmd.sh $i" 2>/dev/null | tail -1 &
done
wait
echo "=== 全部 10 节点已以标准课程体系全速启动！ ==="
