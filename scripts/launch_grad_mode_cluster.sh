#!/bin/bash
# ==============================================================================
# 一键发射：Grad Mode K=32（全集群 24 节点 48 卡，800k SPS，500M 步 10 分钟）
# ==============================================================================
set -e

echo "=== [1/4] 获取 Node 0 内网 IP (MASTER_ADDR) ==="
MASTER=$(/tmp/ndrun/cmd_0 "hostname -I" 2>/dev/null | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
echo "Master IP = $MASTER"

echo "=== [2/4] 生成启动脚本 /tmp/train_cmd_grad.sh ==="
sed "s/__MASTER__/$MASTER/g" scripts/train_cmd_grad_k32.sh > /tmp/train_cmd_grad.sh
chmod +x /tmp/train_cmd_grad.sh

echo "=== [3/4] 杀停旧训练并分发脚本到全部 24 台节点 ==="
for i in $(seq 0 23); do
  /tmp/ndrun/cmd_$i "pkill -9 -f 'python3.*jax_bomb' 2>/dev/null; pkill -9 -f train_real 2>/dev/null; pkill -9 -f multicard 2>/dev/null; truncate -s 0 /root/private_data/train_r$i.log" 2>/dev/null | tail -1 &
done
wait

for i in $(seq 0 23); do
  /tmp/ndrun/scp_$i /tmp/train_cmd_grad.sh /root/private_data/train_cmd.sh >/dev/null 2>&1 &
done
wait
echo "All 24 nodes scripts delivered."

echo "=== [4/4] 同步启动全部 24 节点 ==="
for i in $(seq 0 23); do
  /tmp/ndrun/cmd_$i "bash /root/private_data/train_cmd.sh $i" 2>/dev/null | tail -1 &
done
wait

echo "=== 全部 24 节点已全量发车！进入 40 秒监控自检 ==="
