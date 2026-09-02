#!/bin/bash
set -e
MASTER_IP="172.31.204.214"
MASTER_PORT=29540
echo "=== 开始 10 节点 20 卡分布式同步真实性测试 (MASTER=$MASTER_IP:$MASTER_PORT) ==="

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "pkill -9 -f python3 2>/dev/null; fuser -k 29540/tcp 2>/dev/null" >/dev/null 2>&1 || true
  /tmp/ndrun/scp_$i tests/test_distributed_sync.py /root/private_data/test_distributed_sync.py >/dev/null 2>&1
done

echo "=== 代码已分发至 10 个节点，启动分布式同步测试 ==="

for i in $(seq 0 9); do
  /tmp/ndrun/cmd_$i "source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=10 RANK=$i MASTER_ADDR=$MASTER_IP MASTER_PORT=$MASTER_PORT; python3 -u /root/private_data/test_distributed_sync.py > /root/private_data/sync_test_r$i.log 2>&1 & echo RANK\${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 4
done

echo "=== 等待 15s 收集各节点同步与归约输出 ==="
sleep 15

for i in $(seq 0 9); do
  echo "---------------------- Rank $i 节点测试日志 ----------------------"
  /tmp/ndrun/cmd_$i "cat /root/private_data/sync_test_r$i.log" 2>/dev/null | grep -E "通过|PASS|错误|pmean|all_gather|SHA256|设备" | tail -10
done
