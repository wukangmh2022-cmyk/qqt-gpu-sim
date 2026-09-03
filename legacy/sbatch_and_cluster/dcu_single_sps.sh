#!/usr/bin/env bash
# DCU 单卡 SPS 冒烟：上传最新代码 → 解压 → 单卡小配置训练 → 打印 sps。
# 用法：bash scripts/dcu_single_sps.sh [PORT] [PASS]
#   （机器信息默认取自 nodes_test.txt 第一行 / 环境变量 DCU_TEST_PORT/DCU_TEST_PASS）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PORT="${1:-${DCU_TEST_PORT:-10326}}"
PASS="${2:-${DCU_TEST_PASS:-}}"
HOST="ssh.zzai.scnet.cn"
if [ -z "$PASS" ]; then
  PASS="$(head -1 /tmp/dcu_deploy/nodes_test.txt 2>/dev/null | awk '{print $3}')"
fi
[ -z "$PASS" ] && { echo "DCU_TEST_PASS 未设置"; exit 1; }

WORK=/tmp/ndrun_sps; rm -rf "$WORK"; mkdir -p "$WORK"
cat > "$WORK/cmd.exp" <<EOF
#!/usr/bin/expect -f
set timeout 900
set cmd [lindex \$argv 0]
spawn ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -p $PORT root@$HOST \$cmd
expect { "*assword:*" { send "$PASS\r"; exp_continue } eof }
EOF
cat > "$WORK/scp.exp" <<EOF
#!/usr/bin/expect -f
set timeout 300
set src [lindex \$argv 0]
set dst [lindex \$argv 1]
spawn scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no -P $PORT \$src root@$HOST:\$dst
expect { "*assword:*" { send "$PASS\r"; exp_continue } eof }
EOF
chmod +x "$WORK/cmd.exp" "$WORK/scp.exp"

# 1. 打包最新代码（jax_bomb + levels.json + 校验脚本）
echo "=== [1/3] 打包最新代码 ==="
cd "$ROOT"
rm -rf /tmp/sps_pkg && mkdir -p /tmp/sps_pkg
cp -r jax_bomb /tmp/sps_pkg/ && find /tmp/sps_pkg -name "__pycache__" -type d -exec rm -rf {} +
cp web/assets/maps/levels.json /tmp/sps_pkg/levels.json
(cd /tmp/sps_pkg && tar czf /tmp/jaxbomb_sps.tgz jax_bomb levels.json)

# 2. 上传 + 解压
echo "=== [2/3] 上传并解压到 /root/private_data/qqt-gpu-sim ==="
"$WORK/scp.exp" /tmp/jaxbomb_sps.tgz /root/private_data/ >/dev/null
"$WORK/cmd.exp" "mkdir -p /root/private_data/qqt-gpu-sim && cd /root/private_data && tar xzf jaxbomb_sps.tgz -C qqt-gpu-sim/ && ls qqt-gpu-sim/jax_bomb/jax_env.py && echo UPLOAD_OK" | tail -1

# 3. 单卡 SPS（取第 0 张 DCU；256 帧×2048 envs×2 = 每 iter 1M 帧）
echo "=== [3/3] 单卡 SPS 冒烟（iters=4，含编译时间；看第 2-3 iter 稳定值）==="
"$WORK/cmd.exp" "source /opt/dtk/env.sh >/dev/null 2>&1; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so* 2>/dev/null | head -1); export HIP_VISIBLE_DEVICES=0; cd /root/private_data/qqt-gpu-sim; python3 -c 'import jax; print(\"devices=\", jax.local_device_count())'; export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501; export LSGD_K=0 CKPT_EVERY=999999; timeout 900 python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs 2048 --num-steps 128 --minibatch 2048 --epochs 2 --iters 4 2>&1 | grep -E 'iter|sps|loss|level|关卡|Error|Traceback' | tail -25" | tail -30
echo "=== 完成（上面 sps 为单卡速度；对比基线 20 卡 ~30 万 sps 需 ×20 换算口径）==="
