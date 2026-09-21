#!/usr/bin/env bash
# ==============================================================================
# 昆山集群 6 节点 x 4 卡环境自检与 JAX 依赖探测脚本
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES_FILE="${1:-$SCRIPT_DIR/nodes_current.txt}"

if [ ! -f "$NODES_FILE" ]; then
  echo "错误: 节点文件不存在: $NODES_FILE"
  exit 1
fi

# 1. 读取节点
N_PORT=(); N_HOST=(); N_PASS=()
PENDING_PORT=""; PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | tr -d "\r" | sed "s/^[ \t]*//;s/[ \t]*$//")"
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue

  if [[ "$line" =~ ^ssh\ -p\ ([0-9]+)\ root@([^ ]+) ]]; then
    PENDING_PORT="${BASH_REMATCH[1]}"
    PENDING_HOST="${BASH_REMATCH[2]}"
  elif [ -n "$PENDING_PORT" ] && [ -n "$PENDING_HOST" ]; then
    N_PORT+=("$PENDING_PORT")
    N_HOST+=("$PENDING_HOST")
    N_PASS+=("$line")
    PENDING_PORT=""; PENDING_HOST=""
  fi
done < "$NODES_FILE"

NW=${#N_PORT[@]}
echo "=== 正在检测全部 $NW 台节点的 DCU 状态与 JAX/Python 环境 ==="

WORK="/tmp/scnet_check_work"
rm -rf "$WORK" && mkdir -p "$WORK"

for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOcmd
#!/bin/bash
exec /opt/homebrew/bin/sshpass -p "${N_PASS[$i]}" ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -p ${N_PORT[$i]} root@${N_HOST[$i]} "\$@"
EOcmd
  chmod +x "$WORK/cmd_$i"
done

CHECK_SCRIPT='
echo "--- Node Info: $(hostname) ($(hostname -I | awk "{print \$1}")) ---"
echo ">> [1] DCU 硬件状态 (hy-smi):"
if command -v hy-smi >/dev/null 2>&1; then
  hy-smi -L 2>/dev/null || hy-smi
else
  echo "hy-smi 未找到，尝试 /opt/dtk/bin/hy-smi"
  /opt/dtk/bin/hy-smi -L 2>/dev/null || echo "无法执行 hy-smi"
fi

echo ">> [2] Python 与 JAX 环境检测:"
source /opt/dtk/env.sh 2>/dev/null || true
export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so /public/software/mpi/*/lib/libmpi.so 2>/dev/null | head -1)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

python3 - <<PYCHECK
import sys
print(f"Python: {sys.version.split()[0]}")

try:
    import jax
    devs = jax.devices()
    print(f"JAX: {jax.__version__}, 设备数: {len(devs)}, 设备详情: {[str(d) for d in devs]}")
except Exception as e:
    print(f"JAX 异常/缺失: {e}")

try:
    import optax
    print(f"Optax: {optax.__version__}")
except Exception as e:
    print(f"Optax 缺失: {e}")

try:
    import chex
    print(f"Chex: {chex.__version__}")
except Exception as e:
    print(f"Chex 缺失: {e}")
PYCHECK
'

for i in $(seq 0 $((NW-1))); do
  echo "=================================================="
  echo "检查节点 Rank $i (${N_HOST[$i]}:${N_PORT[$i]}):"
  "$WORK/cmd_$i" "$CHECK_SCRIPT" 2>&1 || echo "节点 $i 连接或探测失败"
done

echo "=================================================="
echo "=== 检测完成 ==="
