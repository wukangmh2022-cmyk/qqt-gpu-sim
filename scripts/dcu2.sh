#!/usr/bin/env bash
# 优化 bench 环境（第二个 DCU）连接 helper。用法：
#   scripts/dcu2.sh <cmd>        # 远端执行单条命令
#   scripts/dcu2.sh              # 交互 shell
set -euo pipefail
_ENV_F="$(cd "$(dirname "$0")/.." && pwd)/.env"
[ -f "$_ENV_F" ] && source "$_ENV_F"
export HOST="${DCU2_USER:-root}@${DCU2_HOST:-ssh.zzai.scnet.cn}"
export PORT="${DCU2_PORT:-10717}"
export PASS="${DCU2_PASS:-}"
if [ -z "$PASS" ]; then echo "DCU2_PASS empty" >&2; exit 1; fi

if [ $# -eq 0 ]; then
  export CMD=""
  echo "交互 shell 请改用: scripts/dcu2.sh 'export CMD=bash; ...' 或手动 ssh"
  exit 1
fi
# 远端 GPU 环境：DTK 库路径 + OpenMPI 预加载（hipfftMp 需要 ompi_* 符号，
# 但它的 NEEDED 里没有 libmpi，只能 LD_PRELOAD 强制加载）。
_DTK_ENV="source /opt/dtk/env.sh >/dev/null 2>&1"
_MPI_PRELOAD="export LD_PRELOAD=/usr/mpi/gcc/openmpi-4.1.7a1/lib/libmpi.so.40"
export CMD="$_DTK_ENV; $_MPI_PRELOAD; $1"
exec "$(cd "$(dirname "$0")" && pwd)/dcu2.exp"
