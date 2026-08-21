#!/bin/bash
# SCNet「模型训练」多卡 DP 启动脚本 —— 全文粘贴到控制台「启动脚本」框
# 16 卡跨实例测速版（每实例 8 卡 × 2 实例 = 16 卡，内部 TCP）
# 平台注入：MASTER_ADDR/WORLD_SIZE/RANK/MASTER_PORT（PyTorch 任务）
#   + 通用 worker0~workerNN（各实例 hostname）；JAX 任务缺注入时脚本自动从 workerN 推导
# 适配任意 每实例卡数×实例数 组合（总卡数不能整除时脚本自动下调 envs/minibatch）
#
# 有损同步（Local SGD）—— 跨机降通信的唯一出路（RCCL 调优实测是死路）：
#   LSGD_K>0 启用，每 K 个 minibatch 才同步一次（默认每 minibatch 同步 =
#   每迭代 512×epochs 次全量 allreduce，20 卡 ~50GB/迭代 → 通信分钟级）。
#   LSGD_K=256 → 4 次同步/迭代 ≈ 194MB ≈ 0.5-1.5s；LSGD_K=128 ≈ 388MB。
#   LSGD_MODE=param|grad（grad=零漂移版，K=1 时与无损逐位一致）；
#   LSGD_BF16=1 流量再减半（通道无损，无稀疏/量化）。
#   LSGD_K=0 = 现状无损路径（每 minibatch pmean 梯度，逐位一致）。
set -e

# ── 自动搜索代码目录（自定义挂载 /mnt/data、private_data、HPC 挂载均可）──
REPO=""
for cand in /root/private_data/qqt-gpu-sim \
            /root/private_data/*/qqt-gpu-sim \
            /mnt/data/qqt-gpu-sim \
            /mnt/data/*/qqt-gpu-sim \
            /mnt/qqt-gpu-sim \
            /public/home/*/qqt-gpu-sim \
            /home/*/qqt-gpu-sim \
            /root/qqt-gpu-sim; do
  [ -d "$cand" ] && REPO="$cand" && break
done
if [ -z "$REPO" ]; then
  echo "ERROR: 找不到 qqt-gpu-sim 代码目录。" >&2
  echo "      补救：自定义挂载源选 qqt-gpu-sim 的父目录、目标挂到 /mnt/data（" >&2
  echo "      容器里路径即为 /mnt/data/qqt-gpu-sim）；" >&2
  echo "      或 Notebook 里 cd /root/private_data && tar xzf qqt-gpu-sim-platform.tar.gz；" >&2
  echo "      或在启动脚本顶部手动指定 REPO=/你的实际路径" >&2
  echo "容器保持存活 1 小时便于排查（可进容器手动放代码后重跑）。" >&2
  sleep 3600
  exit 1
fi
echo "=== REPO=$REPO ==="
cd "$REPO"

# ── 代码版本校验：必须是含 state token 的最终版（6,389,256 参数）──
# 旧版（无 state token，6,383,112 参数）会静默丢失全局状态输入，必须失败而不是硬跑。
# 已知最终版位置：/public/home/actts28ojm/qqt-gpu-sim（HPC 存储，已核对一致），
# 或本机 qqt-gpu-sim/jax_bomb/（打 tar 后经 Notebook/挂载放入容器）。
if [ ! -f "$REPO/jax_bomb/jax_net.py" ]; then
  echo "ERROR: $REPO 缺少 jax_bomb/ 代码。" >&2
  echo "      补救：把最终版 jax_bomb/ 放到 $REPO 下" >&2
  sleep 3600
  exit 1
fi
if ! grep -q "state_dim" "$REPO/jax_bomb/jax_net.py" \
     || ! grep -q "global_vec" "$REPO/jax_bomb/jax_env.py" \
     || ! grep -q "both_states" "$REPO/jax_bomb/jax_train.py"; then
  echo "ERROR: $REPO 的代码是旧版（无 state token，参数 6,383,112）。" >&2
  echo "      需要用最终版替换 jax_bomb/（含 state token，6,389,256 参数）。" >&2
  echo "      快速修复：从本机拷 qqt-gpu-sim/jax_bomb/ 覆盖 $REPO/jax_bomb/" >&2
  echo "      （或把 qqt-gpu-sim-platform.tar.gz 解到 $REPO 父目录）" >&2
  echo "容器保持存活 1 小时便于排查。" >&2
  sleep 3600
  exit 1
fi
echo "=== 代码版本 OK（含 state token）==="

# 依赖兜底：镜像缺 optax 时现场装（平台容器一般可访问 tsinghua pypi）
python3 -c "import optax" 2>/dev/null \
  || pip install optax -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || echo "WARN: optax 自动安装失败（稍后 python 会直接报错）"

# DTK 环境（基础镜像自带，防御性 source）
source /opt/dtk/env.sh 2>/dev/null || true

# 平台容器默认代理会让 JAX rendezvous 超时，必须清掉
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

# 部分 DTK 镜像 hipfftMp 缺 NEEDED libmpi，LD_PRELOAD 兜底（找不到就跳过）
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so* \
           /public/software/mpi/*/lib/libmpi.so*; do
  [ -f "$mpi" ] && export LD_PRELOAD="$mpi" && break
done

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
# ── 分布式变量双保险：平台文档确认 MASTER_ADDR/WORLD_SIZE/RANK/MASTER_PORT 是
# PyTorch 任务注入；若本 JAX 任务未注入（只剩通用 worker0~workerNN = 各实例 hostname），
# 就从 workerN 推导，避免两实例各跑各的 4 卡（数字虚高、consistency 误过）。
if [ "${WORLD_SIZE:-1}" = "1" ] && [ -n "$worker0" ]; then
  nw=$(env | grep -cE '^worker[0-9]+=')
  if [ "$nw" -gt 1 ]; then
    WORLD_SIZE=$nw
    echo "WARN: WORLD_SIZE 未注入，已从 worker0..worker$((nw-1)) 推导为 $nw"
  fi
fi
if [ "${WORLD_SIZE:-1}" -gt 1 ]; then
  if [ -z "${MASTER_ADDR:-}" ] || [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    MASTER_ADDR=${worker0:-127.0.0.1}
    echo "WARN: MASTER_ADDR 未注入，已取 worker0 = $MASTER_ADDR"
  fi
  if [ -z "${RANK:-}" ] || [ "$RANK" = "0" ]; then
    rk=$(hostname 2>/dev/null | sed -n 's/.*-worker-\([0-9][0-9]*\)$/\1/p')
    [ -n "$rk" ] && RANK=$rk && echo "WARN: RANK 未注入，已从 hostname 推导为 $rk"
  fi
fi
# 训练地图分布：论文式全 open + 0-5 随机障碍（每局随机重生）
#   切换回混合地图（纯空50/open带障碍25/corridor25）：注释掉本行
export QQT_ENV_MIX=open_obstacle
echo "=== WORLD_SIZE=$WORLD_SIZE RANK=$RANK MASTER=$MASTER_ADDR:$MASTER_PORT ENV_MIX=$QQT_ENV_MIX ==="

# E512/d2/p4 = 6.38M 参数（单卡实测 47.6K sps；2 卡 89.2K = 1.87×）
# envs/minibatch 按「每卡 2048」放大（16 卡 = 32768），与 2 卡验证负载一致：
#   全局 batch 不随卡数放大 → 每卡只有 128 envs，GEMM 饿死，缩放数据失真
# 本文件为 16 卡跨实例测速版：--iters 5 + --fresh（不接续旧 ckpt）
#   转长训：--iters 2000、去掉 --fresh（自动接续 ckpt/ 最新断点，每小时存盘）
#   换卡数（2/4/8/16/32）：envs/mb 均可整除，脚本自动均分，无需改
# 全量日志落盘（挂载目录两端可看）：长训监控 tail -f $LOG
mkdir -p "$REPO/logs"
LOG="$REPO/logs/multicard_$(date +%Y%m%d_%H%M%S)_r${RANK:-0}.log"

# ── 功耗采样：训练期间后台记录 rocm-smi（功耗/温度/利用率），退出自动停 ──
# 每个实例写自己的文件（文件名带 RANK 区分），跑完在 $REPO/logs/ 下分析
POWER_LOG="$REPO/logs/power_$(date +%Y%m%d_%H%M%S)_r${RANK:-0}.log"
( while true; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$POWER_LOG"
    rocm-smi --showpower --showtemp --showuse >> "$POWER_LOG" 2>&1 \
      || rocm-smi >> "$POWER_LOG" 2>&1 \
      || echo "rocm-smi 不可用" >> "$POWER_LOG"
    sleep 10
  done ) &
POWER_PID=$!
trap 'kill "$POWER_PID" 2>/dev/null' EXIT
echo "=== 功耗采样中: $POWER_LOG ==="

set -o pipefail

# ── 有损同步（Local SGD）参数：提交时通过环境变量/控制台设置 ──
#   例：LSGD_K=256 LSGD_MODE=grad LSGD_BF16=1  →  10 机×2 卡 ≈ 97MB/迭代
LSGD_K=${LSGD_K:-0}
LSGD_MODE=${LSGD_MODE:-param}
LSGD_BF16=${LSGD_BF16:-0}
LSGD_SYNC_STATE=${LSGD_SYNC_STATE:-0}
echo "=== LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE LSGD_BF16=$LSGD_BF16 LSGD_SYNC_STATE=$LSGD_SYNC_STATE ==="

python3 -m jax_bomb.train_real \
  --arch transformer --embed 512 --depth 2 --patch 4 \
  --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 \
  --iters 5 --fresh --lsgd-k "$LSGD_K" --lsgd-mode "$LSGD_MODE" \
  $([ "$LSGD_BF16" = "1" ] && echo --lsgd-bf16) \
  $([ "$LSGD_SYNC_STATE" = "1" ] && echo --lsgd-sync-state) \
  2>&1 | tee "$LOG"
echo "=== 日志文件: $LOG ==="
