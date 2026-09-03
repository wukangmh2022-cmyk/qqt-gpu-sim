#!/usr/bin/env bash
# ============================================================
# DCU 大规模训练启动脚本
# 环境：PyTorch 2.9.0 / py3.11 / Ubuntu22.04 / dtk26.04（海光 DCU）
#
# ⚠️ DCU 没有 nvcc/CUDA 工具链 —— 自定义 CUDA kernel（CudaSim，sim/cuda/）
#    编译不了。因此 **必须 --backend torch**：纯 PyTorch 张量实现，
#    PyTorch 算子自动吃 DCU 的 GPU 加速。--backend auto 会因
#    torch.cuda.is_available()==True 误选 cuda 然后编译失败，禁止用。
#
# 用法：
#   ./scripts/dcu_train.sh                      # 默认大规模配置
#   TOTAL_STEPS=60000000 ./scripts/dcu_train.sh # 覆盖总步数
#   RESUME=ckpt/duel_dcu.pt ./scripts/dcu_train.sh   # 断点续训
#   ARCH=mlp GAE_LAMBDA=0.9 OVERSAMPLE=3 \
#     OPEN_FRACTION=0.34 RING_FRACTION=0.33 RESUME=/root/private_data/duel_rw7.pt \
#     TOTAL_STEPS=420000000 CKPT=/root/private_data/duel_rw8.pt \
#     ./scripts/dcu_train.sh                    # rw8：MLP 从 rw7 续训 + GAE λ A/B
#   SKIP_SMOKE=1 ./scripts/dcu_train.sh         # 跳过冒烟直接大训
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."                     # 项目根目录

PYTHON=${PYTHON:-python}                     # 镜像用 Miniconda，默认 python

# ---------- 可调参数（环境变量覆盖） ----------
NUM_ENVS=${NUM_ENVS:-2048}
ROLLOUT=${ROLLOUT:-128}
MINIBATCHES=${MINIBATCHES:-4}
TOTAL_STEPS=${TOTAL_STEPS:-30000000}         # 3000 万步 ≈ 5 万局（600 tick/局）
CKPT=${CKPT:-ckpt/duel_dcu.pt}
LOG_CSV=${LOG_CSV:-ckpt/train_dcu.csv}
TIME_BUDGET=${TIME_BUDGET:-72000}            # 秒；到点存盘退出（默认 20h）
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-20}
SEED=${SEED:-0}
RESUME=${RESUME:-}
SKIP_SMOKE=${SKIP_SMOKE:-0}
ARCH=${ARCH:-cnn}                            # cnn | mlp（CNN 对照 vs MLP 续训）
GAE_LAMBDA=${GAE_LAMBDA:-}                   # 空 = 用默认 0.95（A/B 用 0.88–0.9）
OVERSAMPLE=${OVERSAMPLE:-}                   # 空 = 用默认 3（1 = 关闭濒死过采样）
OPEN_FRACTION=${OPEN_FRACTION:-0.5}          # 混合地图：open 关占比
RING_FRACTION=${RING_FRACTION:-0.0}          # 环岛关占比（余量 = corridor）

# ---------- 1. 环境检查 ----------
echo "== torch 环境 =="
$PYTHON - <<'EOF'
import torch, sys
print(f"python {sys.version.split()[0]}  torch {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"device count = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device name = {torch.cuda.get_device_name(0)}")
EOF

# 内存守卫依赖 psutil 探测可用内存；缺失时守卫降级为"仅警告"。
# 装齐，让启动前的内存预算真的生效。
$PYTHON -m pip install -q psutil 2>/dev/null || true

# 探测 device：DCU 的 torch 走 cuda 命名空间（torch.cuda.is_available()=True）
DEVICE=$($PYTHON -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
echo "== device = $DEVICE =="
if [ "$DEVICE" != "cuda" ]; then
  echo "警告：torch.cuda.is_available()=False —— DCU 加速未生效，训练会落到 CPU（很慢）。"
  echo "      请确认镜像带 dtk26.04 且 PyTorch 是 DTK 版（pip show torch | grep -i dcu）。"
fi

# ---------- 2. 自检（规则 + 训练；跳过 parity 与 RES 素材测试） ----------
#   parity 要编译 .cu（DCU 无 nvcc）；res_assets 要 pygame（服务器不装）。
echo "== 自检：规则 + 训练（跳过 CUDA parity / RES 素材）=="
$PYTHON -m pytest tests/test_rules.py tests/test_train.py -q -k "not res_assets" || {
  echo "自检失败，先修再训练。"; exit 1; }

# ---------- 3. 小冒烟：确认 DCU 上观测/掩码/PPO 更新能跑通 ----------
if [ "$SKIP_SMOKE" != "1" ]; then
  echo "== 冒烟：64 env × 8192 步 =="
  $PYTHON -m train.train \
    --backend torch --device "$DEVICE" \
    --num-envs 64 --rollout-steps 64 --minibatches 2 \
    --total-steps 8192 --single-stage \
    --ckpt /tmp/duel_smoke.pt --log-csv /tmp/duel_smoke.csv --seed 0
fi

# ---------- 4. 大规模训练 ----------
echo "== 大规模训练：env=$NUM_ENVS rollout=$ROLLOUT mb=$MINIBATCHES steps=$TOTAL_STEPS =="
ARGS=(--backend torch --device "$DEVICE"
      --arch "$ARCH"
      --num-envs "$NUM_ENVS" --rollout-steps "$ROLLOUT" --minibatches "$MINIBATCHES"
      --total-steps "$TOTAL_STEPS" --single-stage
      --snapshot-every "$SNAPSHOT_EVERY"
      --map-mode corridor --open-fraction "$OPEN_FRACTION" --ring-fraction "$RING_FRACTION"
      --ckpt "$CKPT" --log-csv "$LOG_CSV"
      --time-budget "$TIME_BUDGET" --seed "$SEED")
if [ -n "$GAE_LAMBDA" ]; then ARGS+=(--gae-lambda "$GAE_LAMBDA"); fi
if [ -n "$OVERSAMPLE" ]; then ARGS+=(--oversample-dying "$OVERSAMPLE"); fi
if [ -n "$RESUME" ]; then ARGS+=(--resume "$RESUME"); fi
$PYTHON -m train.train "${ARGS[@]}"
echo "== 训练结束，checkpoint=$CKPT =="
