#!/bin/bash
# 单机 8 卡训练编排（DCU / JAX pmap DP，Local SGD 可选）
#
# 与 launch_10nodes.sh（十机二卡主战场）同一套代码、同一份部署包，只是
# 单机运行：WORLD_SIZE=1（JAX 退化为单机 pmap over 8 卡），跨卡梯度同步
# 走机内 RCCL，无需跨机 MPI/RDMA —— 适合模型实验 / 数据并行验证 / 小规模
# 长训。参数默认值与主战场一致（同 arch/embed/depth/...、同 levels 权重、
# 同 crate 奖励），只是 num-envs 按 8 卡缩放（16384 = 8×2048）。
#
# 机器信息：单机八卡机器的 ssh 入口（notebook 风格，带密码）。
#   bash launch_8gpu.sh nodes_8gpu.txt [--deploy] [--resume]
# nodes_8gpu.txt 每行:  <ssh端口> <主机> <密码>     # 只读第 1 行
#   例: 10832 ssh.zzai.scnet.cn 8GPUMACHINEPASS
# --deploy 先推完整部署包（代码+wheels）并跑 setup；已部署过可省（每次都会
#          推增量代码 tgz —— 保证运行的是最新代码，避免忘同步）。
# --resume 接续 ckpt/ 最新断点（默认 --fresh 全新开始；本轮 obs 13→14 通道，旧断点不兼容）。
#
# 流程: 部署(可选) → 自检(import/卡数=8/代码版本) → 启动 train_real →
#       30s 健康检查(空日志/Traceback 即中止) → 提示监控/拉快照。
set -u
NODES_FILE="${1:?用法: launch_8gpu.sh nodes_8gpu.txt [--deploy] [--fresh]}"
shift
DO_DEPLOY=0; FRESH="--fresh"          # 本轮从头训（默认）；--resume 接续旧断点
for a in "$@"; do
  [ "$a" = "--deploy" ] && DO_DEPLOY=1
  { [ "$a" = "--resume" ] || [ "$a" = "--no-fresh" ]; } && FRESH=""
done
[ -f "$NODES_FILE" ] || { echo "找不到 $NODES_FILE"; exit 1; }

# 生产训练参数（与十机二卡 launch_10nodes.sh 同款；改这里即可）
# steps/iter = 2×num_envs×steps = 8.39M；260B 全量 ≈ 31000 iter。试跑用
# ITERS= 覆盖（如 ITERS=100 验证塑形链路）；中断后自动从 CKPT 接续。
ITERS="${ITERS:-31000}"                  # 31000 × 8.39M ≈ 260B 步
NUM_ENVS="${NUM_ENVS:-16384}"            # 8 卡 × 2048（主战场 20 卡 × 1638 ≈ 32768）
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-16384}"
LSGD_K="${LSGD_K:-0}"; LSGD_MODE="${LSGD_MODE:-param}"   # 单机机内通信快，默认逐位一致(K=0)
LEVELS_FILE="${LEVELS_FILE:-}"; LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-empty=0.05,功夫=0.1,比武=0.15}"
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.5}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-30000000000}"  # 30B：开箱奖励长退火（改回 500M 即旧版快退火）
# 探索 novelty：0.01×195 格 = 单局封顶 ~1.95 分，远低于全血伤害 7.5 与击杀 10，
# 不可能压过胜负信号；与 crate 同款 30B 退火
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.01}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-30000000000}"
# 炸墙奖励：每炸一块砖双方各 +0.025（治出生点 3 格死锁——crate 链路太长学不会）
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.05}"
# 动态退火斜率 k：α_dyn = 1-tanh(k·每局击杀率)，击杀率上来塑形自动归零（Pommerman 论文式）
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
# Spawn-Distance 课程：默认不启用（260B 全图均匀采样；论文课程职能已被稠密塑形
# 替代，见 docs/vit_train_log.md §2.4）。要启用：CURRICULUM_JSON=web/assets/maps/curriculum.json
CURRICULUM_JSON="${CURRICULUM_JSON:-}"
CKPT_EVERY="${CKPT_EVERY:-60}"         # 分钟；train_real 默认 60

# ── 读节点（只取第 1 行 = 目标机）──
read -r N_PORT N_HOST N_PASS < "$NODES_FILE"
[ -n "${N_PORT:-}" ] || { echo "nodes_8gpu.txt 第 1 行为空"; exit 1; }
echo "=== 单机 8 卡: ${N_HOST}:${N_PORT} ==="

# 临时 expect 封装（密码内嵌）
WORK=/tmp/ndrun8; rm -rf "$WORK"; mkdir -p "$WORK"
cat > "$WORK/cmd" <<EOF
#!/usr/bin/expect -f
set timeout 300
set cmd [lindex \$argv 0]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $N_PORT root@$N_HOST \$cmd
expect {
  "password:" { send "$N_PASS\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "TIMEOUT" }
}
EOF
cat > "$WORK/scp" <<EOF
#!/usr/bin/expect -f
set timeout 600
set src [lindex \$argv 0]
set dst [lindex \$argv 1]
spawn scp -o StrictHostKeyChecking=accept-new -P $N_PORT \$src root@$N_HOST:\$dst
expect {
  "password:" { send "$N_PASS\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "SCP_TIMEOUT" }
}
EOF
chmod +x "$WORK/cmd" "$WORK/scp"

# ── 1) 部署（可选）：完整包 + setup ──
if [ "$DO_DEPLOY" = "1" ]; then
  echo "=== 部署完整包（代码+wheels）+ setup ==="
  PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || { echo "找不到部署包 $PKG（先构建 dcu_deploy_10node.tar.gz）"; exit 1; }
  "$WORK/scp" "$PKG" /root/private_data/ >/dev/null
  "$WORK/cmd" "cd /root/private_data && tar xzf dcu_deploy_10node.tar.gz; if [ -d dcu_deploy ]; then cd dcu_deploy; fi; bash setup_notebook.sh 2>&1 | tail -4" | tail -5
fi

# ── 2) 推最新代码（每次必做：保证运行的代码就是仓库当前版本）──
echo "=== 推最新代码 ==="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf /tmp/jaxbomb_8gpu && mkdir -p /tmp/jaxbomb_8gpu/scripts
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_8gpu/
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_8gpu/levels.json
mkdir -p /tmp/jaxbomb_8gpu/web/assets/maps
cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_8gpu/web/assets/maps/curriculum.json 2>/dev/null || true
for s in quick_check_levels.py quick_check_bush.py quick_check_crate_semantics.py \
         quick_check_obs_move.py quick_check_js_jax_move.py quick_check_anti_tunnel.py; do
  cp "$ROOT/scripts/$s" /tmp/jaxbomb_8gpu/scripts/ 2>/dev/null
done
find /tmp/jaxbomb_8gpu -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_8gpu && tar czf /tmp/jaxbomb_8gpu.tgz jax_bomb levels.json curriculum.json web scripts)
"$WORK/scp" /tmp/jaxbomb_8gpu.tgz /root/private_data/ >/dev/null
"$WORK/cmd" "cd /root/private_data && rm -rf qqt-gpu-sim && mkdir qqt-gpu-sim && tar xzf jaxbomb_8gpu.tgz -C qqt-gpu-sim && ls qqt-gpu-sim/jax_bomb/jax_env.py && echo UPLOAD_OK" | tail -1

# ── 3) 自检：import + 卡数 = 8 ──
echo "=== 自检（import jax/optax/jax_bomb + 卡数=8）==="
out=$("$WORK/cmd" "source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); cd /root/private_data/qqt-gpu-sim 2>/dev/null || exit 9; python3 -c 'import jax, optax; from jax_bomb.jax_train import ppo_update_gradsync, ppo_update_lsgd; print(\"SELFCHECK_OK\", jax.__version__, \"devices=\", jax.local_device_count())'" 2>/dev/null | grep SELFCHECK_OK | tail -1)
if [ -z "$out" ]; then
  echo "  ✗ 自检失败（setup 未生效 / 环境损坏 / 代码版本不对）"; exit 1
fi
ndev=$(echo "$out" | grep -oE 'devices= [0-9]+' | grep -oE '[0-9]+')
if [ "$ndev" != "8" ]; then
  echo "  ✗ 卡数 $ndev ≠ 8（创建 notebook 时设备数要选 8）"; exit 1
fi
echo "  ✓ $out"

# ── 4) 启动（WORLD_SIZE=1，JAX 单机 pmap over 8 卡）──
echo "=== 启动训练（8 卡，LSGD_K=$LSGD_K $LSGD_MODE，iters=$ITERS，关卡=$LEVELS_FILE 权重=$LEVEL_WEIGHTS 宝箱=$CRATE_REWARD_COEF 探索=$EXPLORE_REWARD_COEF 炸墙=$BRICK_REWARD_COEF 课程=${CURRICULUM_JSON##*/}）==="
"$WORK/cmd" "cd /root/private_data/qqt-gpu-sim; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500; export LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE CKPT_DIR=ckpt CKPT_EVERY=$CKPT_EVERY CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=$CKPT_EVERY; L=\${LEVELS_FILE:+\"--levels \$LEVELS_FILE --level-weights \"\$LEVEL_WEIGHTS\"\"}; C=\${CRATE_REWARD_COEF:+\"--crate-reward-coef \$CRATE_REWARD_COEF --crate-reward-anneal-steps \$CRATE_REWARD_ANNEAL\"}; X=\${EXPLORE_REWARD_COEF:+\"--explore-reward-coef \$EXPLORE_REWARD_COEF --explore-reward-anneal-steps \$EXPLORE_REWARD_ANNEAL\"}; B=\${BRICK_REWARD_COEF:+\"--brick-reward-coef \$BRICK_REWARD_COEF --reward-anneal-k \$REWARD_ANNEAL_K\"}; U=\${CURRICULUM_JSON:+\"--curriculum-json \$CURRICULUM_JSON\"}; E=\${EVAL_VS:+\"--eval-vs \$EVAL_VS --eval-every \${EVAL_EVERY:-200}\"}; nohup python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs $NUM_ENVS --num-steps $NUM_STEPS --minibatch $MINIBATCH --epochs 2 --iters $ITERS --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE \$L \$C \$X \$B \$U \$E $FRESH > /root/private_data/train_8gpu.log 2>&1 & echo STARTED" | tail -1

# ── 5) 30s 后健康检查 ──
echo "=== 等待 30s 做启动健康检查 ==="
sleep 30
last=$("$WORK/cmd" "tail -4 /root/private_data/train_8gpu.log 2>/dev/null || echo NOLOG" | tail -4 | grep -v "spawn ssh\|password:" | tail -2 | tr '\n' ' ' | head -c 200)
if [ -z "$last" ] || echo "$last" | grep -q NOLOG; then
  echo "  ✗ 日志为空（进程没起来）"; exit 1
elif echo "$last" | grep -qiE "Traceback|Error|Exception|exited with"; then
  echo "  ✗ 启动即报错 -> $last"; exit 1
else
  echo "  ✓ $last"
fi

echo "=== 全部拉起，日志 /root/private_data/train_8gpu.log ==="
echo "=== 监控：bash deploy_10node/watch_8gpu.sh nodes_8gpu.txt ==="
echo "=== 拉断点：bash deploy_10node/pull_ckpt_local.sh nodes_8gpu.txt ==="
