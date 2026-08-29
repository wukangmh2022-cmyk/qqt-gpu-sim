#!/bin/bash
# 双机 × 8 卡训练编排（2 rank，每 rank 本地 pmap over 8 DCU；跨机 RCCL/MPI）
#
# 与 launch_10nodes.sh 同一套训练代码，布局差异：
#   - 每台机器 8 卡（自检要求 local_device_count=8）
#   - 两台机器**不同账号** → nodes 文件每行带 用户名@主机 和 各自的上传目录
#   - 代码不上传由脚本推：用户通过网页把 upload_2x8gpu/qqt_upload 文件夹传到
#     两台机器的数据盘并跑 setup_upload.sh（或 launch --setup 代跑）；
#     后续代码迭代用 --push 增量同步，无需再走网页上传。
#
# nodes_2x8.txt 每行: <ssh端口> <用户名@主机> <密码> <上传目录>    # 共 2 行
#   第 1 行 = rank0（coordinator，它的 172.31.x IP 做 MASTER_ADDR）
#   例: 10832 actts28ojm@ssh.zzai.scnet.cn PASS1 /root/private_data
#   密码只用字母/数字（expect 内嵌，含 $ [ ] " 会炸）。
#
# 用法: bash launch_2x8gpu.sh nodes_2x8.txt [--setup] [--push] [--resume]
#   --setup  远程执行 setup_upload.sh（解代码+装依赖+自检；已手动跑过可省）
#   --push   本地重建最新代码 tgz 并 scp 到两台机器覆盖（日常迭代用）
#   --resume 接续各机 ckpt/ 最新断点（默认 --fresh 从头训；本轮 obs 13→14
#            通道，旧断点不兼容）
set -u
NODES_FILE="${1:?用法: launch_2x8gpu.sh nodes_2x8.txt [--setup] [--push] [--resume]}"
shift
DO_SETUP=0; DO_PUSH=0; FRESH="--fresh"          # 本轮从头训（默认）
for a in "$@"; do
  [ "$a" = "--setup" ] && DO_SETUP=1
  [ "$a" = "--push" ] && DO_PUSH=1
  { [ "$a" = "--resume" ] || [ "$a" = "--no-fresh" ]; } && FRESH=""
done
[ -f "$NODES_FILE" ] || { echo "找不到 $NODES_FILE"; exit 1; }

# 生产训练参数（260B 全量口径；试跑用 ITERS=100 覆盖）
# steps/iter = 2×NUM_ENVS×NUM_STEPS = 2×32768×256 ≈ 16.78M；15500 iter ≈ 260B
ITERS="${ITERS:-15500}"
NUM_ENVS="${NUM_ENVS:-32768}"            # 全局 env 数 = 2 机 × 8 卡 × 2048
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32768}"
LSGD_K="${LSGD_K:-0}"; LSGD_MODE="${LSGD_MODE:-param}"   # 仅 2 机：逐位一致精确同步（参数量 7.5M，每 iter ~30MB 可忽略）
LEVELS_FILE="${LEVELS_FILE:-}"; LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-empty=0.05,功夫=0.1,比武=0.15}"
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.5}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-30000000000}"  # 30B 长退火（动态 α 兜底提前失效）
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.01}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-30000000000}"
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.05}"
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
# 课程默认不启用（docs/vit_train_log.md §2.4）；启用：CURRICULUM_JSON=web/assets/maps/curriculum.json
CURRICULUM_JSON="${CURRICULUM_JSON:-}"

# ── 读节点（恰好 2 行：port user@host pass rdir）──
N_PORT=(); N_UH=(); N_PASS=(); N_RDIR=()
while read -r p uh pw rd; do
  [ -z "${p:-}" ] && continue
  N_PORT+=("$p"); N_UH+=("$uh"); N_PASS+=("$pw"); N_RDIR+=("${rd:-/root/private_data}")
done < "$NODES_FILE"
NW=${#N_PORT[@]}
[ "$NW" = "2" ] || { echo "需要恰好 2 台（读到 $NW 行）"; exit 1; }
echo "=== 双机 8 卡: rank0=${N_UH[0]}:${N_PORT[0]} rank1=${N_UH[1]}:${N_PORT[1]} ==="

# ── expect 封装（每节点一份，密码内嵌）──
WORK=/tmp/ndrun2x8; rm -rf "$WORK"; mkdir -p "$WORK"
for i in 0 1; do
  cat > "$WORK/cmd_$i" <<EOF
#!/usr/bin/expect -f
set timeout 300
set cmd [lindex \$argv 0]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p ${N_PORT[$i]} ${N_UH[$i]} \$cmd
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "TIMEOUT_$i" }
}
EOF
  cat > "$WORK/scp_$i" <<EOF
#!/usr/bin/expect -f
set timeout 600
set src [lindex \$argv 0]
set dst [lindex \$argv 1]
spawn scp -o StrictHostKeyChecking=accept-new -P ${N_PORT[$i]} \$src ${N_UH[$i]}:\$dst
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "SCP_TIMEOUT_$i" }
}
EOF
  chmod +x "$WORK/cmd_$i" "$WORK/scp_$i"
done

# ── 1) setup（可选）：跑网页上传文件夹里的 setup_upload.sh ──
if [ "$DO_SETUP" = "1" ]; then
  for i in 0 1; do
    echo "=== rank $i setup（${N_RDIR[$i]}/qqt_upload）==="
    "$WORK/cmd_$i" "cd ${N_RDIR[$i]}/qqt_upload 2>/dev/null || { echo NO_QQT_UPLOAD; exit 9; }; bash setup_upload.sh ${N_RDIR[$i]} 2>&1 | tail -6" | tail -6
  done
fi

# ── 2) push（可选）：本地重建最新代码 tgz 覆盖两台机器 ──
if [ "$DO_PUSH" = "1" ]; then
  echo "=== 推最新代码 ==="
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  rm -rf /tmp/jaxbomb_2x8 && mkdir -p /tmp/jaxbomb_2x8/scripts /tmp/jaxbomb_2x8/web/assets/maps
  cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_2x8/
  cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_2x8/
  cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_2x8/web/assets/maps/ 2>/dev/null || true
  for s in quick_check_levels.py quick_check_bush.py quick_check_crate_semantics.py \
           quick_check_obs_move.py quick_check_js_jax_move.py quick_check_anti_tunnel.py \
           quick_check_push.py; do
    cp "$ROOT/scripts/$s" /tmp/jaxbomb_2x8/scripts/ 2>/dev/null
  done
  find /tmp/jaxbomb_2x8 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
  (cd /tmp/jaxbomb_2x8 && tar czf /tmp/jaxbomb_2x8.tgz jax_bomb levels.json web scripts)
  for i in 0 1; do
    "$WORK/scp_$i" /tmp/jaxbomb_2x8.tgz ${N_RDIR[$i]}/ >/dev/null 2>&1
    "$WORK/cmd_$i" "cd ${N_RDIR[$i]} && tar xzf jaxbomb_2x8.tgz -C qqt-gpu-sim && grep -q push_t qqt-gpu-sim/jax_bomb/jax_env.py && echo RANK${i}_CODE_OK" | tail -1
  done
fi

# ── 3) 自检（任何一台失败即中止）──
echo "=== 自检（import jax/optax/jax_bomb + 卡数=8）==="
ALL_OK=1
for i in 0 1; do
  out=$("$WORK/cmd_$i" "source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); cd ${N_RDIR[$i]}/qqt-gpu-sim 2>/dev/null || exit 9; python3 -c 'import jax, optax; from jax_bomb.jax_train import ppo_update_gradsync, ppo_update_lsgd; print(\"SELFCHECK_OK\", jax.__version__, \"devices=\", jax.local_device_count())'" 2>/dev/null | grep SELFCHECK_OK | tail -1)
  if [ -z "$out" ]; then
    echo "  ✗ rank $i ${N_UH[$i]}: 自检失败（没跑 setup_upload.sh / 环境损坏 / 代码不对）"
    ALL_OK=0
  else
    ndev=$(echo "$out" | grep -oE 'devices= [0-9]+' | grep -oE '[0-9]+$')
    if [ "$ndev" = "8" ]; then
      echo "  ✓ rank $i ${N_UH[$i]}: $out"
    else
      echo "  ✗ rank $i ${N_UH[$i]}: 卡数 $ndev ≠ 8（创建 notebook 时设备数要选 8）"
      ALL_OK=0
    fi
  fi
done
[ "$ALL_OK" = "1" ] || { echo "=== 有节点自检失败，中止启动 ==="; exit 1; }

# ── 4) 取 rank0 内网 IP 做 MASTER_ADDR ──
ip=$("$WORK/cmd_0" "hostname -I" | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
MASTER="${ip:-127.0.0.1}"
[ "$MASTER" = "127.0.0.1" ] && { echo "rank0 内网 IP 获取失败"; exit 1; }
echo "=== MASTER_ADDR=$MASTER ==="

# ── 5) 启动（rank0 先起做 coordinator，rank1 立即跟上）──
echo "=== 启动训练（iters=$ITERS 权重=$LEVEL_WEIGHTS 宝箱=$CRATE_REWARD_COEF 探索=$EXPLORE_REWARD_COEF 炸墙=$BRICK_REWARD_COEF 课程=${CURRICULUM_JSON##*/} fresh=$([ -n "$FRESH" ] && echo yes || echo no)）==="
for i in 0 1; do
  "$WORK/cmd_$i" "cd ${N_RDIR[$i]}/qqt-gpu-sim; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=2 RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500; export LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=30; L=\${LEVELS_FILE:+\"--levels \$LEVELS_FILE --level-weights \"\$LEVEL_WEIGHTS\"\"}; C=\${CRATE_REWARD_COEF:+\"--crate-reward-coef \$CRATE_REWARD_COEF --crate-reward-anneal-steps \$CRATE_REWARD_ANNEAL\"}; X=\${EXPLORE_REWARD_COEF:+\"--explore-reward-coef \$EXPLORE_REWARD_COEF --explore-reward-anneal-steps \$EXPLORE_REWARD_ANNEAL\"}; B=\${BRICK_REWARD_COEF:+\"--brick-reward-coef \$BRICK_REWARD_COEF --reward-anneal-k \$REWARD_ANNEAL_K\"}; U=\${CURRICULUM_JSON:+\"--curriculum-json \$CURRICULUM_JSON\"}; E=\${EVAL_VS:+\"--eval-vs \$EVAL_VS --eval-every \${EVAL_EVERY:-200}\"}; nohup python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs $NUM_ENVS --num-steps $NUM_STEPS --minibatch $MINIBATCH --epochs 2 --iters $ITERS --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE \$L \$C \$X \$B \$U \$E $FRESH > ${N_RDIR[$i]}/train_r$i.log 2>&1 & echo RANK${i}_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 5   # coordinator 先起
done

# ── 6) 30s 健康检查 ──
echo "=== 等待 30s 做启动健康检查 ==="
sleep 30
ALL_OK=1
for i in 0 1; do
  last=$("$WORK/cmd_$i" "tail -4 ${N_RDIR[$i]}/train_r$i.log 2>/dev/null || echo NOLOG" | tail -4 | grep -v "spawn ssh\|password:" | tail -2 | tr '\n' ' ' | head -c 150)
  if [ -z "$last" ] || echo "$last" | grep -q NOLOG; then
    echo "  ✗ rank $i ${N_UH[$i]}: 日志为空（进程没起来）"; ALL_OK=0
  elif echo "$last" | grep -qiE "Traceback|Error|Exception|exited with"; then
    echo "  ✗ rank $i ${N_UH[$i]}: 启动即报错 -> $last"; ALL_OK=0
  else
    echo "  ✓ rank $i ${N_UH[$i]}: $last"
  fi
done
[ "$ALL_OK" = "1" ] || { echo "=== 有节点启动失败，看各自 train_r$i.log 排查 ==="; exit 1; }

echo "=== 全部拉起。日志: ${N_RDIR[0]}/train_r0.log 与 ${N_RDIR[1]}/train_r1.log ==="
echo "=== 断点: 各机 ${N_RDIR[$i]}/qqt-gpu-sim/ckpt/（每 30 分钟）==="
