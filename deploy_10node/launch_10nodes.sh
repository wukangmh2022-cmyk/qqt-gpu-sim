#!/bin/bash
# 10 机 × 2 卡有损同步（Local SGD）训练编排（v2：自检 + 健康检查 + 断点语义修正）
#
# nodes.txt 每行格式:  <ssh端口> <主机> <密码>        # 与现有 notebook 入口一致
#   例: 10326 ssh.zzai.scnet.cn MNIDMRHPC41XQJM
# 第 1 行 = rank 0（coordinator，它的 172.31.x IP 做 MASTER_ADDR）。
# 密码请只用字母/数字（expect 嵌脚本，含 $ [ ] " 会炸）。
#
# 用法:  bash launch_10nodes.sh nodes.txt [--deploy] [--resume]
#   --deploy 先推代码+wheels 并跑 setup（首次必用；已部署过可省）
#   --resume 接续各节点 ckpt/ 最新断点（默认 --fresh 全新开始；本轮 obs 13→14
#   通道，旧断点不兼容）。
#
# 流程: 部署 → 全节点自检(import/卡数/代码版本，失败即中止) → 取 IP →
#       rank0 先起 → 1-9 台 30s 内拉起 → 30s 后健康检查(空日志/Traceback 即中止)
#       → 提示用 watch_10nodes.sh 监控 + pull_ckpt_local.sh 拉快照。
set -u
NODES_FILE="${1:?用法: launch_10nodes.sh nodes.txt [--deploy] [--resume]}"
shift
DO_DEPLOY=0; FRESH="--fresh"          # 本轮从头训（默认）；--resume 接续旧断点
for a in "$@"; do
  [ "$a" = "--deploy" ] && DO_DEPLOY=1
  { [ "$a" = "--resume" ] || [ "$a" = "--no-fresh" ]; } && FRESH=""
done
[ -f "$NODES_FILE" ] || { echo "找不到 $NODES_FILE"; exit 1; }

# 生产训练参数（改这里即可）：steps/iter = 2×32768×256 ≈ 16.78M；
# 260B 全量 ≈ 15500 iter。试跑用 ITERS= 覆盖（如 ITERS=100）。
ITERS="${ITERS:-15500}"
NUM_ENVS="${NUM_ENVS:-32768}"
NUM_STEPS="${NUM_STEPS:-256}"
MINIBATCH="${MINIBATCH:-32768}"
LSGD_K="${LSGD_K:-256}"; LSGD_MODE="${LSGD_MODE:-param}"
# 标准化关卡（241 张 QQ堂地图；levels.json 随部署包分发，auto-detect 找到即启用）
LEVELS_FILE="${LEVELS_FILE:-}"; LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-empty=0.05,功夫=0.1,比武=0.15}"
# 开箱成长 bootstrap 奖励：30B 全局步线性退火（动态 α 兜底：击杀率起来提前失效）
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.5}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-30000000000}"  # 30B：开箱奖励长退火（改回 500M 即旧版快退火）
# 探索 novelty：0.01×195 格 = 单局封顶 ~1.95 分，远低于全血伤害 7.5 与击杀 10；
# 与 crate 同款 30B 退火。日志每 iter 打 explore= 可监控占比
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.01}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-30000000000}"
# 炸墙奖励：每炸一块砖双方各 +0.025（治出生点 3 格死锁——crate 链路太长学不会）
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.05}"
# 动态退火斜率 k：α_dyn = 1-tanh(k·每局击杀率)，击杀率上来塑形自动归零
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
# Spawn-Distance 课程：默认不启用（260B 全图均匀采样；论文课程职能已被稠密塑形
# 替代，见 docs/vit_train_log.md §2.4）。要启用：CURRICULUM_JSON=web/assets/maps/curriculum.json
CURRICULUM_JSON="${CURRICULUM_JSON:-}"

# ── 读节点 ──
N_PORT=(); N_HOST=(); N_PASS=()
while read -r p h pw; do
  [ -z "$p" ] && continue
  N_PORT+=("$p"); N_HOST+=("$h"); N_PASS+=("$pw")
done < "$NODES_FILE"
NW=${#N_PORT[@]}
echo "=== 共 $NW 台节点（rank0 = ${N_HOST[0]}:${N_PORT[0]}）==="
[ "$NW" -lt 2 ] && { echo "至少 2 台"; exit 1; }

# 临时 expect 封装（每节点一份，密码内嵌）
WORK=/tmp/ndrun; rm -rf "$WORK"; mkdir -p "$WORK"
for i in $(seq 0 $((NW-1))); do
  cat > "$WORK/cmd_$i" <<EOF
#!/usr/bin/expect -f
set timeout 300
set cmd [lindex \$argv 0]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p ${N_PORT[$i]} root@${N_HOST[$i]} \$cmd
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "TIMEOUT_$i" }
}
EOF
  cat > "$WORK/scp_$i" <<EOF
#!/usr/bin/expect -f
set timeout 300
set src [lindex \$argv 0]
set dst [lindex \$argv 1]
spawn scp -o StrictHostKeyChecking=accept-new -P ${N_PORT[$i]} \$src root@${N_HOST[$i]}:\$dst
expect {
  "password:" { send "${N_PASS[$i]}\r"; exp_continue }
  "yes/no" { send "yes\r"; exp_continue }
  eof { }
  timeout { puts "SCP_TIMEOUT_$i" }
}
EOF
  chmod +x "$WORK/cmd_$i" "$WORK/scp_$i"
done

# ── 1) 部署（可选）──
if [ "$DO_DEPLOY" = "1" ]; then
  echo "=== 部署代码+wheels+setup ==="
  # 部署包优先取仓库根最新构建的 dcu_deploy_10node.tar.gz（每次 launch 前
  # 用它重建包，避免 /tmp/dcu_deploy 下旧包忘同步）；两者都没有则报错。
  PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || PKG="/tmp/dcu_deploy/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || { echo "找不到部署包（仓库根 dcu_deploy_10node.tar.gz 或 /tmp/dcu_deploy/ 都没有）"; exit 1; }
  for i in $(seq 0 $((NW-1))); do
    echo "  rank $i: scp 包 ($PKG)"
    "$WORK/scp_$i" "$PKG" /root/private_data/ >/dev/null 2>&1
    echo "  rank $i: 解压 + setup"
    # 兼容两种包布局：dcu_deploy/ 子目录（v2）或裸文件（旧包）都正确处理
    "$WORK/cmd_$i" "cd /root/private_data && tar xzf dcu_deploy_10node.tar.gz; if [ -d dcu_deploy ]; then cd dcu_deploy; fi; bash setup_notebook.sh 2>&1 | tail -4" | tail -5
  done
fi

# ── 1.4) 推最新代码（每次必做：保证运行的代码就是仓库当前版本，
#          即使 --deploy 用的是旧部署包也以这里为准）──
echo "=== 推最新代码到全部节点 ==="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf /tmp/jaxbomb_10node && mkdir -p /tmp/jaxbomb_10node/scripts
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_10node/
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_10node/levels.json
for s in quick_check_levels.py quick_check_bush.py quick_check_crate_semantics.py \
         quick_check_obs_move.py quick_check_js_jax_move.py quick_check_anti_tunnel.py; do
  cp "$ROOT/scripts/$s" /tmp/jaxbomb_10node/scripts/ 2>/dev/null
done
find /tmp/jaxbomb_10node -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_10node && tar czf /tmp/jaxbomb_10node.tgz jax_bomb levels.json scripts)
for i in $(seq 0 $((NW-1))); do
  "$WORK/scp_$i" /tmp/jaxbomb_10node.tgz /root/private_data/ >/dev/null 2>&1
  "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim && mkdir qqt-gpu-sim && tar xzf jaxbomb_10node.tgz -C qqt-gpu-sim && ls qqt-gpu-sim/jax_bomb/jax_env.py >/dev/null && echo RANK${i}_CODE_OK" | tail -1
done

# ── 1.5) 全节点自检（无论是否 deploy 都做；任何一台失败即中止）──
echo "=== 自检（import jax/optax/jax_bomb + 卡数）==="
ALL_OK=1
for i in $(seq 0 $((NW-1))); do
  out=$("$WORK/cmd_$i" "source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); cd /root/private_data/qqt-gpu-sim 2>/dev/null || exit 9; python3 -c 'import jax, optax; from jax_bomb.jax_train import ppo_update_gradsync, ppo_update_lsgd; print(\"SELFCHECK_OK\", jax.__version__, \"devices=\", jax.local_device_count())'" 2>/dev/null | grep SELFCHECK_OK | tail -1)
  if [ -z "$out" ]; then
    echo "  ✗ rank $i ${N_HOST[$i]}: 自检失败（setup 未生效 / 环境损坏 / 代码版本不对）"
    ALL_OK=0
  else
    ndev=$(echo "$out" | grep -oE 'devices= [0-9]+' | grep -oE '[0-9]+')
    if [ "$ndev" = "2" ]; then
      echo "  ✓ rank $i ${N_HOST[$i]}: $out"
    else
      echo "  ✗ rank $i ${N_HOST[$i]}: 卡数 $ndev ≠ 2（创建 notebook 时设备数要选 2）"
      ALL_OK=0
    fi
  fi
done
[ "$ALL_OK" = "1" ] || { echo "=== 有节点自检失败，中止启动（修复后重跑，--deploy 可省）==="; exit 1; }

# ── 2) 取各节点 172.31.x IP ──
echo "=== 取各节点 IP ==="
N_IP=()
for i in $(seq 0 $((NW-1))); do
  ip=$("$WORK/cmd_$i" "hostname -I" | grep -oE '172\.31\.[0-9]+\.[0-9]+' | head -1)
  N_IP+=("$ip")
  echo "  rank $i ${N_HOST[$i]} -> $ip"
done
MASTER="${N_IP[0]:-127.0.0.1}"
[ "$MASTER" = "127.0.0.1" ] && { echo "rank0 IP 获取失败"; exit 1; }

# ── 3) 启动（rank0 先起做 coordinator，其余立即跟上）──
echo "=== 启动训练（MASTER_ADDR=$MASTER，LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE，iters=$ITERS，关卡=$LEVELS_FILE 权重=$LEVEL_WEIGHTS 宝箱奖励=$CRATE_REWARD_COEF/$CRATE_REWARD_ANNEAL 探索=$EXPLORE_REWARD_COEF/$EXPLORE_REWARD_ANNEAL 炸墙=$BRICK_REWARD_COEF 课程=${CURRICULUM_JSON##*/}）==="
for i in $(seq 0 $((NW-1))); do
  # LD_PRELOAD 用 glob 取实际 openmpi 版本；LSGD_K/MODE 在本地展开透传
  "$WORK/cmd_$i" "cd /root/private_data/qqt-gpu-sim; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=$NW RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500; export LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=30; L=\${LEVELS_FILE:+"--levels \$LEVELS_FILE --level-weights \"\$LEVEL_WEIGHTS\""}; C=\${CRATE_REWARD_COEF:+\"--crate-reward-coef \$CRATE_REWARD_COEF --crate-reward-anneal-steps \$CRATE_REWARD_ANNEAL\"}; X=\${EXPLORE_REWARD_COEF:+\"--explore-reward-coef \$EXPLORE_REWARD_COEF --explore-reward-anneal-steps \$EXPLORE_REWARD_ANNEAL\"}; B=\${BRICK_REWARD_COEF:+\"--brick-reward-coef \$BRICK_REWARD_COEF --reward-anneal-k \$REWARD_ANNEAL_K\"}; U=\${CURRICULUM_JSON:+\"--curriculum-json \$CURRICULUM_JSON\"}; E=\${EVAL_VS:+\"--eval-vs \$EVAL_VS --eval-every \${EVAL_EVERY:-200}\"}; nohup python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 --num-envs $NUM_ENVS --num-steps $NUM_STEPS --minibatch $MINIBATCH --epochs 2 --iters $ITERS --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE \$L \$C \$X \$B \$U \$E $FRESH > /root/private_data/train_r$i.log 2>&1 & echo RANK$i_STARTED" | tail -1
  [ "$i" = "0" ] && sleep 5   # coordinator 先起
done

# ── 4) 30s 后健康检查：空日志 / 立即 Traceback 视为启动失败并中止 ──
echo "=== 等待 30s 做启动健康检查 ==="
sleep 30
ALL_OK=1
for i in $(seq 0 $((NW-1))); do
  last=$("$WORK/cmd_$i" "tail -4 /root/private_data/train_r$i.log 2>/dev/null || echo NOLOG" | tail -4 | grep -v "spawn ssh\|password:" | tail -2 | tr '\n' ' ' | head -c 150)
  if [ -z "$last" ] || echo "$last" | grep -q NOLOG; then
    echo "  ✗ rank $i ${N_HOST[$i]}: 日志为空（进程没起来）"; ALL_OK=0
  elif echo "$last" | grep -qiE "Traceback|Error|Exception|exited with"; then
    echo "  ✗ rank $i ${N_HOST[$i]}: 启动即报错 -> $last"; ALL_OK=0
  else
    echo "  ✓ rank $i ${N_HOST[$i]}: $last"
  fi
done
if [ "$ALL_OK" = "1" ]; then
  echo "=== 全部拉起，日志 /root/private_data/train_r<i>.log ==="
  echo "=== 监控：bash watch_10nodes.sh nodes.txt（60s 刷新）==="
  echo "=== 拉 rank0 快照：bash pull_ckpt_local.sh nodes.txt ==="
else
  echo "=== 有节点启动失败：看上方日志；修复后重跑本脚本（自动接续，--deploy 可省）==="
  exit 1
fi
