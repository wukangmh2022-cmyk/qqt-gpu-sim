#!/bin/bash
# 24 机 × 2 卡有损同步（Local SGD）训练编排（复制自 launch_10nodes.sh v2：
# 自检 + 健康检查 + 断点语义修正；节点数 = 节点文件行数，本脚本按 24 机准备）
#
# nodes_24x2.txt 兼容两种格式（可混用；# 开头为注释行；第 1 条 = rank 0，
# 即 coordinator，用它的 172.31.x IP 做 MASTER_ADDR）：
#   A) 一行三列:  <ssh端口> <主机> <密码>
#      例: 10326 ssh.zzai.scnet.cn MNIDMRHPC41XQJM
#   B) SCNET 原文（与 10ssh.env 相同，密码在下一行）:
#      ssh -p 11071 root@ssh.zzai.scnet.cn
#      GWIA2GXZSE8SK5H
# 密码请只用字母/数字（expect 嵌脚本，含 $ [ ] " 会炸）。
#
# 用法:  bash launch_24nodes.sh nodes_24x2.txt [--deploy] [--resume]
#   --deploy 先推代码+wheels 并跑 setup（首次必用；已部署过可省）
#   --resume 接续各节点 ckpt/ 最新断点（默认 --fresh 全新开始；本轮 obs 13→14
#   通道，旧断点不兼容）。
#
# 流程: 部署 → 全节点自检(import/卡数/代码版本，失败即中止) → 取 IP →
#       rank0 先起 → 1-23 台 30s 内拉起 → 30s 后健康检查(空日志/Traceback 即中止)
#       → 提示用 watch_24nodes.sh 监控 + pull_ckpt_local.sh 拉快照。
set -u
NODES_FILE="${1:?用法: launch_24nodes.sh nodes_24x2.txt [--deploy] [--resume]}"
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
PATCH="${PATCH:-3}"
ADV_TOP_FRAC="${ADV_TOP_FRAC:-0.25}"
EMA_DECAY="${EMA_DECAY:-0.999}"
LSGD_K="${LSGD_K:-256}"; LSGD_MODE="${LSGD_MODE:-param}"
# 标准化关卡（241 张 QQ堂地图；levels.json 随部署包分发，auto-detect 找到即启用）
LEVELS_FILE="${LEVELS_FILE:-}"; LEVEL_WEIGHTS="${LEVEL_WEIGHTS:-}"
# 极简归一化零和奖励（默认 zero-shaping，完全依托 HP 演进与课程）
CRATE_REWARD_COEF="${CRATE_REWARD_COEF:-0.0}"
CRATE_REWARD_ANNEAL="${CRATE_REWARD_ANNEAL:-0}"
EXPLORE_REWARD_COEF="${EXPLORE_REWARD_COEF:-0.0}"
EXPLORE_REWARD_ANNEAL="${EXPLORE_REWARD_ANNEAL:-0}"
BRICK_REWARD_COEF="${BRICK_REWARD_COEF:-0.0}"
REWARD_ANNEAL_K="${REWARD_ANNEAL_K:-1.2}"
# Spawn-Distance 零规则空间课程（Stage1 贴脸近战 → Stage2 近距破障 → Stage3 中距迷宫 → Stage4 全量 241 图）
CURRICULUM_JSON="${CURRICULUM_JSON:-web/assets/maps/curriculum.json}"

# ── 读节点（兼容 A/B 两种格式，可混用）──
N_PORT=(); N_HOST=(); N_PASS=()
PENDING_PORT=""; PENDING_HOST=""
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"                      # 兼容 Windows 换行
  line="${line#"${line%%[![:space:]]*}"}"   # 去行首空白
  [ -z "$line" ] && continue
  [ "${line:0:1}" = "#" ] && continue
  if [ -n "$PENDING_PORT" ]; then           # B 格式：上一行 ssh 的密码在本行
    N_PORT+=("$PENDING_PORT"); N_HOST+=("$PENDING_HOST"); N_PASS+=("$line")
    PENDING_PORT=""; PENDING_HOST=""
    continue
  fi
  if [[ "$line" =~ ^ssh[[:space:]]+-p[[:space:]]+([0-9]+)[[:space:]]+root@([^[:space:]]+)$ ]]; then
    PENDING_PORT="${BASH_REMATCH[1]}"; PENDING_HOST="${BASH_REMATCH[2]}"
    continue
  fi
  read -r p h pw extra <<< "$line"          # A 格式：一行三列
  [ -n "${p:-}" ] && [ -n "${h:-}" ] && [ -n "${pw:-}" ] || { echo "节点行格式不对: $line"; exit 1; }
  [ -z "${extra:-}" ] || { echo "三列格式行多了字段（只需 端口 主机 密码）: $line"; exit 1; }
  N_PORT+=("$p"); N_HOST+=("$h"); N_PASS+=("$pw")
done < "$NODES_FILE"
[ -z "$PENDING_PORT" ] || { echo "ssh 行后缺密码行: ${PENDING_HOST}:${PENDING_PORT}"; exit 1; }
NW=${#N_PORT[@]}
echo "=== 共 $NW 台节点（rank0 = ${N_HOST[0]}:${N_PORT[0]}）==="
[ "$NW" -lt 2 ] && { echo "至少 2 台"; exit 1; }
[ "$NW" -eq 24 ] || echo "注意: 节点数 $NW ≠ 24（将按 WORLD_SIZE=$NW 继续；掉线降级时可少台重启）"

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
  # 部署包优先取仓库根最新构建的 dcu_deploy_24node.tar.gz（每次 launch 前
  # 用它重建包，避免 /tmp/dcu_deploy 下旧包忘同步）；包内容与节点数无关，
  # 24node 包没有时回落 10node 包/旧位置，两者都没有则报错。
  PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_24node.tar.gz"
  [ -f "$PKG" ] || PKG="$(cd "$(dirname "$0")/.." && pwd)/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || PKG="/tmp/dcu_deploy/dcu_deploy_10node.tar.gz"
  [ -f "$PKG" ] || { echo "找不到部署包（仓库根 dcu_deploy_24node.tar.gz / dcu_deploy_10node.tar.gz 或 /tmp/dcu_deploy/ 都没有）"; exit 1; }
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
rm -rf /tmp/jaxbomb_24node && mkdir -p /tmp/jaxbomb_24node/scripts /tmp/jaxbomb_24node/web/assets/maps
cp -r "$ROOT/jax_bomb" /tmp/jaxbomb_24node/
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_24node/levels.json
cp "$ROOT/web/assets/maps/levels.json" /tmp/jaxbomb_24node/web/assets/maps/levels.json
cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_24node/web/assets/maps/curriculum.json 2>/dev/null || true
cp "$ROOT/web/assets/maps/curriculum.json" /tmp/jaxbomb_24node/curriculum.json 2>/dev/null || true
for s in quick_check_levels.py quick_check_bush.py quick_check_crate_semantics.py \
         quick_check_obs_move.py quick_check_js_jax_move.py quick_check_anti_tunnel.py; do
  cp "$ROOT/scripts/$s" /tmp/jaxbomb_24node/scripts/ 2>/dev/null
done
find /tmp/jaxbomb_24node -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
(cd /tmp/jaxbomb_24node && tar czf /tmp/jaxbomb_24node.tgz jax_bomb levels.json curriculum.json web scripts)
for i in $(seq 0 $((NW-1))); do
  "$WORK/scp_$i" /tmp/jaxbomb_24node.tgz /root/private_data/ >/dev/null 2>&1
  "$WORK/cmd_$i" "cd /root/private_data && rm -rf qqt-gpu-sim && mkdir qqt-gpu-sim && tar xzf jaxbomb_24node.tgz -C qqt-gpu-sim && ls qqt-gpu-sim/jax_bomb/jax_env.py >/dev/null && echo RANK${i}_CODE_OK" | tail -1
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
echo "=== 启动训练（MASTER_ADDR=${MASTER}，LSGD_K=${LSGD_K} LSGD_MODE=${LSGD_MODE}，iters=${ITERS}，patch=${PATCH}，adv_top_frac=${ADV_TOP_FRAC}，ema_decay=${EMA_DECAY}，课程=${CURRICULUM_JSON##*/}）==="
for i in $(seq 0 $((NW-1))); do
  # LD_PRELOAD 用 glob 取实际 openmpi 版本；LSGD_K/MODE 在本地展开透传
  "$WORK/cmd_$i" "cd /root/private_data/qqt-gpu-sim; source /opt/dtk/env.sh 2>/dev/null; unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; export LD_PRELOAD=\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); export WORLD_SIZE=$NW RANK=$i MASTER_ADDR=$MASTER MASTER_PORT=29500; export LSGD_K=$LSGD_K LSGD_MODE=$LSGD_MODE CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=30; L=\${LEVELS_FILE:+"--levels \$LEVELS_FILE --level-weights \"\$LEVEL_WEIGHTS\""}; C=\${CRATE_REWARD_COEF:+\"--crate-reward-coef \$CRATE_REWARD_COEF --crate-reward-anneal-steps \$CRATE_REWARD_ANNEAL\"}; X=\${EXPLORE_REWARD_COEF:+\"--explore-reward-coef \$EXPLORE_REWARD_COEF --explore-reward-anneal-steps \$EXPLORE_REWARD_ANNEAL\"}; B=\${BRICK_REWARD_COEF:+\"--brick-reward-coef \$BRICK_REWARD_COEF --reward-anneal-k \$REWARD_ANNEAL_K\"}; U=\${CURRICULUM_JSON:+\"--curriculum-json \$CURRICULUM_JSON\"}; E=\${EVAL_VS:+\"--eval-vs \$EVAL_VS --eval-every \${EVAL_EVERY:-200}\"}; nohup python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch $PATCH --heads 4 --ff-factor 4 --adv-top-frac $ADV_TOP_FRAC --ema-decay $EMA_DECAY --num-envs $NUM_ENVS --num-steps $NUM_STEPS --minibatch $MINIBATCH --epochs 2 --iters $ITERS --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE \$L \$C \$X \$B \$U \$E $FRESH > /root/private_data/train_r$i.log 2>&1 & echo RANK${i}_STARTED" | tail -1
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
  echo "=== 监控：bash watch_24nodes.sh nodes_24x2.txt（60s 刷新）==="
  echo "=== 拉 rank0 快照：bash pull_ckpt_local.sh nodes_24x2.txt ==="
else
  echo "=== 有节点启动失败：看上方日志；修复后重跑本脚本（自动接续，--deploy 可省）==="
  exit 1
fi
