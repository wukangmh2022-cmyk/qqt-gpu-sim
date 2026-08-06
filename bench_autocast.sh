#!/usr/bin/env bash
# autocast 真实加速基准：同一配置 autocast off/on 各跑 ~3 个迭代，
# 取 csv 平均 sps（端到端 = 模拟 + PPO 更新）做对比。
# 用法：ssh 远端 source env.sh 后执行；结果写到 /root/private_data/bench_ac.csv
set -uo pipefail
source /opt/dtk-26.04/env.sh >/dev/null 2>&1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
cd /root

COMMON=(--backend torch --device cuda
        --num-envs 4096 --rollout-steps 128 --minibatches 4
        --map-mode corridor --arch mlp
        --total-steps 20000000 --single-stage
        --snapshot-every 9999 --time-budget 110)

echo "== [A] autocast OFF =="
python -m train.train "${COMMON[@]}" \
    --ckpt /tmp/ac_off.pt --log-csv /tmp/ac_off.csv >/tmp/ac_off.log 2>&1
echo "off done"

echo "== [B] autocast ON =="
python -m train.train "${COMMON[@]}" --autocast \
    --ckpt /tmp/ac_on.pt --log-csv /tmp/ac_on.csv >/tmp/ac_on.log 2>&1
echo "on done"

echo "=== summary ==="
awk -F, 'NR>1{s+=$NF;n++}END{if(n)printf "autocast OFF  avg_sps=%.0f  (n=%d)\n",s/n,n}' /tmp/ac_off.csv
awk -F, 'NR>1{s+=$NF;n++}END{if(n)printf "autocast ON   avg_sps=%.0f  (n=%d)\n",s/n,n}' /tmp/ac_on.csv
echo "=== tail off ==="; tail -2 /tmp/ac_off.csv
echo "=== tail on ===";  tail -2 /tmp/ac_on.csv
cp /tmp/ac_off.csv /tmp/ac_on.csv /root/private_data/ 2>/dev/null
echo DONE
