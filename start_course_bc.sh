#!/bin/bash
# 重头训练：BC 预训练起点 + README 课程流程（1v1 single-stage）+ 全奖励退火。
# 从 ckpt/bc_init.pt（69 局人类数据 BC 预训练，format v2）起步，白手起家：
#   warmup 150M 启蒙期（只打 bot + 固定陪练，近零权重先打打得过的）
#   → 固定陪练锚点（rw8/5x2/5x3/cnn，ELO 绝对意义）+ 模型池自博弈
#   → 全程：多泡 astar/hunter 常驻固定陪练 + BC 在线引导（coef 0.3）
#   → 探索退火（α=1-tanh(1.2·2·击杀率)，放炮塑形随击杀能力归零）
#   → 自杀重罚 2.0（治自爆）、超时平局（防龟缩）、combo 0.05（逼无伤连击）
#   → flee 追击（对手逃跑才追、不限距离）
# 课程结束（time-budget 存盘）→ python -m train.bc --resume 最终ckpt --epochs 3
#   用人类数据收尾校准（BC 收尾）。
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
source /opt/dtk-26.04/env.sh >/dev/null 2>&1
cd /root
nohup python -m train.train --backend torch --device cuda --arch mlp --single-stage \
  --map-mode corridor --open-fraction 0.5 --ring-fraction 0 --hazard-fraction 0 \
  --num-envs 5632 --total-steps 1_200_000_000 --rollout-steps 128 --minibatches 4 \
  --warmup-steps 150_000_000 --fixed-opp-prob 0.4 --bot-opponents greedy,astar,hunter \
  --fixed-bots astar,hunter \
  --chase-reward 0.02 --chase-adj 0.05 --explore-anneal \
  --suicide-penalty 2.0 --timeout-draw --combo-reward 0.05 --combo-gap-factor 0.9 \
  --bc-data recordings/ --bc-coef 0.3 --bc-batch 256 --bc-every 1 \
  --fixed-ckpt rw8=private_data/duel_rw8.pt --fixed-ckpt 5x2=private_data/duel_5x2.pt \
  --fixed-ckpt 5x3=private_data/duel_5x3.pt --fixed-ckpt cnn=private_data/duel_cnn.pt \
  --snapshot-every 20 --time-budget 43200 --seed 0 --oversample-dying 3 \
  --resume ckpt/bc_init.pt --ckpt private_data/duel_course.pt \
  --log-csv private_data/train_course_bc.csv \
  > train_course_bc.log 2>&1 < /dev/null &
echo "course_bc started pid $!"
