#!/bin/bash
set -e
export PYTHONPATH=/Users/pippo/operater-dev/qqt-gpu-sim
cd /Users/pippo/operater-dev/qqt-gpu-sim
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_182108.npz --scene 比武  --fmt webm --out docs/demo_candidates/cand_01_open_比武.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_174848.npz --scene 中国城 --fmt webm --out docs/demo_candidates/cand_02_open_中国城.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_180228.npz --scene 功夫  --fmt webm --out docs/demo_candidates/cand_03_corr_功夫.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_180635.npz --scene 夺宝  --fmt webm --out docs/demo_candidates/cand_04_corr_夺宝.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_175921.npz --scene 抢包子 --fmt webm --out docs/demo_candidates/cand_05_corr_抢包子.webm --seconds 15
.venv/bin/python scripts/render_gif.py --sim --bot hunter --ckpt ckpt/duel_cnn.pt --scene 矿洞 --fmt webm --out docs/demo_candidates/cand_07_hunter_vs_cnn_矿洞.webm --seconds 15
.venv/bin/python scripts/render_gif.py --sim --bot astar --ckpt ckpt/course_1023m.pt --scene 沙漠 --fmt webm --out docs/demo_candidates/cand_08_astar_vs_1023_沙漠.webm --seconds 15
.venv/bin/python scripts/render_gif.py --sim --bot hunter --ckpt ckpt/course_1023m.pt --scene 雪地 --fmt webm --out docs/demo_candidates/cand_09_hunter_vs_1023_雪地.webm --seconds 15
.venv/bin/python scripts/render_gif.py --sim --bot astar --ckpt ckpt/duel_cnn.pt --scene 水面 --fmt webm --out docs/demo_candidates/cand_10_astar_vs_cnn_水面.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_192030.npz --scene 雪地  --fmt webm --out docs/demo_candidates/cand_11_open_雪地.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_190251.npz --scene 野外  --fmt webm --out docs/demo_candidates/cand_12_open_野外.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260806_014651.npz --scene 矿洞  --fmt webm --out docs/demo_candidates/cand_13_open_矿洞.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_213039.npz --scene 水面  --fmt webm --out docs/demo_candidates/cand_14_open_水面.webm --seconds 15
.venv/bin/python scripts/render_gif.py --sim --bot hunter --ckpt ckpt/course_1023m.pt --map open --scene 沙漠 --fmt webm --out docs/demo_candidates/cand_15_open_hunter_vs_1023_沙漠.webm --seconds 15
echo ALL_DONE
