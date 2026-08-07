#!/bin/bash
set -e
export PYTHONPATH=/Users/pippo/operater-dev/qqt-gpu-sim
cd /Users/pippo/operater-dev/qqt-gpu-sim
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260806_014651.npz --scene 比武 --fmt webm --out docs/demo_candidates/cand_16_open_vs547_比武.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_192030.npz --scene 雪地 --fmt webm --out docs/demo_candidates/cand_17_open_vs597_雪地.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_190251.npz --scene 野外 --fmt webm --out docs/demo_candidates/cand_18_open_vscnn_野外.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_213039.npz --scene 水面 --fmt webm --out docs/demo_candidates/cand_19_open_vs670_水面.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260806_013145.npz --scene 矿洞 --fmt webm --out docs/demo_candidates/cand_20_open_vs793_矿洞.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260806_082136.npz --scene 中国城 --fmt webm --out docs/demo_candidates/cand_21_open_vs1023_中国城.webm --seconds 15
echo ALL_DONE
