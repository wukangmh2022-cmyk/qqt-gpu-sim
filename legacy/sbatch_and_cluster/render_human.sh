#!/bin/bash
set -e
export PYTHONPATH=/Users/pippo/operater-dev/qqt-gpu-sim
cd /Users/pippo/operater-dev/qqt-gpu-sim
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_192206.npz --scene 比武 --fmt webm --out docs/demo_candidates/cand_16_open_human_比武.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_192342.npz --scene 雪地 --fmt webm --out docs/demo_candidates/cand_17_open_human_雪地.webm --seconds 15
.venv/bin/python scripts/render_gif.py --rec recordings/rec_20260805_192425.npz --scene 矿洞 --fmt webm --out docs/demo_candidates/cand_18_open_human_矿洞.webm --seconds 15
echo ALL_DONE
