"""真实训练入口（断点续训默认开启）。

用法与 jax_bomb.multicard_train 相同，区别是面向长训：
  - 默认开启检查点：--ckpt-dir ckpt（每 rank 独立文件 ckpt_<iter>_r<rank>.pkl）
    --ckpt-every 60（分钟），可用环境变量 CKPT_DIR / CKPT_EVERY 覆盖；
  - 默认长训：--iters 2000（~44s/iter 单卡 ≈ 24h；2 卡更快），可用
    ITERS 环境变量或 --iters 覆盖；
  - 启动自动接续 ckpt/ 下最新检查点；--fresh 或删除 ckpt/ 目录 = 全新开始；
  - 收到 SIGTERM/SIGINT（平台抢占/手动停）时当前 iter 结束后存盘退出，
    下次启动自动从断点继续。

示例（10 机×2 卡生产配置，7.46M 参数）：
  python3 -m jax_bomb.train_real \
    --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 \
    --num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 \
    --lsgd-k 256 --lsgd-mode param --iters 2000
"""
import os
import sys

if "CKPT_DIR" not in os.environ:
    os.environ["CKPT_DIR"] = "ckpt"
if "CKPT_EVERY" not in os.environ:
    os.environ["CKPT_EVERY"] = "60"
if "CKPT_LOCAL_DIR" not in os.environ:
    os.environ["CKPT_LOCAL_DIR"] = "ckpt_local"
if "CKPT_LOCAL_EVERY" not in os.environ:
    os.environ["CKPT_LOCAL_EVERY"] = "30"
if not any(a.startswith("--iters") for a in sys.argv[1:]):
    sys.argv.append("--iters")
    sys.argv.append(os.environ.get("ITERS", "2000"))

from jax_bomb.multicard_train import main

if __name__ == "__main__":
    main()
