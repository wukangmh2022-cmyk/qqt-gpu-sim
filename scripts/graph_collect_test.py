#!/usr/bin/env python3
"""阶段 1 原型：CUDA graph 接入 collect 的 DCU 加速验证。

对比 eager（observe→legal_mask→step，early_exit 开）vs graph
（capture 固定轮 + replay）的每 tick 时间。网络前向在 graph 外（阶段 1）。
n 默认 2048（小规模，几秒，不抢训练 GPU 算力 —— 用户允许小函数测试）。
用法：python -m scripts.graph_collect_test --n 2048 --ticks 20
"""
import argparse
import time

import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--ticks", type=int, default=20)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "graph 测试需要 cuda/HIP"
    print(f"[graph_collect] device=cuda n={args.n} ticks={args.ticks}")

    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.5, ring_fraction=0, hazard_fraction=0,
                    timeout_draw=True, combo_reward=0.10, combo_gap_factor=0.9)
    sim = BatchedSim(cfg, args.n, device="cuda", seed=0)
    n = args.n
    actions = torch.zeros(n, 2, 2, dtype=torch.long, device="cuda")

    def sync():
        torch.cuda.synchronize()

    # ---- eager 基准（early_exit 开，auto_reset=False）----
    for _ in range(5):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        sim.step(actions, auto_reset=False)
    sync()
    t0 = time.perf_counter()
    for _ in range(args.ticks):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        sim.step(actions, auto_reset=False)
    sync()
    t1 = time.perf_counter()
    t_eager = (t1 - t0) / args.ticks
    print(f"eager: {t_eager*1000:.2f} ms/tick")

    # ---- graph（固定轮，capture + replay）----
    obs_buf = torch.empty(n, *cfg.obs_shape, dtype=torch.float16, device="cuda")
    mmask_buf = torch.empty(n, 2, 5, dtype=torch.bool, device="cuda")
    bmask_buf = torch.empty(n, 2, 2, dtype=torch.bool, device="cuda")
    reward_buf = torch.empty(n, 2, dtype=torch.float32, device="cuda")
    done_buf = torch.empty(n, dtype=torch.float32, device="cuda")
    winner_buf = torch.empty(n, 2, dtype=torch.float32, device="cuda")
    print("capturing...")
    t0 = time.perf_counter()
    sim.capture_graph(actions, obs_buf, mmask_buf, bmask_buf,
                      reward_buf, done_buf, winner_buf)
    t1 = time.perf_counter()
    print(f"capture: {(t1-t0)*1000:.0f} ms")

    for _ in range(5):
        sim.graph_refill_rand()
        sim.graph_step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.ticks):
        sim.graph_refill_rand()
        sim.graph_step()
    sync()
    t1 = time.perf_counter()
    t_graph = (t1 - t0) / args.ticks
    print(f"graph: {t_graph*1000:.2f} ms/tick  (speedup {t_eager/t_graph:.1f}x)")

    # ---- graph 输出 sanity（reward/done 有限、mask 合法）----
    print(f"  reward 有限: {bool(torch.isfinite(reward_buf).all())}, "
          f"done 有 True: {bool((done_buf > 0).any())}, "
          f"winner 有限: {bool(torch.isfinite(winner_buf).all())}")


if __name__ == "__main__":
    main()
