#!/usr/bin/env python3
"""定位 GPU→CPU 同步点（_local_scalar_dense）的调用栈分布。

用法：python -m scripts.prof_syncs --n 512 --opp astar
输出：每个同步点来源的 top-15 调用栈 + 每 tick 计数。
"""
import argparse
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic
from sim.bots import make_bot
from scripts.prof_env import build_env, tick, sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--opp", default="astar")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[prof_syncs] device={device} n={args.n} ticks={args.ticks} opp={args.opp}")
    _, sim, learner, bot = build_env(args.n, device, args.opp)
    actions = torch.zeros((args.n, 2, 2), dtype=torch.long, device=device)
    for _ in range(3):
        tick(sim, learner, bot, actions)
    sync(device)

    import torch.profiler as prof
    with prof.profile(activities=[prof.ProfilerActivity.CPU],
                      record_shapes=False, with_stack=True) as p:
        for _ in range(args.ticks):
            tick(sim, learner, bot, actions)
        sync(device)

    # 用 group_by_stack_n 聚合同步点来源
    agg = p.key_averages(group_by_stack_n=8)
    rows = []
    for ev in agg:
        if ev.key.startswith("aten::_local_scalar_dense"):
            total = ev.self_cpu_time_total
            cnt = ev.count
            # 栈里第一个非 torch 框架的文件（源码定位）
            src = "?"
            for fr in ev.stack or []:
                s = str(fr)
                if "site-packages/torch" in s or "torch/autograd" in s \
                        or "torch/_ops" in s or "torch/utils" in s \
                        or "torch/_tensor" in s or "torch/_C" in s:
                    continue
                src = s
                break
            rows.append((total, cnt, src))
    rows.sort(reverse=True)
    print(f"\n=== 同步点来源（{args.ticks} tick，共 {sum(c for _, c, _ in rows)} 次）===")
    for total, cnt, src in rows[:20]:
        print(f"  {cnt:5d} 次 ({cnt/args.ticks:.1f}/tick)  {total/args.ticks*1000:7.1f} ms/tick  {src}")
    print(f"  合计 {sum(c for _, c, _ in rows)} 次 / {args.ticks} tick "
          f"= {sum(c for _, c, _ in rows)/args.ticks:.1f}/tick")
    # 打印第一个同步点的完整栈（不过滤），看真实帧格式
    print("\n=== 第一个同步点完整栈（raw）===")
    for ev in agg:
        if ev.key.startswith("aten::_local_scalar_dense") and ev.stack:
            for fr in ev.stack[:14]:
                print(f"  {fr}")
            break


if __name__ == "__main__":
    main()
