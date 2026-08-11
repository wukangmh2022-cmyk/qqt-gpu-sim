#!/usr/bin/env python3
"""Python 插桩：统计每 tick 的 GPU→CPU 同步调用点（item/equal）来源。

monkeypatch torch.Tensor.item 和 torch.equal，用 traceback 记录调用点，
跑 N tick 输出每调用点次数。用于定位热路径的 45 个/tick 同步。
用法：python -m scripts.prof_syncs2 --n 512 --ticks 10 --opp astar
"""
import argparse
import traceback
from collections import Counter

import torch

from scripts.prof_env import build_env, tick, sync

counter = Counter()
_orig_item = torch.Tensor.item
_orig_equal = torch.equal


def _key():
    st = traceback.extract_stack(limit=8)
    for fr in reversed(st[:-1]):
        fn = fr.filename.replace("\\", "/")
        if "/site-packages/torch/" in fn or fn.endswith("/traceback.py") \
                or fn.endswith("/prof_syncs2.py"):
            continue
        return f"{fn.split('/')[-1]}:{fr.lineno}"
    return "?"


def _patched_item(self, *a, **k):
    counter["item " + _key()] += 1
    return _orig_item(self, *a, **k)


def _patched_equal(*a, **k):
    counter["equal " + _key()] += 1
    return _orig_equal(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--opp", default="astar")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[prof_syncs2] device={device} n={args.n} ticks={args.ticks} opp={args.opp}")
    _, sim, learner, bot = build_env(args.n, device, args.opp)
    actions = torch.zeros((args.n, 2, 2), dtype=torch.long, device=device)
    for _ in range(3):
        tick(sim, learner, bot, actions)
    sync(device)

    torch.Tensor.item = _patched_item
    torch.equal = _patched_equal
    try:
        for _ in range(args.ticks):
            tick(sim, learner, bot, actions)
        sync(device)
    finally:
        torch.Tensor.item = _orig_item
        torch.equal = _orig_equal

    print(f"\n=== 同步点来源（{args.ticks} tick，共 {sum(counter.values())} 次，"
          f"{sum(counter.values())/args.ticks:.1f}/tick）===")
    for k, v in counter.most_common(30):
        print(f"  {v:5d} 次 ({v/args.ticks:.1f}/tick)  {k}")


if __name__ == "__main__":
    main()
