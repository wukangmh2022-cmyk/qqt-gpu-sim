"""训练端 vs 启动器通行一致性手动检查（非常驻测试，需要时跑）。

启动器人类走 60Hz 帧级小步（成长后 ~0.105 格/帧），训练端用 step_len
（0.3 格）试探判定 —— 两者都走 sim/move._resolve_axis（同源）。本脚本
在 corridor 真图 + 在场泡 + 满成长速度下铺点，断言"训练说能走 ⇔ 启动器
实际能动"双向一致。

用法：python -m scripts.passability_check
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from play.duel import _player_frame_move  # noqa: E402
from sim.config import SimConfig  # noqa: E402
from sim.move import _EPS  # noqa: E402
from sim.torch_sim import BatchedSim  # noqa: E402


def main() -> None:
    cfg = SimConfig(height=13, width=13, blast=3, max_bombs=10, fuse=60,
                    map_mode="corridor", speed=3.0, tick_hz=10)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.spd_g[0, 0] = 2.1          # 满成长速度：帧步 = 3×2.1/60 = 0.105 格
    sim.blast_cap[0, 0] = 7
    sim.fuse[0, 6, 5] = 30
    sim.owner[0, 6, 5] = 0
    sim.bomb_blast[0, 6, 5] = 3
    sim.fuse[0, 8, 8] = 55
    sim.owner[0, 8, 8] = 1
    sim.bomb_blast[0, 8, 8] = 3

    OFFS = [0.19, 0.195, 0.2, 0.205, 0.21, 0.79, 0.795, 0.8, 0.805, 0.81,
            0.02, 0.5, 0.98]      # 临界偏移（前缘 0.2/0.8）+ 中心 + 边缘
    h, w = 13, 13
    n_pts = len(OFFS) * len(OFFS)
    train_pass = torch.zeros(h * w * n_pts, 4, dtype=torch.bool)
    launch_move = torch.zeros(h * w * n_pts, 4, dtype=torch.bool)
    i = 0
    for r in range(h):
        for c in range(w):
            for fy in OFFS:
                for fx in OFFS:
                    sim.pos[0, 0, 0] = r + fy
                    sim.pos[0, 0, 1] = c + fx
                    mm, _ = sim.legal_mask()
                    train_pass[i] = mm[0, 0, :4]
                    before = (float(sim.pos[0, 0, 0]), float(sim.pos[0, 0, 1]))
                    for mv in range(4):
                        _player_frame_move(sim, mv, 1.0 / 60.0)
                        after = (float(sim.pos[0, 0, 0]), float(sim.pos[0, 0, 1]))
                        launch_move[i, mv] = (abs(after[0] - before[0])
                                              + abs(after[1] - before[1])) > _EPS * 2
                        sim.pos[0, 0, 0] = before[0]
                        sim.pos[0, 0, 1] = before[1]
                    i += 1

    diff = train_pass != launch_move
    n_diff = int(diff.sum())
    only_train = int(((train_pass) & (~launch_move)).sum())
    only_launch = int(((~train_pass) & (launch_move)).sum())
    print(f"总方向断言: {train_pass.numel()} | 不一致: {n_diff}")
    print(f"  训练说能走但启动器没动: {only_train}")
    print(f"  训练说不能走但启动器动了: {only_launch}")
    if n_diff:
        for ii in diff.nonzero()[:8]:
            i_, mv = ii.tolist()
            r_, c_ = divmod(i_ // (len(OFFS) * len(OFFS)), w)
            k = i_ % (len(OFFS) * len(OFFS))
            fy, fx = OFFS[k // len(OFFS)], OFFS[k % len(OFFS)]
            print(f"  点({r_ + fy:.3f},{c_ + fx:.3f}) 方向{mv} "
                  f"训练={bool(train_pass[i_, mv])} 启动器={bool(launch_move[i_, mv])}")
        sys.exit(1)
    print("✓ 训练端与启动器通行判定完全一致")


if __name__ == "__main__":
    main()
