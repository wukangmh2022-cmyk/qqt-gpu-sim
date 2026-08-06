"""验证用户假设：模型不上去杀，是进攻策略缺陷还是地图分布问题。

把 idle 静止靶放在不同行（顶部/中部/底部），模型（纯 open 40% 起点）
去杀 —— 若顶部靶打不死/和局、底部轻松杀，则实锤"进攻策略不覆盖顶部"。

对照组：idle 在 (6,6) 中部、(11,6) 底部；实验组：idle 在 (1,6) 顶部。
同时统计模型前四行占比与超时率（1800 tick 双方存活 = 磨平）。
"""

from __future__ import annotations

import sys

import torch

from sim.bots import make_bot
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.train import load_fixed_checkpoint
from sim.move import center_cell

DEV = "cpu"
CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=1.0, ring_fraction=0.0, hazard_fraction=0.0,
                open_crate_cross=True, hit_attr_penalty=2,
                open_growth_bombs=3, open_growth_blast=3, open_growth_speed=0.84)


def duel_idle_at(sim, net, row: int, episodes: int = 128) -> dict:
    """P0 = 模型，P1 = idle 静止靶放在 (row, 6)（每局固定，不随 reset 变化）。"""
    n = sim.num_envs
    dev = sim.device
    idle = make_bot(sim, "idle")
    sim.reset_all()
    w0 = w1 = dr = rounds = 0
    timeouts = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    row_cnt0 = [0] * sim.cfg.height
    alive_cnt0 = 0
    while rounds < episodes:
        # 每局开始把 P1 钉在 (row, 6)，且保持静止（idle bot 本身不动，这里兜底）
        sim.pos[:, 1, 0] = row + 0.5
        sim.pos[:, 1, 1] = 6.5
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        with torch.no_grad():
            a0 = net.act(obs, mm[:, 0], bm[:, 0], 0)[0]
        a1 = idle.act(obs, mm[:, 1], bm[:, 1], 1)
        _, d, info = sim.step(torch.stack([a0, a1], dim=1))
        cell = center_cell(sim.pos)
        alive0 = sim.alive[:, 0]
        for e in range(n):
            if bool(alive0[e]):
                row_cnt0[int(cell[e, 0, 0])] += 1
                alive_cnt0 += 1
        just = d & ~done
        w0 += int((just & info["winner"][:, 0]).sum())
        w1 += int((just & info["winner"][:, 1]).sum())
        dr += int((just & ~info["winner"][:, 0] & ~info["winner"][:, 1]).sum())
        done |= d
        rounds += int(just.sum())
        # 超时（1800 tick 双方都活着 = 磨平）
        if bool((sim.t >= sim.cfg.max_steps).any()):
            timeouts += int((sim.t >= sim.cfg.max_steps).sum())
            sim.reset_all()
            done.zero_()
        if bool(done.all()):
            sim.reset_all()
            done.zero_()
    tot = max(1, rounds)
    top4 = sum(row_cnt0[:4]) / max(1, alive_cnt0)
    return {"win": w0 / tot, "draw": dr / tot, "loss": w1 / tot,
            "timeout_frac": timeouts / max(1, rounds), "top4": top4}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/course_547m.pt"
    sim = BatchedSim(CFG, 128, device=DEV, seed=0)
    net = load_fixed_checkpoint(path, CFG.obs_shape, DEV)
    net.eval()
    print(f"[{path}] 纯open 40%起点 模型 vs idle静止靶 @不同行\n")
    for row, tag in ((1, "顶部(行1)"), (6, "中部(行6)"), (11, "底部(行11)")):
        r = duel_idle_at(sim, net, row)
        print(f"idle@{tag:<10}: win {r['win']:.1%} / draw {r['draw']:.1%} "
              f"/ loss {r['loss']:.1%}   超时率 {r['timeout_frac']:.1%}   "
              f"模型前4行占比 {r['top4']:.1%}")


if __name__ == "__main__":
    main()
