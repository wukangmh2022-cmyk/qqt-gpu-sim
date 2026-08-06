"""5x3 验收：对战（vs 5x2/rw8/cnn）+ 三件套行为指标（5x3 与 5x2 对照）。

行为指标与"时间差连锁"奖励对齐（用户验收标准）：
- 放炮间隔分布：两次放泡之间的 tick 数 —— 应该变长（治"啪啪啪啪"连丢）；
- 被连锁泡平均剩余引信：被对手连锁点燃时它的引信还剩多少 —— 应该变低
  （对方学会"等快爆的泡续"，我方被连时剩下的引信就少……注意这里测的是
  自己打自己，指标意义 = 双方都会"连老泡"的话，被连泡剩余引信低）；
- 单次连爆泡数：一次爆炸事件连锁引爆的泡数 —— 应该变多。

对战 & 行为指标都跑在 corridor 70%（open 0.3 / ring 0 / hazard 0），
与课程训练主线地图一致。行为指标用 5x3 自战，另跑 5x2 自战做对照。

用法：
    python scripts/acceptance_5x3.py --device mps --episodes 256
"""

from __future__ import annotations

import argparse
import time

import torch

from sim.config import SimConfig
from sim.factory import make_sim
from train.train import load_fixed_checkpoint

CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.3, ring_fraction=0.0, hazard_fraction=0.0)


@torch.no_grad()
def run_duel(sim, net0, net1, episodes: int) -> dict:
    """net0/1 冻结网络（player 0/1），返回胜负统计 + 行为指标。"""
    n = sim.num_envs
    dev = sim.device
    w0 = w1 = draw = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    # 行为指标（只统计 player 0）
    bomb_intervals = []            # 放泡间隔（tick）
    last_bomb = torch.zeros(n, dtype=torch.long, device=dev) - 1000
    chain_rem_fuse = []            # 被连锁点燃的泡：点燃前剩余引信
    chain_bursts = []              # 每次爆炸事件的连锁泡数（>0 时记录）
    rounds = 0
    while rounds < episodes:
        fuse_before = sim.fuse.clone()            # 区分自然走完 vs 被连锁点燃
        live_before = ((sim.owner == 0) & (sim.fuse > 0)).flatten(1).sum(dim=1)
        since0 = sim.since_bomb[:, 0].clone()

        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = net0.act(obs, mm[:, 0], bm[:, 0], 0)[0]
        a1 = net1.act(obs, mm[:, 1], bm[:, 1], 1)[0]
        acts = torch.stack([a0, a1], dim=1)
        _, d, info = sim.step(acts)

        # 放泡间隔：这 tick 放了泡（live 增加）→ 距上次放泡 = since_before + 1
        placed = ((sim.owner == 0) & (sim.fuse > 0)).flatten(1).sum(dim=1) > live_before
        if bool(placed.any()):
            iv = since0[placed] + 1
            bomb_intervals.append(iv.cpu())

        # 连锁：trig 且 点燃前引信 > 0 = 被连锁提前点燃（fuse_before==0 = 自然走完）
        trig = info["trig"]
        if bool(trig.any()):
            chained = trig & (fuse_before > 0)
            burst = int(chained.sum().item())
            if burst > 0:
                chain_bursts.append(burst)
            rem = fuse_before[chained]
            if rem.numel() > 0:
                chain_rem_fuse.append(rem.float().cpu())

        just = d & ~done
        w0 += int((just & info["winner"][:, 0]).sum())
        w1 += int((just & info["winner"][:, 1]).sum())
        draw += int((just & ~info["winner"][:, 0] & ~info["winner"][:, 1]).sum())
        done |= d
        rounds += int(just.sum())
        if bool(done.all()):
            sim.reset_all()
            done.zero_()
            last_bomb.zero_()

    def _mean(xs):
        return float(torch.cat(xs).float().mean()) if xs else float("nan")

    total = w0 + w1 + draw
    return {
        "win0": w0 / total, "draw": draw / total, "win1": w1 / total,
        "bomb_interval": _mean(bomb_intervals),
        "chain_rem_fuse": _mean(chain_rem_fuse),
        "chain_burst": _mean([torch.tensor([b]) for b in chain_bursts]),
        "chain_events": len(chain_bursts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--ckpt-dir", default="ckpt")
    args = ap.parse_args()
    device = args.device
    if device.startswith("mps") and not torch.backends.mps.is_available():
        device = "cpu"
        print("[device] MPS 不可用，回退 CPU")
    torch.manual_seed(0)
    d = args.ckpt_dir

    names = ["5x2", "5x3", "rw8", "cnn"]
    nets = {nm: load_fixed_checkpoint(f"{d}/duel_{nm}.pt", CFG.obs_shape, device)
            for nm in names}
    for nm, net in nets.items():
        print(f"[load] {nm}: obs={net.obs_shape} arch={net.arch}")

    sim = make_sim(CFG, 256, backend="torch", device=device, seed=0)

    print("\n=== 对战（5x3 为 player 0，corridor 70%）===")
    for opp in ["5x2", "rw8", "cnn"]:
        t0 = time.time()
        r = run_duel(sim, nets["5x3"], nets[opp], args.episodes)
        print(f"5x3 vs {opp}: win {r['win0']:.1%} / draw {r['draw']:.1%} / "
              f"loss {r['win1']:.1%}  ({args.episodes} 局, {time.time()-t0:.0f}s)")

    print("\n=== 行为指标（自战，双方同模型）===")
    for nm in ["5x2", "5x3"]:
        t0 = time.time()
        r = run_duel(sim, nets[nm], nets[nm], args.episodes)
        print(f"{nm} 自战: 放炮间隔 {r['bomb_interval']:.1f}tick | "
              f"被连锁泡剩余引信 {r['chain_rem_fuse']:.1f}tick | "
              f"单次连爆 {r['chain_burst']:.2f} 泡 ({r['chain_events']} 次连锁事件, "
              f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
