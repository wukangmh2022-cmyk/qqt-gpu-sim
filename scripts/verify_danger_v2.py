#!/usr/bin/env python3
"""danger_map v2（双缓冲传播）正确性验证：与当前实现逐位一致。

场景覆盖：corridor 多泡连锁、不同 blast 档、open 空场、砖挡火。
不改 blast.py，只对比"新算法"与"现实现"的输出。验证通过后新算法
才替换进 blast.py。

用法：python -m scripts.verify_danger_v2
"""
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
import sim.blast as B


def danger_v2(fuse, wall, blast, fuse_max, brick=None, max_chain=16, exp=2.0,
              early_exit=True):
    """阶段 A 双缓冲传播版：权重 + 剩余距离一起传播，pad 数 ÷14。

    与现实现的差异只在阶段 A 内部（传播结构），阶段 B 完全一致；
    输出必须与现实现逐位一致（danger 显示修复的语义：同 tick 连锁组
    统一取组内最危险值、泡挡火、brick 挡火）。
    """
    should_ee = True if early_exit is None else early_exit
    bombed = fuse > 0
    if should_ee and not bool(bombed.any()):
        return torch.zeros_like(fuse, dtype=torch.float32)
    w_raw = 1.0 - (fuse.float() - 1.0) / float(fuse_max)
    weight = torch.where(fuse > 0, w_raw.clamp_min(0.0).pow(exp),
                         torch.zeros_like(fuse, dtype=torch.float32))
    brick_t = brick if brick is not None else torch.zeros_like(bombed, dtype=torch.bool)
    solid = bombed | brick_t
    not_solid = (~solid).float()
    passable = (~wall).float()
    h, w = fuse.shape[-2], fuse.shape[-1]
    n = fuse.shape[0]
    dev = fuse.device

    if max_chain > 1:
        ww = weight.clone()
        # 双缓冲：波前权重 fw + 剩余传播距离 fd（每格自己的 blast 档）。
        # 内层循环：每方向从波前逐格传播 max_b 次，fd 记录每格剩余距离
        # （原版按 blast 档分组 Σb 次挪格，新法统一 max_b 次 → pad 减半；
        # 传播语义不变：每轮从 newly 泡出发传播自己的 blast 档距离）。
        blast_f = blast.float() if not isinstance(blast, int) \
            else torch.full_like(ww, float(blast))
        max_b = int(blast_f.max()) if blast_f.numel() else 0
        fw = torch.where(bombed, ww, torch.zeros_like(ww))
        fd = torch.where(bombed, blast_f, torch.zeros_like(ww))
        spread = torch.zeros_like(ww)
        for _ in range(max_chain):
            spread.zero_()
            for drow, dcol in B._DIRS:
                fw_p = fw
                fd_p = fd
                for _ in range(max_b):
                    fw1 = B._shift(fw_p, drow, dcol) * passable
                    fd1 = B._shift(fd_p, drow, dcol) * passable
                    fd1 = fd1 - 1.0
                    keep = fd1 >= 0          # 第 b 格（fd1=0）也要记录；耗尽才停
                    fw1 = torch.where(keep, fw1, torch.zeros_like(fw1))
                    spread = torch.maximum(spread, fw1)   # 先记录（覆盖泡格）
                    # 再挡穿透：泡/brick 格记录后不穿（与 rays 同规则）
                    fw1 = fw1 * not_solid
                    fd1 = fd1 * not_solid
                    fw_p, fd_p = fw1, fd1
            # 炮格接收：**只有炮格**接收权重（spread 乘 bombed，与原版一致——
            # 非炮格权重不写进 ww，否则 stage B 会从非炮格 seed 多扩散）。
            # 新权重 = 本轮到位的波前权重（> 现有 → 激活为新波前）。
            spread = spread * bombed.float()
            newly = (spread > ww) & bombed
            ww = torch.maximum(ww, spread)
            fw = torch.where(newly, spread, torch.zeros_like(ww))
            fd = torch.where(newly, blast_f, torch.zeros_like(ww))
            if should_ee and not bool((newly).any()):
                break
        weight = ww

    # 阶段 B（与现实现相同）
    seed = weight * passable
    danger = seed.clone()
    if isinstance(blast, int):
        for drow, dcol in B._DIRS:
            front = seed
            for _ in range(blast):
                front = B._shift(front, drow, dcol) * passable
                danger = torch.maximum(danger, front)
                front = front * not_solid
    else:
        max_b = int(blast.max()) if blast.numel() else 0
        for b in range(1, max_b + 1):
            src = seed * (blast == b)
            for drow, dcol in B._DIRS:
                front = src
                for _ in range(b):
                    front = B._shift(front, drow, dcol) * passable
                    danger = torch.maximum(danger, front)
                    front = front * not_solid
    return danger


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ok = True
    # 场景：corridor 随机多泡（含长链/不同档/砖/墙），每 tick 对比
    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=60, open_fraction=0.5)
    sim = BatchedSim(cfg, 128, device=dev, seed=42)
    sim.reset_all()
    # 随机放泡制造富泡状态（两边用同一 sim 状态）
    from train.model import ActorCritic
    from sim.bots import make_bot
    from scripts.prof_env import tick as ptick
    learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(dev).eval()
    for p in learner.parameters():
        p.requires_grad_(False)
    bot = make_bot(sim, "astar")
    actions = torch.zeros((128, 2, 2), dtype=torch.long, device=dev)
    for i in range(25):
        ptick(sim, learner, bot, actions)
        fuse = sim.fuse
        blast_m = sim._blast_map()
        d1 = B.danger_map(fuse, sim.wall, blast_m, cfg.fuse,
                          sim.brick, cfg.max_chain, early_exit=True)
        d2 = danger_v2(fuse, sim.wall, blast_m, cfg.fuse,
                       sim.brick, cfg.max_chain, early_exit=True)
        if not torch.equal(d1, d2):
            diff = (d1 != d2).sum().item()
            print(f"[MISMATCH] tick {i}: {diff} 格不同, max|d1-d2|="
                  f"{(d1 - d2).abs().max().item():.4f}")
            # 找第一个不同的格子
            idx = (d1 != d2).nonzero()[0]
            print(f"  首个差异 @ {idx.tolist()}  d1={d1[tuple(idx)].item():.4f} "
                  f"d2={d2[tuple(idx)].item():.4f}")
            ok = False
            break
        # 每 5 tick 报告一次危险度活跃情况
        if i % 5 == 0:
            print(f"  tick {i}: 泡格数={int((fuse > 0).sum())} "
                  f"危险非零格={int((d1 > 0).sum())} 一致")
    print("=== PASS ===" if ok else "=== FAIL ===")
    return ok


if __name__ == "__main__":
    main()
