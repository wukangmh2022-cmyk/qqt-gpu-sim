#!/usr/bin/env python3
"""danger_map 阶段 B v2（双缓冲）语义验证：朴素逐炮参考（独立算法）。

阶段 B 语义 = 从每颗炮（修正后的权重）向 4 方向扩散自己的 blast 距离，
泡/brick 挡火（覆盖记录但不穿透）。朴素参考用 Python 逐 env 逐炮循环
（不用 _shift），与生产实现算法完全不同 —— 逐位一致即证明 v2 阶段 B 正确。

stage A（连锁修正）用生产实现内的 v2（已单独验证过），只对 stage B 独立验证。
用法：python -m scripts.verify_danger_stageB
"""
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
import sim.blast as B


def naive_stageB(weight, passable, not_solid, bombed, blast_map, h, w):
    """朴素参考：逐炮 4 方向扩散 blast 距离，泡/brick 挡火。"""
    n = weight.shape[0]
    danger = weight.clone()                       # 炮格自身权重
    d = weight.device
    blast_f = torch.where(bombed, blast_map.float(),
                          torch.zeros_like(weight))
    for e in range(n):
        for r in range(h):
            for c in range(w):
                wgt = weight[e, r, c]
                if wgt <= 0:
                    continue
                b = int(blast_f[e, r, c])
                for drow, dcol in B._DIRS:
                    for s in range(1, b + 1):
                        nr, nc = r + drow * s, c + dcol * s
                        if not (0 <= nr < h and 0 <= nc < w):
                            break
                        dval = danger[e, nr, nc]
                        if wgt > dval:
                            danger[e, nr, nc] = wgt
                        # 泡/brick 挡火：覆盖后不穿透
                        if not bool(not_solid[e, nr, nc]):
                            break
    return danger


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=60,
                    open_fraction=0.5)
    sim = BatchedSim(cfg, 32, device=dev, seed=42)
    sim.reset_all()
    from train.model import ActorCritic
    from sim.bots import make_bot
    from scripts.prof_env import tick as ptick
    learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(dev).eval()
    for p in learner.parameters():
        p.requires_grad_(False)
    bot = make_bot(sim, "astar")
    actions = torch.zeros((32, 2, 2), dtype=torch.long, device=dev)

    ok = True
    for i in range(25):
        ptick(sim, learner, bot, actions)
        fuse = sim.fuse
        bm = sim._blast_map()
        d = B.danger_map(fuse, sim.wall, bm, cfg.fuse, sim.brick,
                         cfg.max_chain, early_exit=True)
        # 生产实现的 stage A 输出（修正后权重）—— 复刻 v2 stage A
        bombed = fuse > 0
        w_raw = 1.0 - (fuse.float() - 1.0) / float(cfg.fuse)
        weight = torch.where(fuse > 0, w_raw.clamp_min(0.0).pow(2.0),
                             torch.zeros_like(fuse, dtype=torch.float32))
        brick_t = sim.brick
        solid = bombed | brick_t
        not_solid = (~solid).float()
        passable = (~sim.wall).float()
        h, w = cfg.height, cfg.width
        n = fuse.shape[0]
        # stage A（生产 v2 逻辑，独立复刻）
        ww = weight.clone()
        blast_f = bm.float()
        max_b = int(blast_f.max())
        fw = torch.where(bombed, ww, torch.zeros_like(ww))
        fd = torch.where(bombed, blast_f, torch.zeros_like(ww))
        spread = torch.zeros_like(ww)
        for _ in range(cfg.max_chain):
            spread.zero_()
            for drow, dcol in B._DIRS:
                fw_p, fd_p = fw, fd
                for _ in range(max_b):
                    fw1 = B._shift(fw_p, drow, dcol) * passable
                    fd1 = B._shift(fd_p, drow, dcol) * passable
                    fd1 = fd1 - 1.0
                    keep = fd1 >= 0
                    fw1 = torch.where(keep, fw1, torch.zeros_like(fw1))
                    spread = torch.maximum(spread, fw1)
                    fw1 = fw1 * not_solid
                    fd1 = fd1 * not_solid
                    fw_p, fd_p = fw1, fd1
            spread = spread * bombed.float()
            newly = (spread > ww) & bombed
            ww = torch.maximum(ww, spread)
            fw = torch.where(newly, spread, torch.zeros_like(ww))
            fd = torch.where(newly, blast_f, torch.zeros_like(ww))
            if not bool((newly).any()):
                break
        # 朴素 stage B 从修正权重出发
        ref = naive_stageB(ww, passable, not_solid, bombed, bm, h, w)
        if not torch.equal(d, ref):
            diff = (d != ref).sum().item()
            idx = (d != ref).nonzero()[0].tolist()
            print(f"[MISMATCH] tick {i}: {diff} 格 @ {idx} "
                  f"prod={d[tuple(idx)].item():.4f} naive={ref[tuple(idx)].item():.4f}")
            ok = False
            break
    print("=== stage B v2 朴素参考逐位一致 PASS ===" if ok else "=== FAIL ===")
    return ok


if __name__ == "__main__":
    main()
