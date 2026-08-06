"""行为克隆（BC）：把人类录像（recordings/*.npz）喂成策略的预训练权重。

原理：人类录像 = (观测, 人类实际动作) 监督样本。BC 直接做**监督学习** ——
最小化模型动作分布与人类动作的交叉熵，把"什么时候放炮 / 什么时候追人 /
怎么躲危险"这些**稀疏奖励学不到的手工行为**直接烧进权重（尤其追杀逃跑对手、
精准放炮这类人类手感）。产出 ckpt 与训练主流程格式兼容（format_version=2），
下段训练 `--resume ckpt/bc_pretrain.pt` 即从人类行为起步，再 PPO 微调。

录像观测 14 通道与训练同构（人类视角 = player 0，pid=1 的局已 swap），
uint8/255 还原。人类动作基本合法，直接全量 CE；放炮用 BCE。

用法：
    python -m train.bc --data recordings/ --arch mlp --epochs 8 \
        --out ckpt/bc_pretrain.pt --device cpu
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import random

import numpy as np
import torch

from sim.config import SimConfig
from train.model import ActorCritic

MOVE_IDLE = 4


def load_recordings(dir_: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """全部录像 → (obs (M,14,13,13) float32, move (M,) long, bomb (M,) long)。"""
    paths = sorted(glob.glob(os.path.join(dir_, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"{dir_}/ 无录像")
    obs_l, mv_l, bm_l = [], [], []
    n_bad = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        o = d["obs"].astype(np.float32) / 255.0
        if o.ndim != 4 or o.shape[1] != 14:
            n_bad += 1
            continue
        a = d["action"]
        obs_l.append(o)
        mv_l.append(a[:, 0].astype(np.int64))
        bm_l.append(a[:, 1].astype(np.int64))
    if n_bad:
        print(f"[bc] 跳过 {n_bad} 个通道不符的录像")
    obs = torch.from_numpy(np.concatenate(obs_l)).float()
    mv = torch.from_numpy(np.concatenate(mv_l)).long()
    bm = torch.from_numpy(np.concatenate(bm_l)).long()
    return obs, mv, bm


def bc_loss(net: ActorCritic, obs: torch.Tensor, mv: torch.Tensor, bm: torch.Tensor,
            pid: int = 0) -> torch.Tensor:
    """监督损失：move 交叉熵 + bomb 二分类交叉熵。"""
    ml, bl, _ = net(obs, pid)
    loss_mv = torch.nn.functional.cross_entropy(ml, mv)
    loss_bm = torch.nn.functional.cross_entropy(bl, bm)   # bomb_head 2 类 logits
    return loss_mv + loss_bm


def bc_update(net: ActorCritic, opt, data: tuple, batch: int, coef: float,
              device: torch.device, seed: int | None = None) -> float:
    """在线 BC 辅助更新（训练循环每迭代调一次）：从人类录像采样一批监督
    样本，做 BC loss 梯度（coef 权重），与 PPO 主力并存 —— 人类"精准放炮/
    追杀/包围"行为持续引导策略，不靠纯预热一次性的脆弱效果。"""
    obs, mv, bm = data
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    idx = torch.randint(0, len(obs), (batch,), generator=g)
    o = obs[idx].to(device)
    m = mv[idx].to(device)
    b = bm[idx].to(device)
    opt.zero_grad()
    loss = bc_loss(net, o, m, b)
    (coef * loss).backward()
    opt.step()
    return float(loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="recordings")
    ap.add_argument("--arch", default="mlp", choices=["cnn", "mlp"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="ckpt/bc_pretrain.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="已有 ckpt（format v2，训练主流程产出）继续 BC 微调 —— "
                         "课程结束后用人类数据收尾校准用："
                         "python -m train.bc --resume 最终ckpt --data recordings/ "
                         "--epochs 3 --out ckpt/final_bc.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    obs, mv, bm = load_recordings(args.data)
    n = obs.shape[0]
    print(f"[bc] {n:,} 样本（{len(glob.glob(os.path.join(args.data, '*.npz')))} 局）"
          f" obs={tuple(obs.shape[1:])} arch={args.arch}")
    # 动作分布概览（数据质量）
    print(f"[bc] 动作分布: 移动 {100*(mv < 4).float().mean():.1f}% / "
          f"IDLE {100*(mv == 4).float().mean():.1f}% / 放泡 {100*bm.float().mean():.2f}%")

    cfg = SimConfig()   # 只取 obs_shape/n_players
    net = ActorCritic(tuple(obs.shape[1:]), arch=args.arch,
                      n_players=cfg.n_players).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        if ck.get("arch") != args.arch:
            raise ValueError(f"--arch {args.arch} 与 resume ckpt 的 {ck.get('arch')} 不一致")
        print(f"[bc] resume {args.resume}（{ck.get('global_step', 0):,} 步）→ BC 微调")
    net.train()

    perm = torch.randperm(n)
    obs, mv, bm = obs[perm], mv[perm], bm[perm]
    steps = max(1, n // args.batch)
    for ep in range(args.epochs):
        tot = 0.0
        for i in range(0, n, args.batch):
            o = obs[i:i + args.batch].to(device)
            m = mv[i:i + args.batch].to(device)
            b = bm[i:i + args.batch].to(device)
            opt.zero_grad()
            loss = bc_loss(net, o, m, b)
            loss.backward()
            opt.step()
            tot += float(loss) * o.shape[0]
        print(f"[bc] epoch {ep + 1}/{args.epochs}  loss {tot / n:.4f}")

    # ---- 产出与训练兼容的 ckpt（format_version=2，可直接 --resume）----
    net.eval()
    from train.curriculum import CurriculumState
    from train.train import ModelPool  # noqa: F401  (pool 类型)
    pool = ModelPool()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "format_version": 2,
        "model": net.state_dict(),
        "obs_shape": tuple(obs.shape[1:]),
        "n_players": cfg.n_players,
        "arch": args.arch,
        "opt": opt.state_dict(),
        "pool": pool.state_dict(),
        "fixed_elo": {},
        "curriculum": {"stage_idx": 0, "episodes_in_stage": 0,
                       "recent_wins": []},
        "global_step": 0,
        "elo": 1000.0,
        "args": {"arch": args.arch, "n_players": cfg.n_players},
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
    }, args.out)
    print(f"[bc] 已存 {args.out} —— 下段训练 `--resume {args.out}` 从人类行为起步")


if __name__ == "__main__":
    main()
