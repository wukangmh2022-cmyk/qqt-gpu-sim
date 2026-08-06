"""炸弹雨躲避特训：MLP 从零、纯生存、100M 步。

与 train/train.py 的关系：**独立**的训练入口（用户要求"单独的一个训练函数"），
不共用课程/混合地图/成长逻辑，只复用 PPO 采集与更新（train/ppo.py）和
模型池（train/model_pool.py）。场景固定为：

- 空地图（open，无墙/砖/宝箱），MLP 网络，从头训练（--resume 可接力）。
- **双方泡数上限强制 0**（bombs_cap=0，max_bombs=0）：玩家放不了泡，
  放泡头被 legal_mask 屏蔽 —— 进攻与放炮机制整体砍掉。
- 环境炸弹雨：每 hazard_wave_ticks（50 tick = 5 秒 @10Hz）一波，每波
  hazard_bombs_min..max（4..30）颗随机落在可通行格；威力 4..8 且随局内
  时间偏向大值 —— 指数采样 v = u^p、p 从 1 线性退火到 0.2，约 60 秒后
  威力几乎总是 7/8（见 sim/config.py hazard_* 注释与 sim/torch_sim.py
  `_hazard_wave`）。环境炸弹 owner=n_players，只出现在危险图通道。
- 评分 = 谁活得久谁赢：保留 hit ±1.2（被环境炸到掉血 / 对方掉血）、
  danger 罚、终局 ±8；**放泡预测/接近/被动奖励全部置零**。

运行（DCU，torch 后端，5632 envs，约 37k sps → 100M 步约 45 分钟）：
    python -m train.dodge_train --backend torch --num-envs 5632 \
        --total-steps 100_000_000 --time-budget 36000 \
        --ckpt ckpt/dodge_rw1.pt --log-csv ckpt/dodge_log.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import torch

from sim.config import N_BOMB, N_MOVES, SimConfig
from sim.factory import make_sim

from .model import ActorCritic
from .model_pool import ModelPool, load_frozen
from .ppo import PPOConfig, SelfPlayRunner, ppo_update
from .train import available_ram_bytes, estimate_peak_bytes


def build_cfg(args) -> SimConfig:
    """躲避特训的固定关卡配置。

    map_mode="open" 且 max_bombs=0：reset_ 的 open 分支把 bombs_cap 置成
    max_bombs=0（hazard 掷关分支也强制 0）→ `_place_bombs` 的 (live < 0)
    恒 False，放泡封死。hazard_fraction=1.0 表示每局都是炸弹雨。
    """
    return SimConfig(
        height=args.map_h,
        width=args.map_w,
        map_mode="open",
        max_steps=args.max_steps,
        max_bombs=0,                    # 双方都不能放泡
        hazard_fraction=args.hazard_fraction,
        hazard_wave_ticks=args.wave_ticks,
        hazard_bombs_min=args.bombs_min,
        hazard_bombs_max=args.bombs_max,
        hazard_blast_min=args.blast_min,
        hazard_blast_max=args.blast_max,
        hazard_ramp_seconds=args.ramp_seconds,
        # 进攻/放炮/接近/被动奖励全砍 —— 纯生存
        place_cover_reward=0.0,
        place_chain_reward=0.0,
        place_dist_reward=0.0,
        approach_reward=0.0,
        passivity_penalty=0.0,
    )


def save_ckpt(path: str, *, learner, opt, pool, global_step, elo, args) -> None:
    """与 train.py 同款格式（format_version 2），play/duel.py 可直接加载。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "format_version": 2,
        "model": learner.state_dict(),
        "obs_shape": learner.obs_shape,
        "n_players": learner.n_players,
        "arch": learner.arch,
        "opt": opt.state_dict(),
        "pool": pool.state_dict(),
        "curriculum": {},              # 无课程，占位保证旧读取方兼容
        "global_step": global_step,
        "elo": elo,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--total-steps", type=int, default=100_000_000)
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "cuda"])
    ap.add_argument("--arch", default="mlp", choices=["cnn", "mlp"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--minibatches", type=int, default=PPOConfig.minibatches)
    ap.add_argument("--max-mem-frac", type=float, default=0.55)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    ap.add_argument("--oversample-dying", type=int,
                    default=PPOConfig.oversample_dying)
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=20,
                    help="每多少次迭代往模型池存快照 + 写一次 ckpt")
    ap.add_argument("--ckpt", default="ckpt/dodge_latest.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-csv", default="ckpt/dodge_log.csv")
    ap.add_argument("--time-budget", type=float, default=11.0 * 3600)
    # 关卡（默认与用户要求一致：13x13 空图，5 秒一波，4..30 颗，威力 4..8，60 秒偏向）
    ap.add_argument("--map-h", type=int, default=13)
    ap.add_argument("--map-w", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--hazard-fraction", type=float, default=1.0)
    ap.add_argument("--wave-ticks", type=int, default=50)
    ap.add_argument("--bombs-min", type=int, default=4)
    ap.add_argument("--bombs-max", type=int, default=30)
    ap.add_argument("--blast-min", type=int, default=4)
    ap.add_argument("--blast-max", type=int, default=8)
    ap.add_argument("--ramp-seconds", type=float, default=60.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = build_cfg(args)
    pcfg = PPOConfig(rollout_steps=args.rollout_steps, lr=args.lr,
                     minibatches=args.minibatches, gae_lambda=args.gae_lambda,
                     oversample_dying=args.oversample_dying)
    elo = 1000.0
    global_step = 0

    ckpt = torch.load(args.resume, map_location=device, weights_only=False) \
        if args.resume else None
    if ckpt and ckpt.get("format_version") != 2:
        raise ValueError("checkpoint 是旧观测布局，无法安全迁移")
    obs_shape = tuple(ckpt["obs_shape"]) if ckpt else cfg.obs_shape
    learner = ActorCritic(obs_shape, arch=args.arch,
                          n_players=cfg.n_players).to(device)
    if ckpt:
        if obs_shape != cfg.obs_shape:
            raise ValueError(
                f"checkpoint obs_shape {obs_shape} ≠ 关卡 {cfg.obs_shape}，拒绝错绑")
        learner.load_state_dict(ckpt["model"])
        elo, global_step = ckpt["elo"], ckpt["global_step"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        random.setstate(ckpt["py_rng"])
    opt = torch.optim.Adam(learner.parameters(), lr=pcfg.lr, eps=1e-5)
    pool = ModelPool()
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
        pool.load_state_dict(ckpt["pool"])
        print(f"[resume] step={global_step} elo={elo:.0f} pool={len(pool)}")
    print(f"[model] arch={learner.arch} params={learner.n_params():,} "
          f"obs_shape={learner.obs_shape} device={device}")
    print(f"[cfg] hazard wave={cfg.hazard_wave_ticks}t "
          f"bombs={cfg.hazard_bombs_min}..{cfg.hazard_bombs_max} "
          f"blast={cfg.hazard_blast_min}..{cfg.hazard_blast_max} "
          f"ramp={cfg.hazard_ramp_seconds:.0f}s max_bombs={cfg.max_bombs}")

    # 内存守卫（复用 train.py 的估算；MLP 实际峰值远小于 CNN 上界，保守）
    avail = available_ram_bytes()
    est = estimate_peak_bytes(cfg, args.num_envs, args.rollout_steps,
                              pcfg.minibatches)
    if not avail:
        print(f"[mem] 无法探测可用内存；预估峰值 {est/1e9:.1f} GB（请留意）")
    else:
        frac = est / avail
        print(f"[mem] 可用 {avail/1e9:.1f} GB  预估峰值 {est/1e9:.1f} GB ({frac:.0%})")
        if frac > args.max_mem_frac:
            raise SystemExit(
                f"拒绝启动：预估峰值 {est/1e9:.1f} GB 超过可用内存的 "
                f"{args.max_mem_frac:.0%}。调小 --num-envs 或调大 --minibatches")

    if len(pool) == 0:
        pool.add(learner, step=global_step, elo=elo)
    sim = make_sim(cfg, args.num_envs, backend=args.backend,
                   device=device, seed=args.seed)
    nets, snaps = [], []
    for _ in range(cfg.n_players - 1):
        snap = pool.sample(elo)
        nets.append(load_frozen(ActorCritic, learner.obs_shape, snap["state"],
                                device, arch=learner.arch,
                                n_players=cfg.n_players))
        snaps.append(snap)
    runner = SelfPlayRunner(sim, learner, nets, pcfg)

    log_f = writer = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        log_f = open(args.log_csv, "a", newline="")
        writer = csv.writer(log_f)
        if log_f.tell() == 0:
            writer.writerow(["step", "stage", "elo", "win_rate", "ep_len",
                             "pg", "vf", "ent", "kl", "clipfrac", "sps"])

    start = time.time()
    it = 0
    per_iter = args.num_envs * pcfg.rollout_steps
    try:
        while global_step < args.total_steps:
            it += 1
            t0 = time.time()
            buf, last_val = runner.collect()
            frac = min(1.0, global_step / max(1, args.total_steps))
            ent_coef = pcfg.entropy_coef + frac * (pcfg.entropy_final - pcfg.entropy_coef)
            stats = ppo_update(learner, opt, buf, last_val, pcfg, ent_coef,
                               autocast=args.autocast)
            global_step += per_iter
            sps = per_iter / max(1e-6, time.time() - t0)

            wr = runner.win_rate()
            done_eps = runner.ep_stats["count"]
            if done_eps:
                elo = pool.update_elo(snaps[0], elo, min(1.0, max(0.0, wr)))

            if it % 10 == 0:
                print(f"[{it:6d}] step={global_step/1e6:.2f}M "
                      f"wr={wr:.3f} elo={elo:.0f} len={runner.mean_ep_len():.0f} "
                      f"ent={stats['ent']:.3f} kl={stats['kl']:+.4f} "
                      f"sps={sps/1e3:.0f}k")
            if log_f:
                writer.writerow([global_step, "dodge", f"{elo:.1f}", f"{wr:.4f}",
                                 f"{runner.mean_ep_len():.1f}"]
                                + [f"{stats[k]:.5f}" for k in
                                   ("pg", "vf", "ent", "kl", "clipfrac")]
                                + [f"{sps:.0f}"])
                log_f.flush()

            if it % args.snapshot_every == 0:
                pool.add(learner, step=global_step, elo=elo)
                save_ckpt(args.ckpt, learner=learner, opt=opt, pool=pool,
                          global_step=global_step, elo=elo, args=args)

            # 每个 rollout 换一次对手，避免过拟合到单一历史策略
            nets, snaps = [], []
            for _ in range(cfg.n_players - 1):
                snap = pool.sample(elo)
                nets.append(load_frozen(ActorCritic, learner.obs_shape,
                                        snap["state"], device, arch=learner.arch,
                                        n_players=cfg.n_players))
                snaps.append(snap)
            runner.opponents = nets
            runner.clear_stats()

            if time.time() - start > args.time_budget:
                print("[budget] 时间预算用尽，存盘退出（--resume 接力）")
                break
    except KeyboardInterrupt:
        print("\n[interrupt] 存盘后退出")
    finally:
        save_ckpt(args.ckpt, learner=learner, opt=opt, pool=pool,
                  global_step=global_step, elo=elo, args=args)
        if log_f:
            log_f.close()
        print(f"[done] step={global_step} ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
