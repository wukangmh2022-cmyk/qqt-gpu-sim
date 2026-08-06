"""吞吐基准：env-steps/s 随并行关卡数的变化。

这是整个项目要展示的核心数字。两点说明：

- 计时前必须 `torch.cuda.synchronize()`，否则量到的是 kernel 入队速度。
- 动作用**预生成的随机整数**而不是走一次策略网络，否则量到的是网络前向，
  不是模拟器。掩码采样也关掉（`--no-mask`）可以看模拟器裸吞吐。

用法：
    python -m bench.throughput --backend torch --device cpu
    python -m bench.throughput --backend cuda --envs 256,1024,4096,16384,65536
"""

from __future__ import annotations

import argparse
import time

import torch

from sim.config import N_BOMB, N_MOVES, SimConfig
from sim.factory import make_sim


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_one(cfg: SimConfig, num_envs: int, backend: str, device: str,
              ticks: int, warmup: int, use_mask: bool) -> dict:
    sim = make_sim(cfg, num_envs, backend=backend, device=device, seed=0)
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    # 预生成动作序列：把随机数生成开销挪出计时窗口
    # 因子化动作：[..., 0] 方向（含 IDLE），[..., 1] 放泡 0/1
    shape = (warmup + ticks, num_envs, cfg.n_players)
    acts = torch.stack([
        torch.randint(0, N_MOVES, shape, generator=gen),
        torch.randint(0, N_BOMB, shape, generator=gen),
    ], dim=-1).to(dev)

    for i in range(warmup):
        sim.step(acts[i])
    sync(dev)

    t0 = time.perf_counter()
    for i in range(ticks):
        if use_mask:
            sim.legal_mask()
        sim.step(acts[warmup + i])
    sync(dev)
    dt = time.perf_counter() - t0

    # 同时量一次 observe：训练时每 tick 都要调，属于热路径
    t1 = time.perf_counter()
    for _ in range(max(1, ticks // 4)):
        sim.observe()
    sync(dev)
    dt_obs = time.perf_counter() - t1

    env_steps = ticks * num_envs
    return {
        "envs": num_envs,
        "sps": env_steps / dt,
        "agent_sps": env_steps * cfg.n_players / dt,
        "us_per_tick": dt / ticks * 1e6,
        "obs_sps": (max(1, ticks // 4) * num_envs) / dt_obs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "cuda"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--envs", default="64,256,1024,4096,16384")
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--no-mask", action="store_true", help="跳过 legal_mask，量裸 step")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimConfig(height=args.size, width=args.size, n_players=args.players)
    print(f"backend={args.backend} device={device} map={args.size}x{args.size} "
          f"P={args.players} ticks={args.ticks} mask={'off' if args.no_mask else 'on'}")
    print(f"{'envs':>8} {'env-steps/s':>14} {'agent-steps/s':>14} "
          f"{'us/tick':>10} {'obs env/s':>12}")
    for num_envs in [int(x) for x in args.envs.split(",")]:
        try:
            r = bench_one(cfg, num_envs, args.backend, device,
                          args.ticks, args.warmup, not args.no_mask)
        except RuntimeError as exc:              # 显存不够就停在上一档
            print(f"{num_envs:>8}  跳过：{exc}")
            break
        print(f"{r['envs']:>8} {r['sps']:>14,.0f} {r['agent_sps']:>14,.0f} "
              f"{r['us_per_tick']:>10.1f} {r['obs_sps']:>12,.0f}")


if __name__ == "__main__":
    main()
