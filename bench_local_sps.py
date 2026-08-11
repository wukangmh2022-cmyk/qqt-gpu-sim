"""本地（M4/MPS）triton 版 SPS 基准：triton_step_full + observe/legal_mask + learner.act。

分段计时 + 内存带宽实测（多次迭代平均），用于推算 Ascend 910B 的 SPS。
用法：.venv/bin/python bench_local_sps.py --n 20000 --ticks 30
"""
import argparse, time, torch
import sys
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.triton_step import triton_step_full
from train.model import ActorCritic

torch.manual_seed(0)


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def bench(fn, it, dev, warmup=3):
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / it


def bw_bench(dev, size_mb=256, it=10):
    """内存带宽：大张量拷 back-to-back 迭代平均，排除单次 launch 抖动。"""
    n = size_mb * 1024 * 1024 // 4
    a = torch.randn(n, device=dev)
    b = torch.empty_like(a)
    sync(dev)
    for _ in range(3):
        b.copy_(a)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(it):
        b.copy_(a)
    sync(dev)
    dt = (time.perf_counter() - t0) / it
    gbs = (a.numel() * 4 * 2) / dt / 1e9
    return gbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ticks", type=int, default=30)
    args = ap.parse_args()
    N = args.n
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                    open_fraction=0.5, timeout_draw=True, combo_reward=0.10)
    sim = BatchedSim(cfg, N, device=dev, seed=0)
    sim.bombs_cap[:] = 10; sim.blast_cap[:] = 7; sim.spd_g[:] = 2.1
    sim.reset_all()

    learner = ActorCritic(cfg.obs_shape, arch="mlp", n_players=2).to(dev).eval()
    for p in learner.parameters():
        p.requires_grad_(False)

    acts = torch.stack([
        torch.randint(0, 5, (N, 2), device=dev),
        torch.randint(0, 2, (N, 2), device=dev),
    ], dim=-1)

    # 预热：每个 kernel 首编一次 + 状态稳定
    for _ in range(3):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        with torch.no_grad():
            learner.act(obs, mm[:, 0], bm[:, 0], 0)
        triton_step_full(sim, acts)

    def t_obs():
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        return obs, mm, bm

    def t_act():
        obs, mm, bm = t_obs()
        with torch.no_grad():
            learner.act(obs, mm[:, 0], bm[:, 0], 0)

    def t_step():
        triton_step_full(sim, acts)

    def t_full():
        obs, mm, bm = t_obs()
        with torch.no_grad():
            learner.act(obs, mm[:, 0], bm[:, 0], 0)
        triton_step_full(sim, acts)

    d_obs = bench(t_obs, args.ticks, dev) * 1e3
    d_act = bench(t_act, args.ticks, dev) * 1e3
    d_step = bench(t_step, args.ticks, dev) * 1e3
    d_full = bench(t_full, args.ticks, dev) * 1e3
    bw = bw_bench(dev)

    print(f"设备: {dev} | N={N} | 内存带宽实测: {bw:.0f} GB/s")
    print(f"--- 每 tick 耗时 (ms) ---")
    print(f"  observe+legal:   {d_obs:.2f}")
    print(f"  observe+act:     {d_act:.2f}")
    print(f"  triton_step:     {d_step:.2f}")
    print(f"  FULL tick:       {d_full:.2f}")
    print(f"--- SPS ---")
    print(f"  仅 triton_step:    {N / (d_step / 1e3):,.0f}")
    print(f"  FULL collect:      {N / (d_full / 1e3):,.0f}")
    print(f"  FULL 含双玩家 act:  {N / ((d_obs + 2 * (d_act - d_obs) + d_step) / 1e3):,.0f}")


if __name__ == "__main__":
    main()
