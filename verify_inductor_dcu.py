"""DCU 验证：_danger_c 级联（npu→inductor→eager）的 parity 与 step SPS。

1) parity：级联（DCU 上 triton 可用 → inductor）vs 强制 eager 的两条 sim，
   同 seed 同动作序列跑 30 tick；逐 tick 比较 danger maxdiff（≤1e-5）与
   状态/掩码/奖励/终局的完全一致。
2) SPS：级联 vs 强制 eager 的 step 均耗时（与 56.4ms 基线同口径：N=8192
   map=open；另报 N=2048 corridor+open_fraction=1.0 的训练配置）。
"""
import sys, time
sys.path.insert(0, ".")
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim

DEV = "cuda"


def make_cfg(map_mode="open", open_fraction=None):
    kw = dict(map_mode=map_mode, speed=3.0, max_steps=1800,
              ring_fraction=0.0, hazard_fraction=0.0,
              crate_speed_only=False, timeout_draw=False,
              combo_reward=0.10, combo_gap_factor=0.9)
    if open_fraction is not None:
        kw["open_fraction"] = open_fraction
    return SimConfig(**kw)


def force_eager(on: bool):
    """开/关 inductor 候选（False → 级联只走 npu→eager，DCU 上即 eager）。"""
    BatchedSim._triton_ok = staticmethod(lambda: not on)  # on=True → 禁 inductor


def run_parity(N=2048, ticks=30):
    cfg = make_cfg(map_mode="corridor", open_fraction=1.0)
    torch.manual_seed(20260816)
    simA = BatchedSim(cfg, N, device=DEV, seed=0)   # 级联（inductor）
    simA.reset_all()
    force_eager(True)
    simB = BatchedSim(cfg, N, device=DEV, seed=0)   # 强制 eager
    simB.reset_all()
    force_eager(False)
    g = torch.Generator(device=DEV).manual_seed(99)

    def acts(sim):
        mmask, bmask = sim.legal_mask()
        mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
        bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
        return torch.stack([mv, bm], dim=-1)

    md_max, obs_max, rew_max, state_eq, done_eq = 0.0, 0.0, 0.0, True, True
    for t in range(ticks):
        a = acts(simA)
        torch.cuda.synchronize()
        rng0 = torch.cuda.get_rng_state().clone()
        ra, da, ia = simA.step(a)
        torch.cuda.synchronize()
        # 回卷默认设备 RNG：simB.step 与 simA.step 用同一随机流（同进程共享
        # 默认生成器，multinomial/step 都会推进它；不回卷则两条 sim 的
        # 宝箱/成长随机值错位，parity 无从对比）。
        torch.cuda.set_rng_state(rng0)
        rb, db, ib = simB.step(a.clone())
        md = (simA._dng_cache - simB._dng_cache).abs().max().item()
        md_max = max(md_max, md)
        oa = simA.observe()
        ob = simB.observe()
        obs_max = max(obs_max, (oa - ob).abs().max().item())
        rew_max = max(rew_max, (ra - rb).abs().max().item())
        state_eq = state_eq and torch.equal(simA.fuse, simB.fuse) \
            and torch.equal(simA.wall, simB.wall) \
            and torch.equal(simA.pos, simB.pos) \
            and torch.equal(simA.alive, simB.alive)
        done_eq = done_eq and torch.equal(da, db) \
            and torch.equal(ia["winner"], ib["winner"])
    ok = md_max <= 1e-5 and obs_max <= 1e-5 and rew_max <= 1e-6 \
        and state_eq and done_eq
    print(f"[parity N={N} ticks={ticks}] danger_maxdiff={md_max:.3e} "
          f"obs_maxdiff={obs_max:.3e} rew_maxdiff={rew_max:.3e} "
          f"state_eq={state_eq} done_eq={done_eq} => {'PASS' if ok else 'FAIL'}", flush=True)


def bench_step(tag, N, cfg, iters=120, warmup=4):
    torch.manual_seed(20260816)
    sim = BatchedSim(cfg, N, device=DEV, seed=0)
    sim.reset_all()
    g = torch.Generator(device=DEV).manual_seed(99)

    def acts():
        mmask, bmask = sim.legal_mask()
        mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=g).view(N, 2)
        bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=g).view(N, 2)
        return torch.stack([mv, bm], dim=-1)

    # warmup 足够长：corridor 成长把 blast 档位（3→7）全部触发编译，
    # 计时区间不含首次编译开销（编译是每档一次性成本，摊到训练全程可忽略）。
    for _ in range(warmup):
        sim.step(acts())
    torch.cuda.synchronize()
    tiers = len(sim._dng_tier)
    t0 = time.time()
    for _ in range(iters):
        sim.step(acts())
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[{tag}] N={N} tiers={tiers} avg_step={dt/iters*1000:6.1f}ms "
          f"({N*iters/dt:,.0f} env-steps/s)", flush=True)


if __name__ == "__main__":
    print("triton importable:", BatchedSim._triton_ok(), flush=True)
    run_parity()
    # SPS：基线同口径（N=8192 map=open，与 56.4ms 那轮一致）
    cfg = make_cfg(map_mode="open")
    force_eager(True)
    bench_step("eager", 8192, cfg)
    force_eager(False)
    bench_step("indct", 8192, cfg)
    # 训练配置（N=2048 corridor+open）：成长把档位拉满后计时
    cfg2 = make_cfg(map_mode="corridor", open_fraction=1.0)
    force_eager(True)
    bench_step("eager", 2048, cfg2, warmup=80)
    force_eager(False)
    bench_step("indct", 2048, cfg2, warmup=80)
    force_eager(False)
