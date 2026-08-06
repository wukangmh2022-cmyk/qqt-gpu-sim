"""Roofline / 预算测算：一个脚本回答四个问题。

1. 这张卡最多能同时塞多少局？（显存探测，翻倍到 OOM）
2. 跑 100 个 batch，每个阶段各花多少时间？（step / observe / mask / 策略前向 / PPO 更新）
3. 瓶颈在 CPU 还是 GPU？（对比"不同步的墙上时间"与"逐阶段同步时间"，
   差值就是 host 侧 Python + launch 开销）
4. 按 60fps、500 万局算，整个训练要跑多久？各 backend 加速比多少？

这是"上机第一件事"用的脚本：拿到 SSH 之后先跑它，再决定要不要开训。
不知道预算就直接开训，是最容易白烧配额的做法。

用法：
    python -m bench.roofline --compare                    # 全 backend 对比 + 预算
    python -m bench.roofline --backend cuda --probe       # 只做显存探测
    python -m bench.roofline --backend cuda --envs 16384 --batches 100
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from sim.config import N_BOMB, N_MOVES, SimConfig
from sim.factory import make_sim
from train.model import ActorCritic
from train.ppo import PPOConfig, RolloutBuffer, ppo_update

# ---------------------------------------------------------------- 计时工具


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def rand_actions(shape: tuple[int, ...], *, device=None, generator=None) -> torch.Tensor:
    """因子化随机动作：[..., 0] 方向（4 向 + IDLE），[..., 1] 放泡 trigger。

    不走掩码采样 —— 这里量的是模拟器，不是策略。非法方向由模拟器自己吃掉
    （撞墙等于原地不动），非法放泡同理，所以随机信号是安全的。
    """
    return torch.stack([
        torch.randint(0, N_MOVES, shape, generator=generator),
        torch.randint(0, N_BOMB, shape, generator=generator),
    ], dim=-1).to(device)


class Timer:
    """逐阶段计时：每个阶段前后都同步，拿到的是该阶段的真实设备耗时。"""

    def __init__(self, dev: torch.device) -> None:
        self.dev = dev
        self.acc: dict[str, float] = {}

    def __call__(self, name: str):
        return _Scope(self, name)

    def add(self, name: str, dt: float) -> None:
        self.acc[name] = self.acc.get(name, 0.0) + dt


class _Scope:
    def __init__(self, timer: Timer, name: str) -> None:
        self.timer, self.name = timer, name

    def __enter__(self):
        sync(self.timer.dev)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.timer.dev)
        self.timer.add(self.name, time.perf_counter() - self.t0)
        return False


# ---------------------------------------------------------------- 显存探测


def probe_max_envs(cfg: SimConfig, backend: str, device: str,
                   start: int = 1024, ceiling: int = 1 << 22) -> dict:
    """翻倍试到 OOM，返回能站住的最大 env 数与每 env 显存开销。

    只探"能不能分配 + 能不能跑一个 tick"。真实训练还要加上 rollout buffer，
    所以最后报的推荐值留了 30% 余量。

    注意 CPU 上不能真的探到 OOM：Linux/macOS 会直接 SIGKILL 整个进程，
    拿不到可捕获的异常。所以非 CUDA 设备用一个保守的硬上限收住。
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        ceiling = min(ceiling, 16384)
    best, per_env = 0, 0.0
    n = start
    while n <= ceiling:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            sim = make_sim(cfg, n, backend=backend, device=device, seed=0)
            acts = rand_actions((n, cfg.n_players), device=dev)
            sim.legal_mask()
            sim.observe()
            sim.step(acts)
            sync(dev)
            if dev.type == "cuda":
                per_env = torch.cuda.max_memory_allocated() / n
            best = n
            del sim, acts
            gc.collect()
            n *= 2
        except (RuntimeError, MemoryError) as exc:
            if isinstance(exc, RuntimeError) and "out of memory" not in str(exc).lower():
                raise
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            break
    total = (torch.cuda.get_device_properties(0).total_memory
             if dev.type == "cuda" else 0)
    return {"max_envs": best, "bytes_per_env": per_env, "total_mem": total,
            "recommended": int(best * 0.7)}


# ---------------------------------------------------------------- 带宽标定


def measure_peak_bw(dev: torch.device, mb: int = 256) -> float:
    """用一次大张量拷贝标定"这台机器实际能跑到的带宽"。

    不去查 spec 峰值：spec 是拿不到的理论值，拿 copy 量出来的数字才是
    roofline 上那条真正的屋顶。
    """
    n = mb * 1024 * 1024 // 4
    src = torch.empty(n, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    sync(dev)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        dst.copy_(src)
    sync(dev)
    dt = time.perf_counter() - t0
    del src, dst
    gc.collect()
    return reps * 2 * n * 4 / dt          # 读+写


def compulsory_bytes_per_tick(cfg: SimConfig) -> int:
    """一个 env 一个 tick 的**强制访存下界**（假设缓存完美、每字节只碰一次）。

    这是 roofline 横轴的分母。真实流量必然更高（连锁要多轮读写 scratch，
    危险图要重复读邻居），所以由它算出的"达成带宽"是保守值 —— 实测值
    离屋顶越近，说明 kernel 越贴近访存极限、越没有优化空间。
    """
    nc = cfg.n_cells
    b = 0
    b += nc * 1              # wall 读
    b += nc * 4 * 2          # fuse 读 + 写
    b += nc * 1 * 2          # owner 读 + 写
    b += nc * 1 * 2 * 2      # covered / trig scratch 各读写一遍（下界：只算 1 轮）
    b += cfg.n_players * (4 * 2 * 2 + 1 * 2 + 4 * 2)   # pos(float2) 读写 + alive 读写 + 双头动作
    b += 4 + 1 + 4           # reward + done + t
    return b


# ---------------------------------------------------------------- 阶段计时


def time_pipeline(cfg: SimConfig, num_envs: int, backend: str, device: str,
                  batches: int, warmup: int, with_nn: bool) -> dict:
    """跑 `batches` 个 batch，逐阶段计时；顺带量一次"不做逐阶段同步"的墙上时间。"""
    dev = torch.device(device)
    sim = make_sim(cfg, num_envs, backend=backend, device=device, seed=0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    # 2v2 随机移动信号：动作预生成，把 RNG 开销挪出计时窗口
    acts = rand_actions((warmup + batches, num_envs, cfg.n_players),
                        device=dev, generator=gen)

    net = ActorCritic(cfg.obs_shape).to(dev) if with_nn else None
    tm = Timer(dev)

    for i in range(warmup):
        sim.legal_mask()
        sim.observe()
        sim.step(acts[i])
    sync(dev)

    for i in range(batches):
        with tm("mask"):
            mmask, bmask = sim.legal_mask()
        with tm("observe"):
            obs = sim.observe()
        if net is not None:
            with tm("policy_fwd"):
                with torch.no_grad():
                    # 2v2：4 个角色都要前向。learner 只占 1 个，其余是冻结对手，
                    # 但推理成本是实打实的 4 份。观测只有共享的一份，视角靠 pid
                    # 传进网络（第一层权重重排），不切片、不拷贝。
                    for pid in range(cfg.n_players):
                        net.act(obs, mmask[:, pid], bmask[:, pid], pid)
        with tm("step"):
            sim.step(acts[warmup + i])

    # 同一段循环，中间不插同步 —— 让 launch 队列自己排下去
    sync(dev)
    t0 = time.perf_counter()
    for i in range(batches):
        mm, bm = sim.legal_mask()
        o = sim.observe()
        if net is not None:
            with torch.no_grad():
                for pid in range(cfg.n_players):
                    net.act(o, mm[:, pid], bm[:, pid], pid)
        sim.step(acts[warmup + (i % batches)])
    sync(dev)
    fused = time.perf_counter() - t0

    del sim, acts, net
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    phase_sum = sum(tm.acc.values())
    return {"phases": tm.acc, "phase_sum": phase_sum, "fused": fused,
            "batches": batches, "envs": num_envs}


def time_ppo_update(cfg: SimConfig, device: str, samples: int = 4096) -> float:
    """量"每条样本的 PPO 更新成本"（秒/样本），用来把训练开销折算进预算。"""
    dev = torch.device(device)
    net = ActorCritic(cfg.obs_shape).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    pcfg = PPOConfig(rollout_steps=16, epochs=2, minibatches=4)
    n_env = max(1, samples // pcfg.rollout_steps)
    buf = RolloutBuffer(pcfg.rollout_steps, n_env, cfg.obs_shape, dev,
                        obs_dtype=torch.float16 if cfg.obs_fp16 else torch.float32)
    buf.ptr = pcfg.rollout_steps
    buf.mmask.fill_(True)
    buf.bmask.fill_(True)
    last_val = torch.zeros(n_env, device=dev)

    ppo_update(net, opt, buf, last_val, pcfg, pcfg.entropy_coef)   # warmup
    sync(dev)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        ppo_update(net, opt, buf, last_val, pcfg, pcfg.entropy_coef)
    sync(dev)
    dt = (time.perf_counter() - t0) / reps
    total = pcfg.rollout_steps * n_env
    del net, opt, buf
    gc.collect()
    return dt / total


# ---------------------------------------------------------------- 预算推算


def budget(sim_sps: float, fwd_sps: float, upd_per_sample: float, args) -> dict:
    """把吞吐换算成"跑完 N 局要多少小时"。

    **逻辑帧率 ≠ 渲染帧率。** 60fps 是画面刷新率，模拟器不需要按它走。
    决策频率取 10Hz（每 100ms 一步）的理由：

    1. 人类反应时间下限就在 200ms 量级，手速极限也做不出 20ms 级变向。
       10Hz 的策略天然就是"人类可执行"的，反而更像人 —— 60Hz 训出来的
       策略会用人做不到的抖动来卡位，看着就假。
    2. 训练墙钟 ∝ 总步数：同样的游戏秒数，10Hz 比 15Hz 少 1/3 的步数，
       墙钟省 1/3（约 37 分钟 → 25 分钟）。
    3. 帧预算线性进入总耗时：10Hz 相对 60Hz 省 6 倍。

    连续坐标在 10Hz 下也不会穿模：默认速度 3 格/秒，一个 tick 位移 0.3 格，
    远小于格宽 1.0，碰撞检测不需要 substep。
    """
    ticks = args.tick_hz * args.game_seconds
    env_steps = args.episodes * ticks
    decisions = env_steps / args.action_repeat          # 需要策略前向的 tick 数
    learner_samples = decisions                        # 自我博弈里 learner 占 1 个位

    t_sim = env_steps / sim_sps
    t_fwd = decisions / fwd_sps if fwd_sps > 0 else 0.0
    t_upd = learner_samples * upd_per_sample
    total = t_sim + t_fwd + t_upd
    return {"ticks_per_ep": ticks, "env_steps": env_steps,
            "t_sim_h": t_sim / 3600, "t_fwd_h": t_fwd / 3600,
            "t_upd_h": t_upd / 3600, "total_h": total / 3600,
            "sessions_12h": total / 3600 / 11.0}


# ---------------------------------------------------------------- 报告


def report_pipeline(res: dict, cfg: SimConfig, peak_bw: float, dev_type: str) -> dict:
    envs, batches = res["envs"], res["batches"]
    env_steps = envs * batches
    print(f"\n--- 阶段耗时（{batches} batch × {envs} env，2v2 随机动作）---")
    print(f"{'阶段':<12}{'总耗时 s':>12}{'ms/batch':>12}{'占比':>8}{'env-steps/s':>16}")
    for name in ("step", "observe", "mask", "policy_fwd"):
        if name not in res["phases"]:
            continue
        t = res["phases"][name]
        print(f"{name:<12}{t:>12.3f}{t / batches * 1e3:>12.3f}"
              f"{t / res['phase_sum'] * 100:>7.1f}%{env_steps / t:>16,.0f}")
    print(f"{'合计':<12}{res['phase_sum']:>12.3f}"
          f"{res['phase_sum'] / batches * 1e3:>12.3f}{100.0:>7.1f}%"
          f"{env_steps / res['phase_sum']:>16,.0f}")

    step_t = res["phases"]["step"]
    sim_sps = env_steps / step_t
    bytes_tick = compulsory_bytes_per_tick(cfg)
    achieved = sim_sps * bytes_tick
    print(f"\n--- roofline（只看 step）---")
    print(f"强制访存下界      {bytes_tick:,} B / env-tick")
    print(f"达成带宽（下界）  {achieved / 1e9:8.2f} GB/s")
    print(f"实测拷贝带宽上限  {peak_bw / 1e9:8.2f} GB/s")
    print(f"屋顶占用率        {achieved / peak_bw * 100:8.1f}%  "
          f"→ {'已贴近访存极限，继续优化要减少访存量而不是加线程' if achieved / peak_bw > 0.4 else '离屋顶还远，说明卡在 launch 开销 / 分支 / 占用率上'}")

    overhead = res["fused"] - res["phase_sum"]
    print(f"\n--- CPU vs GPU ---")
    if dev_type != "cuda":
        print(f"设备 {dev_type}：host 与 device 是同一颗芯片，这一节只在 CUDA 上有意义。")
        print(f"逐阶段同步合计    {res['phase_sum']:8.3f} s")
        print(f"不插同步的墙上钟  {res['fused']:8.3f} s")
    else:
        print(f"逐阶段同步合计    {res['phase_sum']:8.3f} s   （纯设备执行时间）")
        print(f"不插同步的墙上钟  {res['fused']:8.3f} s   （设备 + host 未被掩盖的部分）")
        if overhead > 0.05 * res["phase_sum"]:
            print(f"host 侧未被掩盖   {overhead:8.3f} s  "
                  f"({overhead / res['fused'] * 100:.1f}%)"
                  "  → CPU 是瓶颈：Python 循环 / kernel launch 开销吃掉了带宽")
        else:
            print(f"host 侧未被掩盖   {max(0.0, overhead):8.3f} s"
                  "  → GPU 是瓶颈：host 提交被设备执行完全掩盖，这是想要的状态")
    # 不管在哪台设备上，这条都成立：谁占比最大谁是瓶颈
    worst = max(res["phases"], key=res["phases"].get)
    print(f"阶段级瓶颈        {worst}（占 {res['phases'][worst] / res['phase_sum'] * 100:.1f}%）")
    return {"sim_sps": sim_sps,
            "fwd_sps": (env_steps / res["phases"]["policy_fwd"]
                        if "policy_fwd" in res["phases"] else 0.0)}


def report_budget(name: str, b: dict) -> None:
    print(f"{name:<18}{b['t_sim_h']:>10.1f}{b['t_fwd_h']:>10.1f}"
          f"{b['t_upd_h']:>10.1f}{b['total_h']:>12.1f}{b['total_h'] / 24:>10.1f}"
          f"{b['sessions_12h']:>12.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "cuda"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--envs", type=int, default=0, help="0 = 用探测出来的推荐值")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--players", type=int, default=4, help="2v2 默认 4 人")
    ap.add_argument("--probe", action="store_true", help="只做显存探测")
    ap.add_argument("--compare", action="store_true",
                    help="遍历所有可用 backend，输出加速比与预算对比")
    ap.add_argument("--no-nn", action="store_true", help="跳过策略前向，只量模拟器")
    # 预算模型参数
    ap.add_argument("--episodes", type=int, default=5_000_000)
    ap.add_argument("--tick-hz", type=int, default=10,
                    help="模拟器逻辑帧率（≠ 渲染帧率）。10Hz = 每 100ms 一次决策，"
                         "和 Atari 的 60Hz+frameskip4 是同一个数")
    ap.add_argument("--game-seconds", type=int, default=60)
    ap.add_argument("--action-repeat", type=int, default=1,
                    help="逻辑帧已经是决策频率，默认 1；若把 tick-hz 提到 60 "
                         "则设 4 才等价")
    args = ap.parse_args()

    cfg = SimConfig(height=args.size, width=args.size, n_players=args.players)
    combos = []
    if args.compare:
        combos.append(("torch/cpu", "torch", "cpu"))
        if torch.cuda.is_available():
            combos.append(("torch/cuda", "torch", "cuda"))
            combos.append(("cuda-kernel", "cuda", "cuda"))
        elif torch.backends.mps.is_available():
            combos.append(("torch/mps", "torch", "mps"))
    else:
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        backend = args.backend
        if backend == "auto":
            backend = "cuda" if torch.cuda.is_available() else "torch"
        combos.append((f"{backend}/{dev}", backend, dev))

    print(f"地图 {args.size}x{args.size}  角色 {args.players}（2v2）  "
          f"观测通道 {cfg.n_channels}  max_chain {cfg.max_chain}")
    ticks = args.tick_hz * args.game_seconds
    print(f"预算模型：{args.episodes:,} 局 × {args.tick_hz}Hz 逻辑帧 × "
          f"{args.game_seconds}s = {ticks:,} tick/局 = "
          f"{args.episodes * ticks:.3e} env-steps"
          f"（每 {args.action_repeat} tick 决策一次 → "
          f"{args.tick_hz / args.action_repeat:.0f}Hz，约 "
          f"{1000 * args.action_repeat / args.tick_hz:.0f}ms/步，人类手速量级）")

    results = {}
    for name, backend, device in combos:
        print(f"\n{'=' * 72}\n[{name}]")
        dev = torch.device(device)
        pr = probe_max_envs(cfg, backend, device)
        if pr["total_mem"]:
            print(f"显存探测：能站住 {pr['max_envs']:,} env，"
                  f"约 {pr['bytes_per_env'] / 1024:.1f} KB/env，"
                  f"卡上共 {pr['total_mem'] / 2**30:.1f} GiB "
                  f"→ 推荐并行 {pr['recommended']:,} 局（留 30% 给 rollout buffer）")
        else:
            print(f"显存探测：能站住 {pr['max_envs']:,} env（非 CUDA 设备，不报显存）")
        if args.probe:
            continue

        num_envs = args.envs or min(pr["recommended"] or 1024, 16384)
        peak_bw = measure_peak_bw(dev)
        res = time_pipeline(cfg, num_envs, backend, device,
                            args.batches, args.warmup, not args.no_nn)
        rates = report_pipeline(res, cfg, peak_bw, dev.type)
        upd = time_ppo_update(cfg, device)
        print(f"PPO 更新成本      {upd * 1e6:8.2f} us/样本")
        results[name] = budget(rates["sim_sps"], rates["fwd_sps"], upd, args)
        results[name]["sim_sps"] = rates["sim_sps"]

    if not results:
        return
    print(f"\n{'=' * 72}\n--- 训练总预算（{args.episodes:,} 局）---")
    print(f"{'backend':<18}{'模拟 h':>10}{'前向 h':>10}{'更新 h':>10}"
          f"{'合计 h':>12}{'合计 天':>10}{'11h 会话数':>12}")
    for name, b in results.items():
        report_budget(name, b)

    base = results.get("torch/cpu")
    if base and len(results) > 1:
        print(f"\n--- 模拟器加速比（相对 torch/cpu）---")
        for name, b in results.items():
            print(f"{name:<18}{b['sim_sps']:>16,.0f} env-steps/s"
                  f"{b['sim_sps'] / base['sim_sps']:>10.1f}x")


if __name__ == "__main__":
    main()
