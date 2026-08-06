"""按硬件规格算理论上限：这个 kernel 在 4090 上一秒能跑多少 step。

实测之前先算一遍,是为了知道"跑出来的数字离天花板还有多远"。
没有这条基线,15M steps/s 到底是好还是差根本无从判断。

三条天花板同时存在,取最低的那条:

1. **DRAM 带宽**  每个 env-tick 必须搬的字节数 ÷ 显存带宽
2. **L2 驻留**    状态总量塞得进 L2 时,分母换成 L2 带宽,天花板抬高一个量级
3. **整数指令**   kernel 全是 load/compare/branch/index,几乎没有浮点,
                  受限于 INT32 发射率而不是 FP32 峰值

用法：
    python -m bench.theoretical                    # 全设备 × 默认配置
    python -m bench.theoretical --size 13 --players 4 --envs 65536
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from sim.config import SimConfig


@dataclass(frozen=True)
class Device:
    name: str
    cores: int
    clock_ghz: float
    int_per_clock: float      # 每 core 每 clock 能发多少 INT32 指令
    mem_bw_gb: float
    l2_mb: float
    l2_bw_gb: float           # L2 聚合带宽（厂商不标，按经验取 DRAM 的 4~5 倍）
    tensor_tflops: float      # bf16/fp16 tensor core 稠密峰值

    @property
    def int_ops(self) -> float:
        return self.cores * self.clock_ghz * 1e9 * self.int_per_clock


# Ada / Ampere / Turing 每 SM 都是 FP32 128 lane、其中 64 条兼做 INT32，
# 所以 Ada 和 Ampere 的 int_per_clock 记 0.5，Turing 是独立 64+64 记 1.0。
DEVICES = [
    Device("RTX 4090", 16384, 2.52, 0.5, 1008, 72, 5000, 165.2),
    Device("A100-40GB", 6912, 1.41, 0.5, 1555, 40, 7000, 312.0),
    Device("Tesla T4", 2560, 1.59, 1.0, 320, 4, 1300, 65.0),
]


def state_bytes_per_tick(cfg: SimConfig) -> int:
    """step 一个 env-tick 的强制访存（读+写，假设缓存完美）。"""
    nc = cfg.n_cells
    b = 0
    b += nc * 1              # wall 读
    b += nc * 4 * 2          # fuse 读写
    b += nc * 1 * 2          # owner 读写
    b += nc * 1 * 2 * 2      # covered / trig scratch 读写（下界只算 1 轮）
    b += cfg.n_players * (4 * 2 * 2 + 1 * 2 + 1 * 2 + 4)  # pos(float2) + alive + hold + act
    b += 4 + 1 + 4           # reward + done + t
    return b


def obs_bytes_per_tick(cfg: SimConfig) -> int:
    """observe 一个 env-tick 的写入量。

    观测是 **env 级共享的一份** (2P+3, H, W)，不再每个角色一份：所有通道都
    与"我是谁"无关，视角只是通道置换，由网络第一层的权重索引吸收。
    所以这里既不乘 P，dtype 也是 fp16（默认）而不是 fp32。
    """
    itemsize = 2 if cfg.obs_fp16 else 4
    return cfg.n_channels * cfg.n_cells * itemsize


def instructions_per_tick(cfg: SimConfig) -> int:
    """指令数模型：以"格子访问次数 × 每次约 8 条指令"估算。

    格子访问次数拆解：
      引信递减       1 × ncell
      连锁定轮迭代   max_chain × ncell × 1.2   （扫一遍 + 命中格投射射线）
      清场           1 × ncell
      放泡/移动/判死 P × 约 20（连续坐标要查 2x2 邻格）
    误差在 2~3 倍量级，用来判断"是不是指令受限"够了，不用来报成绩。
    """
    nc = cfg.n_cells
    visits = nc * (2 + 1.2 * cfg.max_chain) + cfg.n_players * 20
    return int(visits * 8)


def nn_flops_per_inference(cfg: SimConfig) -> int:
    """ActorCritic 单样本前向 FLOPs（乘加各算 1）。"""
    c, h, w = cfg.obs_shape
    cells = h * w
    f = 0
    for cin, cout, k in ((c, 16, 3), (16, 32, 3), (32, 64, 3), (64, 8, 1)):
        f += cells * cout * cin * k * k * 2
    flat = 8 * cells
    for fin, fout in ((flat, 128), (128, 128), (128, 64), (64, 5), (128, 64), (64, 1)):
        f += fin * fout * 2
    return f


def analyze(dev: Device, cfg: SimConfig, num_envs: int) -> dict:
    sb = state_bytes_per_tick(cfg)
    ob = obs_bytes_per_tick(cfg)
    instr = instructions_per_tick(cfg)

    resident_mb = num_envs * sb / 1e6
    l2_fits = resident_mb < dev.l2_mb * 0.8
    eff_bw = dev.l2_bw_gb if l2_fits else dev.mem_bw_gb

    step_dram = dev.mem_bw_gb * 1e9 / sb
    step_l2 = dev.l2_bw_gb * 1e9 / sb
    step_instr = dev.int_ops / instr
    # observe 的写入必须落显存（要交给 cuDNN 读），不能算 L2 驻留
    obs_dram = dev.mem_bw_gb * 1e9 / ob

    step_ceiling = min(eff_bw * 1e9 / sb, step_instr)
    pipeline = 1.0 / (1.0 / step_ceiling + 1.0 / obs_dram)
    return {
        "state_bytes": sb, "obs_bytes": ob, "instr": instr,
        "resident_mb": resident_mb, "l2_fits": l2_fits,
        "step_dram": step_dram, "step_l2": step_l2, "step_instr": step_instr,
        "step_ceiling": step_ceiling, "obs_dram": obs_dram, "pipeline": pipeline,
    }


def nn_budget(dev: Device, cfg: SimConfig, env_steps: float, util: float,
              epochs: int = 4) -> dict:
    """网络侧的时间预算：前向（P 个角色）+ 反向（只有 learner 的样本，PPO 多轮）。"""
    f = nn_flops_per_inference(cfg)
    eff = dev.tensor_tflops * 1e12 * util
    t_fwd = env_steps * cfg.n_players * f / eff
    # 反向约等于前向的 2 倍；PPO 对同一批数据重复 epochs 轮
    t_bwd = env_steps * f * 3 * epochs / eff
    return {"flops_per_inf": f, "t_fwd": t_fwd, "t_bwd": t_bwd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=13)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--envs", type=int, default=65536)
    ap.add_argument("--episodes", type=int, default=5_000_000)
    ap.add_argument("--tick-hz", type=int, default=10)
    ap.add_argument("--game-seconds", type=int, default=60)
    ap.add_argument("--nn-util", type=float, default=0.4,
                    help="tensor core 实际利用率；小 batch 小卷积拿不到峰值")
    args = ap.parse_args()

    cfg = SimConfig(height=args.size, width=args.size, n_players=args.players)
    ticks = args.tick_hz * args.game_seconds
    env_steps = args.episodes * ticks

    print(f"配置：{args.size}x{args.size}  {args.players} 人  C={cfg.n_channels}  "
          f"max_chain={cfg.max_chain}  并行 {args.envs:,} 局  "
          f"观测 1 份/env-tick（{'fp16' if cfg.obs_fp16 else 'fp32'} 共享）")
    print(f"预算：{args.episodes:,} 局 × {ticks} tick = {env_steps:.3e} env-steps\n")

    for dev in DEVICES:
        a = analyze(dev, cfg, args.envs)
        print(f"{'=' * 74}\n{dev.name}  "
              f"({dev.cores:,} core @ {dev.clock_ghz} GHz, "
              f"{dev.mem_bw_gb:.0f} GB/s, L2 {dev.l2_mb:.0f} MB, "
              f"{dev.tensor_tflops:.0f} TFLOP/s bf16)")
        print(f"  访存量        state {a['state_bytes']:,} B/env-tick   "
              f"obs {a['obs_bytes']:,} B/env-tick  ({a['obs_bytes'] / a['state_bytes']:.1f}x)")
        print(f"  指令量        约 {a['instr']:,} INT 指令/env-tick")
        print(f"  状态驻留      {a['resident_mb']:.1f} MB  "
              f"{'装得进 L2 → 走 L2 带宽' if a['l2_fits'] else '装不进 L2 → 走 DRAM 带宽'}")
        print(f"  step 上限     DRAM {a['step_dram'] / 1e6:>9,.0f} M/s   "
              f"L2 {a['step_l2'] / 1e6:>9,.0f} M/s   "
              f"指令 {a['step_instr'] / 1e6:>9,.0f} M/s")
        print(f"                → step 天花板 {a['step_ceiling'] / 1e6:,.0f} M env-steps/s "
              f"（{'指令受限' if a['step_instr'] < a['step_dram'] * 1.0 and a['step_instr'] == min(a['step_instr'], a['step_dram'], a['step_l2']) else '访存受限'}）")
        print(f"  observe 上限  {a['obs_dram'] / 1e6:,.0f} M env-steps/s "
              f"（写观测张量必须落显存）")
        print(f"  step+observe  {a['pipeline'] / 1e6:,.0f} M env-steps/s  ← 模拟器实际天花板")

        t_sim = env_steps / a["pipeline"]
        nb = nn_budget(dev, cfg, env_steps, args.nn_util)
        total = t_sim + nb["t_fwd"] + nb["t_bwd"]
        print(f"  {args.episodes / 1e6:.0f}M 局预算   模拟 {t_sim / 60:>7.1f} min   "
              f"前向 {nb['t_fwd'] / 60:>7.1f} min   反向 {nb['t_bwd'] / 60:>7.1f} min   "
              f"合计 {total / 3600:.2f} h")
        share = t_sim / total * 100
        print(f"                模拟只占 {share:.1f}% → "
              f"{'瓶颈在模拟器，继续优化 kernel' if share > 50 else '瓶颈已经转移到神经网络，模拟器不再是问题'}")

    print(f"\n{'=' * 74}")
    print("注意：以上全是纸面上限。真实 kernel 会有 warp 分歧、非完美合并、"
          "launch 开销，\n拿到 2~5 折是正常的。这张表的用途是给实测值一个参照系，"
          "不是承诺。")


if __name__ == "__main__":
    main()
