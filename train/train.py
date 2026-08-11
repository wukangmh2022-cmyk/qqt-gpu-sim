"""训练主循环：自我博弈 + 模型池 + 课程 + 断点续训。

断点续训是硬需求，不是锦上添花：Kaggle 单次会话 12 小时上限、每账号每周
30 小时，1000 万局要靠多账号接力。所以 checkpoint 必须存全：
网络、优化器、全局步数、课程进度、模型池（含各快照 ELO）、RNG 状态。
少存任何一项，接力就会退化成"重新热身"。

用法：
    python -m train.train --num-envs 4096 --total-steps 200_000_000
    python -m train.train --resume ckpt/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import time
import traceback

import torch

from sim.config import N_BOMB, N_MOVES, SimConfig, obs_extra
from sim.factory import make_sim
from sim.bots import make_bot

from .curriculum import CurriculumState, default_curriculum
from .bc import load_recordings, bc_update
from .model import ActorCritic, infer_players
from .model_pool import ModelPool, load_frozen
from .ppo import PPOConfig, SelfPlayRunner, ppo_update


# ---------------------------------------------------------------- 内存预算
# 之前吃过亏：2048 env × 128 rollout 直接整机死机。现在启动前就算好峰值
# 内存上界，超预算立刻拒绝 —— 预测失败，而不是让系统 swap 到死。

def available_ram_bytes() -> int | None:
    """当前可用物理内存（字节）。探测不到返回 None（不设上限，仅警告）。"""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5, check=True).stdout
        counts = {}
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                counts[k.strip()] = int(v.strip().rstrip("."))
            except ValueError:
                continue
        total = (counts.get("Pages free", 0) + counts.get("Pages inactive", 0)
                 + counts.get("Pages speculative", 0)) * page
        return total or None          # 解析失败/为零都按"探测不到"处理
    except Exception:
        return None


def estimate_peak_bytes(cfg: SimConfig, num_envs: int, rollout_steps: int,
                        minibatches: int) -> int:
    """PPO 一个迭代的峰值内存**上界**（字节），启动前就算好。

    峰值主项是更新阶段单个 minibatch 的 CNN 前向+反向中间张量：
    B = num_envs×rollout_steps/minibatches 个样本，每层激活 B×C×H×W×4B，
    反向保守按前向 2 倍算。B 是唯一的大线性项 —— 2048×128/4 = 65,536
    样本会上到 ~11 GB，8GB 机器直接整机死机；512×32/4 = 4,096 只有 ~0.7 GB。
    """
    c, h, w = cfg.obs_shape
    cells = h * w
    B = num_envs * rollout_steps // minibatches
    act_conv = B * cells * (16 + 32 + 64 + 8) * 4 * 2   # 4 层 conv 激活 × 反向
    act_flat = B * cells * 8 * 4 * 2                     # 1x1 压缩后的 flatten
    act_fc = B * 128 * 2 * 4 * 2                         # shared 两个隐层
    mb_obs = B * c * cells * (2 if cfg.obs_fp16 else 4)  # minibatch 的 obs 拷贝
    t = rollout_steps * num_envs
    buf = t * (c * cells * 2 + (N_MOVES + N_BOMB + 4) * 4)   # buffer: obs fp16 + 掩码/动作/统计
    adv = 3 * t * 4                                          # GAE 三份张量
    sim_state = cfg.n_cells * 12 * num_envs                  # 状态平面，量级很小
    return act_conv + act_flat + act_fc + mb_obs + buf + adv + sim_state


def adapt_first_conv(model: ActorCritic, new_shape: tuple[int, int, int],
                     arch: str = "cnn",
                     n_players: int | None = None) -> ActorCritic:
    """人数变化 ⇒ 通道数变化（C = 2P+3）。旧通道原样搬过去，新增对手通道置零。

    置零而不是随机初始化：刚进新阶段时网络行为与上一阶段一致，新对手通道的
    权重由梯度自己长出来，胜率曲线不会先掉一个坑。

    `conv0.weight` 的输入通道维是**视角序**（推导见 train/model.py），四段固定：

        [0]            自己位置          语义固定
        [1]            自己泡泡引信      语义固定
        [2 .. P]       各对手位置        长度 P-1，按较小者截断
        [P+1 .. 2P-1]  各对手泡泡引信    长度 P-1，同上
        [2P .. 2P+2]   墙 / 危险 / 进度  语义固定

    所以搬运 = 头 2 段 + 尾 3 段直接对齐，中间两段各按 min(P) 截断。
    MLP 架构（shared 第一层 Linear 的列块按视角序重排）同样按上述分段搬运
    权重列块；形状相同的键原样拷贝，与 CNN 共用同一套逻辑。
    """
    if tuple(new_shape) == tuple(model.obs_shape):
        return model
    old_p = model.n_players
    if n_players is not None:
        new_p = n_players
    else:
        new_p = infer_players(new_shape[0])
    keep = min(old_p, new_p) - 1            # 能原样继承的对手数
    new_model = ActorCritic(new_shape, arch=arch,
                            n_players=new_p).to(next(model.parameters()).device)
    old_sd, new_sd = model.state_dict(), new_model.state_dict()
    for key in new_sd:
        if key not in old_sd:
            continue
        if old_sd[key].shape == new_sd[key].shape:
            new_sd[key] = old_sd[key].clone()
        elif key == "conv0.weight":
            o, w = old_sd[key], torch.zeros_like(new_sd[key])
            w[:, :2] = o[:, :2]                                  # 自身两通道
            # 墙 / 危险 / 进度在**视角序** [2P..2P+2]，按位置对齐
            # （不能用 -3: —— 新布局视角序尾部是 extra 通道）
            w[:, 2 * new_p:2 * new_p + 3] = o[:, 2 * old_p:2 * old_p + 3]
            if keep > 0:
                w[:, 2:2 + keep] = o[:, 2:2 + keep]              # 对手位置段
                w[:, new_p + 1:new_p + 1 + keep] = o[:, old_p + 1:old_p + 1 + keep]
            new_sd[key] = w
        elif key == "shared.0.weight":      # MLP：第一层 Linear 按视角列块搬运
            o, w = old_sd[key], torch.zeros_like(new_sd[key])
            hw = new_shape[1] * new_shape[2]
            w[:, :2 * hw] = o[:, :2 * hw]
            w[:, (2 * new_p) * hw:(2 * new_p + 3) * hw] = \
                o[:, (2 * old_p) * hw:(2 * old_p + 3) * hw]
            if keep > 0:
                w[:, 2 * hw:(2 + keep) * hw] = o[:, 2 * hw:(2 + keep) * hw]
                w[:, (new_p + 1) * hw:(new_p + 1 + keep) * hw] = \
                    o[:, (old_p + 1) * hw:(old_p + 1 + keep) * hw]
            new_sd[key] = w
    new_model.load_state_dict(new_sd)
    return new_model


def anneal_frac(local_step: int, anneal_steps: int) -> float:
    """熵/学习率线性退火系数，按**本次运行内步数**走到 1。

    用 global_step/total 的话，resume 一进来 frac 就顶着老值 —— 续训从低熵
    直接继续，探索回不来（老 bug）。local_step = 本次运行内步数，fresh 和
    resume 都在 0 起步重新开熵。
    """
    return min(1.0, local_step / max(1, anneal_steps))


def build_opponents(pool: ModelPool, learner: ActorCritic, elo: float,
                    n_players: int, device,
                    fixed_items=(), bot_items=(), fixed_elo=None,
                    warmup: bool = False, fixed_prob: float = 0.0,
                    bot_prob: float = 0.0) -> tuple[list, list]:
    """给每个对手位独立采样一个对手，1v2 时两个对手可以来自不同来源。

    三个来源（课程化，见 sim/bots.py）：
    - `fixed_items`：固定 checkpoint 陪练（5x2/420M/CNN...），ELO 有绝对意义；
    - `bot_items`：规则 bot（random/greedy/astar），启蒙期用；
    - 池子快照：原有自博弈路径，跨通道自动 adapt。

    每个对手位：
    - 热身期（warmup）只用 fixed+bot；
    - 之后以 fixed_prob 概率用 fixed；**以 bot_prob 概率用规则 bot** —— bot
      全程混入（不只 warmup）是防"对手分布单一化"的关键：warmup 后只打接近型
      网络会让模型遗忘"对手可能不接近/躲闪"的分布（实测对静止/躲闪对手会
      贴墙放炮自杀）。bot_prob > 0 保证模型永远见得到非接近型对手；
    - 否则 pool.sample。snaps 与 nets 一一对应：池快照原样返回，
    fixed/bot 返回伪快照 {"name", "elo"}（ELO 更新走 fixed_elo 字典）。
    """
    fixed_elo = fixed_elo or {}
    h, w = learner.obs_shape[1], learner.obs_shape[2]
    nets, snaps = [], []
    for _ in range(n_players - 1):
        cands = list(fixed_items) + (list(bot_items) if warmup else [])
        if cands and (warmup or random.random() < fixed_prob):
            name, net = random.choice(cands)
            nets.append(net)
            snaps.append({"name": name, "elo": fixed_elo.get(name, 1000.0)})
            continue
        if bot_items and random.random() < bot_prob:
            # bot 全程混入：非 warmup 期也有概率打规则 bot（对手多样性锚）
            name, net = random.choice(bot_items)
            nets.append(net)
            snaps.append({"name": name, "elo": fixed_elo.get(name, 1000.0)})
            continue
        snap = pool.sample(elo)
        snap_c = snap["state"]["conv0.weight"].shape[1] if "conv0.weight" in snap["state"] \
            else (snap["state"]["shared.0.weight"].shape[1] // (h * w))
        state = snap["state"]
        if snap_c != learner.obs_shape[0]:
            old = load_frozen(ActorCritic, (snap_c, h, w), state, device,
                              arch=learner.arch, n_players=learner.n_players)
            state = adapt_first_conv(old, learner.obs_shape, arch=learner.arch,
                                     n_players=learner.n_players).state_dict()
        nets.append(load_frozen(ActorCritic, learner.obs_shape, state, device,
                                arch=learner.arch, n_players=learner.n_players))
        snaps.append(snap)
    return nets, snaps


def apply_opp_boost(sim, nets, enabled: bool) -> None:
    """训练难度：按对手类型给 sim 设初始属性增强（见 SimConfig.opp_boost）。

    - 规则 bot（BotWrapper）→ 2：80% 初始（opp_growth_*）；
    - 历史网络（fixed ckpt 陪练 / 模型池快照）→ 1：起点 × opp_hist_mult（2-30%）。
    - 关闭 → 0：双方同起点（对打/测试行为不变）。
    学习侧（pid 0）始终从各自模式起点起步；掉血惩罚对双方生效，
    clamp 回各自（增强后）的起点。1v1 单阶段所有 env 同一对手 → 单一档位。
    """
    if not enabled:
        sim.set_opp_boost(0)
        return
    boost = 2 if any(getattr(n, "is_bot", False) for n in nets) else 1
    sim.set_opp_boost(boost)


def load_fixed_checkpoint(path: str, target_shape, device) -> ActorCritic:
    """把外部 checkpoint 加载成冻结网络并适配到 target obs_shape。

    固定对手用训练时存档（同布局 → 直接载入；跨通道 → adapt_first_conv），
    允许与 learner 不同 arch（比如 cnn 旧档给 mlp learner 当陪练）。
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("arch", "cnn")
    n_p = ck.get("n_players")
    net = ActorCritic(tuple(ck["obs_shape"]), arch=arch, n_players=n_p).to(device)
    net.load_state_dict(ck["model"])
    if tuple(net.obs_shape) != tuple(target_shape):
        net = adapt_first_conv(net, tuple(target_shape), arch=arch,
                               n_players=None).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def update_fixed_elo(fixed_elo: dict, name: str, learner_elo: float,
                     learner_score: float, k: float = 16.0) -> float:
    """固定对手的标准 ELO：learner_score ∈ [0,1]，返回新的 learner_elo。

    与 ModelPool.update_elo 同一套公式，但对手 ELO 存在 `fixed_elo` 字典
    （固定陪练/课程 bot 没有池快照可改）—— 让 learner 的 ELO 有绝对意义：
    "打平/打赢 420M" 直接体现在 ELO 上，而不是只和"上一版自己"比。
    """
    opp = fixed_elo.get(name, 1000.0)
    expected = 1.0 / (1.0 + 10 ** ((opp - learner_elo) / 400.0))
    delta = k * (learner_score - expected)
    fixed_elo[name] = opp - delta
    return learner_elo + delta


def eval_fixed_opponents(sim_cfg, learner, fixed_items, pcfg, device,
                         backend: str = "torch", episodes: int = 128) -> dict:
    """对每个固定对手跑一个小型评估（默认 128 局），返回 name → win_rate。

    训练主循环里跑在独立的小 sim 上，不污染主 buffer；周期调用，
    给出"对 420M/5x2/CNN 的绝对胜率"这个用户关心的硬指标。
    **128 局**（从 256 降）：评估只是统计趋势，128 局噪声可接受；
    而 corridor 100% 爆率下双方满成长（blast=7），rays 火焰计算每 tick
    196 kernel —— 256 局 × 4 对手的评估能把训练节奏拖慢几分钟（实测
    course7 卡 >7min）。评估减半局数即可恢复正常节奏，主训练不受影响。
    """
    sim = make_sim(sim_cfg, 128, backend=backend, device=device, seed=0)
    out = {}
    for name, net in fixed_items:
        # 规则 bot（--fixed-bots，如 astar）**跳过**：BotWrapper 绑定主 sim 的
        # 状态（尺寸 = 主 rollout 的 env 数），在 256-env 小 sim 上跑会跨 sim
        # 错配直接崩；且规则 AI 强度固定、不需要评估，主循环胜负本身在记。
        if getattr(net, "is_bot", False):
            continue
        runner = SelfPlayRunner(sim, learner, [net], pcfg, 1.0)
        runner.clear_stats()
        guard = 0
        while runner.ep_stats["count"] < episodes and guard < 8:
            runner.collect()
            guard += 1
        out[name] = runner.win_rate()
    return out


def save_ckpt(path: str, *, learner, opt, pool, cstate, global_step, elo, args,
              fixed_elo: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "format_version": 2,  # v2 = (N,2P+3,H,W) 共享观测 + conv0.weight
        "model": learner.state_dict(),
        "obs_shape": learner.obs_shape,
        "n_players": learner.n_players,   # 观测含 obs_extra 后无法反推，必须存
        "arch": learner.arch,      # cnn | mlp：加载/评估/对打按它构造网络
        "opt": opt.state_dict(),
        "pool": pool.state_dict(),
        "fixed_elo": fixed_elo or {},
        "curriculum": {"stage_idx": cstate.stage_idx,
                       "episodes_in_stage": cstate.episodes_in_stage,
                       "recent_wins": cstate.recent_wins},
        "global_step": global_step,
        "elo": elo,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)   # 原子替换：会话被强制掐断也不会留下半个文件


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--total-steps", type=int, default=20_000_000,
                    help="env-step 总预算（= num_envs × tick 数）")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "cuda"])
    ap.add_argument("--arch", default="cnn", choices=["cnn", "mlp", "lstm"],
                    help="网络架构：cnn = 3 层 3x3 卷积 + 1x1（默认）；"
                         "mlp = 全局全连接（flat→128→128，参数更少，GEMM 更快）；"
                         "lstm = 局部 7×7 CNN + 相对坐标 + 全局状态 + LSTM（BombermanNet）")
    ap.add_argument("--map-mode", default="open", choices=["open", "corridor"],
                    help="open = 纯空场（默认）；corridor = 左右可炸墙 + 顶部永久墙 "
                         "+ 时间成长（speed 自动降到 3.0、对局 1800 tick）")
    ap.add_argument("--open-fraction", type=float, default=0.5,
                    help="corridor 训练时每局 open 关占比（混合地图，其余 = corridor）。"
                         "open 关纯空场、成长初始 80%% 上限 —— 逼 AI 学真交战，"
                         "防 corridor 里横向刷宝箱的局部最优。0.0 = 纯 corridor")
    ap.add_argument("--ring-fraction", type=float, default=0.0,
                    help="每局环岛关占比（中间固定空旷区 + 四周可炸墙 + 宝箱 100%%）。"
                         "与 open_fraction 之和 ≤ 1（余量 = corridor）")
    ap.add_argument("--hazard-fraction", type=float, default=0.0,
                    help="**融合躲避特训**：每局按该比例掷「炸弹雨躲避关」（其余 = "
                         "正常关）。躲避关：双方禁放泡（bombs_cap=0，放泡头被屏蔽）、"
                         "环境每 hazard_wave_ticks 播一波炸弹雨 —— 谁活得久谁赢。"
                         "与地图类型正交：corridor 混合地图里 open/ring/corridor "
                         "关各自再随机命中躲避模式，正常关与躲避关交替出现 → "
                         "躲避能力进主策略，不牺牲原有进攻/刷箱行为。0 = 不启用。"
                         "已训练档 resume 无缝继续（通道布局不变）")
    ap.add_argument("--crate-speed-only", action="store_true",
                    help="躲避关宝箱只加速度（配合 --hazard-fraction 用）：炸弹雨关"
                         "踩宝箱不再随机升 泡/威/速，只升速度 —— 泡/威在禁放泡关是"
                         "死通道，学了白学；速度才真正影响躲避。正常关不受影响")
    ap.add_argument("--autocast", action="store_true",
                    help="PPO 更新用 fp16 autocast（GPU 上 GEMM 走 fp16，更新提速）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--minibatches", type=int, default=PPOConfig.minibatches,
                    help="PPO 一个 epoch 切几块；minibatch = num_envs×rollout_steps/"
                         "minibatches 是峰值内存的主项，内存紧张就调大它")
    ap.add_argument("--bptt-window", type=int, default=0,
                    help="LSTM 架构的 BPTT 截断窗口（0 = 全序列反传；>0 只反传"
                         "最近 W 步 —— truncated BPTT，910B 实测反向 -78%）")
    ap.add_argument("--max-mem-frac", type=float, default=0.55,
                    help="预估峰值内存占可用内存的比例上限，超了直接拒绝启动")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda,
                    help="GAE λ（默认 0.95）。A/B 实验可用 0.88–0.9：更偏「即时回报」、"
                         "偏差更小但方差更大 —— 在稠密塑形奖励（danger/approach/"
                         "放泡预测）下通常更稳，CNN 训练可以单独试")
    ap.add_argument("--oversample-dying", type=int,
                    default=PPOConfig.oversample_dying,
                    help="濒死/死亡帧过采样倍率（默认 3；1 = 关闭）。掉血/终局帧"
                         "在一局里极稀疏，价值函数在死亡附近样本不足 → 按倍率复制"
                         "进采样池，每 epoch 训练量不变")
    ap.add_argument("--snapshot-every", type=int, default=200,
                    help="每多少次 PPO 迭代往模型池存一个快照")
    ap.add_argument("--single-stage", action="store_true",
                    help="只跑课程第一阶段（1v1），不推进到 1v2 —— 对打 UI 用纯 1v1 模型，"
                         "避免 1v2 权重缩回 1v1 后行为退化（不放炮、往角落偏移）")
    ap.add_argument("--fixed-ckpt", action="append", default=[], metavar="NAME=PATH",
                    help="固定陪练 checkpoint（可重复指定）：启动时加载为冻结网络并"
                         "适配到当前观测，始终作为潜在对手。ELO 走 fixed_elo 字典"
                         "（打平/打赢它直接体现绝对水平）。"
                         "例：--fixed-ckpt 5x2=private_data/duel_5x2.pt")
    ap.add_argument("--bot-opponents", default="",
                    help="逗号分隔的课程 bot 列表（random/greedy/astar，见 sim/bots.py）。"
                         "热身期的主要对手，之后按 --bot-opp-prob 继续混入")
    ap.add_argument("--fixed-bots", default="",
                    help="逗号分隔的**固定陪练 bot**（random/greedy/astar）：与固定 ckpt 一起"
                         "常驻 fixed 阵容，每迭代按 --fixed-opp-prob 被抽中 —— 规则 AI 也"
                         "当固定陪练（如 astar = 进攻/防守混合寻路），保证规则对手持续出现"
                         "（不依赖 --bot-opp-prob 的低概率混入）")
    ap.add_argument("--opp-boost", action="store_true",
                    help="训练难度：对手初始属性按类型增强 —— 历史网络（fixed ckpt + "
                         "模型池快照）初始属性 ×cfg.opp_hist_mult（轻微 2-30%%）；规则 bot "
                         "（astar/greedy/random）初始 80%%（cfg.opp_growth_*）。学习侧仍从"
                         "各自模式起点起步。掉血惩罚对双方生效（clamp 回各自增强后起点）")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="热身步数：global_step < N 时每个对手位只用 bot+固定 ckpt，"
                         "不碰模型池（从近零权重起步先打打得过的，别硬碰历史强模型）")
    ap.add_argument("--fixed-opp-prob", type=float, default=0.0,
                    help="热身结束后每个对手位用固定对手的概率，其余 = 池子快照"
                         "（0.4 = 四成局对固定陪练，六成自博弈）")
    ap.add_argument("--bot-opp-prob", type=float, default=0.0,
                    help="热身结束后规则 bot 的**全程混入概率**（0.2 = 两成局打"
                         "astar/greedy）。bot 只在 warmup 出现会让模型遗忘'对手可能"
                         "不接近/躲闪'的分布（实测对静止对手贴墙放炮自杀）—— 全程"
                         "混入是防遗忘的对手多样性锚。bot_prob 在 fixed_prob 之后独立"
                         "判定")
    ap.add_argument("--lr-final", type=float, default=None,
                    help="学习率线性退火终点（None = 恒为 --lr）")
    ap.add_argument("--ent-anneal-steps", type=int, default=None,
                    help="熵退火跨度（本次运行内步数，resume 时重新开熵；"
                         "None = total_steps）")
    ap.add_argument("--ckpt", default="ckpt/latest.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-csv", default="ckpt/train_log.csv")
    ap.add_argument("--time-budget", type=float, default=11.0 * 3600,
                    help="秒；到点自动存盘退出，留出 Kaggle 12h 的余量")
    ap.add_argument("--timeout-draw", action="store_true",
                    help="超时全员存活 → 血多者胜奖励 × 探索退火（击杀能力上来归零）："
                         "开局超时领先有回报，后期只剩真击杀的固定回报。默认开启"
                         "（SimConfig.timeout_draw=True）")
    ap.add_argument("--combo-reward", type=float, default=0.10,
                    help="combo 连击奖励：不掉血连续造成伤害 = 连击，连击数越高分"
                         "越多、间隔越短分越多（combo_gap_factor^间隔）；掉血打断"
                         "连击 → 逼『无伤压制』（像格斗连段）。默认 0.10，传 0 关闭")
    ap.add_argument("--combo-gap-factor", type=float, default=0.9,
                    help="combo 间隔因子：间隔每 +1 tick 分 × 此值（连击密分高）")
    ap.add_argument("--bc-data", default=None,
                    help="人类录像目录（recordings/*.npz）：每迭代采样一批做 BC "
                         "监督更新（--bc-coef 权重），人类精准放炮/追杀/包围行为持续"
                         "引导策略 —— 稀疏奖励学不到的手工打法直接烧进权重")
    ap.add_argument("--bc-coef", type=float, default=0.5,
                    help="BC 辅助损失权重（PPO 主力 + BC 引导；0 = 纯 PPO）")
    ap.add_argument("--bc-batch", type=int, default=256,
                    help="每次 BC 更新采样的录像样本数")
    ap.add_argument("--bc-every", type=int, default=1,
                    help="每多少迭代做一次 BC 更新")
    ap.add_argument("--explore-anneal", action="store_true",
                    help="探索奖励自适应退火（论文 Pommerman α=1-tanh(k·x)）：探索"
                         "塑形（放炮三件套/连锁/吃箱）乘 α，x=平均每局击杀 —— 前期"
                         "探索满格、后期击杀上来自动归零纯赢比赛。治'后期为刷分放炮'"
                         "卡局部最优。k=1.2 同论文（1v1 x∈[0,1]，曲线仍平滑）")
    ap.add_argument("--explore-anneal-k", type=float, default=1.2,
                    help="退火斜率 k（α=1-tanh(k·2·击杀率)）。默认 1.2 同论文；"
                         "调小=退火更慢，危险惩罚/吃箱等塑形保持更久（2026-08-11 "
                         "用户定：3B 版 α≈0.03 退过头，危险/吃箱信号被缩到 3%，"
                         "corridor 躲炸弹差+吃箱弱同源 → 本阶段用 0.6 慢退火）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                device = torch.device("npu:0")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.map_mode == "corridor":
        # corridor：左右可炸墙 + 顶部永久墙 + 时间成长。
        # 初始 3.0 格/秒（0.3 格/tick），成长 ×2.1 到 6.3 格/秒（0.63 格/tick）；
        # 3 分钟对局 1800 tick。--open-fraction 混合 open 关（纯空场 80% 成长）。
        base = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                         open_fraction=args.open_fraction,
                         ring_fraction=args.ring_fraction,
                         hazard_fraction=args.hazard_fraction,
                         crate_speed_only=args.crate_speed_only,
                         timeout_draw=args.timeout_draw,
                         combo_reward=args.combo_reward,
                         combo_gap_factor=args.combo_gap_factor)
    else:
        base = SimConfig(timeout_draw=args.timeout_draw,
                         combo_reward=args.combo_reward,
                         combo_gap_factor=args.combo_gap_factor)
    stages = default_curriculum(base)
    cstate = CurriculumState()
    pcfg = PPOConfig(rollout_steps=args.rollout_steps, lr=args.lr,
                     minibatches=args.minibatches, gae_lambda=args.gae_lambda,
                     oversample_dying=args.oversample_dying,
                     bptt_window=args.bptt_window)
    elo = 1000.0
    global_step = 0
    fixed_elo: dict[str, float] = {}   # 固定对手 ELO（持久化，resume 恢复）

    ckpt = torch.load(args.resume, map_location=device, weights_only=False) if args.resume else None
    if ckpt and ckpt.get("format_version") != 2:
        raise ValueError(
            "checkpoint 是旧观测布局，不能安全自动迁移。请用共享观测版本重新训练；"
            "新版 checkpoint 含 format_version=2。")
    obs_shape = tuple(ckpt["obs_shape"]) if ckpt else stages[0].cfg.obs_shape
    # n_players：观测含 obs_extra 通道后无法从通道数唯一反推；resume 的旧
    # checkpoint 存了 n_players 字段（7 通道旧档没有则按 (c-3)//2 兜底）。
    # 新训练直接用首阶段人数（含扩展观测时 (c-3)//2 会反推错）。
    ck_np = ckpt.get("n_players") if ckpt else None
    if ck_np is None:
        p0 = (obs_shape[0] - 3) // 2
        ck_np = p0 if 2 * p0 + 3 + obs_extra(p0) == obs_shape[0] \
            else stages[0].cfg.n_players
    learner = ActorCritic(obs_shape, arch=args.arch, n_players=ck_np).to(device)
    ckpt_migrated = False
    if ckpt:
        ck_extra = obs_shape[0] > 2 * ck_np + 3
        if not ck_extra:
            # 旧 7 通道 ckpt → 新 14 通道训练：用 adapt_first_conv 迁移模型
            # （旧通道原样搬 + 新增通道置零，梯度自己长出来）。
            learner = adapt_first_conv(
                learner, stages[0].cfg.obs_shape, arch=args.arch,
                n_players=stages[0].cfg.n_players).to(device)
            ckpt_migrated = True
            print(f"[migrate] 7 通道 → {stages[0].cfg.obs_shape} 通道迁移")
        else:
            # 同布局续训（rw7 14 通道 → rw8 14 通道）：权重形状一致，直接载入。
            learner.load_state_dict(ckpt["model"])
        cstate = CurriculumState(**ckpt["curriculum"])
        elo, global_step = ckpt["elo"], ckpt["global_step"]
        fixed_elo = dict(ckpt.get("fixed_elo") or {})

    stage = stages[cstate.stage_idx]

    # ---------------- 内存守卫（启动前，先于一切大分配）----------------
    # CPU 路径在共享内存上跑 CNN 前向/反向，峰值随 minibatch 线性涨。
    # 不预估就开跑，曾把 8GB 机器直接整机死机（2048×128/4 的 minibatch）。
    # 现在启动前算好上界：超预算立刻打印怎么改，拒绝启动。
    avail = available_ram_bytes()
    est = estimate_peak_bytes(stage.cfg, args.num_envs, args.rollout_steps,
                              pcfg.minibatches)
    if not avail:
        print(f"[mem] 无法探测可用内存；预估峰值 {est/1e9:.1f} GB（请留意系统内存）")
    else:
        frac = est / avail
        print(f"[mem] 可用 {avail/1e9:.1f} GB  预估峰值 {est/1e9:.1f} GB "
              f"({frac:.0%})")
        if frac > args.max_mem_frac:
            raise SystemExit(
                f"拒绝启动：预估峰值 {est/1e9:.1f} GB 超过可用内存的 "
                f"{args.max_mem_frac:.0%}（{avail/1e9:.1f} GB）。"
                f"\n  要么调小 minibatch：--num-envs {args.num_envs} --rollout-steps "
                f"{args.rollout_steps} --minibatches {pcfg.minibatches * 2}"
                f"（当前 minibatch = {args.num_envs * args.rollout_steps // pcfg.minibatches:,} 样本）"
                f"\n  要么调小 env 数：--num-envs {args.num_envs // 2}"
                f"\n  内存是 minibatch 的大线性项，翻倍 minibatches 即减半峰值。")

    learner = adapt_first_conv(learner, stage.cfg.obs_shape,
                                n_players=stage.cfg.n_players)
    # optimizer 必须在最终 learner 确定后创建；adapt_first_conv 可能返回新模型。
    opt = torch.optim.Adam(learner.parameters(), lr=pcfg.lr, eps=1e-5)
    pool = ModelPool()
    if ckpt:
        if ckpt_migrated:
            # 迁移场景：模型被 adapt 成新通道，Adam 动量/池里旧 7 通道快照
            # 形状都对不上 —— 跳过 Adam 状态，池子照搬（build_opponents 会
            # 自动把旧快照适配成新通道数）。
            pool.load_state_dict(ckpt["pool"])
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            random.setstate(ckpt["py_rng"])
            print(f"[resume] step={global_step} stage={cstate.stage_idx} "
                  f"elo={elo:.0f} pool={len(pool)} (Adam 状态重置：通道已迁移)")
        else:
            # checkpoint 所在阶段与恢复阶段一致时参数形状不变，Adam 动量可严格恢复。
            # 若未来允许恢复时直接跳阶段，应显式迁移 optimizer state，不能静默错绑。
            if obs_shape != tuple(stage.cfg.obs_shape):
                raise ValueError(
                    "checkpoint 的 obs_shape 与课程阶段不一致，拒绝错绑 Adam 状态")
            opt.load_state_dict(ckpt["opt"])
            pool.load_state_dict(ckpt["pool"])
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            random.setstate(ckpt["py_rng"])
            print(f"[resume] step={global_step} stage={cstate.stage_idx} "
                  f"elo={elo:.0f} pool={len(pool)}")
    print(f"[model] params={learner.n_params():,} obs_shape={learner.obs_shape} device={device}")
    if len(pool) == 0:
        pool.add(learner, step=global_step, elo=elo)
    sim = make_sim(stage.cfg, args.num_envs, backend=args.backend,
                   device=device, seed=args.seed)

    # ---------------- 人类录像 BC 辅助（可选，--bc-data） ----------------
    # 每迭代从录像采样一批监督样本做 BC 更新（--bc-coef 权重），人类精准放炮/
    # 追杀/包围行为持续引导策略 —— 稀疏奖励学不到的手工打法（锁牢笼封堵等）
    # 直接烧进权重。与 PPO 主力并存，不靠一次性预热。
    bc_data = None
    if args.bc_data:
        bc_data = load_recordings(args.bc_data)
        print(f"[bc] 已加载 {len(bc_data[0]):,} 个人类录像样本（--bc-coef "
              f"{args.bc_coef}，每 {args.bc_every} 迭代更新）")

    # ---------------- 固定陪练与课程 bot（对手池的额外来源）----------------
    # 固定陪练：冻结的外部 checkpoint（5x2/420M/CNN 等），启动时 adapt 到当前
    # 观测；ELO 有绝对意义（打赢它 = 绝对变强，不只是赢过上一版自己）。
    # 课程 bot：全向量化规则策略（random/greedy），**不需要寻路** —— 只用
    # 危险图 + Chebyshev 距离贪心逼近，5632 env 无逐环境循环（见 sim/bots.py）。
    fixed_items: list[tuple[str, ActorCritic]] = []
    fixed_specs: list[tuple[str, str]] = []
    for spec in args.fixed_ckpt:
        name, _, path = spec.partition("=")
        net = load_fixed_checkpoint(path, learner.obs_shape, device)
        fixed_items.append((name, net))
        fixed_specs.append((name, path))
        print(f"[opponent] fixed {name} <- {path} (arch={net.arch} {net.n_players}P)")
    # 固定陪练 bot：把规则 AI（如 astar = 进攻/防守混合寻路）也常驻 fixed 阵容，
    # 和固定 ckpt 一样按 fixed_prob 每迭代被抽中 —— 规则对手持续可见，不靠
    # --bot-opp-prob 的低概率混入（防"只打接近型网络"的分布单一化）。
    for kind in [s.strip() for s in args.fixed_bots.split(",") if s.strip()]:
        if kind not in ("random", "greedy", "astar", "hunter"):
            raise ValueError(f"未知固定 bot 类型: {kind}（可选 random/greedy/astar/hunter）")
        fixed_items.append((kind, make_bot(sim, kind)))
        fixed_specs.append((kind, "bot"))
        print(f"[opponent] fixed bot {kind}")
    if args.warmup_steps > 0 and not args.fixed_ckpt and not args.bot_opponents:
        print("[warn] --warmup-steps 指定了但没有任何固定对手/bot，"
              "热身期退化为纯池子采样")
    bot_items: list[tuple[str, object]] = []
    for kind in [s.strip() for s in args.bot_opponents.split(",") if s.strip()]:
        if kind not in ("random", "greedy", "astar", "hunter"):
            raise ValueError(f"未知 bot 类型: {kind}（可选 random/greedy/astar/hunter）")
        bot_items.append((kind, make_bot(sim, kind)))
        print(f"[opponent] bot {kind}")

    nets, snaps = build_opponents(pool, learner, elo, stage.cfg.n_players, device,
                                  fixed_items=fixed_items, bot_items=bot_items,
                                  fixed_elo=fixed_elo,
                                  warmup=global_step < args.warmup_steps,
                                  fixed_prob=args.fixed_opp_prob,
                                  bot_prob=args.bot_opp_prob)
    # 训练难度：对手初始属性按类型增强（历史网络 ×mult / 规则 bot 80%）
    apply_opp_boost(sim, nets, args.opp_boost)
    runner = SelfPlayRunner(sim, learner, nets, pcfg, stage.opponent_handicap)

    log_f = writer = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        log_f = open(args.log_csv, "a", newline="")
        writer = csv.writer(log_f)
        if log_f.tell() == 0:
            writer.writerow(["step", "stage", "elo", "win_rate", "ep_len",
                             "pg", "vf", "ent", "kl", "clipfrac", "sps",
                             "alpha", "suicide_rate", "bombs_per_ep",
                             "danger_frac", "bc"])

    start = time.time()
    it = 0
    per_iter = args.num_envs * pcfg.rollout_steps
    start_step = global_step          # 本次运行起点：熵/学习率按"本次内步数"退火
    try:
        while global_step < args.total_steps:
            it += 1
            t0 = time.time()
            buf, last_val = runner.collect()
            # 熵系数退火：前期充分探索，后期收敛到接近确定性但**保持随机性下限**
            # （entropy_final=0.03 恒定正熵）。钟摆效应根因：旧版退火到 0.002
            # （几乎确定性）→ 双方策略趋同 → 确定性对称均衡 → PPO 零梯度冻结，
            # 表现为周期摆动 + 同步放炮。常量小熵让双方持续有随机性，破对称。
            # frac 用"本次运行内步数"local_step —— resume 时重新开熵，
            # 不然续训从低熵直接继续，探索回不来（老 bug）。
            anneal = args.ent_anneal_steps or args.total_steps
            frac = anneal_frac(global_step - start_step, anneal)
            ent_coef = pcfg.entropy_coef + frac * (pcfg.entropy_final - pcfg.entropy_coef)
            if args.lr_final is not None:
                lr_now = args.lr + (args.lr_final - args.lr) * frac
                for g in opt.param_groups:
                    g["lr"] = lr_now
            stats = ppo_update(learner, opt, buf, last_val, pcfg, ent_coef,
                               autocast=args.autocast)
            # 人类录像 BC 引导：PPO 主力之外，每 --bc-every 迭代从录像采样一批
            # 做监督更新（人类"该放时才放/追杀/包围"行为持续引导）。
            if bc_data is not None and it % args.bc_every == 0:
                bc_loss_v = bc_update(learner, opt, bc_data, args.bc_batch,
                                      args.bc_coef, device,
                                      seed=args.seed + it)
                stats["bc"] = bc_loss_v
            else:
                stats.setdefault("bc", float("nan"))
            global_step += per_iter
            sps = per_iter / max(1e-6, time.time() - t0)

            wr = runner.win_rate()
            done_eps = runner.ep_stats["count"]
            cstate.episodes_in_stage += done_eps
            for _ in range(done_eps):
                cstate.record(wr)
            if done_eps:
                # slot-0 是池快照 → 池子 ELO；是固定对手/bot → fixed_elo 字典
                if "state" in snaps[0]:
                    elo = pool.update_elo(snaps[0], elo, min(1.0, max(0.0, wr)))
                else:
                    elo = update_fixed_elo(fixed_elo, snaps[0]["name"], elo,
                                           min(1.0, max(0.0, wr)))
            # 探索奖励自适应退火（--explore-anneal，论文 α=1-tanh(k·x)）：
            # x = 平均每局击杀（敌方死亡），击杀能力上来 → α 平滑归零 →
            # 放炮三件套/连锁兑现这些"刷分放炮"塑形自动退场，纯赢比赛。
            # **x 放大 2 倍对齐论文语义**：Pommerman 2v2 x∈[0,2]（最多杀 2），
            # 我们 1v1 上限 1 → x_eff = 2×击杀率，满击杀时 α≈0.016（≈归零）。
            # 每迭代按本迭代击杀率更新（runner.ep_stats 已累计，win=敌死）。
            if args.explore_anneal:
                kx = args.explore_anneal_k * (2.0 * runner.kills_per_ep())
                alpha = 1.0 - math.tanh(kx)
                sim.set_explore_coef(alpha)
            else:
                sim.set_explore_coef(1.0)

            if it % 10 == 0:
                # 健康度指标（指导是否停下找问题）：
                #   suicide_rate = 自爆死亡 / 死亡数（98% 自爆老大难）
                #   bombs_per_ep / danger_frac = 放炮频率 / 站危险区占比（钟摆·磨平信号）
                n_deaths = runner.ep_stats["kills"] + runner.ep_stats["suicide"]
                sui_rate = (runner.ep_stats["suicide"] / max(1, n_deaths)
                            if n_deaths else 0.0)
                ep_cnt = max(1, runner.ep_stats["count"])
                # 站危占比分母 = 本次 collect 总 env-tick（固定 num_envs×rollout_steps），
                # 不是 len_sum（只累计已结束局，本段无终局时是 0 → 失真）
                dang_frac = runner.ep_stats["danger_ticks"] / per_iter
                bombs_ep = runner.ep_stats["bombs"] / ep_cnt
                print(f"[{it:6d}] step={global_step/1e6:.2f}M stage={stage.name} "
                      f"wr={wr:.3f} elo={elo:.0f} len={runner.mean_ep_len():.0f} "
                      f"ent={stats['ent']:.3f} kl={stats['kl']:+.4f} "
                      f"α={sim._explore_coef:.2f} 自杀={sui_rate:.0%} "
                      f"放炮={bombs_ep:.0f}/局 站危={dang_frac:.0%} "
                      f"sps={sps/1e3:.0f}k")
                if fixed_items:
                    # 独立小 sim 上对每个固定陪练跑 256 局 → 绝对胜率
                    fwr = eval_fixed_opponents(
                        stage.cfg, learner, fixed_items, pcfg, device,
                        backend=args.backend)
                    print("    fixed: " + "  ".join(
                        f"{n}={v:.3f}" for n, v in fwr.items()))
            if log_f:
                n_deaths = runner.ep_stats["kills"] + runner.ep_stats["suicide"]
                sui_rate = (runner.ep_stats["suicide"] / max(1, n_deaths)
                            if n_deaths else 0.0)
                ep_cnt = max(1, runner.ep_stats["count"])
                writer.writerow(
                    [global_step, stage.name, f"{elo:.1f}", f"{wr:.4f}",
                     f"{runner.mean_ep_len():.1f}"]
                    + [f"{stats[k]:.5f}" for k in
                       ("pg", "vf", "ent", "kl", "clipfrac")]
                    + [f"{sps:.0f}", f"{sim._explore_coef:.4f}",
                       f"{sui_rate:.4f}",
                       f"{runner.ep_stats['bombs']/ep_cnt:.2f}",
                       f"{runner.ep_stats['danger_ticks']/per_iter:.4f}",
                       f"{stats.get('bc', float('nan')):.5f}"])
                log_f.flush()

            if it % args.snapshot_every == 0:
                pool.add(learner, step=global_step, elo=elo)
                save_ckpt(args.ckpt, learner=learner, opt=opt, pool=pool,
                          cstate=cstate, global_step=global_step, elo=elo,
                          args=args, fixed_elo=fixed_elo)

            # 每个 rollout 换一次对手，避免过拟合到单一历史策略
            nets, snaps = build_opponents(pool, learner, elo, stage.cfg.n_players,
                                          device, fixed_items=fixed_items,
                                          bot_items=bot_items, fixed_elo=fixed_elo,
                                          warmup=global_step < args.warmup_steps,
                                          fixed_prob=args.fixed_opp_prob,
                                          bot_prob=args.bot_opp_prob)
            apply_opp_boost(sim, nets, args.opp_boost)
            runner.opponents = nets
            runner.clear_stats()

            if cstate.should_advance(stage) and cstate.stage_idx + 1 < len(stages) \
                    and not args.single_stage:
                cstate.stage_idx += 1
                cstate.episodes_in_stage = 0
                cstate.recent_wins.clear()
                stage = stages[cstate.stage_idx]
                print(f"[curriculum] → {stage.name}  ({stage.notes})")
                learner = adapt_first_conv(learner, stage.cfg.obs_shape,
                                n_players=stage.cfg.n_players)
                opt = torch.optim.Adam(learner.parameters(), lr=pcfg.lr, eps=1e-5)
                sim = make_sim(stage.cfg, args.num_envs, backend=args.backend,
                               device=device, seed=args.seed + it)
                # bot 绑定着旧 sim 的状态（放泡/位置），换图后必须重建
                bot_items = [(kind, make_bot(sim, kind)) for kind, _ in bot_items]
                # 固定陪练按启动时的 obs_shape 适配过，人数变多后要重新适配
                # （只有 1v1 单阶段训练不触发；跨阶段课程需要时按原路径重载）
                if fixed_items and any(
                        n.obs_shape != learner.obs_shape for _, n in fixed_items):
                    fixed_items = [
                        (name, load_fixed_checkpoint(path, learner.obs_shape, device))
                        for name, path in fixed_specs]
                nets, snaps = build_opponents(pool, learner, elo,
                                              stage.cfg.n_players, device,
                                              fixed_items=fixed_items,
                                              bot_items=bot_items,
                                              fixed_elo=fixed_elo,
                                              warmup=global_step < args.warmup_steps,
                                              fixed_prob=args.fixed_opp_prob,
                                              bot_prob=args.bot_opp_prob)
                apply_opp_boost(sim, nets, args.opp_boost)
                runner = SelfPlayRunner(sim, learner, nets, pcfg,
                                        stage.opponent_handicap)

            if time.time() - start > args.time_budget:
                print("[budget] 时间预算用尽，存盘退出（下个会话 --resume 接力）")
                break
    except KeyboardInterrupt:
        print("\n[interrupt] 存盘后退出")
    except RuntimeError as exc:
        # 兜底：预估算得再准也有意外（比如别的大进程占了内存）。
        # 至少存盘退出，绝不静默崩掉丢进度。
        print(f"\n[error] RuntimeError: {exc}")
        traceback.print_exc()          # 完整栈，方便定位（如本机 torch 索引 bug）
        print("[error] 已尝试存盘。若峰值内存再超，请调大 --minibatches 或调小 --num-envs")
    finally:
        save_ckpt(args.ckpt, learner=learner, opt=opt, pool=pool, cstate=cstate,
                  global_step=global_step, elo=elo, args=args, fixed_elo=fixed_elo)
        if log_f:
            log_f.close()
        print(f"[done] step={global_step} ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
