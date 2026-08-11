"""PPO：rollout 采集 + GAE + 带掩码的裁剪更新。

几个和"多智能体自我博弈"绑定的设计决定：

- **只把 player 0 的轨迹放进 buffer**。对手是冻结的历史快照，在 `no_grad`
  下前向，它们的经验对 learner 没有意义（策略不同分布）。这样 buffer 里
  的数据严格是 on-policy 的。
- **观测是 env 级共享的一份张量**，所有角色前向时读同一块内存，只是把
  `pid` 传给网络（视角 = 第一层权重的通道置换）。buffer 里因此也只存一份，
  而且按 `cfg.obs_fp16` 存 fp16 —— rollout buffer 的显存直接减半。
- **不需要为 done 手工切轨迹**。环境 auto_reset 之后 `next_obs` 已经是新
  局首帧，GAE 里靠 `1 - done` 掐断即可。
- **掩码作用在 logits 上**，采样和重算 log_prob 用的是同一份掩码，
  所以 ratio 是自洽的；掩码必须一起存进 buffer。
- **动作是因子化的 (move, bomb)**，buffer 里存成 (T, N, 2)，两份掩码分开存。
  联合 log_prob 是两个头之和，所以 ratio 仍然是单个标量，PPO 公式一字不改。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from sim.move import center_cell  # noqa: E402

from sim.config import N_BOMB, N_MOVES

from .model import ActorCritic


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.995        # 一局最长 900 tick，折扣要够长才看得到终局奖励
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    entropy_final: float = 0.03
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    oversample_dying: int = 3   # 濒死/死亡帧过采样倍率（1 = 关闭）


class RolloutBuffer:
    """预分配的定长 buffer；T×N 条 player-0 转移。"""

    def __init__(self, steps: int, num_envs: int, obs_shape, device,
                 obs_dtype=torch.float32) -> None:
        c, h, w = obs_shape
        # 观测是 env 级共享的一份，不乘 P；dtype 跟着模拟器（默认 fp16）
        self.obs = torch.zeros((steps, num_envs, c, h, w), dtype=obs_dtype,
                               device=device)
        self.mmask = torch.zeros((steps, num_envs, N_MOVES), dtype=torch.bool, device=device)
        self.bmask = torch.zeros((steps, num_envs, N_BOMB), dtype=torch.bool, device=device)
        self.act = torch.zeros((steps, num_envs, 2), dtype=torch.long, device=device)
        self.logp = torch.zeros((steps, num_envs), device=device)
        self.val = torch.zeros((steps, num_envs), device=device)
        self.rew = torch.zeros((steps, num_envs), device=device)
        self.done = torch.zeros((steps, num_envs), device=device)
        self.dmg = torch.zeros((steps, num_envs), dtype=torch.bool, device=device)
        self.steps = steps
        self.ptr = 0

    def add(self, obs, mmask, bmask, act, logp, val, rew, done, dmg) -> None:
        i = self.ptr
        self.obs[i], self.mmask[i], self.bmask[i] = obs, mmask, bmask
        self.act[i] = act
        self.logp[i], self.val[i], self.rew[i], self.done[i] = logp, val, rew, done
        self.dmg[i] = dmg
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0


def compute_gae(buf: RolloutBuffer, last_val: torch.Tensor, gamma: float, lam: float):
    adv = torch.zeros_like(buf.rew)
    gae = torch.zeros_like(last_val)
    for t in reversed(range(buf.steps)):
        nonterminal = 1.0 - buf.done[t]
        next_val = last_val if t == buf.steps - 1 else buf.val[t + 1]
        delta = buf.rew[t] + gamma * next_val * nonterminal - buf.val[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + buf.val


class SelfPlayRunner:
    """把 sim + learner + 冻结对手串起来，产出一个装满的 buffer。"""

    def __init__(self, sim, learner: ActorCritic, opponents: list, cfg: PPOConfig,
                 handicap: float = 1.0) -> None:
        self.sim = sim
        self.learner = learner
        self.opponents = opponents          # 长度 P-1 的冻结网络列表
        self.cfg = cfg
        self.handicap = handicap
        self.device = next(learner.parameters()).device
        n_players = sim.cfg.n_players
        assert len(opponents) == n_players - 1, "对手数量必须是 P-1"
        self.buf = RolloutBuffer(
            cfg.rollout_steps, sim.num_envs, sim.cfg.obs_shape, self.device,
            obs_dtype=torch.float16 if sim.cfg.obs_fp16 else torch.float32)
        # 统计量：胜/平/负计数和平均局长，用于课程晋级与日志。
        # kills = 敌方死亡数（我方赢的终局数），探索退火的 x 数据源（见 _tally）。
        # 健康度统计（collect 内累积，指导是否停下找问题）：
        #   suicide      自爆次数（死亡 tick 自己名下有在场泡）—— 98% 自爆老大难
        #   bombs        放炮次数
        #   danger_ticks 站危险区 tick 数 / danger_sum 危险值累计 → 站危占比
        self.ep_stats = {"win": 0, "draw": 0, "loss": 0, "len_sum": 0, "count": 0,
                         "kills": 0, "suicide": 0, "bombs": 0,
                         "danger_ticks": 0, "danger_sum": 0.0}
        self._ep_len = torch.zeros((sim.num_envs,), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _opponent_actions(self, obs, mmask, bmask, actions) -> None:
        for k, net in enumerate(self.opponents):
            pid = k + 1
            if getattr(net, "is_bot", False):
                # 课程 bot：规则策略直连，不削弱（它按自己的逻辑打，不是神经网络）
                actions[:, pid] = net.act(obs, mmask[:, pid], bmask[:, pid], pid)
                continue
            # obs 是共享的那一份；视角靠 pid 传给网络，不切片、不拷贝
            a, _, _ = net.act(obs, mmask[:, pid], bmask[:, pid], pid)
            if self.handicap < 1.0:
                # 削弱对手 = 按概率吞掉它的放泡键，移动完全不动。
                # 因子化动作空间的一个附带好处：这里直接把 bomb 位清零就行，
                # 不用像扁平动作空间那样"从非放泡动作里重采"，也就没有
                # 全掩码出 NaN 的隐患。
                keep = torch.rand_like(a[:, 1], dtype=torch.float) <= self.handicap
                a = torch.stack([a[:, 0], a[:, 1] * keep.long()], dim=-1)
            actions[:, pid] = a

    def collect(self) -> tuple[RolloutBuffer, torch.Tensor]:
        sim, cfg = self.sim, self.cfg
        self.buf.reset()
        n = sim.num_envs
        dev = self.device
        n_players = sim.cfg.n_players              # sim 的 SimConfig（cfg 是 PPOConfig）
        # 健康度统计缓冲（每 tick 向量累积，无 host 同步；collect 末尾一次性 sum）
        dng_ch = 2 * n_players + 1                 # 危险通道下标（与 obs 布局一致）
        st_sui = torch.zeros(n, dtype=torch.long, device=dev)
        st_bmb = torch.zeros(n, dtype=torch.long, device=dev)
        st_dt = torch.zeros(n, dtype=torch.long, device=dev)
        st_dv = torch.zeros(n, dtype=torch.float32, device=dev)
        for i in range(cfg.rollout_steps):
            obs = sim.observe()                      # (N, C, H, W) 共享
            mmask, bmask = sim.legal_mask()
            actions = torch.zeros((n, sim.cfg.n_players, 2),
                                  dtype=torch.long, device=obs.device)
            with torch.no_grad():
                a0, logp, value = self.learner.act(obs, mmask[:, 0], bmask[:, 0], 0)
            actions[:, 0] = a0
            self._opponent_actions(obs, mmask, bmask, actions)

            owner_snap = sim.owner.clone()           # 自杀判定：死前自己名下泡数
            fuse_snap = sim.fuse.clone()             # fuse 清场前快照（死后 owner 已置 -1）
            hp0 = sim.hp[:, 0].clone()               # 濒死/死亡帧过采样要用的掉血标志
            reward, done, info = sim.step(actions)
            dmg = (hp0 - sim.hp[:, 0]).clamp(min=0) > 0
            self._ep_len += 1
            self._tally(info, done)
            # 掉血回收（延迟 flush，2026-08-10）：step 内只累积（零同步），
            # 这里降频 flush —— 每 4 tick 一次 + collect 末尾补一次。回收箱
            # 延迟 ≤4 tick 出（掉血补偿资源，拾取时机影响可忽略）；省 3/4 的
            # bool(lost.sum()>0) 同步（每 tick flush 在 collect 里 ~70ms/tick）。
            if i % 4 == 3:
                sim.flush_recycle()
            self.buf.add(obs, mmask[:, 0], bmask[:, 0], a0, logp, value,
                         reward[:, 0], done.float(), dmg)
            # ---- 健康度统计（向量累积，零 host 同步）----
            died0 = info["died"][:, 0]
            own_live = ((owner_snap == 0) & (fuse_snap > 0)).flatten(1).sum(dim=1)
            st_sui += (died0 & (own_live > 0)).long()
            st_bmb += (a0[:, 1] == 1).long()
            # 危险站桩：用**模型决策时**的 obs 危险通道 + 脚下位置（step 前）
            cell = center_cell(sim.pos)
            flat = (cell[:, 0, 0].long() * sim.cfg.width
                    + cell[:, 0, 1].long())
            foot = obs[:, dng_ch].float().flatten(1).gather(1, flat.unsqueeze(1)).squeeze(1)
            st_dt += (foot > 0.04).long()
            st_dv += foot

        self.ep_stats["suicide"] += int(st_sui.sum())
        self.ep_stats["bombs"] += int(st_bmb.sum())
        self.ep_stats["danger_ticks"] += int(st_dt.sum())
        self.ep_stats["danger_sum"] += float(st_dv.sum())
        sim.flush_recycle()    # 末尾补一次：清掉最后 ≤4 tick 累积的回收
        with torch.no_grad():
            *_, last_val = self.learner(sim.observe(), 0)
        return self.buf, last_val

    def _tally(self, info: dict, done: torch.Tensor) -> None:
        """胜负从 step 返回的 info['winner'] 判断，不依赖 reward 阈值。

        默认 win_bonus=0 后，终局 tick 的 reward 可能≈0（掉血/打中净零和），
        靠阈值分不出胜负；info['winner'] 是 reset 前算好的真值。
        """
        if not bool(done.any()):
            return
        winner = info["winner"]                     # (N, P) bool
        win = done & winner[:, 0]                   # player 0 赢
        lose = done & winner[:, 1]                  # 对方赢 = 我输（1v1）
        draw = done & ~win & ~lose
        self.ep_stats["win"] += int(win.sum())
        self.ep_stats["loss"] += int(lose.sum())
        self.ep_stats["draw"] += int(draw.sum())
        finished = done.nonzero(as_tuple=True)[0]
        self.ep_stats["count"] += finished.numel()
        self.ep_stats["len_sum"] += int(self._ep_len[finished].sum())
        self._ep_len[finished] = 0
        # 击杀统计（探索退火的 x 数据源）：敌方（pid≥1）死亡的终局数。
        # 1v1 敌方死亡 = 我方赢的终局（win）。按迭代归一成"平均每局击杀"，
        # 论文 x ∈ [0,2]（2v2 最多杀 2 个）；1v1 上限 1，k 相应取 1.2 曲线仍平滑。
        # 双亡平局/超时不算击杀（win/lose 都不置位）。
        self.ep_stats["kills"] += int(win.sum())

    def mean_ep_len(self) -> float:
        return self.ep_stats["len_sum"] / max(1, self.ep_stats["count"])

    def win_rate(self) -> float:
        total = self.ep_stats["win"] + self.ep_stats["draw"] + self.ep_stats["loss"]
        if total == 0:
            return 0.0
        return (self.ep_stats["win"] + 0.5 * self.ep_stats["draw"]) / total

    def kills_per_ep(self) -> float:
        """平均每局击杀（敌方死亡数）—— 探索退火 x 的归一化值（1v1 上限 1）。"""
        return self.ep_stats["kills"] / max(1, self.ep_stats["count"])

    def clear_stats(self) -> None:
        self.ep_stats = {"win": 0, "draw": 0, "loss": 0, "len_sum": 0, "count": 0,
                         "kills": 0, "suicide": 0, "bombs": 0,
                         "danger_ticks": 0, "danger_sum": 0.0}


def ppo_update(learner: ActorCritic, opt: torch.optim.Optimizer, buf: RolloutBuffer,
               last_val: torch.Tensor, cfg: PPOConfig, entropy_coef: float,
               autocast: bool = False) -> dict:
    adv, ret = compute_gae(buf, last_val, cfg.gamma, cfg.gae_lambda)
    flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
    obs = flat(buf.obs)
    mmask, bmask = flat(buf.mmask), flat(buf.bmask)
    act, old_logp = flat(buf.act), flat(buf.logp)
    adv, ret, old_val = flat(adv), flat(ret), flat(buf.val)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total = obs.shape[0]
    # 濒死/死亡帧过采样：掉血 tick 与终局 tick 在一局里极稀疏（1800 步里通常
    # 只有几个），价值函数在这些状态上几乎没见过样本 → 死亡附近价值估计差、
    # 梯度不稳。把这类帧按 oversample_dying 倍复制进采样池，每 epoch 仍只抽
    # total 个（训练量不变），稀有帧被抽中的概率提高。
    over = getattr(cfg, "oversample_dying", 1)
    base = torch.arange(total, device=obs.device)
    if over > 1:
        rare = flat(buf.dmg) | flat(buf.done).bool()
        if bool(rare.any()):
            rare_idx = rare.nonzero(as_tuple=False).squeeze(1)
            pool = torch.cat([base, rare_idx.repeat(over - 1)])
        else:
            pool = base
    else:
        pool = base

    mb = max(1, total // cfg.minibatches)
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
    n_updates = 0
    # autocast（fp16 训练）：把评估前向+反向包进 autocast，让 GEMM 走 fp16。
    # PPO 更新占端到端 ~68%，是比模拟更大的提速杠杆。loss/ratio 等在
    # autocast 外算（数值稳定），只把网络前向放进 autocast 域。
    ac_ctx = (torch.autocast(device_type=obs.device.type)
              if autocast and obs.device.type != "cpu" else nullcontext())
    for _ in range(cfg.epochs):
        # 从（含过采样副本的）池子里无放回抽 total 个 —— 每 epoch 训练量恒定
        perm = pool[torch.randperm(pool.numel(), device=obs.device)][:total]
        for start in range(0, total, mb):
            sel = perm[start:start + mb]
            with ac_ctx:
                logp, entropy, value = learner.evaluate(
                    obs[sel], mmask[sel], bmask[sel], act[sel])
            ratio = (logp - old_logp[sel]).exp()
            pg = -torch.min(
                ratio * adv[sel],
                ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[sel],
            ).mean()
            # value clipping：RL 的回报尺度在自我博弈里会漂移，限幅更稳
            v_clipped = old_val[sel] + (value - old_val[sel]).clamp(
                -cfg.clip_eps, cfg.clip_eps)
            vf = 0.5 * torch.max((value - ret[sel]) ** 2,
                                 (v_clipped - ret[sel]) ** 2).mean()
            ent = entropy.mean()
            loss = pg + cfg.value_coef * vf - entropy_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                stats["pg"] += float(pg)
                stats["vf"] += float(vf)
                stats["ent"] += float(ent)
                stats["kl"] += float((old_logp[sel] - logp).mean())
                stats["clipfrac"] += float(
                    ((ratio - 1).abs() > cfg.clip_eps).float().mean())
            n_updates += 1
    return {k: v / max(1, n_updates) for k, v in stats.items()}
