"""课程敌人：全张量化的规则策略，作为 PPO 训练的启蒙/陪练对手。

不需要寻路（A*/Dijkstra 在 5632 并行的向量化 sim 里没法写）：候选方向
直接用 Chebyshev 距离 + 危险图打分，撞墙被掩码/墙张量硬过滤 —— 这是
"能追、能躲、会放泡"的最简可学老师，强度和 ELO 地板都够用。

接口对齐冻结网络（train/model.py::ActorCritic.act）：`.act(obs, mmask,
bmask, pid) -> (N, 2)`，SelfPlayRunner 里和网络对手无缝互换（靠
`is_bot = True` 标记区分）。全向量化，5632 env 无逐环境循环。

- random_attack：随机合法移动 + 低概率放泡 —— ELO 地板。
- greedy_attack：朝最近**存活**对手逼近（避雷、不自杀）+ 贴身放泡。
- astar_attack：危险度融合的多源 Dijkstra 价值函数（逃生场 + 逼近场），
  能绕危险区、放完能撤、会连锁老泡 —— 课程强基准（参考 RPG 版 BFS 寻路
  + 泡线避让改造成的代价场版本）。
"""

from __future__ import annotations

from functools import partial

import torch

from sim.config import MOVE_IDLE, N_BOMB, N_MOVES, SimConfig

# (dy, dx)，与 config.DIRS 对齐：0上 1下 2左 3右
_D = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Dijkstra 收敛判定频率（模块级，便于 DCU 调优）：每 CHECK_EVERY 趟才查一次
# `torch.equal`（GPU→CPU 同步点）。实测 corridor 收敛 8-16 趟 → 4 粒度在
# DCU 最优（29.9 vs 8 的 33.3 ms/tick，sync 比 MPS 贵；169 不查最差——
# 空转 kernel 代价 > sync，见 prof_dijkstra_check）。
CHECK_EVERY = 4


def _sample_legal(mask: torch.Tensor) -> torch.Tensor:
    """从合法掩码 (N, K) 均匀采样，全张量、无 CPU 同步。

    cumsum 技巧：合法项贡献区间 [cs_{k-1}, cs_k)，均匀样本落入哪个区间
    就采哪个动作。全掩码时总合法数 0 → clamp 到 1e-6 → 返回 0（调用方
    兜底成 IDLE）。
    """
    n, k = mask.shape
    cs = mask.float().cumsum(dim=-1)
    u = torch.rand(n, 1, device=mask.device) * cs[:, -1:].clamp(min=1e-6)
    return (u > cs).sum(dim=-1).long()


def random_attack(sim, obs, mmask, bmask, pid: int, bomb_rate: float = 0.10):
    """随机合法移动 + bomb_rate 概率放泡（尊重 bmask）。"""
    n = sim.num_envs
    dev = sim.device
    move = _sample_legal(mmask)
    bomb = ((torch.rand(n, device=dev) < bomb_rate) & bmask[:, 1]).long()
    dead = ~sim.alive[:, pid]
    move = torch.where(dead, torch.full_like(move, MOVE_IDLE), move)
    bomb = torch.where(dead, torch.zeros_like(bomb), bomb)
    return torch.stack([move, bomb], dim=-1)


def idle_attack(sim, obs, mmask, bmask, pid: int):
    """完全静止的靶子：永远 MOVE_IDLE、不放泡（对照实验用）。

    用来隔离变量：如果"玩家 vs 训练模型"的自杀现象，在"玩家 vs 静止靶"
    下依然出现，说明与 AI 行为无关；反之则是训练模型的策略缺陷。
    """
    n = sim.num_envs
    dev = sim.device
    move = torch.full((n,), MOVE_IDLE, dtype=torch.long, device=dev)
    place = torch.zeros(n, dtype=torch.long, device=dev)
    return torch.stack([move, place], dim=-1)


def _target_dist(sim, pid: int) -> torch.Tensor:
    """每个 env 中自己到**最近存活对手**的 Chebyshev 距离（格）。

    对手全死 → 1e9（追无可追，交给危险规避兜底 → 游荡）。
    """
    n, p = sim.num_envs, sim.cfg.n_players
    h, w = sim.cfg.height, sim.cfg.width
    r = sim.pos[:, pid, 0].floor().long().clamp(0, h - 1)
    c = sim.pos[:, pid, 1].floor().long().clamp(0, w - 1)
    dist = torch.full((n,), 1e9, dtype=torch.float32, device=sim.device)
    for o in range(p):
        if o == pid:
            continue
        ro = sim.pos[:, o, 0].floor().long().clamp(0, h - 1)
        co = sim.pos[:, o, 1].floor().long().clamp(0, w - 1)
        d = (r - ro).abs().maximum((c - co).abs()).float()
        dist = torch.minimum(dist, torch.where(sim.alive[:, o], d,
                                               torch.full_like(d, 1e9)))
    return dist, r, c


def greedy_attack(sim, obs, mmask, bmask, pid: int):
    """朝最近存活对手逼近：安全格按 (距离 + 2×danger) 取最小，危险格硬过滤。

    逼近打分（每个合法方向的目标格）：
        score = Chebyshev 距离 + 2×danger + 0.05×噪声
    - 硬过滤 danger ≥ 0.35（快爆的泡不钻）；全部被过滤 → 取最低 danger 的
      合法格（逃生态，宁可走火也不原地等死）。
    - 放泡：bmask 允许 且 对手在 blast_cap 内 且 自己当前格 danger < 0.2
      （贴身进攻，不自爆）。
    """
    cfg: SimConfig = sim.cfg
    n, p = sim.num_envs, cfg.n_players
    h, w = cfg.height, cfg.width
    dev = sim.device
    danger = obs.float()[:, 2 * p + 1]                 # (N,H,W)
    blocked = (sim.wall | sim.brick)                   # (N,H,W) 不可通行
    r = sim.pos[:, pid, 0].floor().long().clamp(0, h - 1)
    c = sim.pos[:, pid, 1].floor().long().clamp(0, w - 1)
    own_dng = danger.flatten(1).gather(
        1, (r * w + c).unsqueeze(1)).squeeze(1)

    # 候选 = 4 方向 + IDLE（5 个目标格）。IDLE 的目标格 = 当前格。
    cand_rows = torch.zeros(n, 5, dtype=torch.long, device=dev)
    cand_cols = torch.zeros(n, 5, dtype=torch.long, device=dev)
    cand_rows[:, 0] = (r - 1).clamp(0, h - 1)
    cand_rows[:, 1] = (r + 1).clamp(0, h - 1)
    cand_rows[:, 2] = r
    cand_rows[:, 3] = r
    cand_rows[:, 4] = r
    cand_cols[:, 0] = c
    cand_cols[:, 1] = c
    cand_cols[:, 2] = (c - 1).clamp(0, w - 1)
    cand_cols[:, 3] = (c + 1).clamp(0, w - 1)
    cand_cols[:, 4] = c
    cflat = cand_rows * w + cand_cols                   # (N,5)
    dng_c = danger.flatten(1).gather(1, cflat)          # (N,5)
    blk_c = blocked.flatten(1).gather(1, cflat)         # (N,5)
    # 距离：在每个候选格上直接算到最近存活对手的 Chebyshev（不是当前格+增量）
    dist_c = torch.full((n, 5), 1e9, dtype=torch.float32, device=dev)
    for o in range(p):
        if o == pid:
            continue
        ro = sim.pos[:, o, 0].floor().long().clamp(0, h - 1)
        co = sim.pos[:, o, 1].floor().long().clamp(0, w - 1)
        d = (cand_rows - ro.unsqueeze(1)).abs().maximum(
            (cand_cols - co.unsqueeze(1)).abs()).float()
        d = torch.where(sim.alive[:, o].unsqueeze(1), d,
                        torch.full_like(d, 1e9))
        dist_c = torch.minimum(dist_c, d)
    dist = dist_c[:, MOVE_IDLE]                         # 当前格距离（放泡判断用）

    best = torch.zeros(n, dtype=torch.long, device=dev)     # 兜底 MOVE_UP
    best_scr = torch.full((n,), float("inf"), device=dev)
    escape_scr = torch.full((n,), float("inf"), device=dev)
    escape_dir = torch.zeros(n, dtype=torch.long, device=dev)
    noise = 0.05 * torch.rand(n, 5, device=dev)

    for d in range(N_MOVES):
        legal = mmask[:, d] & ~blk_c[:, d]
        score = dist_c[:, d] + 2.0 * dng_c[:, d] + noise[:, d]
        score = torch.where(legal, score, torch.full_like(score, float("inf")))
        better = score < best_scr
        best_scr = torch.where(better, score, best_scr)
        best = torch.where(better, torch.full_like(best, d), best)
        # 逃生兜底：只看合法 + 最低 danger（不管距离）
        esc = torch.where(legal, dng_c[:, d], torch.full_like(dng_c[:, d], float("inf")))
        e_better = esc < escape_scr
        escape_scr = torch.where(e_better, esc, escape_scr)
        escape_dir = torch.where(e_better, torch.full_like(escape_dir, d), escape_dir)

    # 硬过滤：所有安全格（danger<0.35）里取最优；一个都没有 → 逃生格
    safe = best_scr < float("inf")
    use_safe = safe & (dng_c.gather(1, best.unsqueeze(1)).squeeze(1) < 0.35)
    move = torch.where(use_safe, best, escape_dir)

    can = bmask[:, 1].bool()
    opp_near = dist < sim.blast_cap[:, pid].float()
    place = (can & opp_near & (own_dng < 0.2)).long()
    dead = ~sim.alive[:, pid]
    move = torch.where(dead, torch.full_like(move, MOVE_IDLE), move)
    place = torch.where(dead, torch.zeros_like(place), place)
    return torch.stack([move, place], dim=-1)


def _dijkstra_fields(sim, sources, danger, lam: float = 2.0,
                     max_passes: int | None = None,
                     blocked: torch.Tensor | None = None) -> torch.Tensor:
    """S 场并行的多源 Dijkstra 价值场（Bellman-Ford 全张量化）。

    sources (N, S, H, W) bool → 返回 (N, S, H*W)。每场独立传播，与
    `_dijkstra_field` 逐位一致（同一数学，只是 batch 维从 (N,V) 变 (N,S,V)）。

    **2026-08-10 优化：4 方向邻居 gather 合并** —— 每趟从 4 次独立
    (gather+where+minimum) 合成 1 次批量 gather (N,S,4V) + 一次 min，kernel
    数 ÷2.4（DCU launch-bound 的直接收益）。合并前每趟 12 kernel，合并后 ~5。
    """
    n, s, h, w = sources.shape
    v = h * w
    dev = sim.device
    rc = torch.arange(v, device=dev)
    rr = rc // w
    cc = rc % w
    # 4 方向邻居索引 + 越界掩码一次构建（(4, V)）
    neigh_idx = []
    neigh_oob = []
    for dr, dc in _D:
        nr = rr + dr
        nc = cc + dc
        oob = (nr < 0) | (nr >= h) | (nc < 0) | (nc >= w)
        flat = (nr.clamp(0, h - 1) * w + nc.clamp(0, w - 1))
        neigh_idx.append(flat)
        neigh_oob.append(oob)
    idx_all = torch.stack(neigh_idx, dim=0)          # (4, V)
    oob_all = torch.stack(neigh_oob, dim=0)          # (4, V)
    if blocked is None:
        blocked = sim.wall | sim.brick
    b = blocked.reshape(n, v).float()
    cost = 1.0 + lam * danger.reshape(n, v)
    cost = torch.where(b > 0, torch.full_like(cost, float("inf")), cost)
    dist = torch.full((n, s, v), float("inf"), device=dev)
    dist = torch.where(sources.reshape(n, s, v), torch.zeros_like(dist), dist)
    passes = max_passes or v
    # 邻居索引展开 (1,1,4V) / 越界 (1,1,4,V)，expand 后循环内零重构
    idx_flat = idx_all.reshape(1, 1, -1).expand(n, s, -1)
    oob_3d = oob_all.reshape(1, 1, 4, v).expand(n, s, 4, v)
    for i in range(passes):
        old = dist
        nd = dist.gather(2, idx_flat)                # (N,S,4V)
        nd = nd.reshape(n, s, 4, v)
        nd = torch.where(oob_3d,
                         torch.full_like(nd, float("inf")), nd)
        nd = nd.min(dim=2).values                    # (N,S,V) 4 方向最小
        dist = torch.minimum(dist, nd + cost.unsqueeze(1))
        if i % CHECK_EVERY == CHECK_EVERY - 1 and bool(torch.equal(dist, old)):
            break
    return dist


def _dijkstra_field(sim, sources, danger, lam: float = 2.0,
                    max_passes: int | None = None,
                    blocked: torch.Tensor | None = None) -> torch.Tensor:
    """单场多源 Dijkstra（兼容入口，包装 _dijkstra_fields 取第 0 场）。"""
    return _dijkstra_fields(sim, sources.unsqueeze(1), danger, lam,
                            max_passes, blocked)[:, 0]


def _mode_ticker(sim):
    """astar 行为模式随机切换（全张量，按 env 独立）：0=aggressive 逼近 / 1=flee 远离。

    每 env 一个随机倒计时（60~240 tick），归零时随机换模式 —— 对手"一段时间
    接近、一段时间远离"，课程里教模型应对**多变的接近/远离分布**（治"对不接近
    的对手就乱来"：跟接近型打久了会遗忘撤退，见 build_opponents 的 bot 退场）。
    """
    n = sim.num_envs
    dev = sim.device
    if not hasattr(sim, "_bmode"):
        sim._bmode = torch.randint(0, 2, (n,), device=dev)
        sim._btimer = torch.randint(60, 240, (n,), device=dev)
    # 计数用 getattr（老对象/新 sim 可能没有）。**只在 16 倍 tick 才查**
    # bool(any())（GPU→CPU 同步点）—— 注意不能写成 `%16==0 or bool(...)`，
    # Python 短路会让右侧在非 16 倍 tick 也被求值（每 tick 同步，v1 的 bug）。
    sim._bmode_tick = getattr(sim, "_bmode_tick", 0) + 1
    sim._btimer -= 1
    if sim._bmode_tick % 16 == 0 and bool((sim._btimer <= 0).any()):
        switch = sim._btimer <= 0
        new = torch.randint(0, 2, (n,), device=dev)
        sim._bmode = torch.where(switch, new, sim._bmode)
        sim._btimer = torch.where(
            switch, torch.randint(60, 240, (n,), device=dev), sim._btimer)
    return sim._bmode


def astar_attack(sim, obs, mmask, bmask, pid: int,
                 eat_crates: bool = False):
    """危险度融合的寻路 bot（A*/Dijkstra 价值函数版）——课程强基准。

    价值函数 = 多源 Dijkstra 最短代价场（见 _dijkstra_field）：
      V_safe(c) = 到最近安全格（danger<0.35）的代价  → 逃生
      V_opp(c)  = 到最近存活对手的代价               → 逼近
    决策（每 tick，沿价值函数最速下降）：
      - 自己脚下危险 → 沿 V_safe 下降逃生（平局用 V_opp 破，边逃边找对手）；
      - 自己安全     → 沿 V_opp 下降逼近，硬过滤 danger≥0.5 的格
        （快爆的泡不钻）；全被过滤 → 转逃生；
      - 放泡：合法 且（对手在 blast_cap 十字内，或附近有 fuse≤10 的老泡
        可"时间差连锁"—— 老泡先爆把我这颗拉爆）且 自己脚下 danger<0.2
        且 放完有安全邻格可撤（不自爆）。

    **行为模式**（sim._bmode，BotWrapper 的 mode_fn 随机切换）：
      - aggressive（0）：上述逼近 + 放泡（默认）；
      - flee（1）：沿 -V_opp 远离对手（朝离对手最远的格）+ 只避险不放泡
        —— 教模型应对"不接近/躲闪型"对手，防"跟接近型打久了就遗忘撤退"。

    比贪心一步的 greedy 强一个量级：能绕开危险区、能撤、能连锁老泡，
    是"结合危险图的寻路 AI"的稳定基准（参考 RPG 版 BFS 寻路 + 泡线避让
    的思路，改造成危险度代价）。
    """
    cfg: SimConfig = sim.cfg
    n, p = sim.num_envs, cfg.n_players
    h, w = cfg.height, cfg.width
    dev = sim.device
    danger = obs.float()[:, 2 * p + 1]                 # (N,H,W)
    r = sim.pos[:, pid, 0].floor().long().clamp(0, h - 1)
    c = sim.pos[:, pid, 1].floor().long().clamp(0, w - 1)
    own = r * w + c
    own_dng = danger.flatten(1).gather(1, own.unsqueeze(1)).squeeze(1)

    # --- 确定威胁（自己名下在场泡的爆炸范围）只做"绕行成本"，不动危险通道 ---
    # 危险通道保留时间信息（danger=1-(fuse-1)/fuse_max，新泡只有 0.03），
    # V_safe 的"安全格"定义不能被威胁图干掉（否则逃生场无源、原地等爆）。
    # 自己的泡引信确定 → 它的 blast 是必爆区，作为 +2 的绕行惩罚叠加进
    # 逼近/逃生打分：放完泡会绕开自己的炮火，追对手也不钻进去。
    threat = (sim.owner == pid) & (sim.fuse > 0)       # (N,H,W) bool
    # 膨胀轮数用静态上限（blast_cap ≤ growth_blast_max，每 tick 读
    # `max().item()` 是 GPU→CPU 同步点，砍掉）；batch-max 本就统一膨胀。
    for _ in range(int(cfg.growth_blast_max)):
        nb = threat.clone()
        nb[:, 1:, :] |= threat[:, :-1, :]
        nb[:, :-1, :] |= threat[:, 1:, :]
        nb[:, :, 1:] |= threat[:, :, :-1]
        nb[:, :, :-1] |= threat[:, :, 1:]
        threat = nb

    # --- 两个价值场（每 tick 重算；对手会动、危险图会变）---
    # 障碍 = 墙 | 砖 | 所有在场泡泡 —— 与 legal_mask 的 blocked 一致：
    # 规划穿泡 = 走出永远走不通的路线（旧版不挡泡，泡阵困住自己时 V_safe
    # 算出"穿泡逃生"，esc 打分里 IDLE 反而最小 → 站火海不动，修复点 1）。
    # **单场 ×2（勿用 _dijkstra_fields 合并）**：批量 (N,2,V) 的 3D gather
    # 在 MPS 上比两次 2D gather 慢 2 倍（实测 34→71ms）；DCU 上 launch-bound
    # 可能相反，若在 DCU 验证更快可切回（_dijkstra_fields 已保留）。
    block_all = sim.wall | sim.brick | (sim.fuse > 0)
    V_safe = _dijkstra_field(sim, danger < 0.35, danger, blocked=block_all)
    opp_src = torch.zeros(n, h, w, dtype=torch.bool, device=dev)
    for o in range(p):
        if o == pid:
            continue
        ro = sim.pos[:, o, 0].floor().long().clamp(0, h - 1)
        co = sim.pos[:, o, 1].floor().long().clamp(0, w - 1)
        opp_src.view(n, -1)[torch.arange(n, device=dev), ro * w + co] = sim.alive[:, o]
    V_opp = _dijkstra_field(sim, opp_src, danger, blocked=block_all)

    # --- 5 个候选动作的目标格（4 方向 + IDLE）---
    cand_rows = torch.zeros(n, 5, dtype=torch.long, device=dev)
    cand_cols = torch.zeros(n, 5, dtype=torch.long, device=dev)
    cand_rows[:, 0] = (r - 1).clamp(0, h - 1)
    cand_rows[:, 1] = (r + 1).clamp(0, h - 1)
    cand_rows[:, 4] = r
    cand_cols[:, 2] = (c - 1).clamp(0, w - 1)
    cand_cols[:, 3] = (c + 1).clamp(0, w - 1)
    cand_cols[:, 4] = c
    cand_rows[:, 2] = r
    cand_rows[:, 3] = r
    cand_cols[:, 0] = c
    cand_cols[:, 1] = c
    cflat = cand_rows * w + cand_cols
    Vopp_c = V_opp.gather(1, cflat)                          # (N,5)
    Vsafe_c = V_safe.gather(1, cflat)
    dng_c = danger.flatten(1).gather(1, cflat)
    thr_c = threat.flatten(1).gather(1, cflat).float()
    blk_c = (sim.wall | sim.brick).flatten(1).gather(1, cflat)
    legal = mmask & ~blk_c.bool()
    noise = 0.05 * torch.rand(n, 5, device=dev)

    # --- 逼近/逃生：沿价值函数最速下降，确定威胁区叠 +2 绕行惩罚 ---
    app = Vopp_c + 2.0 * dng_c + 2.0 * thr_c + noise
    app = torch.where(dng_c >= 0.5, torch.full_like(app, float("inf")), app)
    app = torch.where(legal, app, torch.full_like(app, float("inf")))
    app_ok = app.min(dim=1).values.isfinite()
    use_esc = (own_dng >= 0.35) | ~app_ok      # 脚下危险 或 逼近无路 → 转逃生
    # 逃生打分**只用 Vsafe**（安全距离），**不加 Vopp**：对手被泡阵/墙封住时
    # V_opp 只在封锁区内有限，逃生方向的 Vopp=inf 会把 esc 打成全 inf →
    # argmin 走到 fallback → 站火海不动（真 bug：有出路却停原地）。
    # 逃生本质 = "尽快到安全区"，与对手位置无关。
    esc = Vsafe_c * 100.0 + 2.0 * thr_c + noise     # 逃生命安全优先，顺带绕开威胁
    # **停方向逃生禁止（修复点 4）**：自己脚下危险（马上爆）时，停 = 留在
    # 必受伤的格，永远不是逃生选项。旧版（含 +1.0 惩罚）不够：esc=Vsafe×100
    # 尺度下停的 Vsafe 常最小（已在安全格附近），+1.0 压不住 Vsafe 微差 →
    # 危险 0.49 涨到 0.59 连续站死。直接把停方向 esc 打成 inf（禁停），argmin
    # 必选非停方向；真被围死（非停全 inf）时 fallback 兜底允许停。
    esc = torch.where(use_esc.unsqueeze(1) & (torch.arange(5, device=dev) == 4),
                      torch.full_like(esc, float("inf")), esc)
    esc = torch.where(legal, esc, torch.full_like(esc, float("inf")))

    # --- 吃道具层（hunter 专属，高优先级）---
    # 成长属性（泡数/威力/速度）中 **≥2 项低于上限的 70%** 且场上有宝箱 →
    # 朝最近宝箱寻路（走路踩箱升级），优先级高于逼近、低于逃生（命要紧）。
    # 特别适合 corridor：成长全靠踩箱，属性不满时先补属性再打，不无脑冲。
    # **避险（不因吃箱变笨）**：吃箱路径对齐 app 的避险 —— 快爆泡（dng≥0.5）
    # 禁行（Vcrate×100 不能压过危险），且四周有更优非停方向时禁停（防原地
    # 发呆被炸）。危险接近时 use_esc 转逃生（已有，吃箱永不压过逃命）。
    eat_on = torch.zeros(n, dtype=torch.bool, device=dev)
    eat = None
    # 不再用 bool(sim.crate.any()) 提前跳过（GPU→CPU 同步点）—— 向量化
    # 处理：crate 全空时 has_crate 全 False → eat_on 全 False，行为不变，
    # V_crate 空源 8 趟即收敛（同步频率反而更低）。
    if eat_crates and hasattr(sim, "crate"):
        # 各属性上限（corridor 用成长上限；open 无成长字段 → 回退基础值，
        # 此时 fracs 恒 1.0 → hungry 恒 False，天然不触发）
        b_max = float(getattr(cfg, "growth_bombs_max", cfg.max_bombs))
        z_max = float(getattr(cfg, "growth_blast_max", cfg.blast))
        s_max = float(getattr(cfg, "growth_speed_max", 1.0))
        fracs = torch.stack([
            sim.bombs_cap[:, pid].float() / b_max,
            sim.blast_cap[:, pid].float() / z_max,
            sim.spd_g[:, pid].float() / s_max,
        ], dim=0)                                     # (3,n)
        hungry = (fracs < 0.7).sum(dim=0) >= 2        # ≥2 项不满 70%
        has_crate = sim.crate.any(dim=(1, 2))
        eat_on = hungry & has_crate & sim.alive[:, pid]
        # 朝最近宝箱的 Dijkstra 场（与 V_safe/V_opp 同款，泡挡路）
        V_crate = _dijkstra_field(sim, sim.crate, danger, blocked=block_all)
        Vcrate_c = V_crate.gather(1, cflat)           # (N,5)
        eat = Vcrate_c * 100.0 + 2.0 * dng_c + noise  # 吃箱优先，避险
        # 快爆泡禁行（对齐 app/flee）：不钻马上爆的格 —— 之前没有这层，
        # Vcrate×100 压过危险 → 直奔箱子走进爆炸区被炸死（用户实测变笨）。
        eat = torch.where(dng_c >= 0.5,
                          torch.full_like(eat, float("inf")), eat)
        eat = torch.where(legal, eat, torch.full_like(eat, float("inf")))
        # 禁停（防发呆）：四周有**更优非停方向**（更近箱或更安全）就不许停，
        # 停只在"停确实最优"（四周都更远更危险）时允许 —— 否则原地发呆被炸。
        eat_no_idle = torch.where(torch.arange(5, device=dev) < 4,
                                  eat, torch.full_like(eat, float("inf")))\
            .min(dim=1).values
        # 禁停判定：`eat_no_idle (N,) < eat[:, 4:5] (N,1)` 直接广播会变 (N,N)
        # （一维 N 对齐 (N,1) 的最后一维 1）→ expand_as 崩（2026-08-10 实测，
        # hunter 触发 eat 分支、n=5632 时报 expanded 5632≠5）。必须 unsqueeze
        # 成 (N,1) 再比：每行"非停最优" vs "停"。arange(5)==4 的 (5,) 与
        # (N,1) 广播 → (N,5)（5 对齐最后一维，即 IDLE 列），无需 expand_as。
        idle_better = (torch.arange(5, device=dev) == 4) \
            & (eat_no_idle.unsqueeze(1) < eat[:, 4:5])
        eat = torch.where(idle_better,
                          torch.full_like(eat, float("inf")), eat)
        # 吃箱路径必须可达（V_crate 全 inf = 被泡/墙封住到不了箱）→ 回退逼近，
        # 否则 eat_on 时 eat 全 inf → argmin 乱走（跟逃生 fallback 同理）。
        eat_on = eat_on & eat.min(dim=1).values.isfinite()

    score = torch.where(use_esc.unsqueeze(1), esc,
                        torch.where(eat_on.unsqueeze(1),
                                    eat if eat is not None else app, app))
    # 行为模式：flee（1）= 远离对手（-V_opp 最小化）+ 避险；aggressive（0）= 逼近。
    mode = getattr(sim, "_bmode", None)
    if mode is not None:
        # V_opp 被泡阵封住时是 inf → -inf 全等 → argmin 乱走；钳到 1e3 让
        # "远离度"退化到只比危险/威胁（反正都远离不了被封的对手）。
        vopp_f = Vopp_c.clamp(max=1e3)
        flee = -vopp_f + 2.0 * dng_c + 2.0 * thr_c + noise     # 朝最远格 + 避险
        flee = torch.where(dng_c >= 0.5, torch.full_like(flee, float("inf")), flee)
        flee = torch.where(legal, flee, torch.full_like(flee, float("inf")))
        flee = torch.where(use_esc.unsqueeze(1), esc, flee)    # 危险时也先逃
        score = torch.where((mode == 1).unsqueeze(1), flee, score)
    move = score.argmin(dim=1).long()
    # 终极兜底（修复点 2 + 修复点 3）：打分全 inf（被泡/火完全围死，V_safe 无路）时，
    # 选**最小 danger 的合法格**（跟 greedy 的逃生兜底一致）—— 宁可走火也
    # 不原地等死。旧版 argmin 在全 inf 时返回 0 = 向上，被掩码屏蔽后等于
    # 站着不动 → "站火海不动"的另一个来源。
    # **修复点 3（优先非停）**：全 inf 时从**非停**方向里选最小危险 —— 原地停
    # 是"等死"，只要有方向能降低/持平危险就优先走（停只在所有方向都更危险
    # 时才允许）。旧版"最小危险合法格"常选到停（原地危险恰好最低）→ 站火海。
    any_legal = legal.any(dim=1)
    no_path = ~score.isfinite().any(dim=1)
    dng_legal = torch.where(legal, dng_c, torch.full_like(dng_c, float("inf")))
    # 非停方向（idx 0-3）里最小危险；有 → 走它；没有 → 才允许停（idx 4）
    dng_move = torch.where(torch.arange(5, device=dev) < 4,
                           dng_legal, torch.full_like(dng_legal, float("inf")))
    fallback_move = dng_move.argmin(dim=1)
    fallback_idle = dng_legal.argmin(dim=1)          # 含停在全部候选兜底
    has_move_opt = dng_move.min(dim=1).values.isfinite()
    fallback = torch.where(has_move_opt, fallback_move, fallback_idle)
    move = torch.where(no_path & any_legal, fallback, move)

    # --- 放泡：十字内打得到对手 / 连锁老泡 / **近身压制**，且不自爆、能撤 ---
    cap = sim.blast_cap[:, pid].long()
    dr_o = torch.full((n,), 10**9, dtype=torch.long, device=dev)
    dc_o = torch.full((n,), 10**9, dtype=torch.long, device=dev)
    for o in range(p):
        if o == pid:
            continue
        ro = sim.pos[:, o, 0].floor().long().clamp(0, h - 1)
        co = sim.pos[:, o, 1].floor().long().clamp(0, w - 1)
        dr_o = torch.where(sim.alive[:, o], (r - ro).abs(), dr_o)
        dc_o = torch.where(sim.alive[:, o], (c - co).abs(), dc_o)
    aligned_opp = ((dr_o == 0) & (dc_o <= cap)) | ((dc_o == 0) & (dr_o <= cap))
    # 近身压制：曼哈顿距离 ≤ 威力+1（含斜向）就布泡封锁 —— 追到能威胁对手的
    # 距离就放，形成**多泡封锁网**（预算内持续布泡），不再"没对准十字线就永远
    # 不放"（旧版一局只放 1-3 个泡，太保守 → 对手弱 → 练出来的模型也弱）。
    manh = (dr_o + dc_o).clamp(max=10**8)
    near_opp = manh <= (cap + 1).clamp(min=1)
    # 连锁：附近（blast_cap 十字内）有引信 ≤10 的老泡 —— 老泡先爆把我拉爆
    chain = torch.zeros(n, dtype=torch.bool, device=dev)
    bidx = torch.arange(n, device=dev)
    for k in range(1, int(cap.max().item()) + 1):
        for dr, dc in _D:
            nr = (r + dr * k).clamp(0, h - 1)
            nc = (c + dc * k).clamp(0, w - 1)
            inb = ((r + dr * k) >= 0) & ((r + dr * k) < h) \
                & ((c + dc * k) >= 0) & ((c + dc * k) < w)
            f = sim.fuse[bidx, nr, nc]
            chain |= inb & (f > 0) & (f <= 10)
    can = bmask[:, 1].bool()
    can_escape = (legal & (dng_c < 0.35) & ~thr_c.bool()).any(dim=1)
    place = (can & (aligned_opp | near_opp | chain)
             & (own_dng < 0.2) & can_escape).long()
    if mode is not None:
        # flee 模式：不放"进攻泡"（aligned/near/chain 都是进攻导向），但保留
        # **撒雷阻追兵**—— 对手在十字射程内且自己脚下安全、放完能撤时丢一颗
        # 身后雷（封追兵路线），这是逃避型敌人难追的关键（经典"逃跑撒雷"）。
        # 安全门槛（own_dng<0.2 + can_escape）保证不是自杀式乱丢。
        flee_place = (can & (aligned_opp | near_opp)
                      & (own_dng < 0.2) & can_escape).long()
        place = torch.where((mode == 1), flee_place, place)

    dead = ~sim.alive[:, pid]
    move = torch.where(dead, torch.full_like(move, MOVE_IDLE), move)
    place = torch.where(dead, torch.zeros_like(place), place)
    return torch.stack([move, place], dim=-1)


class BotWrapper:
    """把规则策略包装成与冻结网络同接口的对手。

    `.act(obs, mmask_p, bmask_p, pid)` 里的掩码已按 pid 切好
    （(N,5)/(N,2)），与 `ActorCritic.act` 签名一致；`is_bot = True`
    让 SelfPlayRunner 跳过冻结网络的 no_grad/削削弱路径。
    `mode_fn`（可选）：每 tick 决策前调用的行为模式更新器（如 astar 的
    随机接近/远离，见 _mode_ticker）—— 策略内部经 sim 属性读模式。
    """

    is_bot = True

    def __init__(self, sim, fn, name: str, mode_fn=None) -> None:
        self.sim = sim
        self.fn = fn
        self.name = name
        self.mode_fn = mode_fn

    def act(self, obs, mmask, bmask, pid: int):
        if self.mode_fn is not None:
            self.mode_fn(self.sim)
        return self.fn(self.sim, obs, mmask, bmask, pid)

    def __repr__(self) -> str:
        return f"BotWrapper({self.name})"


def make_bot(sim, kind: str, mode: bool = True) -> BotWrapper:
    """按名字构造课程 bot。kind ∈ {"random", "greedy", "astar", "hunter"}。

    `mode=False`（测试用）：astar 不挂随机接近/远离（恒定 aggressive）——
    单元测试断言放泡/逼近行为时用，避免随机 flee 模式干扰；课程/对打默认开。

    `hunter` = **纯进攻寻路 AI**：恒 aggressive（逼近+放泡），不随机 flee。
    与 astar 的进攻/防守混合模式对应 —— 启动器可选「纯进攻」对手，
    练"打一个只会进攻不会逃的强敌"（astar flee 会躲，hunter 一直压上来）。
    """
    if kind == "random":
        return BotWrapper(sim, random_attack, "random")
    if kind == "greedy":
        return BotWrapper(sim, greedy_attack, "greedy")
    if kind == "idle":
        return BotWrapper(sim, idle_attack, "idle")   # 完全静止靶子（对照用）
    if kind == "astar":
        return BotWrapper(sim, astar_attack, "astar",
                          mode_fn=_mode_ticker if mode else None)
    if kind == "hunter":
        # 纯进攻 + **吃道具层**：恒 aggressive（mode=None = 不切 flee），
        # eat_crates=True → 成长属性 ≥2 项不满 70% 且有宝箱时，高优先级寻路
        # 吃箱补属性（特别适合 corridor：成长全靠踩箱）。
        return BotWrapper(sim, partial(astar_attack, eat_crates=True), "hunter")
    raise ValueError(f"未知 bot 类型: {kind}")
