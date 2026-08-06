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

import torch

from sim.config import MOVE_IDLE, N_BOMB, N_MOVES, SimConfig

# (dy, dx)，与 config.DIRS 对齐：0上 1下 2左 3右
_D = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _dijkstra_field(sim, sources, danger, lam: float = 2.0,
                    max_passes: int | None = None,
                    blocked: torch.Tensor | None = None) -> torch.Tensor:
    """多源 Dijkstra 价值场（Bellman-Ford 全张量化，无逐环境循环）。

    V(c) = 从 `sources` 走到格 c 的**最短代价**，进入格 c 的代价 =
    1 + lam×danger[c] —— 危险度直接融进路径代价，绕开危险区是"省代价"
    的自然结果（A* 在网格上 ≡ Dijkstra，Bellman-Ford 是它的向量化写法）。

    13×13=169 格 → 至多 V-1 趟收敛即精确；每趟 = 4 个邻居 gather+min。
    5632 env 就是 (5632,169) 的批运算，一个 for 循环都没有。
    `blocked`：不可通行掩码 (N,H,W) bool；None = wall|brick。调用方要传
    **含在场泡泡**的掩码（逃生/逼近都不该规划穿泡的路径 —— 泡在 legal_mask
    里挡移动，规划穿泡 = 走出永远走不通的路线）。返回 (N, H*W)。
    """
    n, h, w = danger.shape
    v = h * w
    dev = sim.device
    rc = torch.arange(v, device=dev)
    rr = rc // w
    cc = rc % w
    neigh = []
    for dr, dc in _D:
        nr = rr + dr
        nc = cc + dc
        oob = (nr < 0) | (nr >= h) | (nc < 0) | (nc >= w)
        flat = (nr.clamp(0, h - 1) * w + nc.clamp(0, w - 1))
        neigh.append((flat, oob))
    if blocked is None:
        blocked = sim.wall | sim.brick
    b = blocked.reshape(n, v).float()
    cost = 1.0 + lam * danger.reshape(n, v)
    cost = torch.where(b > 0, torch.full_like(cost, float("inf")), cost)
    dist = torch.full((n, v), float("inf"), device=dev)
    dist = torch.where(sources.reshape(n, v), torch.zeros_like(dist), dist)
    passes = max_passes or v
    for i in range(passes):
        old = dist
        for flat, oob in neigh:
            nd = dist.gather(1, flat.view(1, -1).expand(n, -1))
            nd = torch.where(oob.view(1, -1).expand(n, -1),
                             torch.full_like(nd, float("inf")), nd)
            dist = torch.minimum(dist, nd + cost)
        if i % 8 == 7 and bool(torch.equal(dist, old)):
            break                             # 收敛提前停（每 8 趟查一次，省 sync）
    return dist


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
    sim._btimer -= 1
    switch = sim._btimer <= 0
    if bool(switch.any()):
        new = torch.randint(0, 2, (n,), device=dev)
        sim._bmode = torch.where(switch, new, sim._bmode)
        sim._btimer = torch.where(
            switch, torch.randint(60, 240, (n,), device=dev), sim._btimer)
    return sim._bmode


def astar_attack(sim, obs, mmask, bmask, pid: int):
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
    for _ in range(int(sim.blast_cap[:, pid].max().item())):
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
    esc = Vsafe_c * 100.0 + Vopp_c + 2.0 * thr_c + noise     # 逃生命安全优先，顺带逼近
    esc = torch.where(legal, esc, torch.full_like(esc, float("inf")))

    app_ok = app.min(dim=1).values.isfinite()
    use_esc = (own_dng >= 0.35) | ~app_ok
    score = torch.where(use_esc.unsqueeze(1), esc, app)
    # 行为模式：flee（1）= 远离对手（-V_opp 最小化）+ 避险；aggressive（0）= 逼近。
    mode = getattr(sim, "_bmode", None)
    if mode is not None:
        flee = -Vopp_c + 2.0 * dng_c + 2.0 * thr_c + noise     # 朝最远格 + 避险
        flee = torch.where(dng_c >= 0.5, torch.full_like(flee, float("inf")), flee)
        flee = torch.where(legal, flee, torch.full_like(flee, float("inf")))
        flee = torch.where(use_esc.unsqueeze(1), esc, flee)    # 危险时也先逃
        score = torch.where((mode == 1).unsqueeze(1), flee, score)
    move = score.argmin(dim=1).long()
    # 终极兜底（修复点 2）：打分全 inf（被泡/火完全围死，V_safe 无路）时，
    # 选**最小 danger 的合法格**（跟 greedy 的逃生兜底一致）—— 宁可走火也
    # 不原地等死。旧版 argmin 在全 inf 时返回 0 = 向上，被掩码屏蔽后等于
    # 站着不动 → "站火海不动"的另一个来源。
    any_legal = legal.any(dim=1)
    no_path = ~score.isfinite().any(dim=1)
    fallback = torch.where(legal, dng_c, torch.full_like(dng_c, float("inf"))).argmin(dim=1)
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
        place = torch.where((mode == 1), torch.zeros_like(place), place)  # flee 不放泡

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
        # 纯进攻：恒 aggressive（mode=None = 不切 flee；astar_attack 里
        # mode is None 时用 aggressive 评分/放泡路径）
        return BotWrapper(sim, astar_attack, "hunter")
    raise ValueError(f"未知 bot 类型: {kind}")
