"""triton 化 step 整合（Step 1：核心传播）。

核心传播段（计数/放泡/引信/移动/爆炸）用 triton kernel（每 tick ~6 launch），
伤害/清场/终局用 torch（少量 launch）。奖励在 Step 2 补（triton_step_full）。

与 BatchedSim.step 的核心状态逐位一致（cfg 关奖励时验证：
place_*/hit_attr/combo/chain/growth 全关 → 状态无奖励副作用）。
"""
import torch

from .triton_sim import (place_bombs_triton, move_players_triton,
                         resolve_triton, danger_triton)
from .torch_sim import center_cell


def triton_step_core(sim, actions):
    """in-place 更新 sim 核心状态。返回 (placed, covered, triggered, done)。

    对应 BatchedSim.step 的 1-6 步 + 清场 + 终局（不含奖励/place_predict/
    hit_attr/宝箱拾取/连锁兑现）。训练奖励段由 Step 2（triton_step_full）叠加。
    """
    cfg = sim.cfg
    n, p = sim.num_envs, cfg.n_players
    d = sim.pos.device
    move, bomb = actions[..., 0], actions[..., 1]
    alive0 = sim.alive.clone()

    # 1. 引信递减（torch 1 kernel——in-place，地址稳定）
    torch.where(sim.fuse > 0, sim.fuse - 1, sim.fuse, out=sim.fuse)

    # 2. 放泡（triton：计数 kernel + 放泡 kernel = 2 launch）
    placed = place_bombs_triton(cfg, sim.fuse, sim.owner, sim.bomb_blast,
                                sim.pos, sim.alive, bomb,
                                sim.bombs_cap, sim.blast_cap, sim.brick,
                                sim.wall)

    # 3. 被动计时
    sim.since_bomb.add_(1)
    sim.since_bomb[placed] = 0

    # 4. 移动（triton，AABB 滑动）
    blocked = sim.wall | sim.brick | (sim.fuse > 0)
    sm = sim.spd_g
    if sim.speed_mult is not None:
        sm = sm * sim.speed_mult
    sim.pos.copy_(move_players_triton(cfg, sim.pos, move, sim.alive, blocked, sm))

    # 5. 爆炸与连锁（triton explode kernel 链）
    covered, triggered = resolve_triton(sim.fuse, sim.owner, sim.wall,
                                        sim.bomb_blast, sim.brick,
                                        cfg.max_chain,
                                        early_exit=not sim._graph_mode)

    # 5.5 砖/宝箱更新（torch step 第 4 步 resolve 后同款——**不能漏**：
    #     爆炸会摧毁 brick 并转 crate；core 若不更新，下一 tick 的 blocked
    #     = wall|brick|(fuse>0) 就与 torch 分叉，第一波爆炸后级联发散）。
    if cfg.map_mode == "corridor":
        sim.crate.bitwise_or_(sim.brick & covered)   # 炸掉的砖 → 宝箱（in-place）
    sim.brick.bitwise_and_(~covered)                  # 摧毁砖（in-place）

    # 6. 伤害判定（中心格着火 + 无敌期）
    cell = center_cell(sim.pos)
    flat = (cell[..., 0] * cfg.width + cell[..., 1]).clamp(0, cfg.width * cfg.height - 1)
    hit = alive0 & covered.view(n, -1).gather(1, flat)
    invuln_ok = sim.invuln <= 0
    hit_eff = hit & invuln_ok
    hp_new = (sim.hp.to(torch.int32) - hit_eff.to(torch.int32)).clamp(min=0)
    died = hit_eff & (hp_new == 0)
    sim.hp.copy_(hp_new.to(torch.uint8))
    sim.alive.copy_(alive0 & ~died)
    sim.invuln.sub_(1)
    sim.invuln.clamp_(min=0)
    sim.invuln[hit_eff] = cfg.invuln_ticks

    # 7. 清场（触发泡 fuse→0 / owner→-1 / blast→0）
    torch.where(triggered, torch.zeros_like(sim.fuse), sim.fuse, out=sim.fuse)
    torch.where(triggered, torch.full_like(sim.owner, -1), sim.owner,
                out=sim.owner)
    torch.where(triggered, torch.zeros_like(sim.bomb_blast), sim.bomb_blast,
                out=sim.bomb_blast)

    # 8. 计步与终局
    sim.t.add_(1)
    n_alive = sim.alive.sum(dim=1)
    done = (n_alive <= 1) | (sim.t >= cfg.max_steps)
    return placed, covered, triggered, done


def triton_step_full(sim, actions):
    """完整 triton 化 step：核心传播（triton kernel）+ 奖励段（torch，顺序对齐）。

    与 BatchedSim.step 逐位一致（含 place_predict/hit_attr/combo/宝箱拾取/
    连锁兑现/胜负奖励）。核心段的 launch 已 triton 化（~6 launch），奖励段的
    torch 算子在 Ascend（CANN）上 launch 开销低——后续可按需再 triton 化。
    """
    cfg = sim.cfg
    n, p = sim.num_envs, cfg.n_players
    d = sim.pos.device
    move, bomb = actions[..., 0], actions[..., 1]
    alive0 = sim.alive.clone()
    hp_before = sim.hp.clone()

    # 1. 引信递减（torch）
    torch.where(sim.fuse > 0, sim.fuse - 1, sim.fuse, out=sim.fuse)

    # 2. 放泡（triton 计数+放泡 2 launch）
    placed = place_bombs_triton(cfg, sim.fuse, sim.owner, sim.bomb_blast,
                                sim.pos, sim.alive, bomb,
                                sim.bombs_cap, sim.blast_cap, sim.brick,
                                sim.wall)

    # 3. 放泡奖励（torch——需要爆炸前的 fuse/live 状态）
    place_bonus = sim._place_predict_reward(placed, alive0)

    # 4. 被动计时
    sim.since_bomb.add_(1)
    sim.since_bomb[placed] = 0

    # 5. 移动（triton）
    blocked = sim.wall | sim.brick | (sim.fuse > 0)
    sm = sim.spd_g
    if sim.speed_mult is not None:
        sm = sm * sim.speed_mult
    sim.pos.copy_(move_players_triton(cfg, sim.pos, move, sim.alive, blocked, sm))

    # 6. 爆炸与连锁（triton explode 链）
    covered, triggered = resolve_triton(sim.fuse, sim.owner, sim.wall,
                                        sim.bomb_blast, sim.brick,
                                        cfg.max_chain,
                                        early_exit=not sim._graph_mode)

    # 7. 连锁兑现（torch，清场前读 owner/引信）
    chain_bonus_p = torch.zeros(n, p, dtype=torch.float32, device=d)
    if cfg.chain_blast_bonus > 0 and cfg.max_chain > 1:
        nat = triggered & (sim.fuse == 0)
        nat_flat = nat.view(n, -1)
        own_flat = sim.owner.view(n, -1)
        chained_mask = (triggered & ~nat).view(n, -1)
        for pl in range(cfg.n_players):
            fired = (nat_flat & (own_flat == pl)).sum(dim=1).clamp(max=1)
            cross = (chained_mask & (own_flat != pl)).sum(dim=1)
            chain_bonus_p[:, pl] = cfg.chain_blast_bonus * cross * fired

    # 8. 炸砖变箱 + 摧毁砖
    if cfg.map_mode == "corridor":
        sim.crate.bitwise_or_(sim.brick & covered)
    sim.brick.bitwise_and_(~covered)

    # 9. 伤害判定
    cell = center_cell(sim.pos)
    flat = (cell[..., 0] * cfg.width + cell[..., 1]).clamp(0, cfg.width * cfg.height - 1)
    hit = alive0 & covered.view(n, -1).gather(1, flat)
    invuln_ok = sim.invuln <= 0
    hit_eff = hit & invuln_ok
    hp_new = (sim.hp.to(torch.int32) - hit_eff.to(torch.int32)).clamp(min=0)
    died = hit_eff & (hp_new == 0)
    sim.hp.copy_(hp_new.to(torch.uint8))
    sim.alive.copy_(alive0 & ~died)
    own_live_snap = torch.stack([
        (sim.owner == me).flatten(1).sum(dim=1) for me in range(cfg.n_players)
    ], dim=1)
    sim.invuln.sub_(1)
    sim.invuln.clamp_(min=0)
    sim.invuln[hit_eff] = cfg.invuln_ticks

    # 10. 清场
    torch.where(triggered, torch.zeros_like(sim.fuse), sim.fuse, out=sim.fuse)
    torch.where(triggered, torch.full_like(sim.owner, -1), sim.owner,
                out=sim.owner)
    torch.where(triggered, torch.zeros_like(sim.bomb_blast), sim.bomb_blast,
                out=sim.bomb_blast)

    # 11. 计步/终局
    sim.t.add_(1)
    sim._hazard_wave()
    n_alive = sim.alive.sum(dim=1)
    done = (n_alive <= 1) | (sim.t >= cfg.max_steps)

    # 11.5 危险图（triton 阶段 A 连锁 + B 扩散，与 torch step 同源同语义）。
    #     计算后写入 _dng_cache/_dng_sig —— observe() 直接复用（省掉 torch 侧
    #     每 tick 重算 ~44ms 的 danger_map；triton 版 ~1.6ms @N=8192）。
    blast_map = sim._blast_map()
    danger = danger_triton(sim.fuse, sim.wall, sim.fuse > 0, sim.brick,
                           blast_map, cfg.fuse, max_chain=cfg.max_chain,
                           early_exit=not sim._graph_mode)
    sim._dng_cache = danger
    sim._dng_sig = sim._dng_signature()

    # 12. 稠密伤害 + 基础奖励
    dmg = (hp_before - sim.hp.to(torch.int32)).clamp(min=0).float()
    dealt = dmg.sum(dim=1, keepdim=True) - dmg
    reward = (-cfg.step_penalty * alive0.float()
              + cfg.hit_reward * dealt - cfg.hit_reward * dmg
              + sim._explore_coef * place_bonus * alive0.float())

    # 13. 掉血属性惩罚（全量化，回收延迟——与 torch_sim 相同）
    if cfg.hit_attr_penalty > 0 and not sim._graph_mode:
        hit_any = (dmg > 0) & alive0
        pen = cfg.hit_attr_penalty
        nb = torch.clamp(sim.bombs_cap - pen, min=sim._lo_bombs)
        nz = torch.clamp(sim.blast_cap - pen, min=sim._lo_blast)
        ns = torch.max(sim.spd_g - pen * cfg.growth_speed_step, sim._lo_spd)
        lost_all = ((sim.bombs_cap - nb) + (sim.blast_cap - nz)
                    + torch.round((sim.spd_g - ns)
                                  / cfg.growth_speed_step)).long()
        torch.where(hit_any, nb, sim.bombs_cap, out=sim.bombs_cap)
        torch.where(hit_any, nz, sim.blast_cap, out=sim.blast_cap)
        torch.where(hit_any, ns, sim.spd_g, out=sim.spd_g)
        lost_eff = lost_all * hit_any.long()
        for pl in range(cfg.n_players):
            lost_p = lost_eff[:, pl]
            if bool(lost_p.sum() > 0):
                hidx = lost_p.nonzero(as_tuple=True)[0]
                sim._scatter_recycle(hidx, lost_p[hidx])

    # 14. 连锁兑现奖励（探索塑形）
    reward = reward + sim._explore_coef * chain_bonus_p * alive0.float()

    # 15. 宝箱拾取 + 成长（corridor）
    if cfg.map_mode == "corridor":
        cell = center_cell(sim.pos)
        flat = (cell[..., 0] * cfg.width + cell[..., 1]).clamp(0, cfg.width * cfg.height - 1)
        stood = sim.crate.view(n, -1).gather(1, flat)
        reward = reward + cfg.brick_reward * stood.float() * alive0.float()
        for pl in range(cfg.n_players):
            rb = (sim._rand_buf[pl * n:(pl + 1) * n]
                  if sim._rand_buf is not None else torch.rand(n, device=d))
            rec_flat = sim._recycle_crate.view(n, -1).gather(
                1, flat[:, pl].unsqueeze(1)).squeeze(1)
            prob = torch.where(rec_flat.bool(), cfg.recycle_crate_prob,
                               sim.crate_prob)
            hits = stood[:, pl] & (rb < prob) & alive0[:, pl]
            sim._grow_player_vec(pl, hits.long(), alive0[:, pl])
        for pl in range(cfg.n_players):
            sim.crate.view(n, -1)[torch.arange(n, device=d),
                                  flat[:, pl]] = False

    # 15.5 危险区站桩罚（与 torch step 同公式：乘探索退火 _explore_coef）。
    #      danger 已在 11.5 由 triton 计算并缓存，这里直接 gather 脚下格。
    if cfg.danger_penalty > 0:
        cell = center_cell(sim.pos)
        flat = (cell[..., 0] * cfg.width + cell[..., 1]).clamp(0, cfg.width * cfg.height - 1)
        standing = danger.view(n, -1).gather(1, flat)
        reward = reward - sim._explore_coef * cfg.danger_penalty \
            * standing * alive0.float()

    # 16. combo 连击
    if cfg.combo_reward > 0:
        hit_this = (dmg > 0)
        for me in range(cfg.n_players):
            dealt_me = dealt[:, me]
            c = sim._combo[:, me]
            gap = sim.t - sim._last_hit[:, me]
            factor = cfg.combo_gap_factor ** gap.clamp(min=0).float()
            inc = (dealt_me > 0).long()
            combo_now = (c + inc) * inc
            pts = (combo_now.float() * cfg.combo_reward * factor) \
                * dealt_me.clamp(max=1).float()
            reward[:, me] += pts * alive0[:, me].float()
            sim._combo[:, me] = torch.where(inc.bool(), combo_now, c)
            sim._last_hit[:, me] = torch.where(
                inc.bool(), sim.t, sim._last_hit[:, me])
            sim._combo[:, me] = torch.where(
                hit_this[:, me], torch.zeros_like(c), sim._combo[:, me])

    # 17. 胜负奖励
    death_done = done & (n_alive == 1)
    winner = death_done.unsqueeze(1) & sim.alive & (n_alive == 1).unsqueeze(1)
    loser = death_done.unsqueeze(1) & ~sim.alive & (n_alive == 1).unsqueeze(1)
    hp = sim.hp.float()
    all_alive = done & (n_alive == cfg.n_players)
    if cfg.timeout_draw:
        timeout_scale = sim._explore_coef
    else:
        timeout_scale = 1.0
        for me in range(cfg.n_players):
            others = [o for o in range(cfg.n_players) if o != me]
            wins = all_alive & (hp[:, me].unsqueeze(1) > hp[:, others]).all(dim=1)
            loses = all_alive & (hp[:, me].unsqueeze(1) < hp[:, others]).any(dim=1)
            winner[:, me] |= wins
            loser[:, me] |= loses
    if cfg.win_hp_scaled:
        for me in range(cfg.n_players):
            opp_hp = (hp.sum(dim=1) - hp[:, me]) / (p - 1)
            diff = hp[:, me] - opp_hp
            base = (cfg.win_bonus / cfg.max_hp) * diff
            reward[:, me] += base * death_done.float() \
                + base * all_alive.float() * timeout_scale
    else:
        reward = reward + cfg.win_bonus * (winner.float() - loser.float())
        if cfg.timeout_draw:
            for me in range(cfg.n_players):
                opp_hp = (hp.sum(dim=1) - hp[:, me]) / (p - 1)
                diff = hp[:, me] - opp_hp
                reward[:, me] += (cfg.win_bonus / cfg.max_hp) * diff \
                    * all_alive.float() * timeout_scale

    info = {"n_alive": n_alive, "blast": covered, "trig": triggered,
            "died": died, "winner": winner.clone()}
    if sim._graph_mode or bool(done.any()):
        # 终局 env 重置（含自杀判定快照——torch step 的 reset 在 auto_reset）
        sim._last_hit[sim._last_hit < -1e8] = -10**9
        sim.reset_(done)
    return reward, done, info
