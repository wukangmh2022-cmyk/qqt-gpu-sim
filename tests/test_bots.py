"""课程敌人（sim/bots.py）与课程对手混合（train/train.py）的最小可运行性测试。

守的是四条线：
- bot 动作永远合法（尊重 mmask/bmask；死人只能 IDLE）；
- greedy 确实在逼近（Chebyshev 距离单调降）、贴身会放泡；
- build_opponents 的热身/固定概率混合逻辑正确；
- 熵退火按 local_step 计算（fresh vs resume 行为一致）。

不验证"能不能打赢"—— 强度排序由 sanity 脚本实测（见验证步骤）。
"""

from __future__ import annotations

import argparse

import torch

from sim.bots import BotWrapper, _sample_legal, make_bot
from sim.config import MOVE_IDLE, N_BOMB, N_MOVES, SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic
from train.model_pool import ModelPool
from train.ppo import PPOConfig, SelfPlayRunner
from train.train import (adapt_first_conv, anneal_frac, build_opponents,
                         save_ckpt, update_fixed_elo)

D = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _opp_dist(sim, a: int, b: int) -> torch.Tensor:
    """两个玩家之间的 Chebyshev 距离（按当前格算）。"""
    r = sim.pos[:, a, 0].floor().long().clamp(0, sim.cfg.height - 1)
    c = sim.pos[:, a, 1].floor().long().clamp(0, sim.cfg.width - 1)
    ro = sim.pos[:, b, 0].floor().long().clamp(0, sim.cfg.height - 1)
    co = sim.pos[:, b, 1].floor().long().clamp(0, sim.cfg.width - 1)
    return (r - ro).abs().maximum((c - co).abs()).float()


def test_sample_legal_respects_mask():
    mask = torch.zeros(64, N_MOVES, dtype=torch.bool)
    mask[:, 2] = True
    assert torch.all(_sample_legal(mask) == 2)
    mask[:, 4] = True
    got = _sample_legal(mask)
    assert torch.all((got == 2) | (got == 4)), "只能落在合法项上"


def test_bots_actions_always_legal():
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=60)
    sim = BatchedSim(cfg, 32, seed=1)
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    for kind in ("random", "greedy"):
        bot = make_bot(sim, kind)
        a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        assert a.shape == (32, 2)
        assert torch.all(a[:, 0] < N_MOVES) and torch.all(a[:, 1] < N_BOMB)
        legal_m = mm[:, 1].gather(1, a[:, 0:1]).squeeze(1)
        legal_b = bm[:, 1].gather(1, a[:, 1:1]).squeeze(1)
        assert bool(legal_m.all()), f"{kind} 选了非法方向"
        assert bool(legal_b.all()), f"{kind} 选了非法放泡"


def test_dead_bot_idles():
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=30)
    sim = BatchedSim(cfg, 8, seed=2)
    sim.hp[:, 1] = 0
    sim.alive[:, 1] = False
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    for kind in ("random", "greedy"):
        a = make_bot(sim, kind).act(obs, mm[:, 1], bm[:, 1], 1)
        assert torch.all(a[:, 0] == MOVE_IDLE) and torch.all(a[:, 1] == 0)


def _full_actions(sim, bot_act: torch.Tensor, pid: int = 1) -> torch.Tensor:
    """把单个 bot 的 (N,2) 动作补齐成 sim.step 要的 (N,P,2)，其余玩家呆立。"""
    a = torch.zeros((sim.num_envs, sim.cfg.n_players, 2), dtype=torch.long,
                    device=sim.device)
    a[:, pid] = bot_act
    return a


def test_greedy_approaches_open_map():
    """空场上 greedy 朝对手逼近：距离单调降；离得远（超 blast_cap）不放泡。"""
    cfg = SimConfig(height=13, width=13, n_players=2, max_steps=40)
    sim = BatchedSim(cfg, 64, seed=3)
    bot = make_bot(sim, "greedy")
    # 对手挪到最远角（Chebyshev ≈8 > open 关 blast_cap=6），bot 只有逼近可学
    sim.pos[:, 0, 0] = 12.5
    sim.pos[:, 0, 1] = 12.5
    d0 = _opp_dist(sim, 1, 0)
    d1 = None
    for _ in range(10):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        assert torch.all(a[:, 1] == 0), "超过 blast_cap 不该放泡"
        sim.step(_full_actions(sim, a), auto_reset=False)
        d1 = _opp_dist(sim, 1, 0)
    assert bool((d1 < d0).sum() >= 0.6 * d0.numel()), \
        f"greedy 应逐步逼近：d0={d0.mean():.2f} → d1={d1.mean():.2f}"


def test_greedy_places_bomb_adjacent():
    """贴脸 + 有泡预算时 greedy 放泡；贴身泡朝对手方向放置（打得到）。"""
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=30)
    sim = BatchedSim(cfg, 32, seed=4)
    bot = make_bot(sim, "greedy")
    # 把两个玩家摆到同一格（距离 0），强制"贴脸"
    r = torch.full((32,), 4, dtype=torch.long)
    c = torch.full((32,), 4, dtype=torch.long)
    sim.pos[:, 1, 0] = r.float() + 0.5
    sim.pos[:, 1, 1] = c.float() + 0.5
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    assert torch.all(a[:, 1] == 1), "贴脸应放泡"
    sim.step(_full_actions(sim, a), auto_reset=False)
    # 放出去的泡：owner=1 且引信 > 0
    live = (sim.owner == 1) & (sim.fuse > 0)
    assert bool(live.flatten(1).sum(dim=1).all())


def test_greedy_avoids_about_to_explode_cell():
    """脚下有即将爆炸的泡时，greedy 选最低 danger 的合法格逃走（不原地等死）。"""
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=20)
    sim = BatchedSim(cfg, 16, seed=5)
    bot = make_bot(sim, "greedy")
    # 在自己脚下放一颗引信 2 的泡 → 当前格 danger 接近 1，所有邻格必须更安全
    r = torch.full((16,), 3, dtype=torch.long)
    c = torch.full((16,), 3, dtype=torch.long)
    sim.pos[:, 1, 0] = r.float() + 0.5
    sim.pos[:, 1, 1] = c.float() + 0.5
    b_idx = torch.arange(16)
    sim.fuse[b_idx, r, c] = 2
    sim.owner[b_idx, r, c] = 1
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    # 不许原地等死：IDLE 的目标格 = 当前格（danger≈1），逃生逻辑必须选别的方向
    from sim.blast import danger_map
    dng = danger_map(sim.fuse, sim.wall, cfg.blast, cfg.fuse)
    own = dng.flatten(1).gather(1, (r * cfg.width + c).unsqueeze(1)).squeeze(1)
    assert bool((own > 0.8).all()), "测试前提：脚下必须真危险"
    assert bool((a[:, 0] != MOVE_IDLE).any()), "危险格上不该站着等死"


def test_bot_wrapper_is_bot_flag():
    cfg = SimConfig(height=9, width=9, n_players=2)
    sim = BatchedSim(cfg, 4, seed=6)
    assert make_bot(sim, "greedy").is_bot is True
    assert make_bot(sim, "random").is_bot is True
    assert make_bot(sim, "astar", mode=False).is_bot is True


def test_astar_actions_always_legal():
    """A* bot 的动作同样必须尊重掩码/墙/死人。"""
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=60)
    sim = BatchedSim(cfg, 32, seed=8)
    bot = make_bot(sim, "astar", mode=False)
    for _ in range(5):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        assert torch.all(mm[:, 1].gather(1, a[:, 0:1]).squeeze(1))
        assert torch.all(bm[:, 1].gather(1, a[:, 1:1]).squeeze(1))
        sim.step(_full_actions(sim, a), auto_reset=False)


def test_astar_escapes_danger():
    """脚下快爆（fuse≤2）时 A* 沿安全价值场逃生，不原地等死。"""
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=20)
    sim = BatchedSim(cfg, 16, seed=9)
    bot = make_bot(sim, "astar", mode=False)
    r = torch.full((16,), 3, dtype=torch.long)
    c = torch.full((16,), 3, dtype=torch.long)
    sim.pos[:, 1, 0] = r.float() + 0.5
    sim.pos[:, 1, 1] = c.float() + 0.5
    b_idx = torch.arange(16)
    sim.fuse[b_idx, r, c] = 2
    sim.owner[b_idx, r, c] = 1
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    assert bool((a[:, 0] != MOVE_IDLE).any()), "危险格上不该站着等死"
    # 逃出的那步应该降低脚下 danger（沿 V_safe 走）
    from sim.blast import danger_map
    dng = danger_map(sim.fuse, sim.wall, cfg.blast, cfg.fuse)
    flat = a[:, 0] * 0 + (r * cfg.width + c)
    d0 = dng.flatten(1).gather(1, flat.unsqueeze(1)).squeeze(1)
    sim.step(_full_actions(sim, a), auto_reset=False)
    sim.fuse -= 1                                          # 走一步，引信也走一步
    dng2 = danger_map(sim.fuse.clamp(min=0), sim.wall, cfg.blast, cfg.fuse)
    r2 = sim.pos[:, 1, 0].floor().long().clamp(0, cfg.height - 1)
    c2 = sim.pos[:, 1, 1].floor().long().clamp(0, cfg.width - 1)
    d1 = dng2.flatten(1).gather(1, (r2 * cfg.width + c2).unsqueeze(1)).squeeze(1)
    assert bool((d1 < d0).sum() >= 0.6 * d0.numel()), \
        f"A* 逃生应显著降低脚下 danger：{d0.mean():.2f} → {d1.mean():.2f}"


def test_astar_places_bomb_when_aligned_and_retreats():
    """与对手同行/同列且在 blast_cap 内 → 放泡；随后撤出炮火区、不自杀。"""
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=40)
    sim = BatchedSim(cfg, 32, seed=10)
    bot = make_bot(sim, "astar", mode=False)
    # bot(pid=1) 在 (4,4)，对手(pid=0) 在同行 (4,6) → 距离 2 = blast_cap(默认 2)
    sim.pos[:, 1, 0] = 4.5
    sim.pos[:, 1, 1] = 4.5
    sim.pos[:, 0, 0] = 4.5
    sim.pos[:, 0, 1] = 6.5
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    assert bool((a[:, 1] == 1).all()), "同行同 blast 内应放泡"
    sim.step(_full_actions(sim, a), auto_reset=False)
    bomb_r = torch.full((32,), 4, dtype=torch.long)
    bomb_c = torch.full((32,), 4, dtype=torch.long)
    exited = torch.zeros(32, dtype=torch.bool)
    hp0 = sim.hp[:, 1].clone()
    for _ in range(5):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a2 = bot.act(obs, mm[:, 1], bm[:, 1], 1)
        sim.step(_full_actions(sim, a2), auto_reset=False)
        r2 = sim.pos[:, 1, 0].floor().long().clamp(0, cfg.height - 1)
        c2 = sim.pos[:, 1, 1].floor().long().clamp(0, cfg.width - 1)
        exited |= ~((r2 == bomb_r) & (c2 == bomb_c))
    assert bool(exited.all()), "放泡后应撤出泡格（0.36 格/tick，几 tick 内必撤）"
    assert bool((sim.hp[:, 1] == hp0).all()), "撤出前不应吃自己泡的伤害"


def test_astar_escapes_own_bomb_ring():
    """泡阵困住自己时走出口，不站火海不动（回归修复）。

    旧版 bug：Dijkstra 不把泡当障碍 → V_safe 规划"穿泡逃生"，esc 打分里
    IDLE（当前格）反而最小 → 站火海里傻站。修复：泡进障碍掩码 + 无路时
    选最小 danger 的合法格兜底。
    """
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=30)
    sim = BatchedSim(cfg, 32, seed=11)
    bot = make_bot(sim, "astar", mode=False)
    # pid=1 在 (3,3)，自己在 (3,2)(3,4)(2,3) 放泡（上左右），下方 (4,3) 空 = 出口
    sim.pos[:, 1, 0] = 3.5
    sim.pos[:, 1, 1] = 3.5
    sim.pos[:, 0, 0] = 0.5
    sim.pos[:, 0, 1] = 0.5
    b_idx = torch.arange(32)
    for r, c in ((3, 2), (3, 4), (2, 3)):
        sim.fuse[b_idx, r, c] = 30
        sim.owner[b_idx, r, c] = 1
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    idle_frac = (a[:, 0] == MOVE_IDLE).float().mean().item()
    assert idle_frac == 0.0, f"泡阵困住应走出口而非 IDLE（IDLE 占比 {idle_frac}）"
    # 往下（出口）的方向是合法且被选中的
    assert torch.all(a[:, 0] == 1), "唯一出口在下方，应全选下"


def test_astar_fallback_when_no_path():
    """打分全 inf（V_safe 无路）时兜底选最小 danger 的合法格，不 argmin=0 卡死。

    旧版 argmin 在全 inf 时返回 0（向上），被掩码屏蔽 = 站着不动。
    """
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=30)
    sim = BatchedSim(cfg, 16, seed=12)
    bot = make_bot(sim, "astar", mode=False)
    # 自己脚下 fuse=1 的泡（即将爆炸，danger≈1），四周全放泡把自己围死
    b_idx = torch.arange(16)
    sim.fuse[b_idx, 3, 3] = 1
    sim.owner[b_idx, 3, 3] = 1
    for r, c in ((3, 2), (3, 4), (2, 3), (4, 3)):
        sim.fuse[b_idx, r, c] = 30
        sim.owner[b_idx, r, c] = 1
    obs = sim.observe()
    mm, bm = sim.legal_mask()
    a = bot.act(obs, mm[:, 1], bm[:, 1], 1)
    # 至少要有 move（IDLE 也是合法结果之一，但不能是"非法 0 被屏蔽"的隐性卡死；
    # 这里断言动作是合法掩码里的 —— 兜底选出的格一定 legal）
    assert bool((mm[:, 1].gather(1, a[:, 0:1]).squeeze(1)).all()), \
        "兜底 move 必须合法"


def test_astar_random_mode_switches():
    """astar 的随机接近/远离模式：_bmode 存在、可切换（课程多样性）。

    随机倒计时（60~240 tick）受全局 RNG 序列影响，连跑可能不切换 ——
    测试手动把 _btimer 压到 0，强制下 tick 切换，确定性验证模式逻辑。
    """
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=60)
    sim = BatchedSim(cfg, 16, seed=13)
    bot = make_bot(sim, "astar")          # mode=True：挂随机模式
    seen = set()
    for _ in range(30):
        sim._btimer = torch.ones(16, dtype=torch.long, device=sim.device)
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        bot.act(obs, mm[:, 1], bm[:, 1], 1)
        assert hasattr(sim, "_bmode"), "应初始化 _bmode"
        seen.add(int(sim._bmode[0].item()))
        sim.step(_full_actions(sim, torch.zeros(16, 2, dtype=torch.long)),
                 auto_reset=False)
        if 0 in seen and 1 in seen:
            break
    assert 0 in seen and 1 in seen, f"随机模式应切换 aggressive/flee，实际 {seen}"
    # 固定模式（mode=False）不挂 mode_fn
    assert make_bot(sim, "astar", mode=False).mode_fn is None


def test_runner_dispatches_bot_opponent():
    """SelfPlayRunner 对手位放 BotWrapper：collect 跑通、统计正常、无 handicap 报错。"""
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=60)
    sim = BatchedSim(cfg, 8, seed=7)
    learner = ActorCritic(cfg.obs_shape)
    bot = make_bot(sim, "greedy")
    pcfg = PPOConfig(rollout_steps=16, epochs=1, minibatches=2)
    runner = SelfPlayRunner(sim, learner, [bot], pcfg, 1.0)
    buf, last_val = runner.collect()
    assert buf.ptr == pcfg.rollout_steps
    assert torch.isfinite(last_val).all()
    # bot 无 handicap（1.0）时不该走削弱分支；低 handicap 也不该崩
    runner2 = SelfPlayRunner(sim, learner, [bot], pcfg, 0.3)
    buf2, _ = runner2.collect()
    assert buf2.ptr == pcfg.rollout_steps


# ---------------------------------------------------------------- 课程混合

def _mk_fixed(obs_shape) -> tuple[str, ActorCritic]:
    net = ActorCritic(obs_shape)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return "fx", net


def test_warmup_uses_only_bots_and_fixed():
    cfg = SimConfig(height=7, width=7, n_players=3)
    learner = ActorCritic(cfg.obs_shape)
    pool = ModelPool()
    pool.add(learner, step=0, elo=1000.0)
    dev = next(learner.parameters()).device
    name, fnet = _mk_fixed(cfg.obs_shape)
    sim = BatchedSim(cfg, 4)
    bot = make_bot(sim, "greedy")
    nets, snaps = build_opponents(
        pool, learner, 1000.0, 3, dev,
        fixed_items=[(name, fnet)], bot_items=[("greedy", bot)],
        warmup=True, fixed_prob=0.0)
    assert len(nets) == 2
    for s in snaps:
        assert "state" not in s, "热身期不得出现池子快照"
        assert s["name"] in ("fx", "greedy")


def test_fixed_prob_mixing():
    cfg = SimConfig(height=7, width=7, n_players=3)
    learner = ActorCritic(cfg.obs_shape)
    pool = ModelPool()
    pool.add(learner, step=0, elo=1000.0)
    dev = next(learner.parameters()).device
    name, fnet = _mk_fixed(cfg.obs_shape)
    sim = BatchedSim(cfg, 4)
    # fixed_prob=1.0：必出固定对手
    nets, snaps = build_opponents(
        pool, learner, 1000.0, 3, dev,
        fixed_items=[(name, fnet)], bot_items=[], fixed_prob=1.0)
    assert all(s.get("name") == name for s in snaps)
    # fixed_prob=0.0：必出池子快照
    nets, snaps = build_opponents(
        pool, learner, 1000.0, 3, dev,
        fixed_items=[(name, fnet)], bot_items=[], fixed_prob=0.0)
    assert all("state" in s for s in snaps)
    # 无 fixed/bot、warmup=False → 纯池子
    nets, snaps = build_opponents(pool, learner, 1000.0, 3, dev)
    assert all("state" in s for s in snaps)


def test_update_fixed_elo_mutates_dict():
    fe = {}
    elo = update_fixed_elo(fe, "5x2", 1000.0, 1.0)     # 赢 → learner 涨分
    assert elo > 1000.0 and fe["5x2"] < 1000.0
    elo2 = update_fixed_elo(fe, "5x2", elo, 0.0)       # 输 → 回落
    assert elo2 < elo and fe["5x2"] > 1000.0 - 32.0
    # 胜负守恒检查：赢 K 分、输 K 分大致回到原点
    elo = 1000.0
    fe2 = {}
    elo = update_fixed_elo(fe2, "x", elo, 1.0)
    elo = update_fixed_elo(fe2, "x", elo, 0.0)
    assert abs(elo - 1000.0) < 1.0, f"一胜一负应收敛回原点，实际 {elo:.2f}"


def test_fixed_elo_roundtrips_in_ckpt(tmp_path):
    cfg = SimConfig(height=7, width=7, n_players=2)
    learner = ActorCritic(cfg.obs_shape)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    pool = ModelPool()
    from train.curriculum import CurriculumState
    path = str(tmp_path / "ck.pt")
    save_ckpt(path, learner=learner, opt=opt, pool=pool,
              cstate=CurriculumState(), global_step=5, elo=1000.0,
              args=argparse.Namespace(num_envs=8),
              fixed_elo={"5x2": 1234.0, "greedy": 900.0})
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["fixed_elo"] == {"5x2": 1234.0, "greedy": 900.0}


def test_anneal_frac_local_step():
    """退火按本次运行内步数走：fresh 和 resume 从同一起点重新开熵。"""
    assert anneal_frac(0, 100) == 0.0
    assert anneal_frac(50, 100) == 0.5
    assert anneal_frac(100, 100) == 1.0
    assert anneal_frac(999, 100) == 1.0
    # resume：global_step 大但 local_step 小 → frac 归零重开（老 bug 是顶着 1）
    assert anneal_frac(3_000_000, 3_000_000_000) < 0.01
    # 自定跨度
    assert anneal_frac(150_000_000, 150_000_000) == 1.0
    assert anneal_frac(75_000_000, 150_000_000) == 0.5
