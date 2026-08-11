"""炸弹雨（hazard）躲避特训的规则测试。

机制见 sim/config.py 的 hazard_* 注释与 sim/torch_sim.py 的 `_hazard_wave`：
- 泡数上限 0 → 玩家无法放泡（`_place_bombs` 的 live < 0 恒 False + 放泡头
  被 legal_mask 屏蔽）；
- 每 hazard_wave_ticks 播一波，每波 bombs_min..max 颗落在**可通行格**
  （无墙/砖、无在场泡、非活人脚下）；
- 威力 blast_min..blast_max，随局内时间偏向大值（约 ramp 秒后几乎总是
  max-1/max）；
- 环境炸弹 owner = n_players（越界标记）→ 只进危险图通道，不进任何玩家
  引信通道 —— 网络靠危险图躲。

坐标是连续的；与 test_rules 同款约定：tick_hz 钉死 15（0.2 格/tick）。
"""

from __future__ import annotations

import torch

from sim.config import MOVE_IDLE, SimConfig
from sim.torch_sim import BatchedSim


def H(**kw) -> SimConfig:
    """hazard 测试本地构造器：13x13 空图 + 固定 15Hz。"""
    kw.setdefault("speed", 3.0)
    kw.setdefault("tick_hz", 15)
    kw.setdefault("height", 13)
    kw.setdefault("width", 13)
    kw.setdefault("max_bombs", 0)          # 双方不能放泡（特训核心）
    kw.setdefault("hazard_fraction", 1.0)  # 每局都是炸弹雨
    return SimConfig(**kw)


def idle(sim: BatchedSim) -> torch.Tensor:
    """全员松手 + 不放泡的动作张量。"""
    return torch.full((sim.num_envs, sim.cfg.n_players, 2), 0, dtype=torch.long)


def env_bombs(sim: BatchedSim) -> torch.Tensor:
    """(N,H,W) bool：当前在场环境炸弹（owner == n_players 且引信未走完）。"""
    return (sim.owner == sim.cfg.n_players) & (sim.fuse > 0)


# ---------------- 放泡封锁 ----------------

def test_hazard_players_cannot_place_bombs():
    cfg = H()
    sim = BatchedSim(cfg, 1, seed=0)
    assert int(sim.bombs_cap[0, 0]) == 0, "炸弹雨关泡数上限强制 0"
    a = torch.tensor([[[MOVE_IDLE, 1], [MOVE_IDLE, 1]]], dtype=torch.long)
    sim.step(a, auto_reset=False)
    assert int(sim.fuse.sum()) == 0, "泡数上限 0：想放也放不出来"
    assert int(sim.owner.max()) == -1
    # 放泡头被掩码屏蔽（活人也不能选 1）
    mm, bm = sim.legal_mask()
    assert not bool(bm[0, 0, 1]) and not bool(bm[0, 1, 1])
    # 观测的可用泡/上限通道全 0（14 通道尾部）
    obs = sim.observe()
    assert float(obs[0, 2 * cfg.n_players + 4].max()) == 0.0   # avail P 通道
    assert float(obs[0, 2 * cfg.n_players + 6].max()) == 0.0   # cap P 通道


# ---------------- 波次 ----------------

def test_hazard_wave_spawns_every_wave_ticks():
    cfg = H(hazard_wave_ticks=20, hazard_bombs_min=4, hazard_bombs_max=30, hazard_blast_min=1, hazard_blast_max=1,
            max_steps=400)
    sim = BatchedSim(cfg, 1, seed=0)
    a = idle(sim)
    for _ in range(19):
        sim.step(a, auto_reset=False)
    assert int(sim.fuse.sum()) == 0, "首波前没有任何炸弹"
    sim.step(a, auto_reset=False)                     # t=20：第一波
    live = env_bombs(sim)
    n = int(live.sum())
    assert 4 <= n <= 30, f"每波炸弹数 ∈ [4,30]，收到 {n}"
    assert int(sim.fuse[live].min()) == cfg.fuse, "落弹引信从满值开始"
    assert int(sim.owner[live].max()) == cfg.n_players
    # 再跑 20 tick：第二波叠加（旧泡引信仍 >0，新波落在其余可通行格）
    for _ in range(20):
        sim.step(a, auto_reset=False)
    assert int(env_bombs(sim).sum()) >= n, "第二波在旧泡之上叠加"


def test_hazard_bombs_only_on_walkable_not_under_players():
    cfg = H(hazard_wave_ticks=10, hazard_bombs_min=5, hazard_bombs_max=10, hazard_blast_min=1, hazard_blast_max=1,
            max_steps=200)
    sim = BatchedSim(cfg, 2, seed=5)
    sim.wall[0, 0, 0] = True            # 手动种一堵永久墙（open 关本身没墙）
    sim.brick[0, 12, 12] = True         # 手动种一格可炸墙
    a = idle(sim)
    for _ in range(20):                 # 两波
        sim.step(a, auto_reset=False)
    live = env_bombs(sim)
    assert bool(live.any())
    assert not bool((live & sim.wall).any()), "炸弹不能落在永久墙上"
    assert not bool((live & sim.brick).any()), "炸弹不能落在可炸墙上"
    cell = sim.pos.floor().long()
    for pl in range(cfg.n_players):
        oc = sim.alive[:, pl]
        if bool(oc.any()):
            yy, xx = cell[oc, pl, 0], cell[oc, pl, 1]
            assert not bool(live[oc, yy, xx].any()), "炸弹不能落在活人脚下"


# ---------------- 威力与偏置 ----------------

def test_hazard_blast_bias_grows_with_time():
    """早期波威力接近均匀（4..8 均值≈6）；约 ramp 秒后偏向 7/8。"""
    # ramp 1 秒 @15Hz = 15 tick；hazard_wave_ticks=10 → 第二波起已满偏
    cfg = H(hazard_wave_ticks=10, hazard_bombs_min=15, hazard_bombs_max=25,
            hazard_blast_min=4, hazard_blast_max=8, hazard_ramp_seconds=1.0, max_steps=400)

    # 早期：只跑第一波（t=10，ramp 未开始 → p≈1，均匀）
    early_sim = BatchedSim(cfg, 8, seed=3)
    for _ in range(10):
        early_sim.step(idle(early_sim), auto_reset=False)
    eb = early_sim.bomb_blast[env_bombs(early_sim)]
    assert eb.numel() >= 30, "早期样本太少，断言不具统计意义"
    assert int(eb.min()) >= 4 and int(eb.max()) <= 8
    early_mean = float(eb.float().mean())
    assert 5.0 <= early_mean <= 7.0, f"早期波应近似均匀，均值 {early_mean:.2f}"

    # 晚期：跑满 8 波（ramp 已结束 → v≈u^0.2，P(blast≥7)≈0.92）
    late_sim = BatchedSim(cfg, 8, seed=3)
    for _ in range(80):
        late_sim.step(idle(late_sim), auto_reset=False)
    lb = late_sim.bomb_blast[env_bombs(late_sim)]
    assert lb.numel() >= 100
    late_mean = float(lb.float().mean())
    high_frac = float(((lb >= 7)).float().mean())
    assert late_mean > 6.5, f"晚期波威力应偏向大值，均值 {late_mean:.2f}"
    assert high_frac > 0.6, f"晚期波 7/8 占比 {high_frac:.2f}"


def test_hazard_blast_values_in_range_always():
    cfg = H(hazard_wave_ticks=10, hazard_bombs_min=4, hazard_bombs_max=30,
            hazard_blast_min=4, hazard_blast_max=8, hazard_ramp_seconds=60.0, max_steps=600)
    sim = BatchedSim(cfg, 4, seed=11)
    a = idle(sim)
    for _ in range(300):
        sim.step(a, auto_reset=False)
    live = env_bombs(sim)
    assert bool(live.any())
    bl = sim.bomb_blast[live]
    assert int(bl.min()) >= 4 and int(bl.max()) <= 8, "威力必须留在 [4,8]"


# ---------------- 观测通道 ----------------

def test_hazard_bombs_in_danger_channel_only():
    cfg = H(hazard_wave_ticks=20, hazard_bombs_min=10, hazard_bombs_max=15, hazard_blast_min=4, hazard_blast_max=4,
            max_steps=300)
    sim = BatchedSim(cfg, 2, seed=7)
    a = idle(sim)
    for _ in range(20):
        sim.step(a, auto_reset=False)
    live = env_bombs(sim)
    assert bool(live.any())
    obs = sim.observe()
    p = cfg.n_players
    for i in range(p):
        assert float(obs[:, p + i].max()) == 0.0, \
            "环境炸弹（owner=P）不得出现在任何玩家的引信通道"
    danger = obs[:, 2 * p + 1]
    assert bool((danger > 0)[live].all()), "炸弹格在危险图通道必须非零"


# ---------------- 伤害 / 混合占比 ----------------

def test_hazard_bomb_hurts_players():
    """环境炸弹（owner=P）炸到人：掉血、对手拿 hit_reward、自己扣 hit_reward。"""
    cfg = H(max_hp=3, hazard_blast_min=1, hazard_blast_max=1, max_steps=100)
    sim = BatchedSim(cfg, 1, seed=0)
    # 手工摆一颗即将爆炸的环境炸弹，正炸玩家 0（玩家 1 站远处）
    sim.fuse[0, 4, 5] = 1
    sim.owner[0, 4, 5] = cfg.n_players
    sim.bomb_blast[0, 4, 5] = 3
    sim.pos[0, 0] = torch.tensor([4.5, 5.5])   # 玩家 0 站在炸弹右侧格
    sim.pos[0, 1] = torch.tensor([1.0, 1.0])
    a = idle(sim)
    r, _, _ = sim.step(a, auto_reset=False)
    assert int(sim.hp[0, 0]) == 2, "玩家 0 被环境炸弹打掉 1 血"
    assert int(sim.hp[0, 1]) == 3
    assert abs(float(r[0, 0]) - (-cfg.hit_reward - cfg.step_penalty)) < 1e-4, \
        f"挨炸 -hit_reward + 步罚"
    assert abs(float(r[0, 1]) - (cfg.hit_reward - cfg.step_penalty)) < 1e-4, \
        "对手 +hit_reward（我的命更久）"


def test_hazard_fraction_mixed_rolls():
    """hazard_fraction=0.5：一部分 env 是炸弹雨关（上限 0），其余正常。"""
    cfg = H(max_bombs=4, hazard_fraction=0.5)
    sim = BatchedSim(cfg, 64, seed=9)
    haz = sim._hazard
    assert 0 < int(haz.sum()) < 64, "两类关都应出现"
    assert int((sim.bombs_cap[:, 0] == 0)[haz].sum()) == int(haz.sum())
    assert int((sim.bombs_cap[:, 0] == 4)[~haz].sum()) == int((~haz).sum())
