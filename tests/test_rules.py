"""规则测试：每个用例只钉一条 RULES.md 里的规定。

写法上都是手工摆盘（直接改 sim 的状态张量）再跑一个 tick，
比"跑随机策略看有没有崩"有用得多——回归时能直接定位到哪条规则错了。

坐标是连续的，所以移动类用例都写成"跑 k 个 tick，位置应该在哪"，
而不是"跳没跳过去"。默认 speed=3 / tick_hz=15 ⇒ 每 tick 0.2 格。
"""

from __future__ import annotations

import torch

from sim.blast import rays, resolve_explosions
from sim.config import (
    MOVE_DOWN,
    MOVE_IDLE,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    N_BOMB,
    N_MOVES,
    SimConfig,
)
from sim.torch_sim import BatchedSim


def C(**kw):
    """规则测试的本地构造器：**固定 tick_hz=15**，与项目默认 Hz 解耦。

    这类测试是"走 k 个 tick 应该到哪"的步数断言，按 0.2 格/tick（15Hz）调的。
    训练/对打默认改 10Hz（0.3 格/tick）后，这些步数断言会集体错位；
    规则本身与 Hz 无关，所以测试里钉死 15Hz 而不是跟着默认值漂。
    """
    kw.setdefault("speed", 3.0)      # 钉死 0.2 格/tick（与全局 3.6 解耦）
    kw.setdefault("tick_hz", 15)
    return SimConfig(**kw)


def make(cfg: SimConfig | None = None, n: int = 1) -> BatchedSim:
    return BatchedSim(cfg or C(height=7, width=7), n, seed=0)


def clear(sim: BatchedSim) -> None:
    sim.fuse.zero_()
    sim.owner.fill_(-1)
    sim.wall.zero_()


def act(*pairs: tuple[int, int]) -> torch.Tensor:
    """act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)) → (1, P, 2)。"""
    return torch.tensor([[list(p) for p in pairs]], dtype=torch.long)


def hold(sim: BatchedSim, a: torch.Tensor, ticks: int) -> None:
    for _ in range(ticks):
        sim.step(a, auto_reset=False)


# ---------------- 火焰形状 ----------------

def test_rays_is_cross_not_square():
    wall = torch.zeros((1, 7, 7), dtype=torch.bool)
    src = torch.zeros((1, 7, 7), dtype=torch.bool)
    bombed = torch.zeros((1, 7, 7), dtype=torch.bool)
    src[0, 3, 3] = True
    cov = rays(src, wall, bombed, blast=2)
    assert cov[0, 3, 3] and cov[0, 1, 3] and cov[0, 5, 3]
    assert cov[0, 3, 1] and cov[0, 3, 5]
    # 斜角不能着火：这是 blast.py 里"每个方向从原始爆源重新出发"的回归测试
    assert not cov[0, 2, 2] and not cov[0, 4, 4]
    assert int(cov.sum()) == 1 + 4 * 2


def test_wall_blocks_flame_and_is_not_burned():
    wall = torch.zeros((1, 7, 7), dtype=torch.bool)
    wall[0, 3, 4] = True
    src = torch.zeros((1, 7, 7), dtype=torch.bool)
    src[0, 3, 3] = True
    bombed = torch.zeros((1, 7, 7), dtype=torch.bool)
    cov = rays(src, wall, bombed, blast=3)
    assert not cov[0, 3, 4], "墙本身不被点燃"
    assert not cov[0, 3, 5], "墙背后被挡住"
    assert cov[0, 3, 0] and cov[0, 3, 1] and cov[0, 3, 2], "另一侧照常延伸"


def test_bomb_blocks_flame_but_is_ignited():
    """泡挡火（经典炸弹人）：火焰覆盖泡本身并引爆它，但不穿过它继续延伸。"""
    wall = torch.zeros((1, 7, 7), dtype=torch.bool)
    src = torch.zeros((1, 7, 7), dtype=torch.bool)
    bombed = torch.zeros((1, 7, 7), dtype=torch.bool)
    src[0, 3, 3] = True
    bombed[0, 3, 4] = True                    # 一颗泡挡在右侧
    cov = rays(src, wall, bombed, blast=3)
    assert cov[0, 3, 4], "火焰要覆盖到泡所在的格（把它点燃）"
    assert not cov[0, 3, 5], "泡背后被挡住：火焰不能穿透"
    assert not cov[0, 3, 6]
    assert cov[0, 3, 0], "没有泡的一侧照常延伸"


# ---------------- 连锁 ----------------

def test_chain_explosion_propagates():
    fuse = torch.zeros((1, 7, 7), dtype=torch.int16)
    owner = torch.full((1, 7, 7), -1, dtype=torch.int8)
    wall = torch.zeros((1, 7, 7), dtype=torch.bool)
    # 0 号泡这 tick 引爆，间隔 2 格各放一个引信还很长的泡，应该被连锁带走
    for col, f in ((1, 0), (3, 9), (5, 9)):
        fuse[0, 3, col] = f
        owner[0, 3, col] = 0
    cov, trig = resolve_explosions(fuse, owner, wall, blast=2, max_chain=8)
    assert trig[0, 3, 1] and trig[0, 3, 3] and trig[0, 3, 5], "三个泡应全部引爆"
    assert cov[0, 3, 6], "最后一个泡的火焰要覆盖到 col=6"


def test_max_chain_truncates():
    """max_chain=1 时只炸第一颗——这是 RULES.md 里明确记录的近似。"""
    fuse = torch.zeros((1, 7, 7), dtype=torch.int16)
    owner = torch.full((1, 7, 7), -1, dtype=torch.int8)
    wall = torch.zeros((1, 7, 7), dtype=torch.bool)
    fuse[0, 3, 1], owner[0, 3, 1] = 0, 0
    fuse[0, 3, 3], owner[0, 3, 3] = 9, 0
    _, trig = resolve_explosions(fuse, owner, wall, blast=2, max_chain=1)
    assert trig[0, 3, 1] and not trig[0, 3, 3]


def test_long_chain_fully_detonates():
    """长链（9+ 颗泡首尾相接）必须全部引爆，不能尾部漏爆。

    回归（用户实测 bug）：max_chain=8 限制连锁轮数 → 13×13 一行最长 13 颗
    泡需要 12 轮连锁，9+ 颗长链尾部漏爆（danger 预警预测覆盖但 resolve 实际
    没引爆）。修复：max_chain 8→16（SimConfig 默认）。resolve_explosions 用
    SimConfig 的 max_chain 直接验证整条链引爆。
    """
    from sim.config import SimConfig
    from sim.torch_sim import BatchedSim

    cfg = SimConfig(height=13, width=13, max_bombs=20)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.wall.zero_()
    sim.fuse.zero_()
    sim.owner.fill_(-1)
    # 一整行 13 颗泡（blast=1 首尾相接），最左 fuse=0 自然爆，其余满引信
    for i in range(13):
        sim.fuse[0, 6, 0 + i] = 0 if i == 0 else cfg.fuse
        sim.owner[0, 6, 0 + i] = 0
        sim.bomb_blast[0, 6, 0 + i] = 1
    a = torch.tensor([[[4, 0], [4, 0]]], dtype=torch.long)
    _, _, info = sim.step(a, auto_reset=False)
    exploded = int((sim.owner[0] == -1).sum())      # 全被清场（引爆后 owner→-1）
    assert exploded >= 13, f"13 颗长链应全引爆（清场），实际 {exploded}"


def test_danger_and_resolve_agree_on_long_chain():
    """danger_map 预警与 resolve_explosions 实际引爆轮数一致：长链不留尾巴。

    回归：danger 阶段 A 用 max_chain 轮、resolve 用 max_chain-1 次连锁 ——
    旧 max_chain=8 时 9+ 颗链 danger 预测覆盖但 resolve 截断漏爆。
    修复后（max_chain=16）两者都应覆盖整条链，且 danger>0 的泡必被引爆。
    """
    from sim.blast import danger_map

    fuse = torch.zeros((1, 7, 20), dtype=torch.int16)
    owner = torch.full((1, 7, 20), -1, dtype=torch.int8)
    blast = torch.zeros((1, 7, 20), dtype=torch.long)
    wall = torch.zeros((1, 7, 20), dtype=torch.bool)
    for i, f in enumerate((0,) + (10,) * (10 - 1)):     # 10 颗链
        fuse[0, 3, 2 + i] = f
        owner[0, 3, 2 + i] = 0
        blast[0, 3, 2 + i] = 1
    mc = 16
    dng = danger_map(fuse, wall, blast, 10, max_chain=mc)
    _, triggered = resolve_explosions(fuse, owner, wall, blast, mc)
    bomb_cells = fuse > 0
    assert int((bomb_cells & ~triggered).sum()) == 0, "长链必须全引爆"
    assert int(((bomb_cells & (dng > 0.01)) & ~triggered).sum()) == 0, \
        "danger 预测危险的泡必须实际引爆（两者轮数一致）"


def test_danger_map_chain_group_uniform():
    """连锁组危险度应与实际爆炸时刻一致：同一 tick 同爆的一组炮，危险度
    统一为组内最危险（先放的深、后放的浅是 bug —— 显示/训练读到的都是
    danger_map 同一份输出）。

    10 颗横向连炮（blast=1 首尾相接），先放的引信更短（更接近爆炸），
    后放的引信长。max_chain>1 时应收敛到同一危险值；孤立炮保持自身梯度。
    """
    from sim.blast import danger_map

    h, w, fmax = 11, 11, 10
    wall = torch.zeros((1, h, w), dtype=torch.bool)
    fuse = torch.zeros((1, h, w), dtype=torch.int16)
    blast_map = torch.zeros((1, h, w), dtype=torch.long)
    for col in range(1, 11):
        fuse[0, 5, col] = 11 - col              # 先放的引信短、后放的长
        blast_map[0, 5, col] = 1

    new = danger_map(fuse, wall, blast_map, fmax, max_chain=8)
    vals = [float(new[0, 5, col]) for col in range(1, 11)]
    assert max(vals) - min(vals) < 0.15, \
        f"连锁组应同时爆、同色深：{vals}（组内极差 {max(vals)-min(vals):.3f}）"

    old = danger_map(fuse, wall, blast_map, fmax, max_chain=1)
    oldv = [float(old[0, 5, col]) for col in range(1, 11)]
    assert max(oldv) - min(oldv) > 0.5, \
        "回归参照：max_chain=1（无连锁修正）应保留先深后浅的旧渐变"

    # 孤立炮（不相连）必须保持各自引信梯度（指数化后），不被"连锁修正"误伤
    fuse2 = torch.zeros((1, h, w), dtype=torch.int16)
    blast2 = torch.zeros((1, h, w), dtype=torch.long)
    fuse2[0, 3, 3], blast2[0, 3, 3] = 8, 1
    fuse2[0, 3, 8], blast2[0, 3, 8] = 2, 1
    d2 = danger_map(fuse2, wall, blast2, fmax, max_chain=8)
    assert abs(float(d2[0, 3, 3]) - (1 - (8 - 1) / fmax) ** 2) < 1e-3
    assert abs(float(d2[0, 3, 8]) - (1 - (2 - 1) / fmax) ** 2) < 1e-3


# ---------------- 连续移动 ----------------

def test_speed_is_cells_per_second():
    """速度 3 = 一秒走三格：跑满一秒的 tick 数，位移必须正好是 3 格。"""
    cfg = C(height=7, width=7, speed=3.0, tick_hz=15)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), cfg.tick_hz)
    assert abs(float(sim.pos[0, 0, 1]) - 3.5) < 1e-4, "匀速：15 tick × 0.2 = 3 格"
    assert abs(float(sim.pos[0, 0, 0]) - 3.5) < 1e-6, "四方向移动，不产生侧向漂移"


def test_idle_holds_position():
    sim = make()
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    hold(sim, act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), 5)
    assert torch.allclose(sim.pos[0, 0], torch.tensor([3.5, 3.5]))


def test_stops_flush_against_wall_and_never_enters():
    """撞墙是"贴住"而不是"停在上一格中心"——连续坐标的核心差别。"""
    cfg = C(height=7, width=7, radius=0.3)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.wall[0, 3, 5] = True
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 30)
    x = float(sim.pos[0, 0, 1])
    assert x < 5.0 - cfg.radius + 1e-3, "碰撞盒不能捅进墙里"
    assert x > 5.0 - cfg.radius - 0.01, "应该紧贴墙面，而不是停在 4.5"


def test_pressing_into_wall_is_a_legal_noop():
    """方向不是"非法动作"，按向墙只是走不动 —— 这也是掩码要标出来的那一项。"""
    sim = make()
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.wall[0, 3, 4] = True
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 5)   # 先贴住
    before = sim.pos[0, 0].clone()
    mmask, _ = sim.legal_mask()
    assert not bool(mmask[0, 0, MOVE_RIGHT]), "贴住之后这个方向该被掩掉"
    sim.step(act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert torch.allclose(sim.pos[0, 0], before), "按了也不动，但不报错"


def test_out_of_bounds_is_masked():
    sim = make()
    clear(sim)
    sim.pos[0, 0] = torch.tensor([0.5, 0.5])
    hold(sim, act((MOVE_UP, 0), (MOVE_IDLE, 0)), 5)
    mmask, _ = sim.legal_mask()
    assert not bool(mmask[0, 0, MOVE_UP]), "地图外等同于墙"
    assert bool(mmask[0, 0, MOVE_DOWN]) and bool(mmask[0, 0, MOVE_IDLE])


def test_positions_never_leave_map_bounds():
    """防御性边界夹紧：任何路径（含边界格滑动）都不能把角色推出地图。

    这是对"敌人穿出界面"bug 的回归测试 —— _resolve_axis 的 stop_pos 在
    边界格上可能把坐标算到界外，靠移动后的 clamp 兜底。
    """
    cfg = C(height=9, width=9, n_players=2)
    sim = make(cfg)
    clear(sim)
    gen = torch.Generator().manual_seed(3)
    dirs = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT)
    for _ in range(30):
        for corner in ((0.5, 0.5), (cfg.height - 0.5, cfg.width - 0.5)):
            sim.pos[0, 0] = torch.tensor(corner)   # 两个角色都挪到角落
            sim.pos[0, 1] = torch.tensor(corner)
            for d in dirs:                          # 朝每个方向狂按 25 tick
                hold(sim, act((d, d), (d, d)), 25)
        # 乱跑一会，验证任意时刻都在界内
        for _ in range(40):
            mv = torch.multinomial(torch.ones(2, N_MOVES), 1,
                                   generator=gen).view(1, 2)
            bm = torch.zeros(1, 2, dtype=torch.long)
            sim.step(torch.stack([mv, bm], dim=-1), auto_reset=False)
        y, x = sim.pos[0, :, 0], sim.pos[0, :, 1]
        lo, hi = cfg.radius, cfg.height - cfg.radius   # 碰撞盒最贴边但不出界
        assert bool(((y >= lo - 1e-5) & (y <= hi + 1e-5)
                     & (x >= lo - 1e-5) & (x <= hi + 1e-5)).all()), \
            f"角色越界: y={y.tolist()} x={x.tolist()}"


def test_players_pass_through_each_other():
    """连续坐标下不做角色间碰撞：格子版那套 O(P²) 换位/同格消解直接删掉了。"""
    sim = make()
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 2.5])
    sim.pos[0, 1] = torch.tensor([3.5, 4.5])
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_LEFT, 0)), 8)
    assert float(sim.pos[0, 0, 1]) > float(sim.pos[0, 1, 1]), "应该已经互相穿过"


# ---------------- 放泡 ----------------

def test_can_move_and_place_in_the_same_tick():
    """因子化动作空间存在的理由：边跑边放。扁平 6 动作做不到这件事。"""
    cfg = C(height=7, width=7, fuse=10)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.owner[0, 3, 3]) == 0, "泡落在按下那一刻的脚下格"
    assert float(sim.pos[0, 0, 1]) > 3.5, "同一 tick 人已经跑起来了"


def test_can_walk_off_own_bomb_but_not_back_in():
    cfg = C(height=7, width=7, fuse=60)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 10)
    assert int(sim.pos[0, 0, 1].floor()) > 3, "脚下的泡不挡自己出去"
    hold(sim, act((MOVE_LEFT, 0), (MOVE_IDLE, 0)), 20)
    assert float(sim.pos[0, 0, 1]) > 4.0 - cfg.radius - 0.01, "出去之后就回不来了"


def test_bomb_budget_and_no_stacking():
    cfg = C(height=7, width=7, max_bombs=2, fuse=60)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])

    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    assert sim.fuse[0, 3, 0] == cfg.fuse and sim.owner[0, 3, 0] == 0
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 5)      # 走到 col=1
    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.owner[0, 3, 1]) == 0, "第二颗在额度内"
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 5)      # 走到 col=2
    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.owner[0, 3, 2]) == -1, "第三颗超出 max_bombs"


def test_bomb_mask_matches_budget():
    cfg = C(height=7, width=7, max_bombs=1, fuse=60)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    assert bool(sim.legal_mask()[1][0, 0, 1])
    sim.step(act((MOVE_RIGHT, 1), (MOVE_IDLE, 0)), auto_reset=False)
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 5)
    assert not bool(sim.legal_mask()[1][0, 0, 1]), "额度用满，掩码要关掉放泡"


# ---------------- 死亡与奖励 ----------------

def test_owner_is_not_immune_to_own_bomb():
    cfg = C(height=7, width=7, fuse=1, max_hp=1)   # max_hp=1 ≡ 旧版一碰就死
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0
    reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert not bool(sim.alive[0, 0]), "自己的泡照样炸死自己"
    assert reward[0, 0] < -0.9
    assert bool(done[0]) and reward[0, 1] > 0.9, "对手独活得 +1"


def test_simultaneous_death_is_a_draw():
    cfg = C(height=7, width=7, fuse=1, blast=2, max_hp=1)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 2.5])
    sim.pos[0, 1] = torch.tensor([3.5, 4.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0
    # 一 tick 只跑 0.2 格，中心格没变，两人都还在 blast=2 的十字里 → 同 tick 双亡
    reward, done, _ = sim.step(act((MOVE_LEFT, 0), (MOVE_RIGHT, 0)), auto_reset=False)
    assert int(sim.alive[0].sum()) == 0
    assert bool(done[0])
    # 双亡 = 平局：1v1 同时死，双方各造成 1 伤害也各承受 1 伤害，hit_reward 净零，
    # 只剩步罚 —— 没有 ±win_bonus。
    assert torch.allclose(reward[0], torch.full((2,), -cfg.step_penalty),
                          atol=1e-6), f"双亡应是平局: {reward[0].tolist()}"


def test_dealing_damage_gives_positive_reward():
    """打中对手 +hit_reward / 自己掉血 -hit_reward（1v1 对方掉血 = 我造成的伤害）。

    这是"学会躲开自己放炮"的奖励基础：站在自己泡的火焰里 1 tick 就该有负分。
    """
    cfg = C(height=7, width=7, fuse=1, blast=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])          # 玩家离得远远的
    sim.pos[0, 1] = torch.tensor([3.5, 4.5])          # AI 站在 (3,4)，邻格 (3,3) 放泡
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0      # 玩家 0 的泡炸到 AI
    reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.hp[0, 1]) == 4, "AI 应掉 1 血"
    # 玩家 0：打中 +0.5；AI：掉血 -0.5。双方都扣步罚。
    assert torch.allclose(reward[0, 0], torch.tensor(cfg.hit_reward - cfg.step_penalty),
                          atol=1e-5)
    assert torch.allclose(reward[0, 1], torch.tensor(-cfg.hit_reward - cfg.step_penalty),
                          atol=1e-5)


def test_timeout_higher_hp_wins():
    """超时全员存活 → 血多者胜。默认 win_hp_scaled=True：终局分按剩余血量差给。

    win_bonus=8, max_hp=5, 血量 3 vs 1 → 胜者 +8×(3−1)/5=+3.2，败者 −3.2。
    血平超时 = 平局 0（只有步罚）。
    """
    cfg = C(height=7, width=7, max_steps=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    sim.hp[0, 0] = 3
    sim.hp[0, 1] = 1
    reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(done[0]) and int(sim.alive[0].sum()) == 2
    hp_gap = 2.0
    scaled = cfg.win_bonus * hp_gap / cfg.max_hp        # +3.2
    assert torch.allclose(reward[0, 0],
                          torch.tensor(scaled - cfg.step_penalty), atol=1e-5)
    assert torch.allclose(reward[0, 1],
                          torch.tensor(-scaled - cfg.step_penalty), atol=1e-5)


def test_timeout_higher_hp_annealed():
    """超时全员存活（timeout_draw=True 默认）→ 血多者胜 × 退火 α。

    新语义：超时血量差奖励 × _explore_coef（默认 1.0，无退火时满额）。
    win_hp_scaled=False + 死亡终局才给固定 win_bonus —— 超时不是死亡，
    给血量差比例分（8/5×血量差×α）。血量 3 vs 1 → +8×2/5×1.0 = +3.2。
    """
    cfg = C(height=7, width=7, max_steps=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    sim.hp[0, 0] = 3
    sim.hp[0, 1] = 1
    reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(done[0]) and int(sim.alive[0].sum()) == 2
    hp_gap = 2.0
    scaled = cfg.win_bonus * hp_gap / cfg.max_hp * 1.0     # ×α（explore_coef 默认 1.0）
    assert torch.allclose(reward[0, 0],
                          torch.tensor(scaled - cfg.step_penalty), atol=1e-5)
    assert torch.allclose(reward[0, 1],
                          torch.tensor(-scaled - cfg.step_penalty), atol=1e-5)


def test_death_fixed_win_bonus():
    """死亡终局（win_hp_scaled=False 默认）→ 击杀给**固定** ±win_bonus，不看血量差。

    用户定：对手 hp=0（击杀）才是奖励，固定值 —— 残血险胜和满血击杀同分。
    """
    cfg = C(height=7, width=7, fuse=1, blast=1, max_hp=5)
    for win_hp in (5, 1, 3):
        sim = make(cfg)
        clear(sim)
        sim.pos[0, 0] = torch.tensor([6.5, 6.5])          # 幸存者：离远
        sim.pos[0, 1] = torch.tensor([3.5, 4.5])          # 死者站在邻格
        sim.hp[0, 0] = win_hp
        sim.hp[0, 1] = 1
        sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0      # 玩家 0 的泡炸死玩家 1
        reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)),
                                   auto_reset=False)
        assert bool(done[0]) and int(sim.alive[0, 0]) == 1, "玩家 0 应幸存"
        assert int(sim.alive[0, 1]) == 0, "玩家 1 应死"
        got = reward[0, 0]
        expect = cfg.win_bonus + cfg.hit_reward - cfg.step_penalty
        assert torch.allclose(got, torch.tensor(expect), atol=1e-5), \
            f"win_hp={win_hp}: 击杀固定 {cfg.win_bonus}（应 {expect}），实际 {got}"


def test_timeout_even_hp_is_draw():
    cfg = C(height=7, width=7, max_steps=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    reward, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(done[0]) and int(sim.alive[0].sum()) == 2
    assert torch.allclose(reward[0], torch.full((2,), -cfg.step_penalty), atol=1e-5), \
        "血平超时 = 平局，只有步罚"


def test_info_winner_death_and_timeout():
    """info['winner'] 与 PPO._tally 的判据一致：

    - 死亡终局（n_alive==1）→ 存活者 winner=True；
    - 超时全员存活（timeout_draw=True 默认）→ 全 False（平局，reward 血差×退火
      但 ELO/tally 不计胜负 —— 超时不算赢）；
    - 同时死光 → 全 False。
    """
    # (a) 死亡终局：0 号被邻格泡炸死，1 号存活
    cfg = C(height=7, width=7, fuse=1, blast=1, max_hp=5, invuln_ticks=0)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 4.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 1
    for _ in range(cfg.max_hp):                # 每 tick 扣 1 血，烧够 max_hp 次
        _, done, info = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
        sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 1   # 重新种泡
        if bool(done[0]):
            break
    assert bool(done[0]) and bool(info["winner"][0, 1])
    assert not bool(info["winner"][0, 0])

    # (b) 超时全员存活（timeout_draw=True 默认）→ winner 全 False（平局口径）
    cfg = C(height=7, width=7, max_steps=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    sim.hp[0, 0], sim.hp[0, 1] = 3, 1
    _, done, info = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(done[0])
    assert not bool(info["winner"].any()), \
        "timeout_draw=True：超时不进 winner（ELO/tally 计平局），reward 才给血差×退火"

    # (c) 超时血平 → 平局
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    _, done, info = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(done[0]) and not bool(info["winner"].any())

    # (d) 同时死光 → 平局（双亡，1v1 各炸各的）
    cfg = C(height=7, width=7, fuse=1, blast=2, max_hp=5, invuln_ticks=0)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    sim.pos[0, 1] = torch.tensor([5.5, 5.5])
    for _ in range(cfg.max_hp):
        sim.fuse[0, 1, 1], sim.owner[0, 1, 1] = 1, 0   # 0 号脚下
        sim.fuse[0, 5, 5], sim.owner[0, 5, 5] = 1, 1   # 1 号脚下
        _, done, info = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
        if bool(done[0]):
            break
    assert bool(done[0]) and not bool(info["winner"].any())


def test_danger_zone_standing_penalty():
    """站在被泡泡爆炸范围覆盖的格每 tick 扣分，大小 × danger 值（越接近爆越疼）。"""
    cfg = C(height=7, width=7, max_hp=5)
    sim = make(cfg)
    clear(sim)
    # 玩家站在 (3,4)，(3,3) 有一泡 fuse=3 → (3,4) 是危险区。
    # 奖励段 fuse 已递减为 2：danger = (1 - (2-1)/FUSE)^exp（exp=2）
    sim.pos[0, 0] = torch.tensor([3.5, 4.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 3, 1
    reward, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    danger = (1.0 - 1.0 / cfg.fuse) ** 2     # fuse 3→2 后：(fuse-1)/FUSE = 1/45，再平方
    expected = -cfg.step_penalty - cfg.danger_penalty * danger
    assert torch.allclose(reward[0, 0], torch.tensor(expected), atol=1e-5), \
        f"应扣 danger 罚: {reward[0, 0].item():.4f} vs {expected:.4f}"


def test_passivity_penalty_after_idle():
    """连续 passivity_ticks tick 没放泡，之后每 tick 扣分；放一次立即清零。"""
    cfg = C(height=7, width=7, max_hp=5, passivity_ticks=5,
            place_dist_cooldown=0)   # 近身定位关掉：这个用例只钉被动罚
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    for _ in range(cfg.passivity_ticks):          # 5 tick 不放
        sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.since_bomb[0, 0]) == cfg.passivity_ticks
    reward, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    expected = -cfg.step_penalty - cfg.passivity_penalty
    assert torch.allclose(reward[0, 0], torch.tensor(expected), atol=1e-5)
    sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)), auto_reset=False)   # 放一颗
    assert int(sim.since_bomb[0, 0]) == 0, "放泡成功后清零"


def test_no_passivity_penalty_when_bombs_full():
    """放满（在场泡数达 max_bombs）时不扣被动罚 —— 只有"还有预算却摆烂"才被罚。"""
    cfg = C(height=7, width=7, max_hp=5, max_bombs=1, passivity_ticks=3,
            place_dist_cooldown=0)   # 近身定位关掉：这个用例只钉"放满不罚"
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)), auto_reset=False)   # 放满 1 颗
    assert int(sim.since_bomb[0, 0]) == 0
    # 先离开自己泡的爆炸范围（斜角格不在十字火内），避免叠加危险区罚
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])
    for _ in range(cfg.passivity_ticks + 2):          # 超过阈值也不放
        reward, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
        # 放满了：只扣步罚，不扣被动罚
        assert torch.allclose(reward[0, 0], torch.tensor(-cfg.step_penalty), atol=1e-5), \
            f"放满不该扣被动罚: {reward[0, 0].item()}"


def test_midline_spawn_positions():
    """出生点在中线横向排开（2 人：地图中间、相隔 4 格），不再是四角。"""
    cfg = C(height=13, width=13, n_players=2)
    pos = cfg.spawn_pos()
    assert len(pos) == 2
    y0, x0 = pos[0]
    y1, x1 = pos[1]
    assert abs(y0 - 6.5) < 1e-6 and abs(y1 - 6.5) < 1e-6, "都在中线"
    assert abs(x0 - 4.5) < 1e-6 and abs(x1 - 8.5) < 1e-6, "横向排开相隔 4 格"


def test_hp_decrements_and_dies_at_zero():
    """着火每 tick 扣 1 血：烧 1 次 5→4 且存活，连烧 5 次才死。

    火焰是单 tick 闪光，所以每个 tick 前手工种一颗 fuse=1 的泡在邻格，
    该 tick 内必然爆炸并覆盖玩家 —— 不依赖多颗泡的连锁时序。
    """
    cfg = C(height=7, width=7, fuse=1, blast=1, max_hp=5, invuln_ticks=0)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 4.5])   # 玩家中心格 (3,4)
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    for i in range(5):
        sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 1   # 邻格 (3,3)，blast=1 覆盖 (3,4)
        before = int(sim.hp[0, 0])
        sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
        if i < 4:
            assert int(sim.hp[0, 0]) == before - 1, f"第 {i+1} tick 应扣 1 血"
            assert bool(sim.alive[0, 0]), "血没归 0 不该死"
        else:
            assert int(sim.hp[0, 0]) == 0 and not bool(sim.alive[0, 0]), \
                "第 5 tick 血归 0 才死"


def test_hp_resets_to_max_on_reset():
    cfg = C(height=7, width=7, fuse=1, max_hp=5)
    sim = make(cfg)
    clear(sim)
    sim.hp[0, 0] = 1                                   # 残血 1：再烧一次就死
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0
    _, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=True)
    assert bool(done[0]) and bool(sim.alive[0].all())
    assert int(sim.hp[0, 0]) == cfg.max_hp, "auto_reset 后血应回满"


def test_death_is_judged_by_center_cell():
    """判定用中心格而不是整个碰撞盒：半格之内的越界不算被烧到。"""
    cfg = C(height=7, width=7, fuse=1, blast=1)
    sim = make(cfg)
    clear(sim)
    # 碰撞盒压住 col=3（会着火），但中心格是 col=4
    sim.pos[0, 0] = torch.tensor([3.5, 4.2])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 2], sim.owner[0, 3, 2] = 1, 1
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(sim.alive[0, 0])


def test_exploded_cell_is_cleared_and_budget_returned():
    cfg = C(height=7, width=7, fuse=1, max_bombs=1)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([0.5, 0.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.fuse[0, 3, 3]) == 0 and int(sim.owner[0, 3, 3]) == -1
    assert bool(sim.legal_mask()[1][0, 0, 1]), "泡泡爆掉后额度归还"


def test_timeout_is_done_with_no_winner():
    cfg = C(height=7, width=7, max_steps=3)
    sim = make(cfg)
    clear(sim)
    idle = act((MOVE_IDLE, 0), (MOVE_IDLE, 0))
    for _ in range(2):
        _, done, _ = sim.step(idle, auto_reset=False)
        assert not bool(done[0])
    reward, done, _ = sim.step(idle, auto_reset=False)
    assert bool(done[0]) and int(sim.alive[0].sum()) == 2
    assert torch.allclose(reward[0], torch.full((2,), -cfg.step_penalty))


# ---------------- 掩码不会全关 ----------------

def test_dead_player_mask_is_all_true():
    sim = make()
    clear(sim)
    sim.alive[0, 1] = False
    mmask, bmask = sim.legal_mask()
    assert bool(mmask[0, 1].all()) and bool(bmask[0, 1].all())


def test_boxed_in_player_still_has_idle_and_no_bomb():
    """四面被封 + 脚下有泡：放泡头只剩"不放"，方向头至少还有 IDLE。

    格子版里为此写了个"无路可走就整行放开"的兜底分支；因子化动作空间下
    IDLE 和 bomb=0 天然永远合法，那个分支不存在了。
    注意四个方向**不一定**都被掩掉：radius=0.3 时格内还有 0.2 的余量可以挪，
    只有真正贴住了那一侧才会被标成不可走。
    """
    sim = make()
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    for r, c in ((2, 3), (4, 3), (3, 2), (3, 4)):
        sim.wall[0, r, c] = True
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 60, 0   # 引信调长，免得测试期间炸了
    hold(sim, act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), 5)   # 贴到右墙
    mmask, bmask = sim.legal_mask()
    assert not bool(mmask[0, 0, MOVE_RIGHT]), "贴住的方向要掩掉"
    assert bool(mmask[0, 0, MOVE_IDLE]), "IDLE 永远合法"
    assert bool(bmask[0, 0, 0]) and not bool(bmask[0, 0, 1])


# ---------------- 自动重置 ----------------

# ---------------- 共享观测的通道语义 ----------------

def test_shared_obs_shape_and_dtype():
    """观测是 env 级共享的一份 (2P+3+obs_extra, H, W)，默认 fp16。不再每个角色一份。"""
    cfg = C(height=7, width=7, n_players=3)
    sim = make(cfg, n=4)
    obs = sim.observe()
    assert obs.shape == (4,) + cfg.obs_shape
    assert obs.dtype == torch.float16, "cfg.obs_fp16 默认开"
    assert torch.isfinite(obs.float()).all()


def test_obs_channels_are_per_player_and_env_level():
    """0..P-1 是各人位置（splat 质量 1），P..2P-1 是各人名下泡泡引信，尾 3 是环境。"""
    cfg = C(height=7, width=7, n_players=2)
    sim = make(cfg)
    clear(sim)
    p = cfg.n_players
    sim.pos[0, 0] = torch.tensor([2.5, 3.5])       # 正落格心 ⇒ splat 只点一格
    sim.pos[0, 1] = torch.tensor([5.5, 1.5])
    sim.wall[0, 0, 6] = True
    sim.fuse[0, 2, 3], sim.owner[0, 2, 3] = cfg.fuse, 0
    sim.fuse[0, 5, 1], sim.owner[0, 5, 1] = cfg.fuse, 1
    sim.t[0] = 30
    obs = sim.observe()[0].float()

    for i, (r, c) in ((0, (2, 3)), (1, (5, 1))):
        assert obs[i, r, c] == 1.0, f"玩家 {i} 位置通道应在自己格上取 1"
        assert abs(float(obs[i].sum()) - 1.0) < 1e-3, "splat 总质量必须是 1"
        assert obs[p + i, r, c] == 1.0, f"玩家 {i} 名下泡泡引信满格应是 1"
        # 泡泡引信按 owner 分通道，不合并 —— 否则视角就不是纯置换了
        other = 1 - i
        assert obs[p + other, r, c] == 0.0, "别人的泡泡不能出现在我的引信通道里"

    assert obs[2 * p, 0, 6] == 1.0 and float(obs[2 * p].sum()) == 1.0, "墙通道"
    assert float(obs[2 * p + 1].max()) > 0.0, "有泡泡时危险图不能全 0"
    prog = obs[2 * p + 2]
    assert torch.allclose(prog, torch.full_like(prog, 30.0 / cfg.max_steps), atol=1e-3)


def test_dead_player_has_no_position_mass():
    cfg = C(height=7, width=7, n_players=2)
    sim = make(cfg)
    clear(sim)
    sim.alive[0, 1] = False
    obs = sim.observe()[0].float()
    assert float(obs[1].sum()) == 0.0, "死人不该出现在位置通道里"
    assert float(obs[0].sum()) > 0.0


def test_auto_reset_restores_spawns():
    cfg = C(height=7, width=7, fuse=1, max_hp=1)
    sim = make(cfg)
    clear(sim)
    sim.pos[0, 0] = torch.tensor([3.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    sim.fuse[0, 3, 3], sim.owner[0, 3, 3] = 1, 0
    _, done, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=True)
    assert bool(done[0])
    assert bool(sim.alive[0].all()) and int(sim.t[0]) == 0
    assert tuple(sim.pos[0, 0].tolist()) == cfg.spawn_pos()[0]


def test_res_assets_load_and_crop():
    """RES 素材完整性：角色 4×4、炸弹 6 帧、爆炸中心+四臂按 blast 切片。"""
    import os
    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((100, 100))    # 无头环境也要有 surface 才能加载图片
    try:
        from play.res import MOVE_TO_SPRITE_ROW, Res
        from sim.config import MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP

        res = Res(cell=60, blast=2)   # 画布 1.5 倍：素材原生 40px/格 → 60px/格
        assert len(res.players) == 4 and all(len(r) == 4 for r in res.players)
        assert len(res.player_ai) == 4 and all(len(r) == 4 for r in res.player_ai)
        # 角色 85x85 原尺寸 × 1.5 = 128（不裁切）
        for r in res.players:
            for fr in r:
                assert fr.get_size() == (128, 128)
        # 泡泡：单张固定图（呼吸靠渲染时垂直浮动）。素材原生 40px/格，按画布
        # 缩放 1.5 倍到格子大小（60×60，和爆炸中心一致）—— 不缩小，
        # 摆放与人物同款（水平中心对格心、底边对格底线）。
        assert len(res.bombs) == 1 and res.bombs[0].get_size() == (60, 60)
        # 爆炸中心 40×1.5 = 60
        assert res.explo_center.get_size() == (60, 60)
        assert len(res.explo_arms) == 4
        # 行序（用户确认）：下、左、右、上
        assert MOVE_TO_SPRITE_ROW[MOVE_DOWN] == 0
        assert MOVE_TO_SPRITE_ROW[MOVE_LEFT] == 1
        assert MOVE_TO_SPRITE_ROW[MOVE_RIGHT] == 2
        assert MOVE_TO_SPRITE_ROW[MOVE_UP] == 3
        # 臂按 blast=2 从"炸弹边缘端"切 2 格(80px)再 ×1.5 = 120
        assert res.explo_arms[(0, 1)].get_size() == (120, 60)    # 向右：保留最右端
        assert res.explo_arms[(0, -1)].get_size() == (120, 60)   # 向左：保留最左端
        assert res.explo_arms[(1, 0)].get_size() == (60, 120)    # 向下：保留最下端
        assert res.explo_arms[(-1, 0)].get_size() == (60, 120)   # 向上：保留最上端
        assert os.path.exists(os.path.join("res", "bg1.png"))
    finally:
        pygame.quit()


def test_random_rollout_stays_consistent():
    """跑一批随机策略：状态不变量在整个 rollout 里都要成立。"""
    cfg = C(height=9, width=9, n_players=3, wall_density=0.7)
    sim = BatchedSim(cfg, 32, seed=1)
    gen = torch.Generator().manual_seed(7)
    for _ in range(200):
        mmask, bmask = sim.legal_mask()
        mv = torch.multinomial(mmask.float().view(-1, N_MOVES), 1, generator=gen)
        bm = torch.multinomial(bmask.float().view(-1, N_BOMB), 1, generator=gen)
        acts = torch.stack([mv.view(32, 3), bm.view(32, 3)], dim=-1)
        sim.step(acts)
        assert not bool((sim.fuse < 0).any())
        assert not bool(((sim.owner >= 0) & (sim.fuse <= 0)).any()), "owner 与 fuse 必须同步"
        assert not bool((sim.wall & (sim.fuse > 0)).any()), "泡泡不能落在墙里"
        y, x = sim.pos[..., 0], sim.pos[..., 1]
        assert bool(((y >= 0) & (y < cfg.height) & (x >= 0) & (x < cfg.width)).all())
        # 碰撞盒不能压进墙里：中心格必须是可通行的
        cell = sim.pos.floor().long()
        flat = (cell[..., 0] * cfg.width + cell[..., 1])
        wall_hit = sim.wall.view(32, -1).gather(1, flat)
        assert not bool(wall_hit.any()), "角色中心不能落在墙格里"


# ---------------- corridor 模式：可炸墙 + 顶部永久墙 + 时间成长 ----------------

def test_corridor_map_layout():
    """corridor 地图：左右各 4 列可炸墙（brick）、顶部 4 行永久墙、
    出生点在空旷区中心（行 8.5、列 5.5/8.5，13x13）。"""
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4, speed=3.0)
    sim = BatchedSim(cfg, 1, seed=0)
    # 左右各 4 列 brick（列 0-3、9-12），顶部 4 行永久墙（行 0-3）
    assert bool(sim.brick[0, 5, 0]) and bool(sim.brick[0, 5, 12])
    assert not bool(sim.brick[0, 5, 4]) and not bool(sim.brick[0, 5, 8]), "中间 5 列可通行"
    assert not bool(sim.brick[0, 2, 0]), "顶部行是永久墙不是 brick"
    assert bool(sim.wall[0, 2, 6]), "顶部 4 行永久墙"
    assert not bool(sim.wall[0, 8, 6]), "空旷区无永久墙"
    # 出生点：空旷区（行 4..12 × 列 4..8）中心
    p0, p1 = cfg.spawn_pos()
    assert abs(p0[0] - 8.5) < 1e-6 and abs(p1[0] - 8.5) < 1e-6, "空旷区中线"
    assert abs(p0[1] - 5.5) < 1e-6 and abs(p1[1] - 8.5) < 1e-6, "中间 5 列内均分，相隔 3 列"


def test_brick_blocks_flame_and_is_destroyed():
    """brick 挡火（不穿过），被覆盖即摧毁；永久墙挡火且不摧毁。

    corridor 13 宽、corridor_width=5：可通行列 4..8，brick 列 0..3 与 9..12。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4, speed=3.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.fuse.zero_()
    sim.owner.fill_(-1)
    # 手工种一颗威力 2 的泡在 (8,5)；左侧 brick 从列 3 起（列 4 是可通行区）
    sim.pos[0, 0] = torch.tensor([8.5, 4.5])
    sim.pos[0, 1] = torch.tensor([8.5, 8.5])
    sim.fuse[0, 8, 5], sim.owner[0, 8, 5] = 1, 0
    sim.bomb_blast[0, 8, 5] = 2
    assert not bool(sim.brick[0, 8, 4]), "(8,4) 在可通行区，不是 brick"
    assert bool(sim.brick[0, 8, 3]), "(8,3) 是左侧 brick（列 0..3）"
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert not bool(sim.brick[0, 8, 3]), "brick 被火焰覆盖即摧毁"
    assert not bool(sim.fuse[0, 8, 5]) and int(sim.owner[0, 8, 5]) == -1, "泡已爆并清场"
    # 顶部永久墙原样保留
    assert bool(sim.wall[0, 2, 6])


def test_brick_blocks_flame_propagation():
    """brick 挡火：爆源隔着一层 brick 的后方不被烧到。"""
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4, speed=3.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.fuse.zero_()
    sim.owner.fill_(-1)
    sim.pos[0, 0] = torch.tensor([8.5, 4.5])
    sim.pos[0, 1] = torch.tensor([8.5, 8.5])
    # 爆源 (8,5) 威力 1：r=1 到 (8,4)（可通行，被烧），r=2 到 (8,3) 是 brick ——
    # 威力 1 根本到不了 brick，测不出"挡火"。改用威力 2：烧到 (8,3) brick，
    # 但 brick 挡火不穿到 (8,2)（brick 列 0..3 内更左）。
    sim.fuse[0, 8, 5], sim.owner[0, 8, 5] = 1, 0
    sim.bomb_blast[0, 8, 5] = 2
    assert bool(sim.brick[0, 8, 3]) and bool(sim.brick[0, 8, 2])
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert not bool(sim.brick[0, 8, 3]), "brick (8,3) 被烧毁"
    assert bool(sim.brick[0, 8, 2]), "brick (8,2) 挡住火，未被烧到"


def test_growth_crate_pickup():
    """宝箱拾取成长：砖被炸掉 → 变宝箱 → 玩家走到宝箱格开箱（爆率 100% 便于断言）。

    - 开局能力 = start（2泡/2威/1.0速）
    - 炸掉 brick → crate 出现；玩家走到 crate 格 → 三属性之一 +1
    - 玩家 1 没踩宝箱不成长；宝箱开过后消失
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, growth_crate_prob=1.0)
    sim = BatchedSim(cfg, 1, seed=0)
    assert int(sim.bombs_cap[0, 0]) == cfg.growth_bombs_start
    assert int(sim.blast_cap[0, 0]) == cfg.growth_blast_start
    assert abs(float(sim.spd_g[0, 0]) - 1.0) < 1e-6

    # 玩家 0 站 (8.5,5.5)，放威力 3 的泡在 (8,5)，西向炸掉 (8,3) brick → 变 crate
    sim.pos[0, 0] = torch.tensor([10.5, 6.5])   # 远离爆炸
    sim.pos[0, 1] = torch.tensor([10.5, 1.5])
    sim.fuse[0, 8, 5], sim.owner[0, 8, 5] = 1, 0
    sim.bomb_blast[0, 8, 5] = 3
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert not bool(sim.brick[0, 8, 3]), "brick (8,3) 被炸掉"
    assert bool(sim.crate[0, 8, 3]), "炸掉的砖变宝箱"
    # 玩家 0 走到 (8.5,3.5) 踩宝箱 → 开箱成长一次
    before = (int(sim.bombs_cap[0, 0]), int(sim.blast_cap[0, 0]),
              float(sim.spd_g[0, 0]))
    sim.pos[0, 0] = torch.tensor([8.5, 3.5])
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    after = (int(sim.bombs_cap[0, 0]), int(sim.blast_cap[0, 0]),
             float(sim.spd_g[0, 0]))
    grew = (after[0] - before[0]) + (after[1] - before[1]) + \
           int(round((after[2] - before[2]) / cfg.growth_speed_step))
    assert grew == 1, f"踩宝箱应恰好成长 1 次: {before} → {after}"
    assert not bool(sim.crate[0, 8, 3]), "宝箱开过后消失"
    # 玩家 1 没踩宝箱，不成长
    assert int(sim.bombs_cap[0, 1]) == cfg.growth_bombs_start
    assert int(sim.blast_cap[0, 1]) == cfg.growth_blast_start


def test_growth_clamps_at_max():
    """成长上限 clamp：泡数/威力/速度到 max 后不再涨。"""
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, growth_crate_prob=1.0,
                    growth_bombs_max=3, growth_blast_max=3, growth_speed_max=1.2,
                    growth_speed_step=0.1)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.bombs_cap[0, 0] = 3            # 已在 max
    sim.blast_cap[0, 0] = 3
    sim.spd_g[0, 0] = 1.2
    sim.pos[0, 0] = torch.tensor([10.5, 6.5])
    sim.pos[0, 1] = torch.tensor([10.5, 1.5])
    sim.fuse[0, 8, 5], sim.owner[0, 8, 5] = 1, 0
    sim.bomb_blast[0, 8, 5] = 3
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    # 制造宝箱并走到上面开箱（已满级 → clamp 不变）
    sim.crate[0, 8, 3] = True
    sim.pos[0, 0] = torch.tensor([8.5, 3.5])
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.bombs_cap[0, 0]) == 3 and int(sim.blast_cap[0, 0]) == 3
    assert abs(float(sim.spd_g[0, 0]) - 1.2) < 1e-6, "成长 clamp 到上限"


def test_high_speed_never_tunnels_through_wall():
    """高速撞墙不穿模：前缘区间枚举。

    角色从 (3.5, 0.5) 向左，碰撞盒前缘扫过 col 0..1，应贴在 col 0 右壁
    （x≈0.3），不会穿到 col 0 里；向右则正常匀速。
    """
    # 15Hz：speed 4.5 → 0.3 格/tick，5 tick = 1.5 格
    cfg = C(height=7, width=7, speed=4.5)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])
    for _ in range(5):
        sim.step(act((MOVE_LEFT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert float(sim.pos[0, 0, 1]) >= 0.299, "高速向左不得穿墙"
    sim2 = BatchedSim(cfg, 1, seed=0)
    sim2.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim2.pos[0, 1] = torch.tensor([6.5, 6.5])
    for _ in range(5):
        sim2.step(act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert abs(float(sim2.pos[0, 0, 1]) - 2.0) < 1e-4, "高速向右正常移动 0.5+5×0.3"
    # 10Hz：0.45 格/tick（> 1-2r=0.4），前缘跨 2 格边界，同样不穿
    cfg10 = SimConfig(height=7, width=7, speed=4.5, tick_hz=10)
    sim3 = BatchedSim(cfg10, 1, seed=0)
    sim3.pos[0, 0] = torch.tensor([3.5, 0.5])
    sim3.pos[0, 1] = torch.tensor([6.5, 6.5])
    for _ in range(3):
        sim3.step(act((MOVE_LEFT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert float(sim3.pos[0, 0, 1]) >= 0.299, "10Hz 0.45 格/tick 向左也不得穿墙"


def test_brick_reward_on_step():
    """宝箱开箱奖励：**踩箱即给** +brick_reward，与中奖概率无关。

    踩到宝箱 = 必得收集奖励（不管 prob 是多少、成长是否触发）；宝箱开过即消失。
    prob=1.0 时踩箱 = 1 次成长 + 得分；prob=0.0 时踩箱 = 0 次成长、**同样得分**。
    """
    for prob in (1.0, 0.0):
        cfg = SimConfig(height=13, width=13, n_players=2,
                        map_mode="corridor", corridor_width=5, top_wall_rows=4,
                        max_steps=1800, speed=3.0, growth_crate_prob=prob)
        sim = BatchedSim(cfg, 1, seed=0)
        sim.pos[0, 0] = torch.tensor([10.5, 6.5])   # 远离爆炸
        sim.pos[0, 1] = torch.tensor([10.5, 1.5])
        sim.fuse[0, 8, 5], sim.owner[0, 8, 5] = 1, 0
        sim.bomb_blast[0, 8, 5] = 3
        sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
        assert bool(sim.crate[0, 8, 3]), "砖炸掉变宝箱"
        sim.pos[0, 0] = torch.tensor([8.5, 3.5])    # 走到宝箱格
        reward, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)),
                                auto_reset=False)
        # 踩箱必得 brick_reward（只有步罚 + 踩箱奖励；prob=0 时同样得分）
        assert abs(float(reward[0, 0])
                   - (cfg.brick_reward - cfg.step_penalty)) < 1e-5, \
            f"prob={prob} 踩箱应 +brick_reward: {reward[0, 0].item()}"
        assert not bool(sim.crate[0, 8, 3]), "宝箱开过后消失"


def test_place_bomb_no_bonus():
    """放泡**不再直接给奖励**（place_bonus 已移除）。

    实测放泡直接奖励会诱导 corridor 里"横向放炮刷宝箱"成为最优路径，
    主次颠倒。放泡本身应该只由"命中/终局/被动罚"间接驱动。
    关掉近身定位分（place_dist_reward=0）排除门槛奖励的干扰；排除危险罚
    （danger_penalty=0）避免泡中心格的干扰；只有步罚。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_dist_reward=0.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([6.5, 5.5])        # 空旷区，别贴墙
    sim.pos[0, 1] = torch.tensor([6.5, 8.5])
    reward, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                            auto_reset=False)
    assert abs(float(reward[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        f"放泡成功不应有额外奖励，只有步罚: {reward[0, 0].item()}"
    assert abs(float(reward[0, 1]) - (-cfg.step_penalty)) < 1e-5, \
        "没放泡只有步罚"


def test_place_dist_reward_near_enemy():
    """近身定位分（十字辐射外）：炮距敌人越近分越高，带半径 + 冷却门槛。

    敌人不在新泡的辐射十字上（off-cross）→ 距离分；距离 < 半径才给。
    开局 since_bomb = place_dist_cooldown（冷却已完成）→ 第一泡就够格。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_dist_reward=0.01, place_dist_radius=4.0,
                    place_dist_cooldown=15, place_cover_reward=0.0,
                    place_chain_reward=0.0)
    sim = BatchedSim(cfg, 1, seed=0)
    # 玩家 0 在 (6.5,6.5) 放泡；玩家 1 在 (6.5,9.5) —— 距离 3 < 4，
    # 且不在辐射十字上（威力 1：只烧 (5,6)/(7,6)/(6,5)/(6,7)）
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([6.5, 9.5])
    reward, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                            auto_reset=False)
    expected = 0.01 * (1.0 - 3.0 / 4.0) - cfg.step_penalty
    assert abs(float(reward[0, 0]) - expected) < 1e-5, \
        f"off-cross 近身应得距离分: 期望 {expected:.5f} 实得 {reward[0, 0].item()}"
    # 太远（距离 ≥ 半径）不给分：敌人移到 (6.5,11.5) → 距离 5 > 4
    sim.reset_(torch.tensor([True], dtype=torch.bool))
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([6.5, 11.5])
    reward2, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                             auto_reset=False)
    assert abs(float(reward2[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        f"距离 ≥ 半径不应有近身分: {reward2[0, 0].item()}"
    # 冷却：第一泡（开局冷却已完成）给分；紧接着的第二泡 since_bomb 已清零，
    # 不给距离分 —— 连续快速放炮不赚
    sim.reset_(torch.tensor([True], dtype=torch.bool))
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([6.5, 9.5])
    r_first, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                             auto_reset=False)
    assert abs(float(r_first[0, 0]) - expected) < 1e-5, \
        f"开局第一泡（冷却完成）应给距离分: {r_first[0, 0].item()}"
    sim.fuse.zero_(); sim.owner.fill_(-1)        # 清掉刚放的泡，否则放不下去
    r_second, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                              auto_reset=False)
    assert abs(float(r_second[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        "冷却期内（上一泡刚放）不应给近身分"


def test_place_cover_beats_dist_reward():
    """覆盖到敌人时走覆盖分，**不再叠加**近身距离分（防双重计分）。

    敌人正落在辐射十字上（距离 1、同线）→ 覆盖分 +place_cover_reward；
    off-cross 距离分不参与。初始冷却已完成，第一泡即可评估。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_cover_reward=0.02, place_dist_reward=0.01,
                    place_dist_radius=4.0, place_dist_cooldown=15,
                    place_chain_reward=0.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([6.5, 7.5])     # 正东 1 格，在辐射十字上
    reward, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                            auto_reset=False)
    expected = cfg.place_cover_reward - cfg.step_penalty
    assert abs(float(reward[0, 0]) - expected) < 1e-5, \
        f"覆盖应 +{cfg.place_cover_reward} 且不叠加距离分: {reward[0, 0].item()}"


def test_place_chain_reward_near_live_bomb():
    """连锁奖励：新泡火焰**真的**点燃近旁已有泡才给分，× 剩余引信因子。

    关键回归：爆源 seed 必须是"自己脚下那一格"，不能用 `placed.view(n,1,1)`
    广播——那会让 rays 从每个格发火、覆盖≈全图，连锁/覆盖/近身全失去几何。
    这里离已有泡 2 格（在威力 3 的十字上）→ 给分；离 3 格（超出）→ 不给。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_cover_reward=0.0, place_dist_reward=0.0,
                    place_chain_reward=0.06, chain_time_factor=0.5)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([11.5, 11.5])   # 远处，别干扰
    # 已有泡在 (6.5,8.5) 的格 (6,8)：fuse=2（本 tick 递减前剩 2 秒份）
    sim.fuse[0, 6, 8], sim.owner[0, 6, 8] = 2, 1
    reward, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                            auto_reset=False)
    # 玩家 0 威力 3：新泡在 (6,6)，十字烧到 (6,8) 的已有泡（在射程内、真点燃）
    # fuse 已本 tick 递减 → 评估时 fuse=1 → 因子 = 0.5 + 0.5×(1-1/30) ≈ 0.9833
    fuse_frac = 1.0 / cfg.fuse
    w = cfg.chain_time_factor + (1 - cfg.chain_time_factor) * (1 - fuse_frac)
    expected = cfg.place_chain_reward * w - cfg.step_penalty
    assert abs(float(reward[0, 0]) - expected) < 1e-4, \
        f"点燃近旁已有泡应给连锁分: 期望 {expected:.5f} 实得 {reward[0, 0].item()}"
    # 反向检查：已有泡在 3 格外 (6,9) → 烧不到，连锁不给分
    sim.reset_(torch.tensor([True], dtype=torch.bool))
    sim.pos[0, 0] = torch.tensor([6.5, 6.5])
    sim.pos[0, 1] = torch.tensor([11.5, 11.5])
    sim.fuse[0, 6, 9], sim.owner[0, 6, 9] = 2, 1
    reward2, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                             auto_reset=False)
    assert abs(float(reward2[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        f"射程外的已有泡不该给连锁分: {reward2[0, 0].item()}"


def test_chain_time_factor_weights_old_bomb():
    """时间差因子：**连老泡**（快爆）远赚于**连新泡**——专治"啪啪啪"贴脸连丢。

    chain_time_factor=0.15 时：被连锁泡剩余 1 tick（fuse≈0）→ 因子≈1.0；
    剩余满 30 → 因子≈0.15。同配置下分别验证两档的连锁分差 ≈ 6.7 倍。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_cover_reward=0.0, place_dist_reward=0.0,
                    place_chain_reward=0.15, chain_time_factor=0.15,
                    chain_blast_bonus=0.0)
    sim = BatchedSim(cfg, 1, seed=0)

    def once(fuse_init):
        sim.reset_(torch.tensor([True], dtype=torch.bool))
        sim.pos[0, 0] = torch.tensor([6.5, 6.5])
        sim.pos[0, 1] = torch.tensor([11.5, 11.5])     # 远处别干扰
        sim.fuse[0, 6, 8], sim.owner[0, 6, 8] = fuse_init, 1
        r, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)),
                           auto_reset=False)
        return float(r[0, 0])

    old = once(2)                       # fuse=2 → 递减后 1 → 因子 ≈ 0.967（不立刻爆）
    new = once(cfg.fuse)                # fuse=30 → 递减后 29 → 因子 ≈ 0.15
    fuse_frac = 1.0 / cfg.fuse
    w_old = cfg.chain_time_factor + (1 - cfg.chain_time_factor) * (1 - fuse_frac)
    w_new = cfg.chain_time_factor + (1 - cfg.chain_time_factor) * (1 - 29 / cfg.fuse)
    base = -cfg.step_penalty
    assert abs(old - (base + cfg.place_chain_reward * w_old)) < 1e-4, \
        f"连老泡分不对: {old:.5f}"
    assert abs(new - (base + cfg.place_chain_reward * w_new)) < 1e-4, \
        f"连新泡分不对: {new:.5f}"
    assert (old - base) > 5.0 * (new - base), \
        f"连老泡应远赚于连新泡: 老{old - base:.4f} vs 新{new - base:.4f}"


def test_chain_blast_bonus_at_explosion():
    """爆炸时刻连锁兑现：被连锁**提前点燃**的泡每颗给点火源 +0.08。

    P1 在 (6,6) 一颗老泡（fuse=30），P0 在 (6,8) 放一颗新泡。之后把两泡引信
    压到 1/2：下 tick 老泡自然走完引爆 → 火焰把 P0 的新泡连锁点燃（被提前
    触发）→ 点火源 P1 得 chain_blast_bonus；P0 只被点燃，不得分。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_cover_reward=0.0, place_dist_reward=0.0,
                    place_chain_reward=0.0, chain_blast_bonus=0.08)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([6.5, 8.5])     # P0 放泡位 (6,8)
    sim.pos[0, 1] = torch.tensor([6.5, 6.5])     # P1 放泡位 (6,6)
    sim.fuse[0, 6, 6], sim.owner[0, 6, 6] = 30, 1
    r0, _, _ = sim.step(act((MOVE_IDLE, 1), (MOVE_IDLE, 0)), auto_reset=False)
    assert abs(float(r0[0, 0]) + cfg.step_penalty) < 1e-5, "P0 放泡无直接分，只有步罚"
    # 压引信：P1(6,6)→1（下 tick 自然走完），P0(6,8)→2
    sim.fuse[0, 6, 6], sim.fuse[0, 6, 8] = 1, 2
    r2, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    # 本 tick：P1(6,6) fuse 0 自然引爆（点火源 P1）→ 连锁点燃 P0(6,8) 新泡
    assert abs(float(r2[0, 1]) - (cfg.chain_blast_bonus - cfg.step_penalty)) < 1e-5, \
        f"点火源 P1 应得爆炸连锁分: {r2[0, 1].item()}"
    assert abs(float(r2[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        f"P0 被连锁点燃不应得分: {r2[0, 0].item()}"


def test_chain_blast_own_bombs_no_bonus():
    """归属修复（反"自连爆白捡"）：自己引信自然走完的泡点燃**自己的**泡 → 0 分。

    旧版 chained 统计所有被点燃泡（不分 owner），自己连放一排 → 第 1 颗自然爆
    点燃后 5 颗 → 白捡 5×0.08 —— 梯度把 AI 推成"满预算一股脑全丢"。
    现在只给**跨 owner 连锁**：P0 老泡自然爆点燃 P0 自己的新泡 = 0 分。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, danger_penalty=0.0,
                    place_cover_reward=0.0, place_dist_reward=0.0,
                    place_chain_reward=0.0, chain_blast_bonus=0.08)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([11.5, 11.5])    # P0 站远处，别吃自己爆心的伤害
    sim.pos[0, 1] = torch.tensor([11.5, 3.5])     # 对手也远离
    # P0 两泡在 (6,6)/(6,8)（同 owner=0）：老泡自然走完 → 点燃自己的新泡
    sim.fuse[0, 6, 6], sim.owner[0, 6, 6] = 30, 0
    sim.fuse[0, 6, 8], sim.owner[0, 6, 8] = 30, 0
    r0, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    # 压引信：老泡→1 自然走完，新泡→2 被连锁点燃
    sim.fuse[0, 6, 6], sim.fuse[0, 6, 8] = 1, 2
    r2, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    # 本 tick：P0(6,6) 自然爆（点火源 P0）→ 连锁点燃自己的 P0(6,8)
    assert abs(float(r2[0, 0]) - (-cfg.step_penalty)) < 1e-4, \
        f"自连爆(同 owner)应 0 分: {r2[0, 0].item()}"


def test_invuln_after_hit():
    """无敌保护期：被炸掉血后 invuln_ticks 内再被炸不掉血、不触发对方 hit 奖励。

    玩家 0 被玩家 1 的泡炸掉 1 血 → 进入无敌期；紧接着的爆炸不再掉血。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, invuln_ticks=30,
                    chain_blast_bonus=0.0)   # 隔离爆炸连锁兑现，只测无敌语义
    sim = BatchedSim(cfg, 1, seed=0)
    # 玩家 0 站 (5.5,5.5)，玩家 1 的泡在 (5,5)（fuse=1 下一 tick 爆），威力 3
    sim.pos[0, 0] = torch.tensor([5.5, 5.5])
    sim.pos[0, 1] = torch.tensor([11.5, 11.5])
    sim.fuse[0, 5, 5], sim.owner[0, 5, 5] = 1, 1
    sim.bomb_blast[0, 5, 5] = 3
    hp0 = int(sim.hp[0, 0])
    reward, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.hp[0, 0]) == hp0 - 1, "第一炸掉 1 血"
    assert int(sim.invuln[0, 0]) == cfg.invuln_ticks, "掉血后进入无敌期"
    # 同 tick 对方（玩家 1）因玩家 0 掉血得到 hit 奖励（1v1 对掉血=我打的）
    assert float(reward[0, 1]) > 0, "造成伤害应得 hit_reward"
    # 再放一颗泡在脚下（下一 tick 爆），无敌期内不掉血
    sim.fuse[0, 5, 5], sim.owner[0, 5, 5] = 1, 1
    sim.bomb_blast[0, 5, 5] = 3
    reward2, _, _ = sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert int(sim.hp[0, 0]) == hp0 - 1, "无敌期内被炸不掉血"
    assert float(reward2[0, 1]) <= 1e-5, "无敌期造成伤害不得 hit_reward"


def test_open_map_growth_init_and_layout():
    """open 关（混合地图）：纯空场 + 成长初始 = 上限 80%（6/6/2.4）。

    open 关没有砖墙/宝箱（刷分路径被切断），出生点整宽中线均分。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, open_fraction=1.0,
                    open_growth_bombs=6, open_growth_blast=6,
                    open_growth_speed=2.4)
    sim = BatchedSim(cfg, 1, seed=0)
    assert not bool(sim.brick.any()), "open 关无 brick"
    assert not bool(sim.wall.any()), "open 关无永久墙"
    assert int(sim.bombs_cap[0, 0]) == 6, "open 关泡数初始 6"
    assert int(sim.blast_cap[0, 0]) == 6, "open 关威力初始 6"
    assert abs(float(sim.spd_g[0, 0]) - 2.4) < 1e-5, "open 关速度初始 2.4"
    # 出生点整宽中线均分（P=2 → (4.5,6.5)/(8.5,6.5)）
    assert float(sim.pos[0, 0, 1]) == 4.5 and float(sim.pos[0, 0, 0]) == 6.5
    assert float(sim.pos[0, 1, 1]) == 8.5 and float(sim.pos[0, 1, 0]) == 6.5


def test_mixed_map_reset():
    """混合地图：open_fraction=0.5 时两类关（open 空场 / corridor 砖墙）都出现。"""
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, open_fraction=0.5)
    sim = BatchedSim(cfg, 64, seed=0)
    has_open = int((~sim.brick.any(dim=(1, 2))).sum())
    has_corr = int((sim.brick.any(dim=(1, 2))).sum())
    assert 0 < has_open < 64 and 0 < has_corr < 64, \
        f"两类地图都应出现: open={has_open} corr={has_corr}"


def test_corridor_max_growth_speed_no_tunnel():
    """corridor 满成长速度（3.0× = 0.9 格/tick）高速撞 brick 不穿模。

    玩家 0 速度拉到满，向左冲 15 tick（0.9×15=13.5 格，远超到左 brick 墙），
    前缘区间枚举应让它贴在 brick 右壁（radius=0.45 ⇒ x≈4.45）停下，不得穿进 brick 格。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, growth_speed_max=3.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.spd_g[0, 0] = 3.0                 # 满成长：0.9 格/tick
    sim.pos[0, 0] = torch.tensor([6.5, 5.5])
    sim.pos[0, 1] = torch.tensor([6.5, 8.5])
    for _ in range(15):
        sim.step(act((MOVE_LEFT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    x = float(sim.pos[0, 0, 1])
    assert 4.44 <= x <= 4.46, \
        f"0.9 格/tick 高速撞 brick 应贴右壁停（radius=0.45 ⇒ x≈4.45）: x={x:.3f}"


def test_approach_reward_removed():
    """接近/追击奖励已按用户要求移除（approach_reward=0 默认，代码段删除）。

    朝对手移动不再产生接近分 —— 只有步罚。防止未来有人改回 approach_reward
    而漏掉这里的行为回归。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0, open_fraction=0.0,
                    danger_penalty=0.0)
    sim = BatchedSim(cfg, 1, seed=0)
    sim.pos[0, 0] = torch.tensor([6.5, 3.5])
    sim.pos[0, 1] = torch.tensor([6.5, 8.5])          # 距离 5，朝它走
    r1, _, _ = sim.step(act((MOVE_RIGHT, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert abs(float(r1[0, 0]) - (-cfg.step_penalty)) < 1e-5, \
        f"接近奖励已删除：朝对手移动应只有步罚，实际 {r1[0, 0].item()}"


def test_ring_map_layout_and_crate_prob():
    """环岛地图：中间 7×7 永久墙山体（不可行走不可炸）+ 山体外圈 1 格环带**稀疏**
    brick（ring_brick_density，非全充满）+ 四角出生点 + 宝箱爆率 100%。"""
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0,
                    open_fraction=0.0, ring_fraction=1.0, ring_crate_prob=1.0,
                    ring_brick_density=0.4)
    sim = BatchedSim(cfg, 1, seed=0)
    # 中间 7×7 (3..9) 全永久墙 —— 整块不可行走的障碍物区域
    assert bool(sim.wall[0, 3:10, 3:10].all()), "中间 7×7 应是永久墙山体"
    assert not bool(sim.wall[0, 2, 2]) and not bool(sim.wall[0, 10, 10]), \
        "山体之外（环带/角落）无永久墙"
    # 环带 = 山体外圈 1 格（[2:10,2:10] 减 [3:10,3:10] = 3 行 3 列的外壳）
    band = sim.brick[0, 2, 3:10] | sim.brick[0, 10, 3:10] \
        | sim.brick[0, 3:10, 2] | sim.brick[0, 3:10, 10]
    assert bool(band.any()), "环带应有 brick"
    assert not bool(sim.brick[0, 3:10, 3:10].any()), "山体内部不铺 brick"
    # 稀疏：环带 120 格里 brick 占比应远低于全充满（0.4 密度 + 四角清空）
    n_band = int(band.sum())
    assert n_band <= 60, f"环带 brick 应稀疏: {n_band}/120"
    # 四角出生点（(1.5,1.5)/(1.5,11.5)）脚下及四邻无 brick
    for row, col in cfg.ring_spawn_cells():
        assert not bool(sim.brick[0, row, col])
        for drow, dcol in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + drow, col + dcol
            if 0 <= nr < 13 and 0 <= nc < 13:
                assert not bool(sim.brick[0, nr, nc]), \
                    f"出生点四邻无 brick: ({nr},{nc})"
        assert tuple(sim.pos[0, 0].tolist()) == (1.5, 1.5), "玩家 0 出生在左上角"
        assert tuple(sim.pos[0, 1].tolist()) == (1.5, 12.5), "玩家 1 出生在右上角"
    assert float(sim.crate_prob[0]) == 1.0, "环岛宝箱爆率 100%"
    # 环带上放泡 → 炸掉 brick → 变宝箱 → 踩上 100% 成长。
    # 任取一块环带 brick，在它的带外侧一格放泡（威力 1）→ 火焰覆盖 brick 格。
    band_cells = [(r, c) for r in range(3, 10) for c in (2, 10)] + \
                 [(r, c) for r in (2, 10) for c in range(3, 10)]
    hits = [(r, c) for r, c in band_cells if bool(sim.brick[0, r, c])]
    assert len(hits) > 0, "环带应有可炸 brick 格"
    br, bc = hits[0]
    if bc == 2:                                # 西侧：炸弹放 (br,1) 向东
        bx, by = br, 1
    elif bc == 10:                             # 东侧：炸弹放 (br,11) 向西
        bx, by = br, 11
    elif br == 2:                              # 北侧：炸弹放 (1,bc) 向南
        bx, by = 1, bc
    else:                                      # 南侧：炸弹放 (11,bc) 向北
        bx, by = 11, bc
    assert not bool(sim.wall[0, bx, by]) and not bool(sim.brick[0, bx, by]), \
        "炸弹位应在带外空旷格"
    sim.pos[0, 0] = torch.tensor([1.5, 1.5])       # 玩家躲角落，远离爆炸
    sim.pos[0, 1] = torch.tensor([1.5, 11.5])
    sim.fuse[0, bx, by], sim.owner[0, bx, by] = 1, 0
    sim.bomb_blast[0, bx, by] = 1
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    assert bool(sim.crate[0, br, bc]), f"环带 brick 被炸掉 → 宝箱 ({br},{bc})"
    before = int(sim.bombs_cap[0, 0])
    sim.pos[0, 0] = torch.tensor([float(br), float(bc)])
    sim.step(act((MOVE_IDLE, 0), (MOVE_IDLE, 0)), auto_reset=False)
    grew = int(sim.bombs_cap[0, 0]) + int(sim.blast_cap[0, 0]) + \
        (0 if abs(float(sim.spd_g[0, 0]) - 1.0) < 1e-6 else 1)
    assert grew > before, "环岛踩宝箱 100% 成长"


def test_three_map_mixing():
    """三图混合：open_fraction+ring_fraction 时 open/ring/corridor 三类都出现。

    判据：ring 关有山体永久墙（wall 非空）且 crate_prob=1.0；open 关无墙无砖
    （纯空场 + 中心十字宝箱，crate_prob=1.0）；corridor 关 wall 非空且
    crate_prob=growth_crate_prob。
    """
    cfg = SimConfig(height=13, width=13, n_players=2,
                    map_mode="corridor", corridor_width=5, top_wall_rows=4,
                    max_steps=1800, speed=3.0,
                    open_fraction=1/3, ring_fraction=1/3)
    sim = BatchedSim(cfg, 90, seed=0)
    # open 无墙无砖（ring/corridor 都有墙）→ 唯一可辨特征
    open_m = ~sim.brick.any(dim=(1, 2)) & ~sim.wall.any(dim=(1, 2))
    ring_m = sim.wall.any(dim=(1, 2)) & (sim.crate_prob == 1.0)
    corr_m = sim.wall.any(dim=(1, 2)) & (sim.crate_prob == cfg.growth_crate_prob)
    n_open, n_ring, n_corr = int(open_m.sum()), int(ring_m.sum()), int(corr_m.sum())
    assert n_open > 0 and n_ring > 0 and n_corr > 0, \
        f"三类地图都应出现: open={n_open} ring={n_ring} corridor={n_corr}"


# ---------------- 共享观测的扩展通道（1 + 3P，无逃生可达段） ----------------

def test_obs_extra_channel_layout_segments():
    """扩展通道布局逐段钉死：宝箱/无敌/可用泡/上限不重叠（回归：曾覆盖重叠）。

    crate(1) + invuln(P) + avail(P) + cap(P)，每段索引必须互斥；不含逃生
    可达段（逃生通道实验已弃用）。P=2 → 1+3P = 7 个扩展通道 + 2P+3 = 14 总通道。
    """
    from sim.config import n_obs_channels, obs_extra
    for p in (2, 3, 4):
        n = n_obs_channels(p)
        assert n == 2 * p + 3 + obs_extra(p) == 5 * p + 4, n
        assert n == 1 + p + p + p + 2 * p + 3, "逐段索引相加要等于总通道数"
    # obs 实测：无敌段的位置格标记 1
    cfg = SimConfig(height=7, width=7, n_players=2, tick_hz=15)
    sim = BatchedSim(cfg, 1, seed=0)
    clear(sim)
    sim.invuln[0] = torch.tensor([5, 0])          # 玩家 0 无敌
    obs = sim.observe()
    assert obs.shape[1] == 14, obs.shape
    p = 2
    cell = sim.pos[0, 0].floor().long().clamp(0, 6)
    # 无敌段：ch 2P+4 .. 2P+3+P，玩家 0 在其位置格标记 1，玩家 1 无标记
    ch_inv0 = 2 * p + 4
    assert float(obs[0, ch_inv0, cell[0], cell[1]]) == 1.0, "无敌段玩家 0 位置格应为 1"
    assert float(obs[0, ch_inv0].sum()) == 1.0, "无敌段只有玩家 0 一个标记"
    assert float(obs[0, ch_inv0 + 1].sum()) == 0.0, "玩家 1 未无敌，段内无标记"
    # 可用泡段：ch 2P+4+P .. 2P+3+2P，位置格 = avail/cap ∈ (0,1]
    ch_avail = 2 * p + 4 + p
    v = float(obs[0, ch_avail, cell[0], cell[1]])
    assert 0.0 < v <= 1.0, v
    # 上限段：ch 2P+4+2P .. 2P+3+3P，位置格 = cap/growth_bombs_max ∈ (0,1]
    ch_cap = 2 * p + 4 + 2 * p
    v = float(obs[0, ch_cap, cell[0], cell[1]])
    assert 0.0 < v <= 1.0, v
