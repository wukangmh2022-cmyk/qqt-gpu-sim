"""训练侧的最小可运行性测试：只验证形状、梯度和 checkpoint 往返。

不验证"能不能学会"——那需要几百万局。这里守住的是"跑得起来、不 NaN、
断点续训能接上"这条线，这是被 12 小时会话上限反复打断时最容易坏的地方。
"""

from __future__ import annotations

import torch

from sim.config import N_BOMB, N_MOVES, SimConfig, view_perm
from sim.dev import available, resolve_device
from sim.factory import make_sim
from sim.torch_sim import BatchedSim
from train.curriculum import CurriculumState, default_curriculum
from train.model import ActorCritic
from train.model_pool import ModelPool, clone_frozen
from train.ppo import PPOConfig, SelfPlayRunner, compute_gae, ppo_update
from train.train import adapt_first_conv


def test_model_param_budget_and_shapes():
    cfg = SimConfig(height=11, width=11, n_players=2)
    net = ActorCritic(cfg.obs_shape)
    ml, bl, value = net(torch.zeros((4,) + cfg.obs_shape))
    assert ml.shape == (4, N_MOVES) and bl.shape == (4, N_BOMB)
    assert value.shape == (4,)
    n = net.n_params()
    assert 150_000 < n < 260_000, f"参数量 {n} 偏离设计目标（11x11 上约 21 万）"


def test_factored_heads_joint_logp_and_entropy():
    """联合 log_prob = 两个头之和，熵也是。ratio 才能仍然是一个标量。"""
    cfg = SimConfig(height=7, width=7, n_players=2)
    net = ActorCritic(cfg.obs_shape)
    obs = torch.randn((16,) + cfg.obs_shape)
    mm = torch.ones(16, N_MOVES, dtype=torch.bool)
    bm = torch.ones(16, N_BOMB, dtype=torch.bool)
    a, logp, _ = net.act(obs, mm, bm)
    assert a.shape == (16, 2)
    logp2, ent, _ = net.evaluate(obs, mm, bm, a)
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert (ent <= torch.log(torch.tensor(float(N_MOVES * N_BOMB))) + 1e-4).all()
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()


def test_masked_dist_never_picks_illegal():
    logits = torch.randn(64, N_MOVES)
    mask = torch.zeros(64, N_MOVES, dtype=torch.bool)
    mask[:, 2] = True
    dist = ActorCritic.masked_dist(logits, mask)
    assert torch.all(dist.sample() == 2)
    assert torch.isfinite(dist.entropy()).all(), "全掩码之外不能出 NaN"


def test_device_resolution_and_auto_backend_respect_cpu():
    assert available("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    assert not available("cpu:1")
    sim = make_sim(SimConfig(height=7, width=7), 2,
                   backend="auto", device="cpu", seed=0)
    assert isinstance(sim, BatchedSim)
    assert sim.device == torch.device("cpu")


def test_three_player_opponents_receive_initial_attributes():
    cfg = SimConfig(height=9, width=9, n_players=3, map_mode="open")
    sim = BatchedSim(cfg, 16, device="cpu", seed=0)
    mmask, bmask = sim.legal_mask()
    assert torch.all(sim.bombs_cap[:, 1:] > 0)
    assert torch.all(sim.blast_cap[:, 1:] > 0)
    assert torch.all(sim.spd_g[:, 1:] > 0)
    assert torch.all(bmask[:, 1:, 1]), "每个对手开局都应可放第一颗泡"


def test_hybrid_runner_cpu_sim_and_train_devices():
    """Simulator may stay on CPU while rollout storage uses learner device."""
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=40)
    sim = BatchedSim(cfg, 8, device="cpu", seed=0)
    learner = ActorCritic(cfg.obs_shape).to("cpu")
    opp = clone_frozen(learner)
    pcfg = PPOConfig(rollout_steps=8, epochs=1, minibatches=2)
    runner = SelfPlayRunner(sim, learner, [opp], pcfg, measure_timing=True)
    buf, last_val = runner.collect()
    assert sim.device.type == "cpu"
    assert buf.obs.device.type == "cpu"
    assert last_val.device.type == "cpu"
    assert buf.ptr == pcfg.rollout_steps
    assert all(v >= 0.0 for v in runner.last_timing.values())
    opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
    stats = ppo_update(learner, opt, buf, last_val, pcfg, pcfg.entropy_coef)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


def test_ppo_iteration_runs_and_updates():
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=40)
    sim = BatchedSim(cfg, 8, seed=0)
    learner = ActorCritic(cfg.obs_shape)
    opp = clone_frozen(learner)
    pcfg = PPOConfig(rollout_steps=16, epochs=2, minibatches=2)
    runner = SelfPlayRunner(sim, learner, [opp], pcfg)

    before = learner.move_head[-1].weight.detach().clone()
    before_b = learner.bomb_head[-1].weight.detach().clone()
    buf, last_val = runner.collect()
    assert buf.ptr == pcfg.rollout_steps
    assert buf.act.shape == (pcfg.rollout_steps, 8, 2)
    adv, ret = compute_gae(buf, last_val, pcfg.gamma, pcfg.gae_lambda)
    assert adv.shape == buf.rew.shape and torch.isfinite(adv).all()

    opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
    stats = ppo_update(learner, opt, buf, last_val, pcfg, pcfg.entropy_coef)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert not torch.equal(before, learner.move_head[-1].weight), "方向头应被更新"
    assert not torch.equal(before_b, learner.bomb_head[-1].weight), "放泡头应被更新"
    assert 0.0 <= runner.win_rate() <= 1.0


def test_runner_health_stats_collected():
    """collect 的进度/健康度统计（bombs/danger）必须可累积且非负。

    新增的自杀/放炮/危险站桩统计是训练日志判断"是否停下找问题"的依据，
    统计错（如用错 cfg、索引越界）会导致日志失真 —— 这里验证基本完整性。
    """
    cfg = SimConfig(height=7, width=7, n_players=2, max_steps=40)
    sim = BatchedSim(cfg, 8, seed=1)
    learner = ActorCritic(cfg.obs_shape)
    opp = clone_frozen(learner)
    pcfg = PPOConfig(rollout_steps=16)
    runner = SelfPlayRunner(sim, learner, [opp], pcfg)
    runner.collect()
    assert "suicide" in runner.ep_stats
    assert "bombs" in runner.ep_stats
    assert "danger_ticks" in runner.ep_stats
    for k in ("suicide", "bombs", "danger_ticks", "kills", "count"):
        assert runner.ep_stats[k] >= 0, f"{k} 应为非负，实际 {runner.ep_stats[k]}"
    # danger_ticks 统计所有 collect 的 tick（含未结束局）→ 上限 = rollout_steps×envs
    assert 0 <= runner.ep_stats["danger_ticks"] <= pcfg.rollout_steps * 8, \
        f"danger_ticks 不应超过总 collect tick"
    # collect 两次应累积（不重置），clear_stats 后归零
    runner.collect()
    assert runner.ep_stats["bombs"] > 0 or runner.ep_stats["count"] > 0, \
        "两次 collect 应累计健康度"
    runner.clear_stats()
    assert runner.ep_stats["suicide"] == 0 and runner.ep_stats["bombs"] == 0


def test_reward_coefficients_match_config_snapshot():
    """奖励系数快照：防止优化/重构时数值漂移（训练依赖这些默认值）。

    2026-08 人类录像校准后的新值：稀疏主信号加大、稠密塑形压低、
    死信号（dist/chainblst）归零、终局固定击杀 + 超时退火。
    """
    cfg = SimConfig()
    assert cfg.hit_reward == 1.5
    assert cfg.win_bonus == 10.0
    assert cfg.danger_penalty == 0.015
    assert cfg.brick_reward == 0.15  # 0.15→0.05→0.10→2026-08-11 回调 0.15(配合退火放慢)
    assert cfg.place_cover_reward == 0.05
    assert cfg.place_chain_reward == 0.20
    assert cfg.place_dist_reward == 0.0, "近身定位是死信号，应归零"
    assert cfg.chain_blast_bonus == 0.0, "跨主连锁是死信号，应归零"
    assert cfg.combo_reward == 0.10
    assert cfg.timeout_draw is True, "超时血差×退火（默认开）"
    assert cfg.win_hp_scaled is False, "终局击杀固定值（不看血量差）"


def test_opponent_handicap_reduces_bomb_rate():
    """削弱对手应该显著减少对手的放泡次数（否则 handicap 是个空实现）。"""
    cfg = SimConfig(height=9, width=9, n_players=2, max_steps=60)
    pcfg = PPOConfig(rollout_steps=32)

    def bombs_placed(handicap: float) -> int:
        torch.manual_seed(0)
        sim = BatchedSim(cfg, 16, seed=0)
        learner = ActorCritic(cfg.obs_shape)
        runner = SelfPlayRunner(sim, learner, [clone_frozen(learner)], pcfg,
                                handicap=handicap)
        count = 0
        for _ in range(pcfg.rollout_steps):
            obs = sim.observe()
            mm, bm = sim.legal_mask()
            acts = torch.zeros((16, 2, 2), dtype=torch.long)
            runner._opponent_actions(obs, mm, bm, acts)
            count += int(acts[:, 1, 1].sum())
            sim.step(acts)
        return count

    full, weak = bombs_placed(1.0), bombs_placed(0.0)
    assert weak == 0 and full > 0, f"削弱后放泡应归零：{weak} vs {full}"


def test_model_pool_elo_and_eviction():
    net = ActorCritic(SimConfig(height=7, width=7).obs_shape)
    pool = ModelPool(max_size=3)
    for step, elo in ((0, 900.0), (1, 1000.0), (2, 1100.0)):
        pool.add(net, step=step, elo=elo)
    assert len(pool) == 3
    pool.add(net, step=3, elo=1200.0)
    assert len(pool) == 3
    assert min(s["elo"] for s in pool.snapshots) == 1000.0, "应淘汰 ELO 最低的快照"

    snap = pool.snapshots[0]
    old_opp = snap["elo"]
    new_elo = pool.update_elo(snap, 1000.0, learner_score=1.0)
    assert new_elo > 1000.0 and snap["elo"] < old_opp, "赢了 learner 涨、对手跌"


def test_elo_expected_depends_on_opponent_strength():
    """ELO 的期望得分必须按对手实力算 —— 赢强对手比赢弱对手涨得多。

    这是"按对手实力提升排名"的语义：如果哪天真有人把 expected 简化成固定值
    （比如只按胜率加权），这条测试会立刻红。
    """
    def gain(opp_elo: float, score: float) -> float:
        pool = ModelPool()
        snap = {"step": 0, "elo": opp_elo, "state": {}}
        return pool.update_elo(snap, learner_elo=1000.0, learner_score=score) - 1000.0

    # 基准：同分对决 expected=0.5，胜 1.0 得 +8（k/2）
    assert abs(gain(1000.0, 1.0) - 8.0) < 1e-9
    # 赢强对手（1400，expected≈0.0909）涨幅远大于赢同分对手
    assert gain(1400.0, 1.0) > gain(1000.0, 1.0)
    # 平局也要看对手：**对手越强我方 expected 越低**。
    # 平强对手（我方预期只有 ≈0.09 的胜率）→ 超预期，应涨；
    # 平弱对手（预期 ≈0.91）→ 不及预期，应跌。
    assert gain(1400.0, 0.5) > 0.0, "平强对手：0.5-expected>0，超预期应涨"
    assert gain(600.0, 0.5) < 0.0, "平弱对手：0.5-expected<0，不及预期应跌"
    # 输给谁跌得狠由 expected 决定：输弱对手比输强对手跌得多
    assert gain(600.0, 0.0) < gain(1400.0, 0.0)


def test_pool_state_dict_roundtrip():
    net = ActorCritic(SimConfig(height=7, width=7).obs_shape)
    pool = ModelPool()
    pool.add(net, step=5, elo=1050.0)
    other = ModelPool()
    other.load_state_dict(pool.state_dict())
    assert len(other) == 1 and other.snapshots[0]["step"] == 5


def test_adapt_first_conv_preserves_fixed_channels():
    small = SimConfig(height=9, width=9, n_players=2)     # C = 7（旧布局）
    big = SimConfig(height=9, width=9, n_players=3)       # C = 2*3+3+obs_extra(3)=22
    old = ActorCritic(small.obs_shape)
    new = adapt_first_conv(old, big.obs_shape)
    assert new.obs_shape == big.obs_shape
    ow = old.state_dict()["conv0.weight"]
    nw = new.state_dict()["conv0.weight"]
    # 视角序：[0]自己位置 [1]自己引信 [2..P]对手位置 [P+1..2P-1]对手引信
    #        [2P..2P+2]墙/危险/进度 [2P+3..] extra 通道原样保留
    assert torch.equal(nw[:, :2], ow[:, :2]), "自身两通道要原样搬过去"
    assert torch.equal(nw[:, 6:9], ow[:, 4:7]), "墙/危险/进度按视角位置对齐"
    assert torch.equal(nw[:, 2:3], ow[:, 2:3]), "旧对手位置通道保留"
    assert torch.equal(nw[:, 4:5], ow[:, 3:4]), "旧对手引信通道搬到新段内偏移"
    assert torch.count_nonzero(nw[:, 3:4]) == 0, "新增对手位置通道置零"
    assert torch.count_nonzero(nw[:, 5:6]) == 0, "新增对手引信通道置零"
    assert torch.count_nonzero(nw[:, 9:]) == 0, "extra 通道（宝箱/可达/无敌/泡数）置零"
    # 其余层形状不变，应完全复用
    for key in ("conv.2.weight", "shared.0.weight", "move_head.2.weight",
                "bomb_head.2.weight"):
        assert torch.equal(new.state_dict()[key], old.state_dict()[key])


def test_view_perm_is_a_permutation():
    """视角必须是纯置换——一旦某个通道需要求和，权重重排的技巧就失效了。

    新布局：基础 2P+3 参与视角置换，尾部 obs_extra(P) 通道（宝箱/无敌/可用
    泡数/泡数上限）是世界信息，对任意视角**原样保留**在尾部。
    """
    from sim.config import n_obs_channels, obs_extra
    for p in (2, 3, 4):
        for me in range(p):
            perm = view_perm(me, p)
            assert sorted(perm) == list(range(n_obs_channels(p))), (p, me, perm)
            assert perm[0] == me, "视角通道 0 必须是自己的位置"
            assert perm[1] == p + me, "视角通道 1 必须是自己名下泡泡的引信"
            base = 2 * p + 3
            extra = obs_extra(p)
            assert tuple(perm[-extra:]) == tuple(range(base, base + extra)), \
                "尾部扩展通道必须原样保留（不进置换）"


def test_weight_perm_equals_data_gather():
    """置换第一层权重 ≡ 显式 gather 观测通道。这条测试钉住 model.py 的核心技巧。"""
    cfg = SimConfig(height=7, width=7, n_players=3)
    net = ActorCritic(cfg.obs_shape)
    obs = torch.randn((5,) + cfg.obs_shape)
    for pid in range(cfg.n_players):
        view = obs[:, list(view_perm(pid, cfg.n_players))]        # 老办法：真搬数据
        ref = torch.nn.functional.conv2d(
            view, net.conv0.weight, net.conv0.bias, padding=1)
        got = torch.nn.functional.conv2d(
            obs, net.conv0.weight[:, net.inv_perm[pid]], net.conv0.bias, padding=1)
        assert torch.allclose(ref, got, atol=1e-5), f"pid={pid} 第一层输出不一致"
        # 第一层之后与视角无关 ⇒ 第一层相等即整网相等，顺手验一下 logits
        feat = net.shared(net.conv(ref).flatten(1))
        assert torch.allclose(net.move_head(feat), net(obs, pid)[0], atol=1e-5)


def test_curriculum_preserves_base_config():
    base = SimConfig(height=9, width=9, speed=2.0, radius=0.4,
                     tick_hz=20, obs_fp16=False)
    stages = default_curriculum(base)
    assert [s.cfg.n_players for s in stages] == [2, 2, 3, 3]
    for stage in stages:
        assert stage.cfg.speed == 2.0 and stage.cfg.radius == 0.4
        assert stage.cfg.tick_hz == 20 and not stage.cfg.obs_fp16


def test_curriculum_advances_on_win_rate():
    stages = default_curriculum()
    assert [s.cfg.n_players for s in stages] == [2, 2, 3, 3]
    st = CurriculumState()
    for _ in range(600):
        st.record(1.0)
    assert st.should_advance(stages[0]), "胜率达标应提前晋级"
    st2 = CurriculumState()
    for _ in range(600):
        st2.record(0.0)
    assert not st2.should_advance(stages[0])
    st2.episodes_in_stage = stages[0].episodes
    assert st2.should_advance(stages[0]), "跑满盘数也要晋级"


def test_checkpoint_roundtrip(tmp_path):
    import argparse

    from train.train import save_ckpt

    cfg = SimConfig(height=7, width=7, n_players=2)
    learner = ActorCritic(cfg.obs_shape)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    # 走一步优化器，让 state 里有动量，验证续训不会丢 Adam 状态
    learner(torch.zeros((2,) + cfg.obs_shape))[0].sum().backward()
    opt.step()
    pool = ModelPool()
    pool.add(learner, step=10, elo=1010.0)
    cstate = CurriculumState(stage_idx=1, episodes_in_stage=42, recent_wins=[1.0, 0.0])
    path = str(tmp_path / "ckpt.pt")
    save_ckpt(path, learner=learner, opt=opt, pool=pool, cstate=cstate,
              global_step=1234, elo=1010.0, args=argparse.Namespace(num_envs=8))

    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["format_version"] == 2
    assert ck["global_step"] == 1234 and ck["obs_shape"] == cfg.obs_shape
    assert ck["curriculum"]["episodes_in_stage"] == 42
    assert len(ck["pool"]["snapshots"]) == 1
    restored = ActorCritic(cfg.obs_shape)
    restored.load_state_dict(ck["model"])
    opt2 = torch.optim.Adam(restored.parameters(), lr=3e-4)
    opt2.load_state_dict(ck["opt"])
    assert opt2.state_dict()["state"], "Adam 动量必须一起恢复"
    for a, b in zip(learner.parameters(), restored.parameters()):
        assert torch.equal(a, b)
