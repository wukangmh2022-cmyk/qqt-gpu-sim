"""冒烟：位置对称化（随机交换出生点）+ 对手初始属性增强（三档 boost）+ 掉血 clamp。"""
import sys, torch
sys.path.insert(0, "/Users/pippo/operater-dev/qqt-gpu-sim")
from sim.config import SimConfig
from sim.torch_sim import BatchedSim

cfg = SimConfig(map_mode="corridor", open_fraction=0.5, ring_fraction=0.0,
                n_players=2, speed=3.0, max_steps=1200,
                open_crate_cross=True, hit_attr_penalty=2)
N = 64
sim = BatchedSim(cfg, N, device="cpu", seed=7)
print("cfg.opp_hist_mult:", cfg.opp_hist_mult, "opp_growth:", cfg.opp_growth_bombs,
      cfg.opp_growth_blast, cfg.opp_growth_speed)

# ---------- 1) 默认 boost=0：双方同起点（按各模式起点）----------
for e in range(N):
    lb = cfg.open_growth_bombs if bool(sim._is_open[e]) else cfg.growth_bombs_start
    lz = cfg.open_growth_blast if bool(sim._is_open[e]) else cfg.growth_blast_start
    ls = cfg.open_growth_speed if bool(sim._is_open[e]) else cfg.growth_speed_start
    for pl in (0, 1):
        assert sim.bombs_cap[e, pl] == lb, (e, pl, sim.bombs_cap[e, pl].item())
        assert sim.blast_cap[e, pl] == lz
        assert abs(float(sim.spd_g[e, pl]) - ls) < 1e-6
print("boost=0 双方同起点 OK")

# ---------- 2) boost=1（历史网络）：P1 起点 ×1.3，P0 不变 ----------
sim.set_opp_boost(1)
sim.reset_all()
for e in range(N):
    lb = cfg.open_growth_bombs if bool(sim._is_open[e]) else cfg.growth_bombs_start
    lz = cfg.open_growth_blast if bool(sim._is_open[e]) else cfg.growth_blast_start
    ls = cfg.open_growth_speed if bool(sim._is_open[e]) else cfg.growth_speed_start
    exp_b = int(round(lb * cfg.opp_hist_mult))
    exp_z = int(round(lz * cfg.opp_hist_mult))
    exp_s = ls * cfg.opp_hist_mult
    assert int(sim.bombs_cap[e, 1]) == exp_b, (e, sim.bombs_cap[e,1].item())
    assert int(sim.blast_cap[e, 1]) == exp_z
    assert abs(float(sim.spd_g[e, 1]) - exp_s) < 1e-6
    assert sim.bombs_cap[e, 0] == lb and sim.blast_cap[e, 0] == lz
print("boost=1 历史网络 ×1.3（P1 增强、P0 起点不变）OK")

# ---------- 3) boost=2（规则 bot）：P1 = 80%（6/6/1.68），P0 不变 ----------
sim.set_opp_boost(2)
sim.reset_all()
for e in range(N):
    lb = cfg.open_growth_bombs if bool(sim._is_open[e]) else cfg.growth_bombs_start
    assert sim.bombs_cap[e, 1] == 6 and sim.blast_cap[e, 1] == 6
    assert abs(float(sim.spd_g[e, 1]) - 1.68) < 1e-6
    assert sim.bombs_cap[e, 0] == lb
print("boost=2 规则 bot 80%（6/6/1.68）P0 起点不变 OK")

# ---------- 4) boost=2 掉血惩罚：clamp 回增强后起点（6/6/1.68）----------
#   开局 P1 在 80% 地板：掉血不掉属性（与 learner 在地板一致，先成长才能被扣）
oe = [e for e in range(N) if bool(sim._is_open[e])][0]
sim.pos[oe, 1] = torch.tensor([10.5, 2.5]); sim.pos[oe, 0] = torch.tensor([10.5, 1.5])
sim.bombs_cap[oe, 1] = 6; sim.blast_cap[oe, 1] = 6; sim.spd_g[oe, 1] = 1.68
sim.hp[oe] = 3; sim.invuln[oe] = 0
sim.fuse[oe, 10, 2] = 1; sim.owner[oe, 10, 2] = 0     # P0 的泡炸 P1
cb = int(sim.crate[oe].sum())
act = torch.zeros(N, 2, 2, dtype=torch.long)
sim.step(act, auto_reset=False)
assert int(sim.hp[oe, 1]) == 2
assert int(sim.bombs_cap[oe, 1]) == 6, "地板掉血不掉属性（clamp 回增强起点 6）"
assert int(sim.crate[oe].sum()) == cb, "地板掉血不生箱（守恒）"
print("boost=2 地板（6/6/1.68）掉血不掉属性、不生箱 OK")

#   满属性（7/7/2.1）挨炸 → 扣回 80% 起点（6/6/1.8），lost=1+1+2=4 回收 4 箱
sim.bombs_cap[oe, 1] = 7; sim.blast_cap[oe, 1] = 7; sim.spd_g[oe, 1] = cfg.growth_speed_max
sim.hp[oe] = 3; sim.invuln[oe] = 0
sim.fuse[oe, 10, 2] = 1; sim.owner[oe, 10, 2] = 0
cb = int(sim.crate[oe].sum())
sim.step(act, auto_reset=False)
assert int(sim.hp[oe, 1]) == 2
assert int(sim.bombs_cap[oe, 1]) == 6, "满属性掉血扣 1 层回地板 6"
assert int(sim.blast_cap[oe, 1]) == 6
assert abs(float(sim.spd_g[oe, 1]) - (cfg.growth_speed_max - 2*cfg.growth_speed_step)) < 1e-5
assert int(sim.crate[oe].sum()) == cb + 4, "lost=4 → 回收 4 箱"
print("boost=2 满属性掉血扣回 6/6/1.8 + 回收+4 OK")

# ---------- 5) 位置对称化：纯 open 关 P0/P1 出生点约半交换、属性随 pid ----------
cfg2 = SimConfig(map_mode="corridor", open_fraction=1.0, ring_fraction=0.0,
                 n_players=2, speed=3.0, max_steps=300,
                 open_crate_cross=True, hit_attr_penalty=2)
sim2 = BatchedSim(cfg2, 128, device="cpu", seed=3)
sim2.set_opp_boost(2)
sim2.reset_all()
# open 出生点 (6,4)/(6,8)：统计 P0 在 (6,4) 的比例 ≈50%
pos0 = sim2.pos[:, 0]; pos1 = sim2.pos[:, 1]
left0 = int(((pos0[:, 1] < 6.0)).sum())
total = sim2.num_envs
print(f"P0 在左侧出生点比例: {left0}/{total} ({left0/total:.0%}) —— 应≈50%（混合有少量 corridor 关）")
assert 0.3 < left0 / total < 0.7, "P0 出生侧应随机化（~50%）"
# 属性随 pid：P0 = 起点、P1 = 80%，与出生侧无关
for e in range(128):
    assert sim2.bombs_cap[e, 0] == cfg2.open_growth_bombs
    assert sim2.bombs_cap[e, 1] == 6
print("位置对称化：P0 出生侧 ~50% 随机、属性按 pid 绑定 OK")

# ---------- 6) 混合关（open 50%）：交叉验证 —— 交换只动位置不动属性 ----------
sim3 = BatchedSim(SimConfig(map_mode="corridor", open_fraction=0.5, ring_fraction=0.0,
                            n_players=2, speed=3.0, max_steps=300,
                            open_crate_cross=True, hit_attr_penalty=2),
                  N, device="cpu", seed=9)
sim3.set_opp_boost(1)
sim3.reset_all()
for e in range(N):
    lb = cfg.open_growth_bombs if bool(sim3._is_open[e]) else cfg.growth_bombs_start
    lz = cfg.open_growth_blast if bool(sim3._is_open[e]) else cfg.growth_blast_start
    # 无论出生在哪侧，P0 都应该是 learner 起点、P1 是 ×1.3
    assert sim3.bombs_cap[e, 0] == lb, (e, sim3.bombs_cap[e].tolist())
    assert int(sim3.bombs_cap[e, 1]) == int(round(lb * cfg.opp_hist_mult))
    assert sim3.blast_cap[e, 0] == lz
print("混合关属性随 pid 绑定（位置交换不串属性）OK")
print("SMOKE BOOST OK")
