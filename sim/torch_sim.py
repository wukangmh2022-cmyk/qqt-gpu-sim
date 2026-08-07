"""批量参考模拟器：纯 PyTorch 张量运算，cpu / mps / cuda 都能跑。

定位有两个：
1. **正确性基准**。规则以 RULES.md 为准，CUDA kernel 必须和它逐 tick 一致
   （`tests/test_parity.py`）。这一版可读、可断点、可在 Mac 上跑。
2. **性能对照**。同一批 env 数下和 CUDA 版比 env-steps/s，是 benchmark 的分母。

布局：所有状态都是 (N, ...) 的张量，N = 并行关卡数。角色维度 P <= 4，
在 Python 侧展开成小循环 —— 单关卡内的"玩家并行"是伪需求，
真正的并行度来自 N。

坐标是连续的 float32。每个 tick 角色沿最后按下的方向匀速前进
`speed / tick_hz` 格；放泡是独立的 trigger，和移动同帧生效。
"""

from __future__ import annotations

import torch

from .blast import danger_map, rays, resolve_explosions
from .config import DIRS, SimConfig
from .mapgen import make_bricks, make_ring_bricks, make_ring_walls, make_walls
from .move import center_cell, move_players
from .obs import encode_obs, legal_mask


class BatchedSim:
    def __init__(
        self,
        cfg: SimConfig,
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        n, h, w, p = num_envs, cfg.height, cfg.width, cfg.n_players
        d = self.device
        self.wall = torch.zeros((n, h, w), dtype=torch.bool, device=d)
        self.brick = torch.zeros((n, h, w), dtype=torch.bool, device=d)
        self.crate = torch.zeros((n, h, w), dtype=torch.bool, device=d)  # 宝箱：砖炸掉后变
        self.fuse = torch.zeros((n, h, w), dtype=torch.int16, device=d)
        self.owner = torch.full((n, h, w), -1, dtype=torch.int8, device=d)
        # 每颗泡自己的威力（成长系统：放泡那一刻按当前档位快照）。
        # 引信归 0 引爆时用这颗泡存的 blast 而不是"当前 t 的 blast"。
        self.bomb_blast = torch.zeros((n, h, w), dtype=torch.int16, device=d)
        self.pos = torch.zeros((n, p, 2), dtype=torch.float32, device=d)
        self.alive = torch.ones((n, p), dtype=torch.bool, device=d)
        self.hp = torch.full((n, p), cfg.max_hp, dtype=torch.uint8, device=d)
        self.since_bomb = torch.zeros((n, p), dtype=torch.int32, device=d)
        self.t = torch.zeros((n,), dtype=torch.long, device=d)
        # 成长系统状态（corridor 用）：每个玩家独立的 泡数上限/威力/速度倍率。
        # open 模式恒为 cfg 默认值；corridor 开局 start、由成长事件随机递增。
        self.bombs_cap = torch.zeros((n, p), dtype=torch.long, device=d)
        self.blast_cap = torch.zeros((n, p), dtype=torch.long, device=d)
        self.spd_g = torch.ones((n, p), dtype=torch.float32, device=d)
        # 无敌保护期：被炸伤后剩余无敌 tick（>0 时掉血无效、不触发对方 hit 奖励）
        self.invuln = torch.zeros((n, p), dtype=torch.long, device=d)
        # 炸弹雨模式开关（每局独立掷，见 reset_）：True = 本局是炸弹雨关
        self._hazard = torch.zeros((n,), dtype=torch.bool, device=d)
        # open 关标记（混合地图）：True = 本局是纯空场 open 关（时间成长 + 掉血惩罚
        # 只对 open 关生效；corridor/ring 走踩箱成长，不受影响）
        self._is_open = torch.zeros((n,), dtype=torch.bool, device=d)
        # 每 env 的宝箱成长爆率（环岛 100%，corridor 用 cfg.growth_crate_prob）
        self.crate_prob = torch.full((n,), cfg.growth_crate_prob, device=d)
        # 回收宝箱标记：掉血回收生成的箱（_scatter_recycle）→ 踩到**必升**
        # （recycle_crate_prob=1.0，不掷全局爆率）—— 掉多少层补多少箱、
        # 踩了必还原，总量守恒可核算（用户定：受伤爆出来的才是 100%）。
        self._recycle_crate = torch.zeros((n, h, w), dtype=torch.bool, device=d)
        # combo 连击状态（combo_reward>0 时用）：不掉血连续造成伤害 = 连击，
        # 连击数越高分越多、间隔越短分越多；掉血（被打）打断连击。
        self._combo = torch.zeros((n, p), dtype=torch.long, device=d)
        self._last_hit = torch.full((n, p), -10**9, dtype=torch.long, device=d)
        # open 关宝箱布局缓存（懒初始化，cfg 固定）：排除格 / 随机回收池 / 十字格
        self._open_excl: torch.Tensor | None = None
        self._open_avail: torch.Tensor | None = None
        self._open_cross: list[tuple[int, int]] | None = None
        # 每 env 地图类型（0=corridor 1=open 2=ring）与掉血 clamp 起点（per-env**per
        # 玩家**，各模式/各侧起点不同：learner=open_growth_*/growth_*_start，
        # 增强对手 =×opp_hist_mult 或 opp_growth_*）。reset_ 按地图分支填；
        # 掉血惩罚 clamp 用（见 step 的 hit_attr_penalty 块）。
        self._map_kind = torch.zeros((n,), dtype=torch.long, device=d)
        self._lo_bombs = torch.full((n, p), cfg.growth_bombs_start, device=d)
        self._lo_blast = torch.full((n, p), cfg.growth_blast_start, device=d)
        self._lo_spd = torch.full((n, p), cfg.growth_speed_start, device=d)
        # 对手初始属性增强（训练难度，训练侧 build_opponents 后 set_opp_boost）：
        #   0 = 对手与 learner 同起点（对称，默认）
        #   1 = 历史网络（fixed ckpt / 池快照）→ 起点 × opp_hist_mult
        #   2 = 规则 bot → opp_growth_*（80%）
        # reset_ 按它对 pid 1 设初始属性与掉血 clamp 起点。
        self._opp_boost = torch.zeros((n,), dtype=torch.long, device=d)
        # 探索奖励自适应退火系数（论文 α = 1-tanh(k·x)）：训练侧随击杀能力
        # 每迭代 set_explore_coef，step() 里探索类塑形（放炮三件套/连锁/吃箱）
        # 乘此系数 —— 前期=1 探索满格，后期击杀上来自动归零，纯赢比赛。
        # 默认 1.0（对打/测试/无退火训练行为不变）。
        self._explore_coef = 1.0
        # 掉血回收的全局排除集（三类出生点四邻 + open 十字带，懒缓存）
        self._gen_excl: torch.Tensor | None = None
        # 对打窗口可给玩家侧单独提速（见 move_players 的 speed_mult）：
        # 形状 (n, p)，None 表示全 1.0 —— 训练路径保持 None，行为不变
        self.speed_mult: torch.Tensor | None = None
        # CUDA graph 随机源（capture 外预填，capture 内只读）：None = 用设备 RNG
        self._rand_buf: torch.Tensor | None = None
        self.reset_all()

    # ---------------- 场地生成 / 重置 ----------------

    def _make_walls(self, count: int) -> torch.Tensor:
        return make_walls(self.cfg, count, self.gen, self.device)

    def _make_bricks(self, count: int) -> torch.Tensor:
        return make_bricks(self.cfg, count, self.gen, self.device)

    def reset_all(self) -> None:
        self.reset_(torch.ones((self.num_envs,), dtype=torch.bool, device=self.device))

    def reset_(self, mask: torch.Tensor) -> None:
        """就地重置 mask 为 True 的关卡。

        **混合地图**：每局按 open_fraction 掷 open/corridor 两类关。
        - open 关：纯空场（无墙无砖无宝箱），成长初始 = 上限的 80%
          （open_growth_*），出生点 = 整宽中线均分 —— 逼 AI 学真交战。
        - corridor 关：顶部永久墙 + 左右 brick + 宝箱成长，出生点贴脸。
        观测无地图类型标记，网络靠状态差异自然适配两类地图。
        """
        count = int(mask.sum())
        if count == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]
        # open 标记本轮先清（防"上次是 open、这次是 corridor"的 env 残留），
        # open 分支下面再置 True —— 不能在末尾清，掉血惩罚整局要读它。
        self._is_open[idx] = False
        if self.cfg.map_mode != "corridor":
            # 纯 open 训练（或空场测试）：固定能力无成长。
            # 对手（pid 1）初始属性按 _opp_boost（训练难度）。
            self.wall[idx] = False
            self.brick[idx] = False
            b0 = torch.full((count,), self.cfg.max_bombs, dtype=torch.float,
                            device=self.device)
            z0 = torch.full((count,), self.cfg.blast, dtype=torch.float,
                            device=self.device)
            s0 = torch.full((count,), 1.0, dtype=torch.float, device=self.device)
            self.bombs_cap[idx, 0] = self.cfg.max_bombs
            self.blast_cap[idx, 0] = self.cfg.blast
            self.spd_g[idx, 0] = 1.0
            ob, oz, os = self._opp_start(idx, b0, z0, s0)
            self.bombs_cap[idx, 1] = ob
            self.blast_cap[idx, 1] = oz
            self.spd_g[idx, 1] = os
            self._map_kind[idx] = 1                     # open
            self._lo_bombs[idx, 0] = self.cfg.max_bombs
            self._lo_blast[idx, 0] = self.cfg.blast
            self._lo_spd[idx, 0] = 1.0
            self._lo_bombs[idx, 1] = ob
            self._lo_blast[idx, 1] = oz
            self._lo_spd[idx, 1] = os
            spawns = torch.tensor(
                self.cfg.spawn_pos(), dtype=torch.float32, device=self.device)
            self.pos[idx] = spawns.unsqueeze(0).expand(count, -1, -1)
            # 位置对称化：约一半 env 交换 P0/P1 出生点（消除"模型恒打物理左侧"偏置）。
            # 属性按 pid 绑定，与出生侧无关 —— 位置与属性解耦。
            # 注意：链式高级索引 `self.pos[idx][sw, 0] = x` 不写回原张量，
            # 必须取出本地拷贝交换后再一步写回。
            pos_sel = self.pos[idx].clone()
            sw = (torch.rand(count, generator=self.gen) < 0.5).to(self.device)
            if bool(sw.any()):
                tmp = pos_sel[sw, 0].clone()
                pos_sel[sw, 0] = pos_sel[sw, 1]
                pos_sel[sw, 1] = tmp
            self.pos[idx] = pos_sel
        else:
            # **混合地图**：每局按 open_fraction / ring_fraction 随机掷三类关
            # （余量 = corridor）：
            #   open 关：纯空场（无墙无砖无宝箱），成长初始 = 上限 80%，
            #     出生点整宽中线均分 —— 逼 AI 学真交战。
            #   ring 关：中间 7×7 永久墙山体（不可行走不可炸）+ 环带**稀疏**
            #     brick（ring_brick_density，不是全充满）+ 宝箱爆率 100%
            #     + 四角出生点 —— 中央障碍、周边可炸、四角有立足之地。
            #   corridor 关：顶部永久墙 + 左右 brick + 宝箱成长，出生点贴脸。
            # 观测无地图类型标记，网络靠状态差异自然学会适配。
            r = torch.rand(count, generator=self.gen)
            is_open = r < self.cfg.open_fraction
            is_ring = ~is_open & (r < self.cfg.open_fraction + self.cfg.ring_fraction)
            open_idx = idx[is_open]
            ring_idx = idx[is_ring]
            corr_idx = idx[~(is_open | is_ring)]

            if open_idx.numel():
                self.wall[open_idx] = False
                self.brick[open_idx] = False
                no = int(open_idx.numel())
                b0 = torch.full((no,), float(self.cfg.open_growth_bombs),
                                dtype=torch.float, device=self.device)
                z0 = torch.full((no,), float(self.cfg.open_growth_blast),
                                dtype=torch.float, device=self.device)
                s0 = torch.full((no,), float(self.cfg.open_growth_speed),
                                dtype=torch.float, device=self.device)
                self.bombs_cap[open_idx, 0] = self.cfg.open_growth_bombs
                self.blast_cap[open_idx, 0] = self.cfg.open_growth_blast
                self.spd_g[open_idx, 0] = self.cfg.open_growth_speed
                ob, oz, os = self._opp_start(open_idx, b0, z0, s0)
                self.bombs_cap[open_idx, 1] = ob
                self.blast_cap[open_idx, 1] = oz
                self.spd_g[open_idx, 1] = os
                self.crate_prob[open_idx] = 1.0        # open 宝箱 100% 有东西（踩到必升）
                self._is_open[open_idx] = True         # 标记：掉血惩罚 + 宝箱回收生效
                self._map_kind[open_idx] = 1           # open
                self._lo_bombs[open_idx, 0] = self.cfg.open_growth_bombs
                self._lo_blast[open_idx, 0] = self.cfg.open_growth_blast
                self._lo_spd[open_idx, 0] = self.cfg.open_growth_speed
                self._lo_bombs[open_idx, 1] = ob
                self._lo_blast[open_idx, 1] = oz
                self._lo_spd[open_idx, 1] = os
                # 中心十字开局池在 **crate 清零之后**统一撒（见下方
                # _place_open_cross_crates 调用 —— 这里撒会被 self.crate[idx]=False 清掉）。
            if ring_idx.numel():
                # 环岛：中间 7×7 永久墙山体（不可行走不可炸）+ 环带稀疏 brick
                #   （ring_brick_density，出生点及四邻已清空）+ 四角出生点。
                # 玩家在山体外围绕圈 —— 中央障碍、周边可炸、四角有立足之地。
                self.wall[ring_idx] = make_ring_walls(
                    self.cfg, int(ring_idx.numel()), self.device)
                self.brick[ring_idx] = make_ring_bricks(
                    self.cfg, int(ring_idx.numel()), self.gen, self.device)
                nr = int(ring_idx.numel())
                b0 = torch.full((nr,), float(self.cfg.growth_bombs_start),
                                dtype=torch.float, device=self.device)
                z0 = torch.full((nr,), float(self.cfg.growth_blast_start),
                                dtype=torch.float, device=self.device)
                s0 = torch.full((nr,), float(self.cfg.growth_speed_start),
                                dtype=torch.float, device=self.device)
                self.bombs_cap[ring_idx, 0] = self.cfg.growth_bombs_start
                self.blast_cap[ring_idx, 0] = self.cfg.growth_blast_start
                self.spd_g[ring_idx, 0] = self.cfg.growth_speed_start
                ob, oz, os = self._opp_start(ring_idx, b0, z0, s0)
                self.bombs_cap[ring_idx, 1] = ob
                self.blast_cap[ring_idx, 1] = oz
                self.spd_g[ring_idx, 1] = os
                self.crate_prob[ring_idx] = self.cfg.ring_crate_prob
                self._map_kind[ring_idx] = 2           # ring
                self._lo_bombs[ring_idx, 0] = self.cfg.growth_bombs_start
                self._lo_blast[ring_idx, 0] = self.cfg.growth_blast_start
                self._lo_spd[ring_idx, 0] = self.cfg.growth_speed_start
                self._lo_bombs[ring_idx, 1] = ob
                self._lo_blast[ring_idx, 1] = oz
                self._lo_spd[ring_idx, 1] = os
            if corr_idx.numel():
                self.wall[corr_idx] = self._make_walls(int(corr_idx.numel()))
                self.brick[corr_idx] = self._make_bricks(int(corr_idx.numel()))
                nc = int(corr_idx.numel())
                b0 = torch.full((nc,), float(self.cfg.growth_bombs_start),
                                dtype=torch.float, device=self.device)
                z0 = torch.full((nc,), float(self.cfg.growth_blast_start),
                                dtype=torch.float, device=self.device)
                s0 = torch.full((nc,), float(self.cfg.growth_speed_start),
                                dtype=torch.float, device=self.device)
                self.bombs_cap[corr_idx, 0] = self.cfg.growth_bombs_start
                self.blast_cap[corr_idx, 0] = self.cfg.growth_blast_start
                self.spd_g[corr_idx, 0] = self.cfg.growth_speed_start
                ob, oz, os = self._opp_start(corr_idx, b0, z0, s0)
                self.bombs_cap[corr_idx, 1] = ob
                self.blast_cap[corr_idx, 1] = oz
                self.spd_g[corr_idx, 1] = os
                self.crate_prob[corr_idx] = self.cfg.growth_crate_prob
                self._map_kind[corr_idx] = 0           # corridor
                self._lo_bombs[corr_idx, 0] = self.cfg.growth_bombs_start
                self._lo_blast[corr_idx, 0] = self.cfg.growth_blast_start
                self._lo_spd[corr_idx, 0] = self.cfg.growth_speed_start
                self._lo_bombs[corr_idx, 1] = ob
                self._lo_blast[corr_idx, 1] = oz
                self._lo_spd[corr_idx, 1] = os

            all_pos = torch.zeros(count, self.cfg.n_players, 2,
                                  dtype=torch.float32, device=self.device)
            if open_idx.numel():
                open_spawn = self._open_spawns()
                all_pos[is_open] = open_spawn.unsqueeze(0).expand(
                    int(open_idx.numel()), -1, -1)
            if ring_idx.numel():
                ring_spawn = self._ring_spawns()
                all_pos[is_ring] = ring_spawn.unsqueeze(0).expand(
                    int(ring_idx.numel()), -1, -1)
            if corr_idx.numel():
                corr_spawn = torch.tensor(
                    self.cfg.spawn_pos(), dtype=torch.float32,
                    device=self.device)
                all_pos[~ (is_open | is_ring)] = corr_spawn.unsqueeze(0).expand(
                    int(corr_idx.numel()), -1, -1)
            self.pos[idx] = all_pos
            # 位置对称化（混合地图）：约一半 env 交换 P0/P1 出生点（消除"模型恒打
            # 物理左侧"偏置）。属性按 pid 绑定，与出生侧无关 —— 位置与属性解耦。
            # 链式高级索引赋值不写回，必须本地交换后一步写回。
            pos_sel = self.pos[idx].clone()
            sw = (torch.rand(count, generator=self.gen) < 0.5).to(self.device)
            if bool(sw.any()):
                tmp = pos_sel[sw, 0].clone()
                pos_sel[sw, 0] = pos_sel[sw, 1]
                pos_sel[sw, 1] = tmp
            self.pos[idx] = pos_sel

        self.crate[idx] = False
        self._recycle_crate[idx] = False          # 回收标记随局清（防跨局残留）
        self._combo[idx] = 0                      # 连击随局清零
        self._last_hit[idx] = -10**9
        # 开局**中心十字宝箱**（open 关开局属性池）：横竖各 2 排 ≈46 格、
        # 100% 有东西、踩到必升一属性 —— 必须在上面 crate 清零**之后**撒。
        # 属性是稀缺资源：开局给一池，掉血扣的会随机回收补充（总量守恒）。
        if self.cfg.open_crate_cross:
            open_reset = idx[self._is_open[idx]]
            if open_reset.numel():
                self._place_open_cross_crates(open_reset)
        self.fuse[idx] = 0
        self.owner[idx] = -1
        self.bomb_blast[idx] = 0
        self.alive[idx] = True
        self.hp[idx] = self.cfg.max_hp
        # 新一局算"冷却已完成"：开局第一泡也享受近身定位分（place_dist_cooldown
        # < passivity_ticks，不影响被动罚的 60 tick 门槛）。
        self.since_bomb[idx] = self.cfg.place_dist_cooldown
        self.invuln[idx] = 0
        self.t[idx] = 0

        # 炸弹雨模式：每局独立掷（与地图类型正交）。炸弹雨关强制泡数上限 0
        # （玩家无法放泡 —— _place_bombs 的 live < 0 恒 False；放泡头也被
        # legal_mask 屏蔽），环境炸弹由 _hazard_wave 每 hazard_wave_ticks
        # tick 播撒（owner = n_players，只进危险图通道）。
        if self.cfg.hazard_fraction > 0:
            hz_r = (torch.rand(count, generator=self.gen)
                    < self.cfg.hazard_fraction).to(self.device)
            self._hazard[idx] = hz_r
            hz_idx = idx[hz_r]
            if hz_idx.numel():
                self.bombs_cap[hz_idx] = 0
        else:
            self._hazard[idx] = False

    def set_opp_boost(self, v: int) -> None:
        """训练侧设置对手初始属性增强档（见 _opp_boost 注释：0 同起点 / 1 历史
        网络 ×opp_hist_mult / 2 规则 bot 80%）。所有 env 用同一档（1v1 单阶段
        每迭代对手一致；1v2 需 per-env 时另做）。"""
        self._opp_boost.fill_(int(v))

    def set_explore_coef(self, v: float) -> None:
        """训练侧设置探索奖励退火系数（论文 α = 1-tanh(k·x)，随击杀能力平滑
        归零）。step() 里探索类塑形（放炮三件套/连锁兑现/吃箱）乘此系数。"""
        self._explore_coef = float(v)

    def _opp_start(self, idx: torch.Tensor, b0: torch.Tensor, z0: torch.Tensor,
                   s0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对手（pid 1）初始属性按 `_opp_boost[idx]` 三档向量化计算：
        0 = 与 learner 同起点（b0/z0/s0）；1 = 历史网络 ×opp_hist_mult
        （round + clamp 上限）；2 = 规则 bot opp_growth_*（80%）。"""
        cfg = self.cfg
        bo = self._opp_boost[idx]
        m = cfg.opp_hist_mult
        b = torch.where(bo == 2, torch.full_like(b0, cfg.opp_growth_bombs),
                        torch.where(bo == 1,
                                    (b0 * m).round().clamp(max=cfg.growth_bombs_max),
                                    b0))
        z = torch.where(bo == 2, torch.full_like(z0, cfg.opp_growth_blast),
                        torch.where(bo == 1,
                                    (z0 * m).round().clamp(max=cfg.growth_blast_max),
                                    z0))
        s = torch.where(bo == 2, torch.full_like(s0, cfg.opp_growth_speed),
                        torch.where(bo == 1,
                                    (s0 * m).clamp(max=cfg.growth_speed_max),
                                    s0))
        return b.long(), z.long(), s

    def _open_spawns(self) -> torch.Tensor:
        """open 关出生点：整宽中线均分（P=2 时 (4.5,6.5)/(8.5,6.5)）。"""
        cfg = self.cfg
        h, w = float(cfg.height), float(cfg.width)
        row = (h - 1) / 2.0 + 0.5
        cols = [(w - 1) * (i + 1) / (cfg.n_players + 1) + 0.5
                for i in range(cfg.n_players)]
        return torch.tensor([[row, c] for c in cols],
                            dtype=torch.float32, device=self.device)

    def _open_geometry(self) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """open 关布局缓存（懒初始化，cfg 固定，只算一次）：

        - excl：open 排除格集合（全局平铺坐标）—— open 出生点及其 4 邻 +
          **中心十字带**（十字格是开局池专属，随机回收不落在这里）。
        - cross：中心十字格（行列中心 ±{0,1}，横竖各 2 排），开局池专用。
        """
        cfg = self.cfg
        if self._open_excl is not None:
            return self._open_excl, self._open_cross
        h, w = cfg.height, cfg.width
        excl: set[tuple[int, int]] = set()
        # 出生点及其 4 邻（open 关开局脚下必无箱）。必须用 open 关自己的出生点
        # `_open_spawns()`（整宽中线均分），不能用 cfg.spawn_cells() —— 后者在
        # map_mode="corridor" 时返回的是 corridor 出生点（右侧），不对。
        for s in self._open_spawns().tolist():
            rr, cc = int(s[0]), int(s[1])
            excl.add((rr, cc))
            for dr, dc in DIRS:
                nr, nc = rr + int(dr), cc + int(dc)
                if 0 <= nr < h and 0 <= nc < w:
                    excl.add((nr, nc))
        cy = (h - 1) // 2                     # 行中心（13 → 6）
        cx = (w - 1) // 2                     # 列中心（13 → 6）
        cross: set[tuple[int, int]] = set()
        # 横竖各 2 排：行带 {cy-1, cy} 全宽 ∪ 列带 {cx-1, cx} 全高
        # （13×13：26+26−4 重叠 = 48 格 ≈46，再扣出生点四邻后落图 ~44）。
        for cc in range(w):
            for rr in (cy - 1, cy):
                if (rr, cc) not in excl:
                    cross.add((rr, cc))
        for rr in range(h):
            for cc in (cx - 1, cx):
                if (rr, cc) not in excl:
                    cross.add((rr, cc))
        cross = sorted(cross)
        excl |= set(cross)
        excl_flat = torch.tensor(
            [rr * w + cc for rr, cc in excl],
            dtype=torch.long, device=self.device)
        self._open_excl, self._open_cross = excl_flat, cross
        return excl_flat, cross

    def _recycle_excl(self) -> torch.Tensor:
        """掉血回收的**全模式**排除格（全局平铺坐标，懒缓存）：
        三类出生点四邻（open 整宽均分 / corridor / ring 四角） + open 中心十字带。
        回收宝箱只落在无墙无砖可通行格，且绝不叠出生点脚下/开局十字池。
        """
        cfg = self.cfg
        if self._gen_excl is not None:
            return self._gen_excl
        h, w = cfg.height, cfg.width
        excl: set[tuple[int, int]] = set()
        spawn_sets = [self._open_spawns().tolist(),
                      [(float(r), float(c)) for r, c in cfg.spawn_cells()],
                      [(float(r), float(c)) for r, c in cfg.ring_spawn_cells()]]
        for spawns in spawn_sets:
            for s in spawns:
                rr, cc = int(s[0]), int(s[1])
                excl.add((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + int(dr), cc + int(dc)
                    if 0 <= nr < h and 0 <= nc < w:
                        excl.add((nr, nc))
        excl |= set(self._open_geometry()[1])     # open 十字带
        flat = torch.tensor(sorted(rr * w + cc for rr, cc in excl),
                            dtype=torch.long, device=self.device)
        self._gen_excl = flat
        return flat

    def _place_open_cross_crates(self, idx: torch.Tensor) -> None:
        """open 关开局**中心十字宝箱**：横竖各 2 排 —— 行带 {cy-1, cy}
        全宽 + 列带 {cx-1, cx} 全高（13×13 → 2×13 + 2×13 − 2×2 重叠 = 48 格
        ≈46，扣除出生点四邻后落图 ≈42），100% 有东西、踩到必升一属性。
        出生点及其四邻已在 cross 定义里排除（开局脚下必无箱）。
        向量化批量落格（(ne, K) 广播索引），和 corridor 的 brick 生成同风格。
        """
        _, cross = self._open_geometry()
        rows = torch.tensor([r for r, _ in cross], dtype=torch.long,
                            device=self.device)
        cols = torch.tensor([c for _, c in cross], dtype=torch.long,
                            device=self.device)
        self.crate[idx.unsqueeze(1), rows.unsqueeze(0), cols.unsqueeze(0)] = True

    def _scatter_recycle(self, idx: torch.Tensor, counts: torch.Tensor) -> None:
        """掉血回收（**全地图模式**）：在**无墙无砖可通行格**随机撒宝箱。

        counts (k,) 是每 env 撒的个数（= 该玩家实际被扣的层数，见 hit 块）。
        排除出生点四邻 + open 十字带（回收不叠开局池）；corridor/ring 有砖墙，
        只能撒在无墙无砖格（否则永远踩不到 = 白回收）。每 env 可用格不同
        （corridor 砖布局随机）→ per-env 处理，掉血低频可接受。
        不放回抽样（randperm）：掉 N 层必现 N 个新箱，严格守恒。
        """
        cfg = self.cfg
        total = int(counts.sum())
        if total == 0:
            return
        excl = self._recycle_excl()
        avail = (~self.wall[idx] & ~self.brick[idx]).view(-1, cfg.height * cfg.width)
        avail = avail.clone()
        avail[:, excl] = False
        for i, e in enumerate(idx.tolist()):
            c = int(counts[i])
            if c <= 0:
                continue
            cells = avail[i].nonzero(as_tuple=True)[0]
            if cells.numel() == 0:
                continue
            k = min(c, int(cells.numel()))
            pick = cells[torch.randperm(cells.numel(), generator=self.gen)[:k]
                         .to(self.device)]
            self.crate[e, pick // cfg.width, pick % cfg.width] = True
            # 标记回收箱：踩到必升（recycle_crate_prob=1.0，不掷全局爆率）
            self._recycle_crate[e, pick // cfg.width, pick % cfg.width] = True

    def _ring_spawns(self) -> torch.Tensor:
        """环岛关出生点：场地**四角**（山体在中间，玩家围着它绕圈）。"""
        return torch.tensor(
            self.cfg.ring_spawns(), dtype=torch.float32, device=self.device)

    # ---------------- 对外接口 ----------------

    def observe(self) -> torch.Tensor:
        return encode_obs(
            self.cfg, self.wall, self.fuse, self.owner, self.pos, self.alive,
            self.t, self.brick, self.bomb_blast,
            crate=self.crate, invuln=self.invuln, bombs_p=self.bombs_cap,
        )

    def legal_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (move_mask (N,P,5), bomb_mask (N,P,2))。"""
        return legal_mask(
            self.cfg, self.wall, self.fuse, self.owner, self.pos, self.alive,
            self.brick, self.bombs_cap,
        )

    # ---------------- CUDA graph 热路径（训练用，跳过 PPO 依赖的 mask 奖励） ----------------

    def capture_graph(self, actions: torch.Tensor, obs_buf: torch.Tensor,
                      mmask_buf: torch.Tensor, bmask_buf: torch.Tensor,
                      reward_buf: torch.Tensor, done_buf: torch.Tensor,
                      winner_buf: torch.Tensor) -> None:
        """把 step 及其输出固定进 CUDA graph（仅 cuda 后端）。

        要求 step 全 in-place（已满足）+ 输出写入固定 buffer。graph 消除每
        tick ~几十个 kernel 的 launch 开销 —— 当前 mask 16ms 大头就是 launch。
        首图后 graph_step() 每 tick 一次 replay。
        """
        self._g = torch.cuda.CUDAGraph()
        self._obs_buf, self._mmask_buf, self._bmask_buf = obs_buf, mmask_buf, bmask_buf
        self._reward_buf, self._done_buf, self._winner_buf = \
            reward_buf, done_buf, winner_buf
        self._actions = actions
        # 随机源 buffer：capture 外填新值，capture 内只读（HIP stream-capture 不允许
        # 设备 RNG）。每玩家开箱 1 个 + 属性 1 个 = n×(2P+1) 够用。
        self._rand_buf = torch.empty(
            self.num_envs * (2 * self.cfg.n_players + 1), device=actions.device)
        # warmup（graph 外的同形状前向，分配各层工作区）
        for _ in range(3):
            self._graph_body()
        torch.cuda.synchronize()
        with torch.cuda.graph(self._g):
            self._graph_body()
        torch.cuda.synchronize()

    def _graph_body(self) -> None:
        """graph 捕获/回放的都是这一段：观察→掩码→step，输出写固定 buffer。"""
        self._obs_buf.copy_(self.observe())
        mm, bm = self.legal_mask()
        self._mmask_buf.copy_(mm)
        self._bmask_buf.copy_(bm)
        reward, done, info = self.step(self._actions, auto_reset=False)
        self._reward_buf.copy_(reward)
        self._done_buf.copy_(done.float())
        self._winner_buf.copy_(info["winner"].float())

    def graph_step(self) -> None:
        self._g.replay()

    def graph_refill_rand(self) -> None:
        """replay 前调用：给随机源 buffer 填新值（capture 外，graph 只读）。"""
        if self._rand_buf is not None:
            self._rand_buf.uniform_()

    def graph_actions(self) -> torch.Tensor:
        return self._actions

    def _bombs_p(self) -> torch.Tensor:
        """当前每个玩家的泡数上限 (N,P)（corridor 逐人成长；open 恒为 max_bombs）。"""
        return self.bombs_cap

    def state_dict(self) -> dict[str, torch.Tensor]:
        """给 parity 测试用的完整状态快照。"""
        return {
            "wall": self.wall.clone(),
            "brick": self.brick.clone(),
            "crate": self.crate.clone(),
            "fuse": self.fuse.clone(),
            "owner": self.owner.clone(),
            "bomb_blast": self.bomb_blast.clone(),
            "pos": self.pos.clone(),
            "alive": self.alive.clone(),
            "hp": self.hp.clone(),
            "since_bomb": self.since_bomb.clone(),
            "t": self.t.clone(),
        }

    def load_state_dict(self, snap: dict[str, torch.Tensor]) -> None:
        for key, val in snap.items():
            getattr(self, key).copy_(val)

    # ---------------- 一个 tick ----------------

    def _blast_map(self) -> torch.Tensor:
        """(N,H,W) int32：每格泡泡的威力。0（手工种泡/未设）回退 cfg.blast。"""
        return torch.where(self.bomb_blast > 0,
                           self.bomb_blast.long(), self.cfg.blast)

    def _grow_player_vec(self, pl: int, hits_per_env: torch.Tensor,
                         alive_mask: torch.Tensor) -> None:
        """玩家 pl 本 tick 的成长（向量化，GPU 批量，**CUDA graph 兼容**）。

        hits_per_env (n,)：该玩家每 env 触发的成长次数（0/1）。三属性均匀
        分配用 `_rand_buf`（捕获外预填）读取，不用设备 RNG；按 env 就地
        scatter_add 后 clamp 到上限。死人不成长（alive_mask 过滤）。
        **零 host 同步、零设备 RNG** —— graph capture 可过。
        """
        cfg = self.cfg
        n = self.num_envs
        d = self.pos.device
        hits = hits_per_env * alive_mask.long()
        # 每 env 读一个随机数决定升哪个属性（hits 为 0 的 env 该值被忽略）。
        # 随机数从 `_rand_buf`（捕获外预填）读取，不用设备 RNG —— graph 兼容。
        rb = (self._rand_buf[cfg.n_players * n + pl * n:
                             cfg.n_players * n + (pl + 1) * n]
              if self._rand_buf is not None
              else torch.rand(n, device=d))
        attr = (rb * 3).floor().long()               # 0/1/2 均匀
        if cfg.crate_speed_only:
            # 躲避（hazard）关宝箱只加速度：泡/威在禁放泡关是死通道，学了白学。
            # 按 `_hazard` **逐 env** 生效 —— 融合训练里正常关不受影响，
            # 仍三属性随机成长（"特殊模式宝箱只加速度"的语义）。
            attr = torch.where(self._hazard,
                               torch.full((n,), 2, dtype=torch.long, device=d), attr)
        add_bombs = (attr == 0).long() * hits
        add_blasts = (attr == 1).long() * hits
        add_spd = (attr == 2).long() * hits
        self.bombs_cap[:, pl] = torch.clamp(
            self.bombs_cap[:, pl] + add_bombs, max=cfg.growth_bombs_max)
        self.blast_cap[:, pl] = torch.clamp(
            self.blast_cap[:, pl] + add_blasts, max=cfg.growth_blast_max)
        self.spd_g[:, pl] = torch.min(
            self.spd_g[:, pl] + add_spd.float() * cfg.growth_speed_step,
            torch.full_like(self.spd_g[:, pl], cfg.growth_speed_max))

    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """actions: (N, P, 2) long，[..., 0] 是方向（含 IDLE），[..., 1] 是放泡 0/1。

        auto_reset=True 时终局关卡会就地重置，因此紧接着调用 observe()
        拿到的是新一局的首帧 —— 和主流向量化环境的语义一致。
        """
        cfg = self.cfg
        n = self.num_envs
        p = cfg.n_players                       # chase 块用（fleeing 张量）
        alive0 = self.alive.clone()
        hp_before = self.hp.clone()
        pos_before = self.pos.clone()       # 接近奖励：本 tick 移动前的位置
        move, bomb = actions[..., 0], actions[..., 1]
        d = self.pos.device
        # 成长能力（corridor）：每个玩家独立随机成长状态（bombs_cap/blast_cap/spd_g）；
        # open 模式这些张量恒为 cfg 默认值，行为与旧版逐位一致。
        bombs_p = self.bombs_cap          # (n,p)
        blast_p = self.blast_cap          # (n,p)
        spd_p = self.spd_g                # (n,p)

        # 1. 引信递减（in-place：地址稳定，CUDA graph 兼容）
        torch.where(self.fuse > 0, self.fuse - 1, self.fuse, out=self.fuse)
        # 2. 放泡（在移动前，落在这一 tick 的起始中心格：按下即落在脚下）。
        #    放泡那一刻按当前档位快照（bombs_p 上限、blast_p 威力存进 bomb_blast）。
        placed = self._place_bombs(bomb, alive0, bombs_p, blast_p)
        # 放泡奖励（即时信号，一次性）：覆盖敌人 + 连锁快爆的泡（见 _place_predict_reward）。
        # **early return**：没有放泡成功的 tick 直接跳过（火焰预测整图传播很贵 ——
        # corridor 满成长 blast=7 时 rays 每 tick 196 kernel，放泡只占 ~10% tick，
        # 其余 90% 白跑会让训练/评估显著变慢；且 _place_predict_reward 内部已对
        # 无奖励参数 early return）。CUDA graph 安全：无设备 RNG。
        if bool(placed.any()):
            place_bonus = self._place_predict_reward(placed, alive0)
        else:
            place_bonus = torch.zeros(n, cfg.n_players, device=d)
        # 被动计时：没放泡的活人 +1 tick，放成功的清零（in-place）
        self.since_bomb.add_(1)
        self.since_bomb[placed] = 0
        # 3. 连续移动 + AABB 滑动碰撞。速度 = 基础速 × 玩家成长倍率 × 对打玩家倍率。
        blocked = self.wall | self.brick | (self.fuse > 0)
        sm = spd_p
        if getattr(self, "speed_mult", None) is not None:
            sm = sm * self.speed_mult        # (n,p) × (1,p)：玩家侧倍率（对打用）
        self.pos.copy_(move_players(cfg, self.pos, move, alive0, blocked, sm))
        # 4. 爆炸与连锁：每颗泡用自己存的威力；brick 挡火但被覆盖即摧毁。
        #    **宝箱模式**：炸掉的砖变宝箱（crate），谁走到谁开 —— 不需要归属图。
        covered, triggered = resolve_explosions(
            self.fuse, self.owner, self.wall, self._blast_map(),
            cfg.max_chain, self.brick,
        )
        # 爆炸时刻的连锁兑现（chain_blast_bonus）：被**连锁提前点燃**的泡每颗
        # 给**点火源**（引信自然走完的那颗泡的主人）+0.08，奖励"先放 → 别处续
        # → 最后连起来一起爆"的布网+牵引。同一 tick 自然走完的一排泡 k=0 一分
        # 不赚 → 免疫"啪啪啪啪"贴脸连丢（放置时预测是预告，这是真兑现）。
        # 必须在清场（fuse→0、owner→-1）**之前**读 owner/引信；
        # 归点火源玩家（每 env 至多 1 颗自然泡，多颗时双方都计）。
        # **归属修复（反"自连爆白捡"）**：只给**跨 owner 连锁** —— 我引信自然走完
        # 的泡点燃**对手的**泡才算战术（借对手的雷区引爆/扩散）；点燃**自己的**
        # 泡（自己连放一排自爆）= 0 分。旧版 chained 统计所有被点燃泡（不分
        # owner），自己连放 6 颗 → 第 1 颗自然爆点燃后 5 颗 → 白捡 5×0.08=0.4，
        # 梯度把 AI 推成"满预算一股脑全丢"（实测间隔 ≤3tick 占 73%）。
        chain_bonus_p = torch.zeros(n, cfg.n_players, dtype=torch.float32,
                                    device=d)
        if cfg.chain_blast_bonus > 0 and cfg.max_chain > 1:
            nat = triggered & (self.fuse == 0)               # 引信自然走完 = 点火源
            nat_flat = nat.view(n, -1)
            own_flat = self.owner.view(n, -1)
            chained_mask = (triggered & ~nat).view(n, -1)    # 被连锁点燃的泡（n, cells）
            for pl in range(cfg.n_players):
                fired = (nat_flat & (own_flat == pl)).sum(dim=1).clamp(max=1)
                # 只数"被点燃的、且 owner ≠ 我"的泡 —— 跨 owner 连锁才算
                cross = (chained_mask & (own_flat != pl)).sum(dim=1)
                chain_bonus_p[:, pl] = cfg.chain_blast_bonus * cross * fired
        if cfg.map_mode == "corridor":
            self.crate.bitwise_or_(self.brick & covered)   # 炸掉的砖 → 宝箱（in-place）
        self.brick.bitwise_and_(~covered)                  # 摧毁砖（in-place）
        # 5. 伤害判定：以移动后的**中心格**是否着火为准（同 tick 同时结算）。
        #    着火扣 1 血，血归 0 才算死 —— 不再"一碰就死"（max_hp=1 等价旧版）。
        #    **无敌保护期**：被炸伤后 invuln_ticks 内被炸不掉血、不触发对方 hit
        #    奖励（打断"连炮往死里整对手"）；danger 图照常显示（无敌只挡掉血）。
        cell = center_cell(self.pos)
        flat = cell[..., 0] * cfg.width + cell[..., 1]
        hit = alive0 & covered.view(n, -1).gather(1, flat)
        invuln_ok = self.invuln <= 0                 # (n,p) 无敌期结束才能掉血
        hit_eff = hit & invuln_ok                    # 实际扣血命中
        hp_new = (self.hp.to(torch.int32) - hit_eff.to(torch.int32)).clamp(min=0)
        died = hit_eff & (hp_new == 0)               # 只有这 tick 血扣到 0 才计一次死亡
        self.hp.copy_(hp_new.to(torch.uint8))    # in-place
        self.alive.copy_(alive0 & ~died)         # in-place
        # 自杀判定快照：**爆炸清场前**记录每玩家在场泡数（owner/fuse 马上会被
        # 置 -1/0，清场后查"死时有没有自己泡"恒为 0 → 自杀重罚永不触发）。
        # 用 `owner == me` 计数：fuse 已在本 tick 递减（爆炸泡 fuse 1→0），
        # 但 owner 尚未清 —— 只数 owner==me 才能把"刚炸死我的那颗泡"算进去。
        own_live_snap = torch.stack([
            (self.owner == me).flatten(1).sum(dim=1)
            for me in range(cfg.n_players)], dim=1)   # (n,p) 死前在场泡数
        # 无敌期：每 tick 递减（≥0）；实际掉血的人重新进入无敌期
        self.invuln.sub_(1)
        self.invuln.clamp_(min=0)
        self.invuln[hit_eff] = cfg.invuln_ticks
        # 6. 清场，泡泡额度自然归还（owner 置 -1），威力同步清空（in-place）
        torch.where(triggered, torch.zeros_like(self.fuse), self.fuse, out=self.fuse)
        torch.where(triggered, torch.full_like(self.owner, -1), self.owner,
                    out=self.owner)
        torch.where(triggered, torch.zeros_like(self.bomb_blast), self.bomb_blast,
                    out=self.bomb_blast)
        # 7. 计步与终局
        self.t.add_(1)
        # 炸弹雨波次（hazard 模式）：在 t 累计之后、终局判定之前 ——
        # 新落炸弹 fuse 满值，不参与本 tick 结算。
        self._hazard_wave()
        n_alive = self.alive.sum(dim=1)
        done = (n_alive <= 1) | (self.t >= cfg.max_steps)

        # 稠密伤害信号：掉血要疼、打中要赚。否则"站自己泡上挨烧"零成本，
        # 危险图再清楚网络也没有动力躲。1v1 里对方掉血 = 我的泡干的。
        # （不放炮直接奖励：实测它诱导 corridor 横向刷宝箱，主次颠倒。）
        dmg = (hp_before - self.hp.to(torch.int32)).clamp(min=0).float()
        dealt = dmg.sum(dim=1, keepdim=True) - dmg           # 除自己外被造成多少伤害
        # 探索奖励退火：放炮三件套（place_bonus）乘 _explore_coef（论文 α），
        # 前期=1 鼓励炸墙/捡道具探索，后期随击杀能力自动归零（不再为刷分放炮）。
        # 命中/胜负/安全塑形（hit/win/danger/passivity）不乘 —— 主信号恒生效。
        reward = (-cfg.step_penalty * alive0.float()
                  + cfg.hit_reward * dealt - cfg.hit_reward * dmg
                  + self._explore_coef * place_bonus * alive0.float())
        # 掉血惩罚 + 宝箱回收（**全地图模式**，hit_attr_penalty>0）：被炸到掉血的
        # 玩家，泡/威/速各扣 hit_attr_penalty 层（clamp 回各自模式的起点：open →
        # open_growth_*，corridor/ring → growth_*_start）；扣掉的以宝箱**随机可
        # 通行格回收**（corridor/ring crate_prob<1 需踩中才升回；open=1.0 必升）
        # —— **每扣 1 层回收 1 箱**，属性总量守恒：炸人 = 抢属性资源，躲泡少掉血
        # = 保属性 + 不让对方捡走，逼真格斗。
        if cfg.hit_attr_penalty > 0:
            for pl in range(cfg.n_players):
                # 全模式生效：不再限定 _is_open（corridor/ring 掉血同样扣属性）。
                # _map_kind/_lo_* 已由 reset_ 按地图分支填好各自 clamp 起点。
                hit_pl = (dmg[:, pl] > 0) & alive0[:, pl]
                if not bool(hit_pl.any()):
                    continue
                hidx = hit_pl.nonzero(as_tuple=True)[0]
                pen = cfg.hit_attr_penalty
                nb = torch.clamp(self.bombs_cap[hidx, pl] - pen,
                                 min=self._lo_bombs[hidx, pl])
                nz = torch.clamp(self.blast_cap[hidx, pl] - pen,
                                 min=self._lo_blast[hidx, pl])
                ns = torch.max(
                    self.spd_g[hidx, pl] - pen * cfg.growth_speed_step,
                    self._lo_spd[hidx, pl])
                # 实际被扣的层数（clamp 到模式起点：起点以下无层可扣）。
                # 每 env 在起点时扣 0 → 不掉层也不生箱（严格守恒，不凭空增池）。
                lost = ((self.bombs_cap[hidx, pl] - nb)
                        + (self.blast_cap[hidx, pl] - nz)
                        + torch.round((self.spd_g[hidx, pl] - ns)
                                      / cfg.growth_speed_step)).long()
                self.bombs_cap[hidx, pl] = nb
                self.blast_cap[hidx, pl] = nz
                self.spd_g[hidx, pl] = ns
                if bool(lost.sum() > 0):
                    self._scatter_recycle(hidx, lost)
        # 爆炸时刻连锁兑现（探索塑形，乘退火系数）：被连锁提前点燃的泡每颗
        reward = reward + self._explore_coef * chain_bonus_p * alive0.float()
        # 接近奖励（approach）：朝最近对手移动（距离在 approach_dist 内且**正在
        # 缩短**）→ +approach_reward × 缩短量。只有"正在接近"才得分：原地贴脸/
        # 兜圈子不刷分。治"隔着半场对射不逼近"。向量化、零 host 同步。
        # 接近奖励（approach）：**两个互补项，专治"隔着半场对射不逼近"和
        # "追不上逃跑的对手"**：
        #   a) **接近**（对手不动/靠近你时）：朝对手移动、距离在缩短 → 每缩短
        #      1 格 +approach_reward。只有"正在接近"才得分（原地贴脸/兜圈不刷）。
        #   b) **主动追击**（对手逃跑时）：距离不变/拉大也要有收益 —— 按
        #      方向余弦：位移向量朝"对手方向"的分量 ≥0 → 每朝对手推进 1 格给
        #      chase_reward（即使距离没缩短，因为对手也在跑）。**只在对手逃跑时
        #      给**（对手位移朝"离开我"方向）：flee 的 astar 跑到哪追到哪
        #      （治"击杀不了躲避形态"）；对手不逃不给（approach 管近距接近，
        #      不会无脑冲）。chase_adj 是追击的"距离阻力"：对手越远，每格推进
        #      分越打折（1/(1+d×chase_adj)）—— 就近追杀最赚，跨场追也有正分。
        if cfg.approach_reward > 0 or cfg.chase_reward > 0:
            d_before = torch.cdist(pos_before, pos_before)   # (n,p,p) 欧氏
            d_after = torch.cdist(self.pos, self.pos)
            close = d_before < cfg.approach_dist             # (n,p,p) 接敌区
            shrink = (d_before - d_after).clamp(min=0)       # 接近量 ≥0
            # 排除自己（对角 0）与死亡玩家；每玩家对"最近有效对手"取最大缩短
            mask = close & alive0.unsqueeze(1) & alive0.unsqueeze(2)
            gain = (shrink * mask.float()).amax(dim=2)       # (n,p) 每玩家最大接近
            # 贴脸门控：接近后距离仍 < approach_gate 才给（治"隔半场空跑刷接近分"）
            near_gate = (d_after < cfg.approach_gate) & alive0.unsqueeze(1) \
                & alive0.unsqueeze(2)
            gate_ok = near_gate.any(dim=2)                   # (n,p) 有对手在贴脸距离内
            reward = reward + (cfg.approach_reward * gain * gate_ok.float()
                               * alive0.float())
            # 追击项（chase_reward>0）：**只在对手逃跑时给**（fleeing）。
            # 之前用"贴脸门控（距离<3格）"→ 躲避形态 astar（flee 持续远离）一跑出
            # 3 格外追击就零收益 → 模型放弃追杀 → 磨平（用户实测"击杀不了躲避
            # 形态的 astar"）。换成**对手逃跑判定**：对手位移朝"离开我"方向
            # （fleeing=True）→ 追击分**不限距离**照给（距离阻力 w 保留，远追分
            # 少但为正）—— flee 的 astar 跑到哪追到哪；对手原地/靠近
            # （fleeing=False）→ 追击分 0（approach 管近距离接近，不追不逃的
            # 对手 = 不会无脑冲刷分）。
            if cfg.chase_reward > 0:
                alive_opp = alive0.unsqueeze(1) & alive0.unsqueeze(2)  # (n,p,p)
                # 排除自己（对角）与死亡；每玩家取最近存活对手的距离
                d_cur = d_after.clone()
                d_cur = torch.where(alive_opp, d_cur, torch.full_like(d_cur, 1e9))
                nearb = d_cur.argmin(dim=2)                  # (n,p) 最近对手下标
                opp_idx = nearb.unsqueeze(2).expand(-1, -1, 2)
                opp_pos = torch.gather(self.pos, 1, opp_idx)   # (n,p,2)
                disp = self.pos - pos_before                  # (n,p,2) 本 tick 位移
                to_opp = opp_pos - pos_before                 # (n,p,2) 指向对手
                n_to = to_opp.norm(dim=2).clamp(min=1e-6)
                cos = (disp * to_opp).sum(dim=2) / (n_to * (disp.norm(dim=2).clamp(min=1e-6)))
                cos = cos.clamp(min=0.0)                       # 只有朝对手方向才算
                dist_f = d_after.gather(1, nearb.unsqueeze(2)).squeeze(2)  # (n,p)
                w = 1.0 / (1.0 + cfg.chase_adj * dist_f)       # 距离阻力
                chase_gain = (disp.norm(dim=2) * cos * w)      # 有效追击位移（格）
                # 对手逃跑判定：对手位移朝"离开我"方向（fleeing=True）。
                # 用"从我指向对手"的方向点积对手位移 —— 对手在远离我即逃跑。
                fleeing = torch.zeros(n, p, dtype=torch.bool, device=d)
                for me in range(p):
                    for opp in range(p):
                        if opp == me:
                            continue
                        opp_disp = self.pos[:, opp] - pos_before[:, opp]
                        to_opp_cur = self.pos[:, opp] - self.pos[:, me]
                        fleeing[:, me] |= \
                            ((opp_disp * to_opp_cur).sum(dim=-1) > 0) \
                            & alive0[:, opp]
                reward = reward + (cfg.chase_reward * chase_gain
                                   * fleeing.float() * alive0.float())
        # 宝箱拾取（corridor）：角色移动后站在宝箱格上 → 掷 growth_crate_prob 开箱。
        # **奖励与概率解耦**：踩到宝箱**必得** brick_reward（收集是密集正向信号）；
        # 成长（属性升级）仍由 hits 决定（stood & rb<prob）。未命中宝箱也消失。
        # **CUDA graph 兼容**：随机数从 `rand_buf`（捕获外预填的固定 buffer）
        # 读取，不用设备 RNG；每玩家固定读 n 个（每人至多踩 1 个箱够用），
        # 用 stood mask 过滤 —— 零 host 同步、零设备 RNG，capture 可过。
        if cfg.map_mode == "corridor":
            cell = center_cell(self.pos)
            flat = (cell[..., 0] * cfg.width + cell[..., 1])       # (n,p)
            stood = self.crate.view(n, -1).gather(1, flat)          # (n,p) 脚下有宝箱
            # 踩箱即得分（与概率无关）。**不乘退火系数**：成长是每局从起点重新
            # 开始的物理必需（炸墙→宝箱→变强），探索奖励退火只针对"刷分放炮"
            # 的塑形（place_bonus/chain），吃箱/成长信号恒生效 —— 否则后期模型
            # 停止成长直接废掉（Pommerman 后期已吃满道具可退，我们每局重新开始）。
            reward = reward + cfg.brick_reward * stood.float() * alive0.float()
            for pl in range(cfg.n_players):
                rb = (self._rand_buf[pl * n:(pl + 1) * n]
                      if self._rand_buf is not None
                      else torch.rand(n, device=d))
                # 命中判定：**回收箱必升**（recycle_crate_prob=1.0，掉血回收的
                # 箱踩了必还原 —— 掉多少层补多少，总量守恒可核算）；
                # 普通炸砖宝箱掷全局爆率（corridor=growth_crate_prob，环岛=100%）。
                rec_flat = self._recycle_crate.view(n, -1).gather(
                    1, flat[:, pl].unsqueeze(1)).squeeze(1)
                prob = torch.where(rec_flat.bool(), cfg.recycle_crate_prob,
                                   self.crate_prob)
                hits = stood[:, pl] & (rb < prob) \
                    & alive0[:, pl]                              # (n,) 命中（开箱成功）
                self._grow_player_vec(pl, hits.long(), alive0[:, pl])
            # 清掉所有被踩过的宝箱（不管命中与否，开箱即消失）。
            # **只把踩了的格子写 False**：踩的 → 0，没踩的保持原值 ——
            # 之前用 ~stood 会把"没踩的玩家脚下空地"误写成 crate（真 bug）。
            for pl in range(cfg.n_players):
                crate_flat = self.crate.view(n, -1)
                cur = crate_flat.gather(1, flat[:, pl].unsqueeze(1))
                new = torch.where(stood[:, pl].unsqueeze(1),
                                  torch.zeros_like(cur), cur)
                crate_flat.scatter_(1, flat[:, pl].unsqueeze(1), new)
                # 同步清掉被踩的回收标记（防 stale 标记把后续普通箱误判为回收箱）
                rec_flat = self._recycle_crate.view(n, -1)
                rec_cur = rec_flat.gather(1, flat[:, pl].unsqueeze(1))
                rec_new = torch.where(stood[:, pl].unsqueeze(1),
                                      torch.zeros_like(rec_cur), rec_cur)
                rec_flat.scatter_(1, flat[:, pl].unsqueeze(1), rec_new)
        # 危险区站桩罚：站在"被在场泡泡爆炸范围覆盖"的格每 tick 扣分，
        # 大小 × danger值（1-(fuse-1)/FUSE，越接近爆炸越疼）。
        # danger_map 和观测危险通道同源（同一状态、同一函数）→ 网络有直接的监督。
        # **乘 _explore_coef（探索退火）**：前期防自杀引导（站自己泡旁要疼），
        # 后期归零 —— 和放炮塑形一起退掉，只靠真实胜负信号（hit/suicide/win）。
        danger = danger_map(self.fuse, self.wall, self._blast_map(), cfg.fuse,
                            self.brick)
        cell = center_cell(self.pos)
        flat = cell[..., 0] * cfg.width + cell[..., 1]
        standing = danger.view(n, -1).gather(1, flat)
        reward = reward - self._explore_coef * cfg.danger_penalty * standing \
            * alive0.float()
        # 久不放炮罚：**有泡泡预算（还能放）**但连续 passivity_ticks tick 没放才扣；
        # 放满了（在场泡数达到当前档位上限）不扣 —— 只有消极摆烂被罚。
        has_budget = torch.zeros((n, cfg.n_players), dtype=torch.bool, device=d)
        for me in range(cfg.n_players):
            live = ((self.owner == me) & (self.fuse > 0)).flatten(1).sum(dim=1)
            has_budget[:, me] = live < bombs_p[:, me]
        passive = (self.since_bomb >= cfg.passivity_ticks) & alive0 & has_budget
        reward = reward - cfg.passivity_penalty * passive.float()
        # combo 连击奖励（combo_reward>0）：**不掉血**连续造成伤害 = 连击。
        #   连击数越高分越多（combo × combo_reward × 间隔因子），间隔越短分越多
        #   （factor = combo_gap_factor^(间隔 tick)，连击越密分越高）；
        #   **掉血（被打）打断连击**（self._combo=0）—— 防"互怼刷连击"，
        #   逼出"不掉血打伤害"的干净压制（像格斗游戏的连段）。死亡/终局清连击。
        if cfg.combo_reward > 0:
            hit_this = (dmg > 0)                 # (n,p) 本 tick 掉血
            for me in range(cfg.n_players):
                dealt_me = dealt[:, me]          # 本 tick 对对手造成的伤害
                c = self._combo[:, me]
                gap = self.t - self._last_hit[:, me]
                factor = cfg.combo_gap_factor ** gap.clamp(min=0).float()
                # 造成伤害 → 连击 +1（同一 tick 多击只 +1）并计分
                inc = (dealt_me > 0).long()
                combo_now = (c + inc) * inc      # 没造成伤害保持，造成则 +1
                pts = (combo_now.float() * cfg.combo_reward * factor) * dealt_me.clamp(max=1).float()
                reward[:, me] += pts * alive0[:, me].float()
                self._combo[:, me] = torch.where(
                    inc.bool(), combo_now, c)
                self._last_hit[:, me] = torch.where(
                    inc.bool(), self.t, self._last_hit[:, me])
                # 掉血打断：自己掉血 → 连击清零
                self._combo[:, me] = torch.where(
                    hit_this[:, me], torch.zeros_like(c), self._combo[:, me])
        # 终局胜负：
        #   死亡终局（n_alive==1）→ 唯一存活着胜 / 死者输；
        #   超时全员存活（n_alive==P）→ **血多者胜**，血平局 = 平局 0；
        #   同时死光（n_alive==0）→ 平局 0。
        # **超时全员存活（timeout_draw=True）→ 平局 0 分**：去掉"血多者胜"，
        # 让"领先龟缩到超时"没有任何回报（旧版 1.6 分是龟缩的诱惑）。
        # 死亡终局（n_alive==1）的 winner/loser 判定不变；info 与 PPO._tally /
        # ELO 判据同步：超时不再计入胜/负，只计平局。
        death_done = done & (n_alive == 1)
        winner = death_done.unsqueeze(1) & self.alive & (n_alive == 1).unsqueeze(1)
        loser = death_done.unsqueeze(1) & ~self.alive & (n_alive == 1).unsqueeze(1)
        hp = self.hp.float()                       # (N, P)
        all_alive = done & (n_alive == cfg.n_players)
        # 终局给分掩码：死亡终局必给；超时全员存活仅在 timeout_draw=False
        # （血多者胜模式）给分，timeout_draw=True（平局模式）→ 0 分。
        terminal_scaled = death_done | (all_alive & ~cfg.timeout_draw)
        if not cfg.timeout_draw:
            # 旧行为（timeout_draw=False）：超时血多者胜
            for me in range(cfg.n_players):
                others = [o for o in range(cfg.n_players) if o != me]
                wins = all_alive & (hp[:, me].unsqueeze(1) > hp[:, others]).all(dim=1)
                loses = all_alive & (hp[:, me].unsqueeze(1) < hp[:, others]).any(dim=1)
                winner[:, me] |= wins
                loser[:, me] |= loses
        # **终局 reward 按剩余血量比例**给（win_hp_scaled=True，默认）：
        #   +win_bonus/max_hp × (自己剩余血 − 对手平均剩余血)，仅终局 tick
        #   （死亡终局，或 timeout_draw=False 的超时）。
        # 反 reward-hacking：旧版离散 ±win_bonus 只看"谁活着"，领先 1 血时
        # AI 会练出"残血跟对手换命照样 +8"（把领先优势报销成无差别胜利）。
        # 新版干净击杀拿满、残血险胜按比例少拿（满血差击杀 8.0，残血 1 血险胜
        # 1.6），引导"少掉血、干净击杀"。
        if cfg.win_hp_scaled:
            for me in range(cfg.n_players):
                others = [o for o in range(cfg.n_players) if o != me]
                opp_hp = hp[:, others].mean(dim=1)
                diff = hp[:, me] - opp_hp
                reward[:, me] += (cfg.win_bonus / cfg.max_hp) * diff \
                    * terminal_scaled.float()
        else:
            # 旧行为（win_hp_scaled=False 可回退对比）：离散 ±win_bonus 只看胜负
            reward = reward + cfg.win_bonus * (winner.float() - loser.float())
        # **自杀重罚**（suicide_penalty>0）：死亡 tick 自己名下有在场泡 → 额外
        # 负奖励。实测 course8 打贴身快攻对手 98% 死因是自爆 —— 泡是几 tick 前
        # 放的，终局 -8 的 credit 归因太弱，模型把"激进放炮"和"几秒后的死"脱钩。
        # 死亡时刻 + 自爆判定（own_live_snap 死前快照）即时重罚，让"站自己泡上"
        # 本身变贵。
        if cfg.suicide_penalty > 0:
            for me in range(cfg.n_players):
                sui = died[:, me] & (own_live_snap[:, me] > 0)
                reward[:, me] -= cfg.suicide_penalty * sui.float()

        info = {"n_alive": n_alive, "blast": covered, "trig": triggered,
                "died": died, "winner": winner.clone()}
        if auto_reset:
            self.reset_(done)
        return reward, done, info

    def _place_predict_reward(self, placed: torch.Tensor,
                              alive0: torch.Tensor) -> torch.Tensor:
        """放泡当 tick 的即时奖励（(N,P)，一次性，只给放置成功的人）。

        在放置成功、本 tick 的爆炸尚未发生之前，用**当前在场状态**预测这颗
        新泡的火焰覆盖（与观测危险图同源的 rays 传播），奖励三类"有价值的放泡"：

        1. **覆盖敌人**（辐射范围照到敌人）：每人 +place_cover_reward（小分，
           无论引信长短先照到就赚）；乱放地雷/围困不赚。
        2. **连锁快爆的泡**（艺高人胆大：往已有泡上续）：火焰能点燃的现有泡，
           每颗 +place_chain_reward × **剩余引信因子**。因子随被连锁泡的引信
           剩余递减（剩余越短 → 连锁越快兑现 → 分越高，见 config 注释），
           天然奖励"往快爆的泡上续"和连锁反应。
        3. **近身定位**（火焰覆盖不到敌人时的兜底）：炮位到敌人的距离越近
           分越高（`place_dist_reward × (1 - d/radius)`），**带门槛**：
           敌人在 place_dist_radius 内、且放炮前已连续 place_dist_cooldown
           tick 没放炮（冷却 = 天然限频）。只有覆盖不到的敌人（off-cross）
           才走这条路 —— 已经照到的敌人不叠加，避免双重计分。

        一次性计入：只在放置成功的那一刻评估，之后的移动/爆炸不重复给分。
        零 host 同步以外的常规张量运算（rays 的 max 同步与 danger 惩罚同款）。
        """
        cfg = self.cfg
        n, p = self.num_envs, cfg.n_players
        w = cfg.width
        if not (cfg.place_cover_reward or cfg.place_chain_reward
                or cfg.place_dist_reward):
            return torch.zeros(n, p, device=self.pos.device)
        live = self.fuse > 0                       # 在场泡（含刚放的）
        blast_map = self._blast_map()
        # 剩余引信因子：被连锁泡 fuse 剩余 f（引信已本 tick 递减过，f=0 即将爆炸）
        #   → factor = cf + (1-cf)×(1 - f/FUSE)，f=0 → 1.0，新泡(f≈FUSE) → ≈cf
        fuse_frac = (self.fuse.float() / float(cfg.fuse)).clamp(0.0, 1.0)
        weight = (cfg.chain_time_factor
                  + (1.0 - cfg.chain_time_factor) * (1.0 - fuse_frac))
        cell = center_cell(self.pos)
        flat_cell = cell[..., 0] * w + cell[..., 1]          # (n,p) 玩家所在格
        # 刚放的泡占的格：连锁判定要排除自己（新泡覆盖自己格 ≠ 连锁自己）
        placed_map = torch.zeros(n, w * self.cfg.height,
                                 dtype=torch.bool, device=self.pos.device)
        placed_map.scatter_(1, flat_cell, placed)
        placed_map = placed_map.view(n, self.cfg.height, w)

        cover_pts = torch.zeros(n, p, device=self.pos.device)
        dist_pts = torch.zeros(n, p, device=self.pos.device)
        chain_pts = torch.zeros(n, p, device=self.pos.device)
        # 近身冷却：放炮前已连续 ≥place_dist_cooldown tick 没放炮才算"冷静放炮"
        # （评估发生在 since_bomb.add_ 之前，这里读的是放炮前的值；
        #  开局 since_bomb=0，第一泡不给近身分，之后的给 —— 天然限频）。
        cooldown_ok = self.since_bomb >= cfg.place_dist_cooldown
        cell_f = cell.float()                          # (n,p,2) 格心坐标
        for me in range(p):
            # 爆源 seed = 只在自己脚下那一个格（且真的放了泡）。之前用
            # placed[:, me].view(n,1,1) 会把"放没放"广播到全图 → rays 从每个
            # 格都发火 → 覆盖≈全图，cover/chain/dist 全部失去几何意义。
            seed = torch.zeros(n, self.cfg.height * w, dtype=torch.bool,
                               device=self.pos.device)
            seed.scatter_(1, flat_cell[:, me].unsqueeze(1), placed[:, me].view(n, 1))
            seed = seed.view(n, self.cfg.height, w)
            cov = rays(seed, self.wall, live, blast_map, self.brick)   # (n,h,w)
            cov_flat = cov.view(n, -1)
            # 1) 覆盖敌人：取每个敌人的中心格是否着火（死者不计）
            for o in range(p):
                if o == me:
                    continue
                under = cov_flat.gather(1, flat_cell[:, o].unsqueeze(1)).squeeze(1)
                cover_pts[:, me] += (under & alive0[:, o]).float()
                # 3) 近身定位：覆盖不到（off-cross）的敌人才走距离分，
                #    且必须在半径内 + 冷却（限频）—— 两个门槛都过才算
                if cfg.place_dist_reward > 0:
                    d = (cell_f[:, me] - cell_f[:, o]).norm(dim=-1)   # (n,)
                    # 只在真的放了炮才给近身分：seed 全 False 时 cov 全 False，
                    # ~under 恒 True，不加 seed 门槛会让"根本没放炮"也白拿分
                    ok = placed[:, me] & (~under) & (d < cfg.place_dist_radius) \
                        & cooldown_ok[:, me] & alive0[:, o]
                    gain = 1.0 - (d / cfg.place_dist_radius)
                    dist_pts[:, me] += (ok * gain.clamp(min=0.0)).float()
            # 2) 连锁：覆盖到现有泡（排除自己刚放的），× 剩余引信因子
            chained = cov & live & (self.owner >= 0) & ~placed_map
            chain_pts[:, me] = (weight * chained.float()).flatten(1).sum(dim=1)
        bonus = (cfg.place_cover_reward * cover_pts
                 + cfg.place_dist_reward * dist_pts
                 + cfg.place_chain_reward * chain_pts)
        return bonus * alive0.float()

    def _place_bombs(self, bomb: torch.Tensor, alive0: torch.Tensor,
                     bombs_p: torch.Tensor, blast_p: torch.Tensor) -> torch.Tensor:
        """放置泡泡并返回 (N,P) 放置成功掩码（清零 since_bomb 用）。

        bombs_p/blast_p (N,P)：**每个玩家**当前成长的泡数上限与威力
        （corridor 逐人独立；open 恒为 cfg 默认值）。威力在放泡那一刻
        按该玩家当前档位快照进 bomb_blast（爆炸用泡自己的威力）。
        脚下是墙/brick 也不能放泡。
        """
        cfg = self.cfg
        n, w = self.num_envs, cfg.width
        flat_fuse = self.fuse.view(n, -1)
        flat_owner = self.owner.view(n, -1)
        flat_blast = self.bomb_blast.view(n, -1)
        flat_brick = self.brick.view(n, -1)
        cell = center_cell(self.pos)
        placed = torch.zeros((n, cfg.n_players), dtype=torch.bool, device=self.pos.device)
        for me in range(cfg.n_players):
            idx = (cell[:, me, 0] * w + cell[:, me, 1]).unsqueeze(1)
            live = ((self.owner == me) & (self.fuse > 0)).flatten(1).sum(dim=1)
            cur_f = flat_fuse.gather(1, idx).squeeze(1)
            on_brick = flat_brick.gather(1, idx).squeeze(1).bool()
            ok = (
                alive0[:, me]
                & (bomb[:, me] == 1)
                & (cur_f <= 0)
                & ~on_brick
                & (live < bombs_p[:, me])
            )
            placed[:, me] = ok
            flat_fuse.scatter_(
                1, idx,
                torch.where(ok, torch.full_like(cur_f, cfg.fuse), cur_f).unsqueeze(1),
            )
            cur_o = flat_owner.gather(1, idx).squeeze(1)
            flat_owner.scatter_(
                1, idx, torch.where(ok, torch.full_like(cur_o, me), cur_o).unsqueeze(1)
            )
            cur_b = flat_blast.gather(1, idx).squeeze(1)
            flat_blast.scatter_(
                1, idx,
                torch.where(ok, blast_p[:, me].to(cur_b.dtype), cur_b).unsqueeze(1),
            )
        return placed

    def _hazard_wave(self) -> None:
        """炸弹雨波次（hazard 关专用，config 的 hazard_* 注释）。

        每 hazard_wave_ticks tick 一次（t>0，首波在 5 秒后）：每关掷
        hazard_bombs_min..max 颗，落在**可通行格**（无墙/砖、无在场泡、
        非活人脚下），威力按"局内进度 → 偏向大值"采样：指数
        p = 1 - 0.8×min(1, t/ramp)，v = u^p（u∈[0,1)），
        blast = min + floor(v×(max-min+1)) —— 开局均匀，到 ramp 秒后
        v 几乎总是 > 0.8 → 威力几乎总是 max-1/max（4..8 → 7/8）。
        环境炸弹 owner = n_players（越界标记）→ 不进任何玩家的引信通道，
        只出现在危险图通道（网络正是靠它躲的）。训练热路径直接 step()
        用设备 RNG（CUDA graph 捕获路径不覆盖此模式）。
        """
        cfg = self.cfg
        if cfg.hazard_fraction <= 0:
            return
        n, h, w, p = self.num_envs, cfg.height, cfg.width, cfg.n_players
        d = self.pos.device
        wave = self._hazard & (self.t > 0) & (self.t % cfg.hazard_wave_ticks == 0)
        wave_idx = wave.nonzero(as_tuple=True)[0]
        if wave_idx.numel() == 0:
            return
        # 可通行格：无墙/砖、无在场泡；活人脚下排除（死者格不占位置）
        free = (~self.wall) & (~self.brick) & (self.fuse <= 0)      # (n,h,w)
        cell = center_cell(self.pos)
        flat = cell[..., 0] * w + cell[..., 1]                      # (n,p)
        for pl in range(p):
            free.view(n, -1).scatter_(
                1, flat[:, pl].unsqueeze(1),
                (~self.alive[:, pl]).unsqueeze(1))
        free_flat = free.view(n, -1)
        cnt = torch.randint(cfg.hazard_bombs_min, cfg.hazard_bombs_max + 1,
                            (wave_idx.numel(),), device=d)
        frac = (self.t[wave_idx].float()
                / (cfg.hazard_ramp_seconds * cfg.tick_hz)).clamp(0.0, 1.0)
        exp = (1.0 - 0.8 * frac).repeat_interleave(cnt)
        u = torch.rand(int(cnt.sum()), device=d)
        v = u.pow(exp)
        blast = (cfg.hazard_blast_min
                 + (v * (cfg.hazard_blast_max - cfg.hazard_blast_min + 1)).floor())
        blast = blast.clamp(cfg.hazard_blast_min,
                            cfg.hazard_blast_max).to(torch.int16)
        off = 0
        for i in range(wave_idx.numel()):
            e = int(wave_idx[i].item())
            c = int(cnt[i].item())
            pool = free_flat[e].nonzero(as_tuple=True)[0]
            c = min(c, int(pool.numel()))
            if c <= 0:
                continue
            pick = pool[torch.randperm(pool.numel(), device=d)[:c]]
            y, x = pick // w, pick % w
            self.fuse[e, y, x] = cfg.fuse
            self.owner[e, y, x] = p
            self.bomb_blast[e, y, x] = blast[off:off + c]
            off += c
