"""课程学习：把"1v1 打基础 → 削弱版 1v2 → 完全体 1v2"写成可配置的阶段表。

两个硬性约束体现在这里：
- 阶段之间 **复用权重**。人数变化会改变输入通道数（C = 2P+3），
  新增对手通道置零、旧通道原样保留，见 `train.py::adapt_first_conv`。
- 每个阶段都有 **通过标准和早停**。某阶段跑到一半盘数还看不到胜率上涨，
  应该停下来调奖励/超参，而不是硬跑满。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from sim.config import SimConfig


@dataclass
class Stage:
    name: str
    cfg: SimConfig
    episodes: int                 # 该阶段的目标对局数
    win_rate_target: float        # 通过标准
    opponent_handicap: float = 1.0  # <1 表示削弱对手（放泡概率打折）
    notes: str = ""
    # 双维度课程（2026-08-12 起）：对手与地图都按阶段编排。
    # bots：本阶段规则 bot 组合（空 = 沿用 --bot-opponents 全局配置）。
    # bot_prob：本阶段 bot 混入概率（0 = 沿用全局 --bot-opp-prob）。
    # self_play：True 时进入天梯 —— build_opponents 的 warmup 关掉，对手以
    #   池子快照（ELO 就近采样）为主，bot 按 bot_prob 少量混入防遗忘。
    bots: tuple[str, ...] = ()
    bot_prob: float = 0.0
    self_play: bool = False


def default_curriculum(base: SimConfig = SimConfig()) -> list[Stage]:
    """1000 万局预算下的默认切分（50 万 / 200 万 / 350 万 / 400 万）。"""
    # 只改人数，其余字段完整继承。手写 SimConfig(...) 容易在新增 speed、radius、
    # obs_fp16 等字段时静默退回默认值，课程阶段会因此和传入的 base 不一致。
    two = lambda: replace(base, n_players=2)    # noqa: E731
    three = lambda: replace(base, n_players=3)  # noqa: E731
    return [
        Stage("s1-1v1-base", two(), 500_000, 0.55,
              notes="验证奖励函数和网络：曲线不涨就立刻停，别浪费配额"),
        Stage("s1b-1v1-pool", two(), 1_500_000, 0.60,
              notes="模型池就近采样，练到能稳定压制历史版本"),
        Stage("s2-1v2-weak", three(), 3_500_000, 0.50, opponent_handicap=0.5,
              notes="直接上两个削弱对手（放泡概率减半），跳过'幽灵对手'"),
        Stage("s3-1v2-full", three(), 4_500_000, 0.50, opponent_handicap=1.0,
              notes="完全体 1v2；胜率长期低于 40% 说明上一阶段结束太早"),
    ]


@dataclass
class CurriculumState:
    stage_idx: int = 0
    episodes_in_stage: int = 0
    recent_wins: list[float] = field(default_factory=list)

    def record(self, win_score: float, window: int = 2000) -> None:
        self.recent_wins.append(win_score)
        if len(self.recent_wins) > window:
            del self.recent_wins[: len(self.recent_wins) - window]

    def win_rate(self) -> float:
        return sum(self.recent_wins) / max(1, len(self.recent_wins))

    def should_advance(self, stage: Stage) -> bool:
        enough = self.episodes_in_stage >= stage.episodes
        passed = len(self.recent_wins) >= 500 and self.win_rate() >= stage.win_rate_target
        return enough or passed


def lstm_curriculum() -> list[Stage]:
    """LSTM 从零课程（2026-08-12）：**敌人与地图双维度循序渐进**。

    敌人维度：random/greedy 启蒙 → greedy → +astar → +hunter → 池子天梯。
    地图维度：open 纯空场 → open/corridor 50% 混合 → +环岛 +随机立柱(0.3)
            → 立柱 0.5 + 环岛 30% → 全混合常驻（立柱 0.5）。
    防过拟合设计：
    - **地图**：多关混合（open_fraction/ring_fraction）+ wall_density 随机立柱
      密度克制递增 0 → 0.3 → 0.5（经典炸弹人奇数行列图案随机保留，每局不同）。
    - **敌人**：课程阶段 bot 为主（bot_prob 0.9）；s5 起 self_play=True 进入
      天梯（池子 ELO 就近采样为主），bot_prob 降到 0.3 少量混入 —— 防只打
      接近型网络的分布单一化（"对手可能不接近/躲闪"的遗忘）。
    对局总预算 ~900 万局（50/100/150/200/400 万），跑不完也能按胜率提前晋级。
    """
    return [
        Stage("s1-open-basic", SimConfig(map_mode="open"),
              500_000, 0.55, bots=("random", "greedy"), bot_prob=0.9,
              notes="纯空场学移动/放泡/躲避：random 靶子 + greedy 初阶；曲线不涨立刻停"),
        Stage("s2-mix-wall",
              SimConfig(map_mode="corridor", open_fraction=0.5),
              1_000_000, 0.60, bots=("greedy",), bot_prob=0.9,
              notes="一半空场一半 corridor：学炸墙开图（顶墙+左右可炸墙+宝箱成长）"),
        Stage("s3-pillar-astar",
              SimConfig(map_mode="corridor", open_fraction=0.3,
                        ring_fraction=0.2, wall_density=0.3),
              1_500_000, 0.55, bots=("greedy", "astar"), bot_prob=0.9,
              notes="随机立柱密度 0.3（克制起步）+ 环岛 20% + 对手加 astar（会躲会攻）"),
        Stage("s4-hunter-dense",
              SimConfig(map_mode="corridor", open_fraction=0.3,
                        ring_fraction=0.3, wall_density=0.5),
              2_000_000, 0.55, bots=("astar", "hunter"), bot_prob=0.9,
              notes="立柱 0.5 + 环岛 30% + hunter 纯进攻强敌（地图敌人双升级）"),
        Stage("s5-ladder",
              SimConfig(map_mode="corridor", open_fraction=0.35,
                        ring_fraction=0.3, wall_density=0.5),
              4_000_000, 0.55, bots=("greedy", "astar", "hunter"), bot_prob=0.3,
              self_play=True,
              notes="天梯自我对弈：池子快照为主（ELO 就近采样），bot 少量混入防遗忘"),
    ]
