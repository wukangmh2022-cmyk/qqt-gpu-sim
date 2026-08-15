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
    """LSTM 从零课程（2026-08-12，v3）：**敌人与地图双维度循序渐进**。

    敌人维度：random/greedy 启蒙 → greedy → +astar → +hunter → 池子天梯。
    地图维度（用户定，训练不用环岛）：
      - open：纯空场 → 随机散点立柱（wall_density 渐进），学基础 + 绕柱。
      - **corridor 前期就引入**（可炸墙多 → 宝箱奖励稠密，学习效率高），
        障碍三维循序渐进，前期简单后期障碍多样：
        * 顶墙行数 top_wall_rows **2 → 3 → 4**（默认形态是 4 行永久墙，
          "变化着来"：前期少留活动空间、后期顶到默认形态）；
        * 通道宽度 corridor_width **7 → 5**（左右可炸墙列 3 → 4）；
        * 边缘连续横/纵 brick 段 wall_density **0 → 0.25 → 0.45**
          （贴可通行区边缘与顶墙下方，连续段而非散点，放边缘不放中间）。
      - 中后期按 open_fraction 混合 open 关（练纯空场交战 + 立柱泛化）。
      设计权衡：前期（s2）顶墙少/通道宽/无额外段 → 不困难；后期（s4/s5）
      障碍多样（顶墙 4 + 通道 5 + 边缘连续段）→ 保留地图障碍感知能力，
      泛化到 ring 等训练外地图（eval_lstm_ring.py 测）。
      环岛/特殊设计地图只用于泛化测试，不进训练。
    对局总预算 ~900 万局（50/100/150/200/400 万），跑不完也能按胜率提前晋级。
    """
    return [
        Stage("s1-open-basic", SimConfig(map_mode="open"),
              500_000, 0.55, bots=("random", "greedy"), bot_prob=0.9,
              notes="纯空场学移动/放泡/躲避：random 靶子 + greedy 初阶；曲线不涨立刻停"),
        Stage("s2-corridor-easy",
              SimConfig(map_mode="corridor", open_fraction=0.3,
                        top_wall_rows=2, corridor_width=7),
              1_000_000, 0.60, bots=("greedy",), bot_prob=0.9,
              notes="corridor 前期引入且**简单**：顶墙 2 行 + 宽通道 7 + 无额外段"
                    " —— 稠密宝箱奖励学炸墙开图，不困难"),
        Stage("s3-corridor-mid",
              SimConfig(map_mode="corridor", open_fraction=0.3,
                        top_wall_rows=3, corridor_width=5, wall_density=0.25),
              1_500_000, 0.55, bots=("greedy", "astar"), bot_prob=0.9,
              notes="顶墙 3 行 + 通道收窄 5 + 边缘连续段起步（~9 块）"
                    " + 对手加 astar（会躲会攻）"),
        Stage("s4-corridor-hard",
              SimConfig(map_mode="corridor", open_fraction=0.4,
                        top_wall_rows=4, corridor_width=5, wall_density=0.45),
              2_000_000, 0.55, bots=("astar", "hunter"), bot_prob=0.9,
              notes="默认形态顶墙 4 + 通道 5 + 边缘连续段最密 + open 混合 40%"
                    " + hunter 纯进攻强敌（障碍感知泛化）"),
        Stage("s5-ladder",
              SimConfig(map_mode="corridor", open_fraction=0.45,
                        top_wall_rows=4, corridor_width=5, wall_density=0.45),
              4_000_000, 0.55, bots=("greedy", "astar", "hunter"), bot_prob=0.3,
              self_play=True,
              notes="天梯自我对弈：池子快照为主（ELO 就近采样），bot 少量混入防遗忘"),
    ]


def cnn_curriculum(base: SimConfig = SimConfig()) -> list[Stage]:
    """CNN 对打寻路 AI 专精课程（2026-08-15 重写）。

    **上一版教训（2026-08-15 训练日志诊断）**：旧课程末尾 s7 天梯用自博弈
    快照当对手，wr 0.93 是对越来越弱的自己历史版本 —— 最终 ckpt 对纯 astar
    只剩 0.45（连 duel_cnn 基线 0.46 都没保住）。**训练评估口径必须与目标
    环境一致**：对手 = 纯规则 bot（astar/hunter），无自博弈快照、无天梯。
    此时训练内 wr 直接 = 对目标 bot 的胜率（已验证 NPU/CPU 上 bot 行为一致：
    NPU eval 0.443 vs CPU eval 0.471，同 ckpt 同配置）。

    目标：launcher open80（corridor + open_fraction=1.0 + 80% 成长 8/6/1.68）
    打 astar + hunter 各 90%+。从 duel_cnn 基线起步：
      open80 astar 0.463 / open80 hunter 0.297（本地 eval_cnn_bots 多 seed 合并）。
    先专项（greedy 热身 → astar → hunter），再混合收尾（两个都保 90% 防
    专项间互相遗忘）。bot_prob=1.0：课程阶段非 self_play 走 warmup 分支，
    build_opponents 恒从 fixed+bot 选，fixed 为空 → 对手恒为规则 bot。
    """
    open80 = replace(base, open_fraction=1.0, open_growth_bombs=8,
                     open_growth_blast=6, open_growth_speed=1.68)
    return [
        Stage("s1-open80-astar", open80, 600_000, 0.95,
              bots=("astar",), bot_prob=1.0,
              notes="open80 × 纯 astar 专项（基线 0.463 → 0.95）：launcher 空场景"
                    "同款 + 最终目标对手之一。上一版败因正是最后被天梯弱对手"
                    "带偏，本阶段全程只打 astar"),
        Stage("s2-open80-hunter", open80, 600_000, 0.95,
              bots=("hunter",), bot_prob=1.0,
              notes="open80 × 纯 hunter 专项（基线 0.297 → 0.95）：旧课程缺口 ——"
                    "hunter 从未在 launcher 环境专项练过"),
        Stage("s3-open80-both", open80, 800_000, 0.90,
              bots=("astar", "hunter"), bot_prob=1.0,
              notes="open80 × astar+hunter 混合收尾：最终验收口径（两个都要 "
                    "90%+），混合防专项间互相遗忘。全程无自博弈快照"),
    ]
