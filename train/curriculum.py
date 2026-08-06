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
