"""历史模型池 + ELO —— 自我博弈防"策略循环"的关键。

只打最新版的自己，很容易陷入剪刀石头布式的军备竞赛：新策略专克上一版，
但绝对水平不涨。做法是把快照存进池子，按 ELO 就近采样对手。
"""

from __future__ import annotations

import copy
import random

import torch


class ModelPool:
    def __init__(self, max_size: int = 12, k: float = 16.0) -> None:
        self.max_size = max_size
        self.k = k
        self.snapshots: list[dict] = []   # {"step", "elo", "state"}

    def add(self, model: torch.nn.Module, step: int, elo: float = 1000.0) -> None:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.snapshots.append({"step": step, "elo": elo, "state": state})
        if len(self.snapshots) > self.max_size:
            # 丢掉 ELO 最低的那个，而不是最旧的：保留多样化的强对手
            worst = min(range(len(self.snapshots)), key=lambda i: self.snapshots[i]["elo"])
            self.snapshots.pop(worst)

    def sample(self, current_elo: float, near_prob: float = 0.7) -> dict:
        """70% 概率抽 ELO 相近的（势均力敌才学得到东西），30% 完全随机。"""
        if not self.snapshots:
            raise RuntimeError("模型池为空")
        if random.random() < near_prob:
            near = [s for s in self.snapshots if abs(s["elo"] - current_elo) < 120]
            if near:
                return random.choice(near)
        return random.choice(self.snapshots)

    def update_elo(self, snapshot: dict, learner_elo: float, learner_score: float) -> float:
        """standard ELO：learner_score ∈ [0,1]（胜=1 平=0.5 负=0）。返回新的 learner_elo。"""
        opp = snapshot["elo"]
        expected = 1.0 / (1.0 + 10 ** ((opp - learner_elo) / 400.0))
        delta = self.k * (learner_score - expected)
        snapshot["elo"] = opp - delta
        return learner_elo + delta

    def state_dict(self) -> dict:
        return {"max_size": self.max_size, "k": self.k, "snapshots": self.snapshots}

    def load_state_dict(self, d: dict) -> None:
        self.max_size = d["max_size"]
        self.k = d["k"]
        self.snapshots = d["snapshots"]

    def latest(self) -> dict | None:
        return max(self.snapshots, key=lambda s: s["step"]) if self.snapshots else None

    def __len__(self) -> int:
        return len(self.snapshots)


def load_frozen(model_cls, obs_shape, state: dict, device, arch: str = "cnn",
                n_players: int | None = None) -> torch.nn.Module:
    """把快照实例化成一个冻结的推理网络（不参与梯度）。

    `arch` 必须和快照一致（cnn/mlp），否则 state_dict 键不匹配会报错。
    `n_players`：观测含 obs_extra 通道后无法从通道数唯一反推人数，需显式传
    （旧 7 通道快照可省，内部按 (c-3)//2 兜底）。
    """
    net = model_cls(obs_shape, arch=arch, n_players=n_players).to(device)
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def clone_frozen(model: torch.nn.Module) -> torch.nn.Module:
    net = copy.deepcopy(model).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net
