"""人类玩家轨迹录制 → BC（行为克隆）训练数据。

对局内按 10Hz tick 采样：**step 前的完整网络观测**（与 PPO rollout 对齐，
obs → act → step 的时序）+ 人类**实际执行**的动作（帧级移动聚合后的方向
+ 放泡）。观测 14 通道全部归一化到 [0,1]（见 sim/obs.py encode_obs），
存档用 uint8 量化（×255 取整，误差 <1/255），训练侧 /255 还原 ——
13×13 的 1800 tick 一局 ≈ 4.4MB，100 局 ≈ 440MB。

npz 内容（一局一个文件）：
    obs      (T, C, H, W) uint8    完整共享观测（人类玩家视角 = player 0）
    action   (T, 2) int8            人类动作 [move(0..4), bomb(0/1)]
    reward   (T,) float16           人类该 tick 的奖励（PPO 预热的归因参考）
    done     (T,) uint8             1 = 该 tick 终局
    pid      int8                   录的人类是物理 0 还是 1（P1 的 obs 已 swap 成
                                    player0 视角 —— 训练 learner 恒为 player0）
    meta     dict                   地图/场景/对手/seed/成长配置

用法（duel 内部，见 play/duel.py）：
    rec = Recorder()
    rec.begin(meta)          # 新一局
    rec.add(obs, action, reward, done)   # 每 10Hz tick
    rec.finish()             # 存盘（对局结束/重开/ESC）
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import torch


class Recorder:
    """一个对局一个 Recorder 实例；begin/add/finish 每局循环。

    max_ticks 防挂机/超长对局：duel 对局 auto_reset=False，双方都不死时 t 会
    超过 max_steps 一直涨，录制若不设上限会无限累积（内存 + 每 tick 开销 →
    掉帧）。超过上限自动落盘并停录本局。
    """

    def __init__(self, out_dir: str = "recordings", max_ticks: int = 2400) -> None:
        self.out_dir = out_dir
        self.max_ticks = max_ticks
        self._obs: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._rew: list[np.ndarray] = []
        self._done: list[np.ndarray] = []
        self._meta: dict = {}
        self.active = False

    # ---- 每局生命周期 ----
    def begin(self, meta: dict | None = None) -> None:
        self._obs, self._act, self._rew, self._done = [], [], [], []
        self._meta = dict(meta or {})
        self.active = True

    def add(self, obs: torch.Tensor, action: torch.Tensor,
            reward: torch.Tensor, done: torch.Tensor) -> None:
        """obs (1,C,H,W) 共享观测（人类玩家视角）；action (1,2) [move,bomb]；
        reward/done 标量（人类该 tick 的奖励/终局标记）。"""
        if not self.active:
            return
        if len(self._obs) >= self.max_ticks:
            # 超长对局（双方都不死，t 超过 max_steps 仍一直涨）→ 落盘并停录，
            # 防止内存/每 tick 开销无限增长把渲染拖到 20-40fps。
            self.finish()
            return
        o = obs[0].float()
        # uint8 量化：全通道值域 [0,1]，×255 取整，训练侧 /255 还原。
        self._obs.append(np.round(o.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        self._act.append(np.asarray(action[0].cpu().numpy(), dtype=np.int8))
        self._rew.append(np.asarray(float(reward), dtype=np.float16))
        self._done.append(np.asarray(bool(done), dtype=np.uint8))

    @property
    def ticks(self) -> int:
        return len(self._obs)

    def finish(self, min_ticks: int = 16) -> str | None:
        """把本局写入 npz；tick 太少（秒开秒退）丢弃。返回文件路径或 None。"""
        if not self.active or len(self._obs) < min_ticks:
            self.active = False
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"rec_{ts}.npz")
        np.savez_compressed(
            path,
            obs=np.stack(self._obs),          # (T,C,H,W) uint8
            action=np.stack(self._act),       # (T,2) int8
            reward=np.asarray(self._rew, dtype=np.float16),
            done=np.asarray(self._done, dtype=np.uint8),
            pid=np.int8(self._meta.get("pid", 0)),
            scale=np.float32(255.0),
            meta=np.asarray([str(self._meta)], dtype=object),
        )
        self.active = False
        return path
