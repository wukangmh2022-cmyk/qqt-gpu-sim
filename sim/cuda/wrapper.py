"""CUDA backend 的 Python 包装：JIT 编译 kernel + SoA 状态管理。

对外 API 和 `BatchedSim` 完全一致（step / observe / legal_mask /
state_dict / load_state_dict），因此训练脚本和 parity 测试可以直接换 backend。

内部状态用 **(cell, env)** 布局存放，env 在最内层 —— 这是合并访存的关键。
`state_dict()` 会转置回 (N, H, W) 的"人类可读"布局，只在测试/调试时调用。
"""

from __future__ import annotations

import os

import torch

from ..config import N_BOMB, N_MOVES, SimConfig
from ..mapgen import make_walls

_EXT = None


def _load_ext():
    """首次调用时 JIT 编译 bomber_kernels.cu（约 30~60 秒，之后走缓存）。"""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        here = os.path.dirname(os.path.abspath(__file__))
        _EXT = load(
            name="qqt_bomber_cuda",
            sources=[os.path.join(here, "bomber_kernels.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _EXT


class CudaSim:
    def __init__(self, cfg: SimConfig, num_envs: int, seed: int = 0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaSim 需要可用的 CUDA 设备")
        self.ext = _load_ext()
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device("cuda")
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

        n, nc, p = num_envs, cfg.n_cells, cfg.n_players
        d = self.device
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=d)  # noqa: E731
        self.wall = z((nc, n), torch.uint8)
        self.fuse = z((nc, n), torch.int32)
        self.owner = torch.full((nc, n), -1, dtype=torch.int8, device=d)
        self.pos = z((p * 2, n), torch.float32)
        self.alive = torch.ones((p, n), dtype=torch.uint8, device=d)
        self.hp = torch.full((p, n), cfg.max_hp, dtype=torch.uint8, device=d)
        self.since_bomb = torch.zeros((p, n), dtype=torch.int32, device=d)
        self.t = z((n,), torch.int32)
        self.done = z((n,), torch.uint8)
        self.reward = z((p, n), torch.float32)
        self.act_buf = z((p * 2, n), torch.int32)
        # 连锁爆炸用的三块 scratch，一次分配复用，避免每 tick 申请显存
        self.covered = z((nc, n), torch.uint8)
        self.trig = z((nc, n), torch.uint8)
        self.expanded = z((nc, n), torch.uint8)
        # 观测是 env 级共享的一份 (N, 2P+3, H, W)，不乘 P；dtype 跟 cfg.obs_fp16
        self.obs_buf = z((n,) + cfg.obs_shape,
                         torch.float16 if cfg.obs_fp16 else torch.float32)
        self.mmask_buf = z((n, p, N_MOVES), torch.uint8)
        self.bmask_buf = z((n, p, N_BOMB), torch.uint8)
        self.reset_all()

    # ---------------- 重置（host 侧 torch 算子，不在热路径上）----------------

    def reset_all(self) -> None:
        self.reset_(torch.ones((self.num_envs,), dtype=torch.bool, device=self.device))

    def reset_(self, mask: torch.Tensor) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]
        walls = make_walls(self.cfg, count, self.gen, self.device)   # (count, H, W)
        self.wall[:, idx] = walls.reshape(count, -1).t().to(torch.uint8)
        self.fuse[:, idx] = 0
        self.owner[:, idx] = -1
        self.alive[:, idx] = 1
        self.hp[:, idx] = self.cfg.max_hp
        self.since_bomb[:, idx] = 0
        self.t[idx] = 0
        spawns = torch.tensor(self.cfg.spawn_pos(), dtype=torch.float32,
                              device=self.device)            # (P, 2) 浮点格中心
        self.pos[:, idx] = spawns.reshape(-1, 1).expand(-1, count)

    # ---------------- 热路径 ----------------

    def step(
        self, actions: torch.Tensor, auto_reset: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """actions: (N, P, 2) long。内部转成 (P*2, N) 的 SoA 布局。"""
        cfg = self.cfg
        n, p = self.num_envs, cfg.n_players
        self.act_buf.copy_(actions.reshape(n, p * 2).t().to(torch.int32))
        self.ext.step(
            self.wall, self.fuse, self.owner, self.pos, self.alive, self.hp,
            self.since_bomb, self.t, self.act_buf, self.reward, self.done,
            self.covered, self.trig, self.expanded, cfg.height, cfg.width,
            cfg.n_players, self.num_envs, cfg.fuse, cfg.blast, cfg.max_bombs,
            cfg.max_steps, cfg.max_chain, cfg.step_penalty, cfg.radius,
            cfg.step_len, cfg.hit_reward, cfg.win_bonus, cfg.danger_penalty,
            cfg.passivity_penalty, cfg.passivity_ticks,
            cfg.max_hp, int(cfg.win_hp_scaled),
        )
        reward = self.reward.t().clone()          # (N, P)
        done = self.done.bool().clone()
        n_alive = self.alive.sum(dim=0).long()
        # 终局胜负与参考实现同规则（sim/torch_sim.py step 末尾）：
        #   n_alive==1 → 唯一存活着胜；
        #   超时全员存活（n_alive==P）→ 血多者胜，血平局 = 平局 0；
        #   同时死光（n_alive==0）→ 平局 0。
        # kernel 只把这些写进 reward（win_bonus），这里在 reset 前用同一状态
        # 补出 (N, P) 布尔 winner，供 PPO 的 _tally 用。
        alive_t = self.alive.t().bool()           # (N, P)
        hp_t = self.hp.t()                        # (N, P)
        winner = done.unsqueeze(1) & alive_t & (n_alive == 1).unsqueeze(1)
        loser = done.unsqueeze(1) & ~alive_t & (n_alive == 1).unsqueeze(1)
        if bool((done & (n_alive == cfg.n_players)).any()):
            hp = hp_t.float()
            all_alive = done & (n_alive == cfg.n_players)
            for me in range(cfg.n_players):
                others = [o for o in range(cfg.n_players) if o != me]
                wins = all_alive & (hp[:, me].unsqueeze(1) > hp[:, others]).all(dim=1)
                loses = all_alive & (hp[:, me].unsqueeze(1) < hp[:, others]).any(dim=1)
                winner[:, me] |= wins
                loser[:, me] |= loses
        if auto_reset:
            self.reset_(done)
        return reward, done, {"n_alive": n_alive, "winner": winner}

    def observe(self) -> torch.Tensor:
        cfg = self.cfg
        self.ext.observe(
            self.wall, self.fuse, self.owner, self.pos, self.alive, self.t,
            self.obs_buf, cfg.height, cfg.width, cfg.n_players, self.num_envs,
            cfg.fuse, cfg.blast, cfg.max_steps, cfg.n_channels,
        )
        return self.obs_buf

    def legal_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        self.ext.mask(
            self.wall, self.fuse, self.owner, self.pos, self.alive,
            self.mmask_buf, self.bmask_buf, cfg.height, cfg.width, cfg.n_players,
            self.num_envs, cfg.max_bombs, cfg.radius, cfg.step_len,
        )
        return self.mmask_buf.bool(), self.bmask_buf.bool()

    # ---------------- 与参考实现互换状态（仅测试/调试用）----------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        n, h, w, p = self.num_envs, cfg.height, cfg.width, cfg.n_players
        # .t() 之后是非连续视图，reshape 前必须 contiguous()
        soa = lambda x: x.t().contiguous()  # noqa: E731
        return {
            "wall": soa(self.wall).reshape(n, h, w).bool(),
            "fuse": soa(self.fuse).reshape(n, h, w).to(torch.int16),
            "owner": soa(self.owner).reshape(n, h, w).clone(),
            "pos": soa(self.pos).reshape(n, p, 2).long(),
            "alive": soa(self.alive).reshape(n, p).bool(),
            "hp": soa(self.hp).reshape(n, p).clone(),
            "since_bomb": soa(self.since_bomb).reshape(n, p).clone(),
            "t": self.t.long().clone(),
        }

    def load_state_dict(self, snap: dict[str, torch.Tensor]) -> None:
        n = self.num_envs
        to_soa = lambda x, dt: x.reshape(n, -1).t().contiguous().to(dt)  # noqa: E731
        self.wall.copy_(to_soa(snap["wall"], torch.uint8))
        self.fuse.copy_(to_soa(snap["fuse"], torch.int32))
        self.owner.copy_(to_soa(snap["owner"], torch.int8))
        self.pos.copy_(to_soa(snap["pos"], torch.int32))
        self.alive.copy_(to_soa(snap["alive"], torch.uint8))
        self.hp.copy_(to_soa(snap["hp"], torch.uint8))
        self.since_bomb.copy_(to_soa(snap["since_bomb"], torch.int32))
        self.t.copy_(snap["t"].to(torch.int32))

