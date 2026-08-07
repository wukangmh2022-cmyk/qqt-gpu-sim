"""录像回放：用对战同一套逻辑重放录像（人类动作驱动 sim 真实推进）。

录像存的本质 = 每 tick 的人类动作（[方向, 放泡]，见 play/recorder.py）。回放
不做任何 obs 猜测：构造与采集一致的 sim（地图/成长起点/种子），**人类侧重放
录像动作、对手侧 AI 重新决策**，sim.step 真实推进 —— 泡泡出现/爆炸/道具拾取/
角色移动全是 sim 算的，与对打画面完全同步。

ReplaySim 是核心（render_gif.py 也用它批量渲染 GIF/webm）。

用法：
    python -m play.replay               # 列表选录像（输入编号）
    python -m play.replay <npz路径>     # 直接播放
    python -m play.replay --frames 60   # 播放 60 帧退出（headless 冒烟用）

10Hz 播放（与录制同频），60fps 渲染插值（角色平滑、爆炸淡出）；
空格暂停、←→ 逐 tick、↑↓ 调速（0.5/1/2/4×）、R 重头、D 切危险图、Q/ESC 退出。
"""

from __future__ import annotations

import argparse
import ast
import glob
import os

import numpy as np
import pygame
import torch

from .duel import (CELL, MOVE_DOWN, MOVE_IDLE, WALK_HZ, _load_cjk_font,
                   _swap_player_channels, build_static, draw_grid)
from .res import Res
from sim.config import SimConfig
from sim.factory import make_sim
from sim.bots import make_bot
from train.train import load_fixed_checkpoint

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUD_H = 52
MOVE_GLYPH = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "停"}


def list_recordings(dir_: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(dir_, "*.npz")), reverse=True)
    return paths


def load_rec(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    return {
        "action": d["action"], "pid": int(d["pid"]), "meta": meta,
    }


def make_cfg(meta: dict) -> SimConfig:
    """按录像 meta 构造与采集一致的 sim 配置（地图 + 成长起点）。"""
    m = meta.get("map", "corridor")
    b = meta.get("bombs", 2)
    z = meta.get("blast", 2)
    sp = meta.get("speed", 1.0)
    if m == "open":
        return SimConfig(map_mode="corridor", open_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5,
                         open_growth_bombs=int(b), open_growth_blast=int(z),
                         open_growth_speed=float(sp))
    if m == "ring":
        return SimConfig(map_mode="corridor", ring_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5,
                         growth_bombs_start=int(b), growth_blast_start=int(z),
                         growth_speed_start=float(sp))
    return SimConfig(map_mode="corridor", max_steps=1800, speed=3.0, max_hp=5,
                     growth_bombs_start=int(b), growth_blast_start=int(z),
                     growth_speed_start=float(sp))


class ReplaySim:
    """录像重放驱动器：人类侧 replay 录的动作，对手侧 AI 决策，sim 真实推进。

    step(t) 推进到录像 tick t —— 每调一次就 step 一次 sim，泡泡/爆炸/道具
    完全由 sim 计算。渲染直接 draw_grid(sim, ...)，与对打同画面。
    """

    def __init__(self, path: str):
        d = np.load(path, allow_pickle=True)
        self.act = d["action"]               # (T,2) [move, bomb] 人类动作
        self.T = self.act.shape[0]
        self.pid = int(d["pid"])             # 人类物理侧 0/1
        self.meta = ast.literal_eval(str(d["meta"][0])) if d["meta"].size else {}
        self.cfg = make_cfg(self.meta)
        self.sim = make_sim(self.cfg, 1, backend="torch", device="cpu",
                            seed=self.meta.get("seed", 0))
        self.opp_pid = 1 - self.pid
        # 对手：meta.opp → bot:X / ckpt 文件名 / 其他（无对手数据 → idle 兜底）
        opp = self.meta.get("opp", "")
        self.opp_bot = None
        self.opp_net = None
        if isinstance(opp, str) and opp.startswith("bot:"):
            self.opp_bot = make_bot(self.sim, opp.split(":", 1)[1])
        elif isinstance(opp, str) and "human" in opp:
            # 对手是人类：录像只存了人类侧动作（对手侧没数据）→ 用 astar 当
            # 对手重放（有互动；否则对手 idle 站着画面呆）。
            self.opp_bot = make_bot(self.sim, "astar")
        elif isinstance(opp, str) and opp.endswith(".pt"):
            ck = os.path.join(PROJ, "ckpt", opp)
            if os.path.exists(ck):
                self.opp_net = load_fixed_checkpoint(ck, self.cfg.obs_shape, "cpu")
                self.opp_net.eval()
            else:
                self.opp_bot = make_bot(self.sim, "idle")   # ckpt 不在本地 → 兜底
        else:
            self.opp_bot = make_bot(self.sim, "idle")       # 对手未知 → 兜底

    def step(self, t: int) -> tuple:
        """推进到 tick t（人类 replay act[t]，对手 AI 决策）。返回 (info, obs)。"""
        obs = self.sim.observe()
        mm, bm = self.sim.legal_mask()
        a_h = torch.tensor([[int(self.act[t, 0]), int(self.act[t, 1])]],
                           dtype=torch.long)
        if self.opp_bot is not None:
            a_o = self.opp_bot.act(obs, mm[:, self.opp_pid],
                                   bm[:, self.opp_pid], self.opp_pid)
        else:
            with torch.no_grad():
                o = _swap_player_channels(obs) if self.opp_pid == 1 else obs
                a_o = self.opp_net.act(o, mm[:, self.opp_pid],
                                       bm[:, self.opp_pid], 0)[0]
        actions = torch.zeros(1, 2, 2, dtype=torch.long)
        actions[0, self.pid] = a_h
        actions[0, self.opp_pid] = a_o
        _, _, info = self.sim.step(actions)
        return info, self.sim.observe()


class Replay:
    def __init__(self, path: str, scene: str = "比武") -> None:
        self.path = path
        self.rec = load_rec(path)
        self.T = self.rec["action"].shape[0]
        self.t = 0
        self.pause = False
        self.speed = 1.0
        self.show_danger = True
        self.loop = True
        self.rec_sim = ReplaySim(path)
        self.sim = self.rec_sim.sim
        self.cfg = self.rec_sim.cfg
        pygame.init()
        w, h = self.cfg.width, self.cfg.height
        self.screen = pygame.display.set_mode((w * CELL, h * CELL + HUD_H))
        name = os.path.basename(path)
        pygame.display.set_caption(f"录像回放 · {name}")
        self.font = _load_cjk_font(20)
        self.font_s = _load_cjk_font(16)
        self.res = Res(cell=CELL, blast=self.cfg.growth_blast_max, scene=scene)
        self.face = {0: MOVE_DOWN, 1: MOVE_DOWN}
        self.anim = {0: 0, 1: 0}
        self._static = None
        self._info = None
        self._prev_pos = None
        self.acc = 0.0
        self.tick_dt = 0.1 / self.speed

    def _advance(self, to_t: int) -> None:
        """把 sim 推进到 tick to_t（逐 tick step，重放人类动作）。"""
        while self.rec_sim.sim.t[0] < to_t and self.t < self.T - 1:
            self._prev_pos = self.sim.pos[0].float().clone()
            self._info, _ = self.rec_sim.step(int(self.rec_sim.sim.t[0]))
        self._static = build_static(self.res, self.sim)

    def draw(self) -> None:
        s = self.screen
        if self.t < self.T:
            self._advance(self.t)
            # face：本 tick 位移方向
            if self._prev_pos is not None:
                dp = self.sim.pos[0] - self._prev_pos
                for pid in range(2):
                    dx, dy = float(dp[pid, 1]), float(dp[pid, 0])
                    if abs(dx) > 0.05 or abs(dy) > 0.05:
                        self.face[pid] = 0 if dy < 0 else 1 if dy > 0 \
                            else 2 if dx < 0 else 3
        self.anim[0] = int(self.t / 10.0 * WALK_HZ) % 4
        self.anim[1] = self.anim[0]
        obs = self.sim.observe()
        rpos = self.sim.pos[0].float()
        expl = (self._info["blast"][0].bool(), self._info["trig"][0].bool()) \
            if self._info is not None else None
        draw_grid(s, self.res, self.sim, obs, rpos, self.face, self.anim,
                  expl, static=self._static)
        # HUD
        act = self.rec["action"][self.t] if self.t < self.T else (4, 0)
        meta = self.rec["meta"]
        rew = 0.0
        hx = 12
        hud = self.font.render(
            f"t {self.t}/{self.T}  动作 {MOVE_GLYPH.get(int(act[0]), '?')}"
            f"{'💣' if int(act[1]) else ''}   "
            f"{'  ⏸' if self.pause else ''}  {self.speed:.0f}×",
            True, (235, 235, 240))
        s.blit(hud, (hx, self.cfg.height * CELL + 8))
        meta_s = self.font_s.render(
            f"pid={meta.get('pid', self.rec['pid'])}  地图={meta.get('map', '?')}  "
            f"对手={meta.get('opp', '?')}  成长 {meta.get('bombs', '?')}/"
            f"{meta.get('blast', '?')}/{meta.get('speed', '?')}   "
            f"[空格暂停 ←→逐帧 ↑↓调速 R重头 D危险 Q退出]",
            True, (170, 170, 185))
        s.blit(meta_s, (hx, self.cfg.height * CELL + 28))
        pygame.draw.rect(s, (60, 58, 70), (12, self.cfg.height * CELL + 2,
                                           self.cfg.width * CELL - 24, 4))
        if self.T > 1:
            pygame.draw.rect(s, (90, 200, 130),
                             (12, self.cfg.height * CELL + 2,
                              int((self.cfg.width * CELL - 24) * self.t / (self.T - 1)), 4))

    def handle(self, e) -> bool:
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if e.key == pygame.K_SPACE:
                self.pause = not self.pause
            elif e.key == pygame.K_LEFT:
                self.t = max(0, self.t - 1)
                self.pause = True
            elif e.key == pygame.K_RIGHT:
                self.t = min(self.T - 1, self.t + 1)
                self.pause = True
            elif e.key == pygame.K_UP:
                self.speed = min(4.0, self.speed * 2)
                self.tick_dt = 0.1 / self.speed
            elif e.key == pygame.K_DOWN:
                self.speed = max(0.5, self.speed / 2)
                self.tick_dt = 0.1 / self.speed
            elif e.key == pygame.K_r:
                self.t = 0
                self.rec_sim = ReplaySim(self.path)
                self.sim = self.rec_sim.sim
                self._info = None
                self._prev_pos = None
            elif e.key == pygame.K_d:
                self.show_danger = not self.show_danger
        return True

    def run(self, max_frames: int | None = None) -> None:
        clock = pygame.time.Clock()
        frames = 0
        while True:
            for e in pygame.event.get():
                if not self.handle(e):
                    pygame.quit()
                    return
            dt = clock.tick(60) / 1000.0
            if not self.pause:
                self.acc += dt
                while self.acc >= self.tick_dt:
                    self.t += 1
                    self.acc -= self.tick_dt
                    if self.t >= self.T:
                        if not self.loop:
                            self.t = self.T - 1
                            self.pause = True
                            break
                        self.t = 0
            self.draw()
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                pygame.quit()
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None,
                    help="npz 录像路径；缺省 = 列出 recordings/ 让用户选")
    ap.add_argument("--dir", default="recordings", help="录像目录（默认 recordings/）")
    ap.add_argument("--frames", type=int, default=None, help="播放 N 帧退出")
    args = ap.parse_args()

    path = args.path
    if path is None:
        recs = list_recordings(args.dir)
        if not recs:
            print(f"[replay] {args.dir}/ 下没有录像（先打几局，一局结束自动存 npz）")
            return
        print(f"[replay] {args.dir}/ 共 {len(recs)} 个录像：")
        for i, p in enumerate(recs):
            try:
                d = np.load(p, allow_pickle=True)
                T = d["action"].shape[0]
                meta = ast.literal_eval(str(d["meta"][0])) if d["meta"].size else {}
                pid = int(d["pid"])
                tag = (f"tick{T:4d}  pid{pid}  地图{meta.get('map', '?')}  "
                       f"对手{meta.get('opp', '?')}  成长{meta.get('bombs', '?')}/"
                       f"{meta.get('blast', '?')}/{meta.get('speed', '?')}")
            except Exception as exc:
                tag = f"(读取失败: {exc})"
            print(f"  [{i}] {os.path.basename(p)}  {tag}")
        while True:
            try:
                s = input("输入编号播放（q 退出）: ").strip()
            except EOFError:
                return
            if s.lower() in ("q", "quit"):
                return
            try:
                idx = int(s)
            except ValueError:
                continue
            if 0 <= idx < len(recs):
                path = recs[idx]
                break
    Replay(path).run(max_frames=args.frames)


if __name__ == "__main__":
    main()
