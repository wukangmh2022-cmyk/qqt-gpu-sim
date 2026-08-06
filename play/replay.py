"""录像回放：从 recordings/*.npz 逐 tick 重放，检查人类轨迹录得对不对。

**真实场景渲染**：从 obs 通道构造 FakeSim（play/rec_state.py），复用 duel 的
build_static + draw_grid —— 背景大图/砖块贴图/角色精灵/泡泡素材/危险红区/
血条/无敌罩，与对打完全一致（不再是简单色块）。逐 tick 显示 obs 原始状态
（step 前快照）→ 泡泡/道具/位置完全同步，无插值时序偏差。

10Hz 播放（与录制同频）；空格暂停、←→ 逐帧（暂停时逐 tick 检查动作）、
↑↓ 调速（0.5/1/2/4×）、R 重头、D 切危险图、Q/ESC 退出。

用法：
    python -m play.replay               # 列表选录像（输入编号）
    python -m play.replay <npz路径>     # 直接播放
    python -m play.replay --frames 60   # 播放 60 帧退出（headless 冒烟用）
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import sys

import numpy as np
import pygame
import torch

from .rec_state import FakeSim
from .duel import CELL, MOVE_DOWN, WALK_HZ, _load_cjk_font, build_static, draw_grid

# ---- CJK 字体（与 duel 同款：常见中文字体路径，找不到回退默认）----
_CJK_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]

HUD_H = 52
MOVE_GLYPH = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "停"}


def list_recordings(dir_: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(dir_, "*.npz")), reverse=True)
    return paths


def load_rec(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float32) / 255.0          # (T,C,H,W) 还原 [0,1]
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    return {
        "obs": obs, "action": d["action"], "reward": d["reward"],
        "done": d["done"], "pid": int(d["pid"]), "meta": meta,
    }


def player_center(plane: np.ndarray) -> tuple[float, float] | None:
    """从双线性 splat 平面（格下标坐标）反推角色亚格位置（格中心坐标）。"""
    tot = float(plane.sum())
    if tot < 1e-6:
        return None
    h, w = plane.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy = float((plane * ys).sum()) / tot + 0.5          # fy 坐标 → 格中心
    cx = float((plane * xs).sum()) / tot + 0.5
    return cy, cx


def make_cfg(map_: str):
    from sim.config import SimConfig
    if map_ == "open":
        return SimConfig(map_mode="corridor", open_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5)
    if map_ == "ring":
        return SimConfig(map_mode="corridor", ring_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5)
    return SimConfig(map_mode="corridor", max_steps=1800, speed=3.0, max_hp=5)


class Replay:
    def __init__(self, path: str, scene: str = "比武") -> None:
        self.path = path
        self.rec = load_rec(path)
        self.obs = self.rec["obs"]
        self.T = self.obs.shape[0]
        _, self.C, self.H, self.W = self.obs.shape
        self.t = 0
        self.pause = False
        self.speed = 1.0
        self.show_danger = True
        self.loop = True
        self.scene = scene
        self.meta = self.rec["meta"]
        self.cfg = make_cfg(self.meta.get("map", "corridor"))
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.W * CELL, self.H * CELL + HUD_H))
        name = os.path.basename(path)
        pygame.display.set_caption(f"录像回放 · {name}")
        self.font = _load_cjk_font(20)
        self.font_s = _load_cjk_font(16)
        from .res import Res
        self.res = Res(cell=CELL, blast=self.cfg.growth_blast_max, scene=scene)
        self.fake = FakeSim(self.cfg)
        self.face = {0: MOVE_DOWN, 1: MOVE_DOWN}
        self.anim = {0: 0, 1: 0}
        self._static = None
        self.acc = 0.0
        self.tick_dt = 0.1 / self.speed

    # ---- 绘制一帧（t 时刻，真实 RES 渲染）----
    def draw(self) -> None:
        s = self.screen
        o = torch.from_numpy(self.obs[self.t].astype(np.float32))
        self.fake.set_frame(o, self.cfg)
        if self._static is None:
            self._static = build_static(self.res, self.fake)
        # face：位移方向（逐 tick 近似）
        if self.t > 0:
            fx = FakeSim(self.cfg)
            fx.set_frame(torch.from_numpy(self.obs[self.t - 1].astype(np.float32)),
                         self.cfg)
            dp = self.fake.pos[0] - fx.pos[0]
            for pid in range(2):
                dx, dy = float(dp[pid, 1]), float(dp[pid, 0])
                if abs(dx) > 0.05 or abs(dy) > 0.05:
                    self.face[pid] = 0 if dy < 0 else 1 if dy > 0 \
                        else 2 if dx < 0 else 3
        self.anim[0] = int(self.t / 10.0 * WALK_HZ) % 4
        self.anim[1] = self.anim[0]
        draw_grid(s, self.res, self.fake, None, self.fake.pos[0],
                  self.face, self.anim, None, static=self._static)
        # HUD：tick 进度 / 动作 / 奖励 / meta（叠在真实画面上）
        act = self.rec["action"][self.t]
        rew = float(self.rec["reward"][self.t])
        meta = self.meta
        done = bool(self.rec["done"][self.t])
        hx = 12
        hud = self.font.render(
            f"t {self.t}/{self.T}  "
            f"动作 {MOVE_GLYPH.get(int(act[0]), '?')}"
            f"{'💣' if int(act[1]) else ''}   "
            f"奖励 {rew:+.3f}{'  [终局]' if done else ''}"
            f"{'  ⏸' if self.pause else ''}  {self.speed:.0f}×",
            True, (235, 235, 240))
        s.blit(hud, (hx, self.H * CELL + 8))
        meta_s = self.font_s.render(
            f"pid={meta.get('pid', self.rec['pid'])}  地图={meta.get('map', '?')}  "
            f"对手={meta.get('opp', '?')}  成长 {meta.get('bombs', '?')}/"
            f"{meta.get('blast', '?')}/{meta.get('speed', '?')}   "
            f"[空格暂停 ←→逐帧 ↑↓调速 R重头 D危险 Q退出]",
            True, (170, 170, 185))
        s.blit(meta_s, (hx, self.H * CELL + 28))
        # 进度条
        pygame.draw.rect(s, (60, 58, 70), (12, self.H * CELL + 2,
                                           self.W * CELL - 24, 4))
        if self.T > 1:
            pygame.draw.rect(s, (90, 200, 130),
                             (12, self.H * CELL + 2,
                              int((self.W * CELL - 24) * self.t / (self.T - 1)), 4))

    # ---- 事件 ----
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
    ap.add_argument("--frames", type=int, default=None,
                    help="播放 N 帧退出（headless 冒烟用）")
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
                T = d["obs"].shape[0]
                try:
                    meta = ast.literal_eval(str(d["meta"][0]))
                except Exception:
                    meta = {}
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
