"""录像回放：从 recordings/*.npz 逐 tick 重放，检查人类轨迹录得对不对。

直接画观测通道（不需要 sim / 不需要网络）：
    墙/砖   ch4    >0.5 → 深灰砖格
    玩家0   ch0    双线性 splat → 重心反推亚格位置（人类视角，自己=绿、对手=红）
    玩家1   ch1    同上
    引信    ch2/ch3 fuse_norm >0 → 该格有泡（画蓝圈，随引信淡）
    危险    ch5    可选（D 键开关）淡红覆盖
    宝箱    ch7    >0.5 → 金色小方块
    进度    ch6    t/max_steps → 状态栏进度条

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


def _load_cjk_font(size: int) -> pygame.font.Font:
    for path in _CJK_FONT_PATHS:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except pygame.error:
                continue
    return pygame.font.Font(None, size)


CELL = 52
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


class Replay:
    def __init__(self, path: str) -> None:
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
        pygame.init()
        self.screen = pygame.display.set_mode((self.W * CELL, self.H * CELL + HUD_H))
        name = os.path.basename(path)
        pygame.display.set_caption(f"录像回放 · {name}")
        self.font = _load_cjk_font(20)
        self.font_s = _load_cjk_font(16)
        # 静态背景格（浅色棋盘）只画一次，回放帧 blit 后叠动态层
        self.bg = pygame.Surface((self.W * CELL, self.H * CELL))
        self.bg.fill((46, 44, 54))
        for r in range(self.H):
            for c in range(self.W):
                if (r + c) % 2:
                    pygame.draw.rect(self.bg, (52, 50, 60),
                                     (c * CELL, r * CELL, CELL, CELL))
        self.acc = 0.0
        self.tick_dt = 0.1 / self.speed

    # ---- 绘制一帧（t 时刻）----
    def draw(self) -> None:
        s = self.screen
        s.blit(self.bg, (0, 0))
        o = self.obs[self.t]                               # (C,H,W)
        wall = o[2 * 2]                                    # ch4 墙|砖
        crate = o[2 * 2 + 3] if self.C > 2 * 2 + 3 else None
        danger = o[2 * 2 + 1]
        fuse0, fuse1 = o[2], o[3]
        # 危险覆盖（D 键开关）
        if self.show_danger:
            danger_ov = pygame.Surface((self.W * CELL, self.H * CELL),
                                       pygame.SRCALPHA)
            for r in range(self.H):
                for c in range(self.W):
                    dv = float(danger[r, c])
                    if dv > 0:
                        alpha = min(90, int(110 * dv))
                        pygame.draw.rect(
                            danger_ov, (255, 60, 60, alpha),
                            (c * CELL, r * CELL, CELL, CELL))
            s.blit(danger_ov, (0, 0))
        # 格子实体：墙/砖 + 宝箱
        for r in range(self.H):
            for c in range(self.W):
                x, y = c * CELL, r * CELL
                if float(wall[r, c]) > 0.5:
                    pygame.draw.rect(s, (96, 88, 78), (x, y, CELL, CELL))
                    pygame.draw.rect(s, (120, 110, 98), (x, y, CELL, CELL), 2)
                elif crate is not None and float(crate[r, c]) > 0.5:
                    pygame.draw.rect(s, (200, 170, 40),
                                     (x + CELL // 4, y + CELL // 4,
                                      CELL // 2, CELL // 2), border_radius=4)
        # 泡泡（引信 >0）：蓝圈，随引信透明度变化
        for r in range(self.H):
            for c in range(self.W):
                for fv, col in ((float(fuse0[r, c]), (90, 170, 255)),
                                (float(fuse1[r, c]), (255, 130, 90))):
                    if fv > 0:
                        x, y = c * CELL, r * CELL
                        rad = int(CELL * 0.36)
                        pygame.draw.circle(s, col, (x + CELL // 2, y + CELL // 2),
                                           rad, 2)
        # 角色：位置通道重心反推亚格位置；自己(通道0)=绿、对手(通道1)=红
        for ch, col, tag in ((0, (80, 220, 120), "自己"), (1, (240, 100, 100), "对手")):
            pos = player_center(o[ch])
            if pos is None:
                continue
            py_, px_ = pos
            x, y = int(px_ * CELL), int(py_ * CELL)
            pygame.draw.circle(s, col, (x, y), int(CELL * 0.38))
            pygame.draw.circle(s, (255, 255, 255), (x, y), int(CELL * 0.38), 2)
        # HUD：tick 进度 / 动作 / 奖励 / meta
        act = self.rec["action"][self.t]
        rew = float(self.rec["reward"][self.t])
        meta = self.rec["meta"]
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
