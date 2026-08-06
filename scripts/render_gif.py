"""从录像 / 实况对打渲染 GIF/webm（复用 duel 完整 RES 渲染 + sim 真实推进）。

模式一 --rec <npz>：录像重放 —— 用 ReplaySim（play/replay.py）驱动：人类侧重放
   录的动作、对手侧 AI 决策、sim.step 真实推进。泡泡/爆炸/道具/移动全是 sim
   算的 → 与对打画面完全同步（无 obs 猜测/无滞后/无瞬移）。30fps 渲染插值。
模式二 --sim：实况对打 —— 真实 sim 跑 hunter/astar bot vs 网络 ckpt。

用法：
    python scripts/render_gif.py --rec recordings/rec_x.npz --scene 比武 \
        --out docs/demo_x.webm --seconds 15 --fmt webm
    python scripts/render_gif.py --sim --bot hunter \
        --ckpt ckpt/duel_cnn.pt --scene 英雄 --out docs/demo_hunter_vs_cnn.webm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import pygame
import torch

from play.duel import CELL, MOVE_DOWN, WALK_HZ, build_static, draw_grid
from play.replay import ReplaySim
from play.res import Res

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_frames(frames_dir: str, out: str, fps: int, fmt: str) -> None:
    """PNG 序列 → GIF / webm（浏览器 <video controls> 可播放/暂停）。"""
    if fmt == "gif":
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                        "-i", os.path.join(frames_dir, "frame_%05d.png"),
                        "-vf", "split[s0][s1];[s0]palettegen=max_colors=64[p];"
                               "[s1][p]paletteuse=dither=bayer",
                        "-loop", "0", out], check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-framerate", str(fps),
                        "-i", os.path.join(frames_dir, "frame_%05d.png"),
                        "-c:v", "libvpx-vp9", "-b:v", "1.5M",
                        "-pix_fmt", "yuva420p", out], check=True,
                       capture_output=True)


def run_rec(path: str, scene: str, out: str, seconds: int,
            fmt: str = "gif") -> None:
    """录像重放：人类动作驱动 sim 真实推进（ReplaySim），30fps 插值渲染。
    泡泡/爆炸/道具全部由 sim 计算 → 与对打完全同步。"""
    rs = ReplaySim(path)
    sim, cfg, meta = rs.sim, rs.cfg, rs.meta
    fps = 30
    sf = fps // 10                      # 10Hz 逻辑 → 30fps 渲染插值
    total = min(seconds * fps, rs.T * sf)
    res = Res(cell=CELL, blast=cfg.growth_blast_max, scene=scene)
    face = {0: MOVE_DOWN, 1: MOVE_DOWN}
    anim = {0: 0, 1: 0}
    screen = pygame.display.set_mode((cfg.width * CELL, cfg.height * CELL))
    obs = sim.observe()
    static = build_static(res, sim)
    prev_pos = sim.pos[0].float().clone()
    info = None
    with tempfile.TemporaryDirectory() as td:
        for fr in range(total):
            frac = (fr % sf) / sf
            if fr % sf == 0:                     # 每 tick 开头 step（重放人类动作）
                prev_pos = sim.pos[0].float().clone()
                info, obs = rs.step(int(sim.t[0]))
                dp = sim.pos[0].float() - prev_pos
                for pid in range(2):
                    dx, dy = float(dp[pid, 1]), float(dp[pid, 0])
                    if abs(dx) > 0.05 or abs(dy) > 0.05:
                        face[pid] = 0 if dy < 0 else 1 if dy > 0 \
                            else 2 if dx < 0 else 3
                static = build_static(res, sim)
            rpos = prev_pos + (sim.pos[0].float() - prev_pos) * frac
            expl = (info["blast"][0].bool(), info["trig"][0].bool()) \
                if info is not None and frac < 0.7 else None
            anim[0] = int(fr / fps * WALK_HZ) % 4
            anim[1] = anim[0]
            draw_grid(screen, res, sim, obs, rpos, face, anim, expl,
                      static=static)
            pygame.image.save(screen, os.path.join(td, f"frame_{fr:05d}.png"))
        encode_frames(td, out, fps, fmt)
    print(f"[{fmt}] {out} ({total} 帧 / {total/fps:.0f}s @{fps}fps) "
          f"地图={meta.get('map')} 对手={meta.get('opp')} 皮肤={scene}")


def run_sim(bot_kind: str, ckpt: str | None, ckpt_b: str | None,
            scene: str, out: str, seconds: int, map_: str,
            fmt: str = "gif") -> None:
    sys.path.insert(0, PROJ)
    from sim.bots import make_bot
    from sim.config import SimConfig
    from sim.torch_sim import BatchedSim
    from train.train import load_fixed_checkpoint

    def make_cfg(m: str) -> SimConfig:
        if m == "open":
            return SimConfig(map_mode="corridor", open_fraction=1.0,
                             max_steps=1800, speed=3.0, max_hp=5)
        if m == "ring":
            return SimConfig(map_mode="corridor", ring_fraction=1.0,
                             max_steps=1800, speed=3.0, max_hp=5)
        return SimConfig(map_mode="corridor", max_steps=1800, speed=3.0,
                         max_hp=5)

    cfg = make_cfg(map_)
    sim = BatchedSim(cfg, 1, device="cpu", seed=3)
    bot = make_bot(sim, bot_kind)
    net = load_fixed_checkpoint(ckpt, cfg.obs_shape, "cpu") if ckpt else None
    net_b = load_fixed_checkpoint(ckpt_b, cfg.obs_shape, "cpu") if ckpt_b else None
    if net:
        net.eval()
    if net_b:
        net_b.eval()
    res = Res(cell=CELL, blast=cfg.growth_blast_max, scene=scene)
    fps = 30
    sf = fps // 10
    total = seconds * fps
    face = {0: MOVE_DOWN, 1: MOVE_DOWN}
    anim = {0: 0, 1: 0}
    screen = pygame.display.set_mode((cfg.width * CELL, cfg.height * CELL))
    obs = sim.observe()
    static = build_static(res, sim)
    with tempfile.TemporaryDirectory() as td:
        for fr in range(total):
            frac = (fr % sf) / sf
            if fr % sf == 0:
                mm, bm = sim.legal_mask()
                with torch.no_grad():
                    if net is not None:
                        a0 = net.act(obs, mm[:, 0], bm[:, 0], 0)[0]
                    else:
                        a0 = bot.act(obs, mm[:, 0], bm[:, 0], 0)
                    if net_b is not None:
                        a1 = net_b.act(obs, mm[:, 1], bm[:, 1], 0)[0]
                    else:
                        a1 = bot.act(obs, mm[:, 1], bm[:, 1], 1)
                prev_pos = sim.pos[0].float().clone()
                _, _, info = sim.step(torch.stack([a0, a1], dim=1))
                dp = sim.pos[0].float() - prev_pos
                for pid in range(2):
                    dx, dy = float(dp[pid, 1]), float(dp[pid, 0])
                    if abs(dx) > 0.05 or abs(dy) > 0.05:
                        face[pid] = 0 if dy < 0 else 1 if dy > 0 \
                            else 2 if dx < 0 else 3
                obs = sim.observe()
                static = build_static(res, sim)
            rpos = prev_pos + (sim.pos[0].float() - prev_pos) * frac
            expl = (info["blast"][0].bool(), info["trig"][0].bool()) \
                if frac < 0.7 else None
            anim[0] = int(fr / fps * WALK_HZ) % 4
            anim[1] = anim[0]
            draw_grid(screen, res, sim, obs, rpos, face, anim, expl,
                      static=static)
            pygame.image.save(screen, os.path.join(td, f"frame_{fr:05d}.png"))
        encode_frames(td, out, fps, fmt)
    print(f"[{fmt}] {out} ({total} 帧 / {total/fps:.0f}s @{fps}fps) "
          f"{bot_kind} vs {os.path.basename(ckpt or ckpt_b or '?')} 皮肤={scene}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=None, help="录像 npz 路径")
    ap.add_argument("--sim", action="store_true", help="实况对打模式")
    ap.add_argument("--bot", default="hunter", choices=["hunter", "astar", "greedy"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ckpt-b", default=None)
    ap.add_argument("--map", default="corridor")
    ap.add_argument("--scene", default="比武")
    ap.add_argument("--out", default="docs/demo.webm")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--fmt", default="webm", choices=["gif", "webm"])
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))     # 先建窗口：convert_alpha 需要 video mode
    if args.rec:
        run_rec(args.rec, args.scene, args.out, args.seconds, args.fmt)
    else:
        run_sim(args.bot, args.ckpt, args.ckpt_b, args.scene, args.out,
                args.seconds, args.map, args.fmt)
    pygame.quit()


if __name__ == "__main__":
    main()
