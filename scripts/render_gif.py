"""从录像 / 实况对打渲染 GIF（复用 duel 的完整 RES 皮肤渲染）。

模式一 --rec <npz>：录像状态回放 —— 读录像 obs（14 通道）逐帧构造伪 sim，
   走 build_static + draw_grid 渲染（皮肤/精灵/Z 排序/危险红区/泡泡），
   不重跑 sim（录像轨迹是确定的）。10Hz 与录像同频。
模式二 --sim：实况对打 —— 真实 sim 跑 hunter/astar bot vs 网络 ckpt，
   step 的 info[blast/trig] 直接画爆炸动画（更精彩）。

用法：
    python scripts/render_gif.py --rec recordings/rec_x.npz --scene 比武 \
        --out docs/demo_x.gif --seconds 15
    python scripts/render_gif.py --sim --bot hunter \
        --ckpt ckpt/duel_cnn.pt --scene 英雄 --out docs/demo_hunter_vs_cnn.gif
    python scripts/render_gif.py --sim --bot hunter \
        --ckpt ckpt/course_1023m.pt --scene 矿洞 --out docs/demo_hunter_vs_1023.gif
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile

import numpy as np
import pygame
import torch

from play.duel import (CELL, MOVE_DOWN, WALK_HZ, _load_cjk_font, build_static,
                       draw_grid)
from play.rec_state import FakeSim, explosion_between
from play.res import Res
from play.replay import player_center

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_cfg(map_: str) -> "SimConfig":
    from sim.config import SimConfig
    if map_ == "open":
        return SimConfig(map_mode="corridor", open_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5)
    if map_ == "ring":
        return SimConfig(map_mode="corridor", ring_fraction=1.0, max_steps=1800,
                         speed=3.0, max_hp=5)
    return SimConfig(map_mode="corridor", max_steps=1800, speed=3.0, max_hp=5)


def encode_frames(frames_dir: str, out: str, fps: int, fmt: str) -> None:
    """PNG 序列 → GIF / webm。
    gif：palette 两遍压体积；webm（vp9）：浏览器 <video controls> 可播放/暂停，
    体积小画质高（预览/README 首选）。"""
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


def render_frames(sim_or_fake, res, explosion_fn, total_frames, frames_dir,
                  face, anim, now):
    """通用：逐帧渲染存 PNG。sim_or_fake 提供 build_static 需要的状态。
    explosion_fn(i) -> (blast|None, trig|None)。"""
    h, w = sim_or_fake.cfg.height, sim_or_fake.cfg.width
    screen = pygame.display.set_mode((w * CELL, h * CELL))
    static = build_static(res, sim_or_fake)
    for i in range(total_frames):
        sim_or_fake.set_frame  # noqa (rec 模式在循环外已填)
        # face：从位置差推断移动方向（0上 1下 2左 3右）
        if i > 0:
            pass  # 由调用方维护
        explosion = explosion_fn(i)
        draw_grid(screen, res, sim_or_fake, None, sim_or_fake.pos[0],
                  face, anim, explosion, static=static)
        pygame.image.save(screen, os.path.join(frames_dir, f"frame_{i:05d}.png"))


def _interp_fake(cur: FakeSim, nxt: FakeSim, frac: float, cfg,
                 explode) -> FakeSim:
    """中间帧状态：位置在 cur→nxt 插值；**泡泡/宝箱用 cur（tick t 开始状态）**
    —— 录像 obs[t] 是 step 前快照，放炮/吃箱/爆炸发生在 tick t **内**、到
    obs[t+1] 才反映。若用 nxt：泡/宝箱会在区间开头（玩家还在 A 点）就出现/
    消失 → 视觉"提前"（看起来像动作延迟）。用 cur：泡在玩家放炮后（位置到
    B 点）的下一区间才出现、宝箱在玩家到达瞬间被吃 → 动作与状态同步。"""
    f = FakeSim(cfg)
    f.wall.copy_(cur.wall)
    f.brick.copy_(cur.brick)
    f.crate.copy_(cur.crate)
    f.t.copy_(cur.t)
    f.invuln.copy_(cur.invuln)
    f.hp.copy_(cur.hp)
    f.alive.copy_(cur.alive)
    for pid in range(2):
        if bool(cur.alive[0, pid]):
            f.pos[0, pid] = cur.pos[0, pid] + \
                (nxt.pos[0, pid] - cur.pos[0, pid]) * frac
        else:
            f.pos[0, pid] = cur.pos[0, pid]
    f.fuse.copy_(cur.fuse)
    f.owner.copy_(cur.owner)
    if explode is not None:
        trig = explode[1]
        f.fuse[0][trig] = 0
        f.owner[0][trig] = -1
    return f


def run_rec(path: str, scene: str, out: str, seconds: int,
            fmt: str = "gif") -> None:
    d = np.load(path, allow_pickle=True)
    obs = d["obs"].astype(np.float32) / 255.0
    try:
        meta = ast.literal_eval(str(d["meta"][0]))
    except Exception:
        meta = {}
    cfg = make_cfg(meta.get("map", "corridor"))
    fps = 30
    sf = fps // 10                      # 10Hz 录像 → 30fps：每 tick 3 帧插值
    T = obs.shape[0]
    total = min(seconds * fps, (T - 1) * sf)
    res = Res(cell=CELL, blast=cfg.growth_blast_max, scene=scene)
    # 预计算每 tick 快照
    snaps = []
    for i in range(T):
        f = FakeSim(cfg)
        f.set_frame(obs[i], cfg)
        snaps.append(f)
    face = {0: MOVE_DOWN, 1: MOVE_DOWN}
    anim = {0: 0, 1: 0}
    prev_pos = snaps[0].pos[0].clone()
    screen = pygame.display.set_mode((cfg.width * CELL, cfg.height * CELL))
    with tempfile.TemporaryDirectory() as td:
        for fr in range(total):
            tick = fr // sf
            frac = (fr % sf) / sf
            cur = snaps[tick]
            nxt = snaps[min(tick + 1, T - 1)]
            # 爆炸发生在 tick 内 step 时刻（obs[t+1] 才反映）→ 画在区间**后半**
            # （frac≥0.5），前半仍显示 cur 的泡（step 前状态，同步不提前）
            expl = explosion_between(cur, nxt, cfg) if frac >= 0.5 else None
            mid = _interp_fake(cur, nxt, frac, cfg, expl)
            # face：cur tick 相对上一 tick 的位移
            dp = cur.pos[0] - prev_pos
            for pid in range(2):
                dx, dy = float(dp[pid, 1]), float(dp[pid, 0])
                if abs(dx) > 0.05 or abs(dy) > 0.05:
                    face[pid] = 0 if dy < 0 else 1 if dy > 0 \
                        else 2 if dx < 0 else 3
            anim[0] = int(fr / fps * WALK_HZ) % 4
            anim[1] = anim[0]
            draw_grid(screen, res, mid, None, mid.pos[0], face, anim, expl,
                      static=build_static(res, mid))
            pygame.image.save(screen, os.path.join(td, f"frame_{fr:05d}.png"))
            if tick < T - 1:
                prev_pos = cur.pos[0].clone()
        encode_frames(td, out, fps, fmt)
    print(f"[{fmt}] {out} ({total} 帧 / {total/fps:.0f}s @{fps}fps) "
          f"地图={meta.get('map')} 皮肤={scene}")


def run_sim(bot_kind: str, ckpt: str | None, ckpt_b: str | None,
            scene: str, out: str, seconds: int, map_: str,
            fmt: str = "gif") -> None:
    import sys
    sys.path.insert(0, PROJ)
    from sim.bots import make_bot
    from sim.config import SimConfig
    from sim.torch_sim import BatchedSim
    from train.train import load_fixed_checkpoint

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
    sf = fps // 10                      # 10Hz 逻辑 → 30fps 渲染插值
    total = seconds * fps
    face = {0: MOVE_DOWN, 1: MOVE_DOWN}
    anim = {0: 0, 1: 0}
    screen = pygame.display.set_mode((cfg.width * CELL, cfg.height * CELL))
    obs = sim.observe()
    static = build_static(res, sim)
    with tempfile.TemporaryDirectory() as td:
        for fr in range(total):
            frac = (fr % sf) / sf
            if fr % sf == 0:                     # 每 tick 开头 step 一次
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
            # 爆炸只在 tick 后半段淡出（前 0.7 显示）
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
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--fmt", default="gif", choices=["gif", "webm"])
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
