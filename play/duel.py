"""和训练好的 AI 对打（你 = player 0，AI = player 1）。

控制：
    方向键          按住持续移动；**最后按下的方向优先**（按住↑轻点← 切到←）
    空格            放一个泡泡（trigger，按一次放一个）
    R               重新开局
    ESC             结束对局退出（无 Python 启动器了，浏览器版用 web/）
    Q               退出

逻辑 10Hz（与训练一致），渲染 60fps + 位置插值 → 平滑滑动。

渲染用 res/ 素材：
    背景 bg1.png；炸弹 bomb1..6 呼吸动画；爆炸中心+四方向臂按 blast 切片；
    角色 4×4 精灵图（行走动画，AI 染红调）；放泡/爆炸音效。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

# 无窗口场景（冒烟 / 截图）用 dummy 驱动，交互模式用 cocoa。
# 必须在 import pygame 之前决定。
_headless = ("--screenshot" in sys.argv or "--auto-ticks" in sys.argv
              or bool(os.environ.get("SDL_VIDEODRIVER_DUMMY")))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy" if _headless else "cocoa")

import pygame  # noqa: E402

import torch  # noqa: E402

from sim.config import (DIRS, MOVE_DOWN, MOVE_IDLE, MOVE_LEFT, MOVE_RIGHT,  # noqa: E402
                        MOVE_UP, SimConfig, obs_extra)
from sim.move import _EPS, _resolve_axis  # noqa: E402
from sim.torch_sim import BatchedSim  # noqa: E402
from sim.bots import make_bot  # noqa: E402
from sim.obs import local_view_features  # noqa: E402
from train.model import ActorCritic, infer_players  # noqa: E402
from train.train import adapt_first_conv  # noqa: E402

from .recorder import Recorder  # noqa: E402
from .res import MOVE_TO_SPRITE_ROW, Res  # noqa: E402

# P0（第一人类玩家）：WASD 移动 + 空格放泡（左手侧）。
# P1（第二人类玩家）：方向键移动 + 回车放泡（右手侧，双人同屏对打互不干扰）。
# 用户要求 P1 用方向键（方便双打时安排位置），P0 让出方向键改用 WASD。
DIR_KEYS = {
    pygame.K_w: MOVE_UP,
    pygame.K_s: MOVE_DOWN,
    pygame.K_a: MOVE_LEFT,
    pygame.K_d: MOVE_RIGHT,
}
MV_TO_KEY = {mv: key for key, mv in DIR_KEYS.items()}
DIR_GLYPH = {MOVE_UP: "W", MOVE_DOWN: "S", MOVE_LEFT: "A", MOVE_RIGHT: "D"}
DIR_KEYS1 = {
    pygame.K_UP: MOVE_UP,
    pygame.K_DOWN: MOVE_DOWN,
    pygame.K_LEFT: MOVE_LEFT,
    pygame.K_RIGHT: MOVE_RIGHT,
}
MV_TO_KEY1 = {mv: key for key, mv in DIR_KEYS1.items()}
DIR_GLYPH1 = {MOVE_UP: "↑", MOVE_DOWN: "↓", MOVE_LEFT: "←", MOVE_RIGHT: "→"}

# 状态栏中文需要 CJK 字体：pygame 默认字体（Font(None)）不含中文字形，
# 渲染出来全是方块。按常见系统中文字体路径逐个找，找不到回退默认（方块兜底）。
_CJK_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",              # macOS
    "/System/Library/Fonts/STHeiti Light.ttc",         # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",      # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux
    "C:/Windows/Fonts/msyh.ttc",                       # Windows
]


def _load_cjk_font(size: int) -> pygame.font.Font:
    for path in _CJK_FONT_PATHS:
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except pygame.error:
                continue
    return pygame.font.Font(None, size)   # 兜底：默认字体（中文可能显示方块）

CELL = 60                     # 画布放大 1.5 倍：素材原生 40px/格 → 60px/格
GRID = 13                     # 地图边长（与 SimConfig 默认一致）
WALK_HZ = 6.0                 # 走路动画帧率（4 帧循环，60fps 时间驱动）
# 锚点模型（无偏移）：所有站立的精灵（角色/泡泡/宝箱）锚点在**所在格底边正中心**
# (c*CELL+CELL/2, (r+1)*CELL)。精灵图只往锚点上方展开（左/上/右可超出，永不越过
# 底线）：
#   角色：切片底部贴锚点 → blit_y = 锚点y − 高（脚踩格底，脚底 = 切片切线底部）
#   泡泡/宝箱：圆心贴锚点 → blit_y = 锚点y − 高/2
# 不做额外像素偏移 —— 精灵图本身脚底在切片底部（4×4 行走图 340px = 4×85 精确
# 切分，无错位），偏移只会引入观感偏差。
# 条件自动转向触发阈值：角色中心到格边缘线 < TURN_EPS 格才触发（fx<=TURN_EPS 偏左、
# fx>=1-TURN_EPS 偏右…）。0.4 = 中心偏离格中线的 80% 区域都触发 —— 只有贴中线
# 正冲（中间 20%）才不拐，转向靠墙/泡时几乎总生效；0.2 只允许"差一点跨格"的
# 窄带触发，微操转向经常拐不动。TURN_EPS=0.5 会连走中线都乱拐（无脑拐弯），
# 0.4 留出正中窄带给直线贴墙/贴泡的路径。
TURN_EPS = 0.4
WALL_TINT = (60, 110, 240, 110)   # 不可炸墙（永久）：蓝色半透明
BRICK_TINT = (235, 200, 60, 110)  # 可炸墙（brick）：黄色半透明
CRATE_TINT = (60, 220, 90, 120)   # 宝箱（砖炸掉后）：绿色半透明，走到即开
DANGER_TINT = (200, 30, 60, 70)   # 危险区：半透明红
DEAD_FADE = (30, 30, 30, 200)     # 死亡角色：压暗


def ckpt_channels(ckpt: str) -> int:
    """checkpoint 的观测通道数（判断是否用了扩展观测，决定 cfg.obs_extra_enabled）。"""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return int(ck["obs_shape"][0])


def load_ai(ckpt: str, device) -> ActorCritic:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    arch = ck.get("arch", "cnn")              # 旧 ckpt 无 arch 字段 → 默认 cnn
    c_ck = ck["obs_shape"][0]
    # 旧档无 n_players 字段：按通道布局反推真实人数。**C=9 有歧义**：
    # 5P+4 解出 P=1（单人训练不存在），实际 9 = 2P+3 的 3 人课程档 ——
    # 必须先按 2P+3 解释，否则 infer_players 误判成 1 人 → 适配被跳过
    # （14 通道观测撞 9 通道权重，RuntimeError）。
    n_p = ck.get("n_players")
    if n_p is None:
        if (c_ck - 3) % 2 == 0:
            n_p = (c_ck - 3) // 2              # 2P+3：7→2、9→3（歧义按此）
        elif (c_ck - 4) % 5 == 0:
            n_p = (c_ck - 4) // 5              # 5P+4：14→2
        else:
            n_p = infer_players(c_ck)
    net = ActorCritic(tuple(ck["obs_shape"]), arch=arch, n_players=n_p).to(device)
    net.load_state_dict(ck["model"])
    # 训练走课程会推进到 3 人（C=9）；对打是 1v1（C=7/14）。用课程同款的
    # adapt_first_conv 把权重缩回 2 人，保留学到的自身通道与第一个对手通道。
    # 目标通道数固定 2 人布局：7 通道旧档 → 关扩展观测；14 通道新档 → 保留扩展
    # （3 人旧档 9 通道 → 也适配成 14 通道 2 人扩展观测，与 cfg 一致）。
    extra = c_ck > 2 * 2 + 3                  # 7 = 2 人基础布局
    two_c = 2 * 2 + 3 + (obs_extra(2) if extra else 0)
    two_shape = (two_c, ck["obs_shape"][1], ck["obs_shape"][2])
    if tuple(net.obs_shape) != two_shape:
        net = adapt_first_conv(net, two_shape, arch=arch,
                               n_players=None).to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def ai_action(net, obs, mmask, bmask, device, pid: int = 1) -> torch.Tensor:
    """AI 决策：pid 决定视角（0 = 玩家 0，1 = 玩家 1）。

    单人打 AI 时 AI 是 player 1（pid=1，默认）；观战模式双方各用各的 pid ——
    之前 a0/a1 都取 pid=1，导致两个 bot 每 tick 动作完全相同（观战里两个
    角色同轨迹平移），玩家 0 的朝向/行走帧也永不更新（一张图平移）。
    """
    with torch.no_grad():
        a, _, _ = net.act(obs, mmask[:, pid], bmask[:, pid], pid)
    return a


def _swap_player_channels(obs: torch.Tensor) -> torch.Tensor:
    """把物理玩家 0/1 的 per-player 通道互换，让物理 P1 的模型"自己 = 通道 0"。

    通道布局（P=2，见 sim/config.py 与 sim/obs.py）：
      0,1 玩家位置；2,3 玩家引信；4墙 5危险 6进度；7宝箱(共享)；
      8,9 无敌；10,11 可用泡；12,13 泡上限 —— 每段内 0↔1 互换。
    共享通道（墙/危险/进度/宝箱）不动。重排后模型用 pid=0 视角，
    "玩家0、玩家1 都认为自己是玩家0"（各自读到自己在通道 0）。

    **通道数自适应**：7 通道旧观测（obs_extra 关闭，如纯 bot/human 对打无 ckpt）
    只有 位置/引信 两组 per-player 通道，扩展段（8..13）不存在 —— 只换前两组，
    不能按 14 通道硬索引（否则越界崩：P0=寻路AI + P1=人类录制时 obs=7 通道）。
    """
    p = 2
    base = 2 * p + 3                    # 7 = 宝箱（共享，不动）
    c = obs.shape[1]
    idx = list(range(c))
    segs = [range(0, p), range(p, 2 * p)]              # 位置 / 引信（恒有）
    if c > base:                                       # 14 通道才有的扩展段
        segs += [range(base + 1, base + 1 + p),             # 无敌 (8,9)
                 range(base + 1 + p, base + 1 + 2 * p),     # 可用泡 (10,11)
                 range(base + 1 + 2 * p, base + 1 + 3 * p)]  # 上限 (12,13)
    for seg in segs:
        seg = list(seg)
        idx[seg[0]], idx[seg[1]] = idx[seg[1]], idx[seg[0]]
    return obs[:, idx]


# ---------------------------------------------------------------- 渲染


# 动作方向表（与 sim.config.DIRS 对齐：0上 1下 2左 3右）—— 条件自动转向的
# 对角格（原方向 + 转向方向 dy/dx 之和）计算用。
_DY = (-1, 1, 0, 0)
_DX = (0, 0, -1, 1)


# ---------------------------------------------------------------- 玩家帧级移动
# 玩家移动走 60Hz 渲染帧实时执行（只动玩家 0 的坐标）：按键按下那一帧就移动，
# 距离 = 速度 × 该帧真实时长 —— 轻点 1 帧 (~16ms) 走 ~0.06 格、按住每帧都走，
# 微操距离与按键时长真正成正比（轻点 vs 稍长按 trigger 的距离不同）。
# 碰撞/自动转向复用 sim/move.py 的 _resolve_axis 同款规则；AI、泡泡、爆炸照旧
# 10Hz tick 推进，两套路径互不干扰（AI 决策/step 走公共库，与测试同源）。


def _player_blocked_flat(sim) -> torch.Tensor:
    """当前帧玩家碰撞掩码（墙|砖|在场泡），展平成 (1, H*W) 供 _resolve_axis。"""
    return (sim.wall[0] | sim.brick[0] | (sim.fuse[0] > 0)).view(1, -1)


def _probe_move(sim, mv: int) -> bool:
    """方向 mv 是否真的能走（与 legal_mask 同款判定：按 step_len 试探碰撞消解）。"""
    cfg = sim.cfg
    y = sim.pos[0, 0, 0].reshape(1)
    x = sim.pos[0, 0, 1].reshape(1)
    bf = _player_blocked_flat(sim)
    step = cfg.step_len
    dy = torch.tensor([DIRS[mv][0] * step], dtype=sim.pos.dtype, device=sim.pos.device)
    dx = torch.tensor([DIRS[mv][1] * step], dtype=sim.pos.dtype, device=sim.pos.device)
    if dy.item() != 0:
        ny = _resolve_axis(y + dy, dy, x, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, True)
        return bool(abs(float(ny - y)) > _EPS * 2)
    nx = _resolve_axis(x + dx, dx, y, y, x, bf, cfg.radius,
                       cfg.height, cfg.width, False)
    return bool(abs(float(nx - x)) > _EPS * 2)


def _auto_turn(sim, move: int) -> int:
    """条件自动转向（手感辅助，**逐帧**生效，只作用于玩家 0）：

    - 中心到格边缘线 < TURN_EPS 才触发（fx<=TURN_EPS 偏左、fx>=1-TURN_EPS 偏右、
      fy 同理）—— 贴中线直冲不拐；
    - 转向方向由"偏哪边"决定：上/下被挡 → 偏左转左(2)/偏右转右(3)；
      左/右被挡 → 偏上转上(0)/偏下转下(1)；
    - **侧前对角格可通行才触发**（原方向与转向方向 dy/dx 之和），否则不转。
    """
    cfg = sim.cfg
    h, w = cfg.height, cfg.width
    if move >= 4 or _probe_move(sim, move):
        return move
    gy_f = float(sim.pos[0, 0, 0])
    gx_f = float(sim.pos[0, 0, 1])
    fx = gx_f - math.floor(gx_f)
    fy = gy_f - math.floor(gy_f)
    if move in (0, 1):          # 上/下被挡 → 偏左转左、偏右转右
        alt = 2 if fx <= TURN_EPS else (3 if fx >= 1.0 - TURN_EPS else None)
    else:                        # 左/右被挡 → 偏上转上、偏下转下
        alt = 0 if fy <= TURN_EPS else (1 if fy >= 1.0 - TURN_EPS else None)
    if alt is None:
        return move
    r_int, c_int = int(gy_f), int(gx_f)
    nr = r_int + _DY[move] + _DY[alt]
    nc = c_int + _DX[move] + _DX[alt]
    if 0 <= nr < h and 0 <= nc < w and \
            not bool(sim.wall[0, nr, nc]) and \
            not bool(sim.brick[0, nr, nc]) and \
            int(sim.fuse[0, nr, nc]) <= 0:
        return alt
    return move


def _player_frame_move(sim, mv: int, dt: float) -> None:
    """玩家 60Hz 帧级移动：沿 mv 走 speed×成长×玩家倍率×dt 格，AABB 碰撞消解。

    只动玩家 0 的坐标；与 sim/move.py::move_players 同款碰撞（贴墙滑动、
    脚下刚放的泡放行）。dt 钳到一个 tick 的时长（0.1s）—— 帧率再低也不会
    单帧位移超过模拟器自身单步上限（防高速穿模）。
    """
    cfg = sim.cfg
    if mv == MOVE_IDLE or not bool(sim.alive[0, 0]):
        return
    sm = float(sim.spd_g[0, 0])
    if sim.speed_mult is not None:
        sm *= float(sim.speed_mult[0, 0])
    dist = cfg.speed * sm * min(dt, 0.1)
    if dist <= 0:
        return
    y = sim.pos[0, 0, 0].reshape(1)
    x = sim.pos[0, 0, 1].reshape(1)
    bf = _player_blocked_flat(sim)
    dy = torch.tensor([DIRS[mv][0] * dist], dtype=sim.pos.dtype, device=sim.pos.device)
    dx = torch.tensor([DIRS[mv][1] * dist], dtype=sim.pos.dtype, device=sim.pos.device)
    if dy.item() != 0:
        ny = _resolve_axis(y + dy, dy, x, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, True)
        sim.pos[0, 0, 0] = float(ny)
    if dx.item() != 0:
        nx = _resolve_axis(x + dx, dx, y, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, False)
        sim.pos[0, 0, 1] = float(nx)


def _auto_turn1(sim, move: int) -> int:
    """P1（第二人类）的条件自动转向 —— 与 P0 同款规则，作用玩家 1。"""
    return _auto_turn_for(sim, move, pid=1)


def _auto_turn_for(sim, move: int, pid: int) -> int:
    """条件自动转向（任意玩家）：中心到格边缘 < TURN_EPS 才触发，方向由偏哪边
    决定，侧前对角格可通行才转（与 P0 的 _auto_turn 同规则，pid 参数化）。"""
    cfg = sim.cfg
    h, w = cfg.height, cfg.width
    if move >= 4 or _probe_move_for(sim, move, pid):
        return move
    gy_f = float(sim.pos[0, pid, 0])
    gx_f = float(sim.pos[0, pid, 1])
    fx = gx_f - math.floor(gx_f)
    fy = gy_f - math.floor(gy_f)
    if move in (0, 1):          # 上/下被挡 → 偏左转左、偏右转右
        alt = 2 if fx <= TURN_EPS else (3 if fx >= 1.0 - TURN_EPS else None)
    else:                        # 左/右被挡 → 偏上转上、偏下转下
        alt = 0 if fy <= TURN_EPS else (1 if fy >= 1.0 - TURN_EPS else None)
    if alt is None:
        return move
    r_int, c_int = int(gy_f), int(gx_f)
    nr = r_int + _DY[move] + _DY[alt]
    nc = c_int + _DX[move] + _DX[alt]
    if 0 <= nr < h and 0 <= nc < w and \
            not bool(sim.wall[0, nr, nc]) and \
            not bool(sim.brick[0, nr, nc]) and \
            int(sim.fuse[0, nr, nc]) <= 0:
        return alt
    return move


def _probe_move_for(sim, mv: int, pid: int) -> bool:
    """方向 mv 对玩家 pid 是否真的能走（同 legal_mask 判定）。"""
    cfg = sim.cfg
    y = sim.pos[0, pid, 0].reshape(1)
    x = sim.pos[0, pid, 1].reshape(1)
    bf = _player_blocked_flat(sim)
    step = cfg.step_len
    dy = torch.tensor([DIRS[mv][0] * step], dtype=sim.pos.dtype, device=sim.pos.device)
    dx = torch.tensor([DIRS[mv][1] * step], dtype=sim.pos.dtype, device=sim.pos.device)
    if dy.item() != 0:
        ny = _resolve_axis(y + dy, dy, x, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, True)
        return bool(abs(float(ny - y)) > _EPS * 2)
    nx = _resolve_axis(x + dx, dx, y, y, x, bf, cfg.radius,
                       cfg.height, cfg.width, False)
    return bool(abs(float(nx - x)) > _EPS * 2)


def _player_frame_move1(sim, mv: int, dt: float) -> None:
    """P1（第二人类）的帧级移动 —— 同 P0 微操手感，只动玩家 1 的坐标。"""
    cfg = sim.cfg
    if mv == MOVE_IDLE or not bool(sim.alive[0, 1]):
        return
    sm = float(sim.spd_g[0, 1])
    if sim.speed_mult is not None:
        sm *= float(sim.speed_mult[0, 1])
    dist = cfg.speed * sm * min(dt, 0.1)
    if dist <= 0:
        return
    y = sim.pos[0, 1, 0].reshape(1)
    x = sim.pos[0, 1, 1].reshape(1)
    bf = _player_blocked_flat(sim)
    dy = torch.tensor([DIRS[mv][0] * dist], dtype=sim.pos.dtype, device=sim.pos.device)
    dx = torch.tensor([DIRS[mv][1] * dist], dtype=sim.pos.dtype, device=sim.pos.device)
    if dy.item() != 0:
        ny = _resolve_axis(y + dy, dy, x, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, True)
        sim.pos[0, 1, 0] = float(ny)
    if dx.item() != 0:
        nx = _resolve_axis(x + dx, dx, y, y, x, bf, cfg.radius,
                           cfg.height, cfg.width, False)
        sim.pos[0, 1, 1] = float(nx)


def build_static(res: Res, sim) -> pygame.Surface:
    """背景层（纯地面/背景，**不含墙/砖 tile**）。

    墙/砖是有高度的竖立物（tile 底边对齐格底、向上延伸超一格），角色站在
    它们上方一格时脚部应被遮挡。这靠 **Z = 行 Y 的画家算法** 解决（见 draw_grid）：
    所有精灵（墙/砖 tile、角色、泡泡…）带一个 z = 所在行，按 z 升序（远→近）
    逐个 blit，近行（z 大）后画 → 自然盖住远行精灵伸下来的部分。所以墙/砖
    不能烧进静态整图，要作为独立精灵参与 Z 排序。
    """
    cfg = sim.cfg
    h, w = cfg.height, cfg.width
    static = pygame.Surface((w * CELL, h * CELL))
    if res.bg is not None:
        # 背景**整体缩放**（比例 = CELL/40，与所有素材一致）后从左上角铺一张：
        # 原图 600×520 = 15×13 个 40×40 格子，缩放成 900×780 = 15×13 个 60×60
        # 格子 —— 高度正好 = 地图 13 行，宽度多出的 2 列被画布裁掉。格子线因此
        # 与渲染格精确对齐（帮玩家判断位置）。**不平铺**（平铺会显示多于 13 列
        # 的格子、格子错位）；也不非等比拉伸（格子会变形）。
        bw, bh = res.bg.get_size()
        if (bw, bh) == (w * CELL, h * CELL):
            static.blit(res.bg, (0, 0))            # 已是匹配尺寸，直接贴
        else:
            sc = CELL / 40.0
            sb = pygame.transform.smoothscale(
                res.bg, (max(1, round(bw * sc)), max(1, round(bh * sc))))
            static.blit(sb, (0, 0))
    else:
        static.fill((45, 42, 50))
    return static


def _tile_sprites(sim, res: Res) -> list:
    """墙/砖 tile 精灵列表：[(z, x, y, surf), ...]，z = 所在行（Z 排序用）。

    底边对齐格底（tile 向上延伸超一格 → 会盖住上方格角色的脚部，正确）。
    """
    h, w = sim.cfg.height, sim.cfg.width
    sprites = []
    wall = sim.wall[0].cpu().numpy()
    brick = sim.brick[0].cpu().numpy() if hasattr(sim, "brick") else None
    for r in range(h):
        for c in range(w):
            if wall[r, c] and res.wall_tile is not None:
                tiles = res.wall_tile
                t = (tiles[(r * 7 + c * 13) % len(tiles)]
                     if isinstance(tiles, list) else tiles)
                tw, th = t.get_size()
                sprites.append((r, c * CELL + (CELL - tw) // 2,
                                r * CELL + CELL - th, t))
            elif brick is not None and brick[r, c] and res.brick_tile is not None:
                tiles = res.brick_tile
                t = (tiles[(r * 7 + c * 13) % len(tiles)]
                     if isinstance(tiles, list) else tiles)
                tw, th = t.get_size()
                sprites.append((r, c * CELL + (CELL - tw) // 2,
                                r * CELL + CELL - th, t))
    return sprites


_danger_overlay: pygame.Surface | None = None    # 危险区整层 overlay（模块级复用）

# 渲染缓存：危险区 / 泡泡 / 宝箱 / 墙·砖 tile 精灵只在 tick（10Hz）变化，渲染帧（60fps）直接
# 复用，不再每帧 cpu().numpy()+nonzero()+逐格循环（泡泡多、危险区大时正是
# 帧率波动源 —— "每帧重复算同一份"）。以 sim.t[0]（tick 号）为 key。
_rcache = {"tick": -1, "danger_any": False,
           "bomb_idx": None, "crate_idx": None, "tiles": []}


def draw_grid(screen, res: Res, sim, obs, rpos, face, anim_frame,
              explosion, static=None, status_surf=None, hud_surf=None) -> None:
    """RES 版渲染：静态层 → 危险/爆炸/泡泡/宝箱 → 角色（精灵+血条）→ HUD。

    rpos: (P,2) 浮点格坐标（已插值）。face: 每玩家最后移动方向（持久化）。
    anim_frame: 每玩家行走动画帧。explosion: 最近一次 (blast, trig) 或 None。
    static: 静态层缓存（build_static 产出，10Hz 重建）；为 None 时只画背景。
    status_surf/hud_surf: 预渲染文本 surface（避免每帧 font.render，掉帧源）。
    """
    cfg = sim.cfg
    h, w = cfg.height, cfg.width
    danger = obs[0, 2 * cfg.n_players + 1].float() if obs is not None else None
    pos = sim.pos[0].floor().long().clamp(0, h - 1)
    fuse = sim.fuse[0]
    owner = sim.owner[0]
    alive = sim.alive[0]
    hp = sim.hp[0]

    # 背景层（不含墙/砖 tile）：整图一次 blit
    if static is not None:
        screen.blit(static, (0, 0))
    elif res.bg is not None:
        screen.blit(res.bg, (0, 0))
    else:
        screen.fill((45, 42, 50))

    # tick 缓存：危险区 / 泡泡 / 宝箱 / 墙·砖行层（10Hz 重建，渲染帧复用）。
    # **完全复刻训练的危险图** —— obs 通道 2P+1 就是 danger_map 的同一份输出
    # （与训练每 tick 的 danger 惩罚同源），值 = 1-(fuse-1)/FUSE：刚放下引信
    # 最长（危险≈0.03 几乎无色），越接近爆炸越趋近 1.0 → 越红。
    tick = int(sim.t[0]) if sim.t is not None else -1
    if _rcache["tick"] != tick:
        _rcache["tick"] = tick
        _rcache["danger_any"] = False
        if danger is not None and bool((danger > 0.04).any()):
            global _danger_overlay
            if _danger_overlay is None or \
                    _danger_overlay.get_size() != (w * CELL, h * CELL):
                _danger_overlay = pygame.Surface((w * CELL, h * CELL),
                                                 pygame.SRCALPHA)
            _danger_overlay.fill((0, 0, 0, 0))
            arr = danger.cpu().numpy()
            ys, xs = arr.nonzero()
            for y, x, v in zip(ys.tolist(), xs.tolist(), arr[ys, xs].tolist()):
                a = min(255, int(20 + 235 * v))   # 危险值 0→1：红色强度线性爬升
                _danger_overlay.fill((255, 30, 60, a),
                                     (x * CELL, y * CELL, CELL, CELL))
            _rcache["danger_any"] = True
        live = (owner >= 0) & (fuse > 0)
        _rcache["bomb_idx"] = live.cpu().numpy().nonzero()
        _rcache["crate_idx"] = (sim.crate[0].cpu().numpy().nonzero()
                                if hasattr(sim, "crate") else None)
        _rcache["tiles"] = _tile_sprites(sim, res)

    # 危险区整层（地面标记，画在背景上、所有精灵底下）
    if danger is not None and _rcache["danger_any"]:
        screen.blit(_danger_overlay, (0, 0))

    # ---- 画家算法（Z = 所在行 Y）：所有精灵带 z，按 z 升序（远→近）逐个画 ----
    # 2D 游戏的遮挡几乎全由 Y 决定：近行（z 大）后画 → 自然盖住远行精灵伸过来
    # 的部分。墙/砖 tile 底边对齐格底、向上延伸超一格 —— 所以角色站在砖上方
    # 一格时，砖 tile（z=砖行）后画，正好盖住角色脚（"脚踩在砖上"的正确修法）。
    # 同 z 内顺序不影响正确性（同格精灵不互遮脚部），无需稳定排序。
    items: list[tuple[int, int, int, object]] = []   # (z, x, y, surf)

    # 墙/砖 tile（tick 缓存，10Hz）
    items.extend((z, x, y, s) for z, x, y, s in _rcache["tiles"])

    # 爆炸段：中心格用中心图；臂图按 blast 从"炸弹边缘端"切好，尊重挡火规则。
    if explosion is not None:
        blast, trig = explosion
        if trig is not None:
            trig_cells = trig.nonzero(as_tuple=True)
            for t in range(len(trig_cells[0])):
                r, c = int(trig_cells[0][t]), int(trig_cells[1][t])
                items.append((r, c * CELL, r * CELL, res.explo_center))
            for t in range(len(trig_cells[0])):
                sr, sc = int(trig_cells[0][t]), int(trig_cells[1][t])
                for (dr, dc), arm in res.explo_arms.items():
                    # 段索引按实际爆炸格数 n 算（素材按最大威力切片，实际短时
                    # 取内段，火焰头不丢）。
                    n = 0
                    for k in range(1, res.blast + 1):
                        r, c = sr + dr * k, sc + dc * k
                        if not (0 <= r < h and 0 <= c < w):
                            break
                        if not bool(blast[r, c]):
                            break
                        n += 1
                    for k in range(1, n + 1):
                        r, c = sr + dr * k, sc + dc * k
                        if dr == 0:
                            idx = (k - 1) if dc > 0 else (n - k)
                            seg = arm.subsurface((idx * CELL, 0, CELL, CELL))
                        else:
                            idx = (k - 1) if dr > 0 else (n - k)
                            seg = arm.subsurface((0, idx * CELL, CELL, CELL))
                        items.append((r, c * CELL, r * CELL, seg))

    # 泡泡：**底部贴格底线**（站在格子底线上，像放在地上）—— 圆形图高 60px，
    # 圆心若放格底会整颗压进下一格（"严重偏下"），底部对齐则泡中心 = 格中线、
    # 视觉正好在格内。
    now = pygame.time.get_ticks() / 1000.0
    bob = int(round(math.sin(now * 2 * math.pi) * 3))
    bomb_surf = res.bombs[0]
    bw, bh = bomb_surf.get_size()
    for r, c in zip(*_rcache["bomb_idx"]):
        r, c = int(r), int(c)
        cx = c * CELL + CELL / 2.0
        bx = int(cx - bw / 2)
        by = (r + 1) * CELL - bh + bob              # 底边 = 格底线
        items.append((r, bx, by, bomb_surf))

    # 宝箱（砖被炸掉后变）：三张道具图轮流展示，上下浮动；底部同样贴格底线。
    if hasattr(sim, "crate") and res.props and _rcache["crate_idx"]:
        prop_w, prop_h = res.props[0].get_size()
        prop_idx = int(now * 2) % len(res.props)
        prop_surf = res.props[prop_idx]
        for r, c in zip(*_rcache["crate_idx"]):
            r, c = int(r), int(c)
            cx = c * CELL + CELL / 2.0
            px = int(cx - prop_w / 2)
            py = (r + 1) * CELL - prop_h + bob       # 底边 = 格底线
            items.append((r, px, py, prop_surf))

    # 角色：z = 脚所在行 = int(插值中心 y)（帧底边 = 中心格底边）。
    chars = []               # (z, bx, by, surf, wudi, wx, wy, hpv, mx)
    for pid, is_ai in ((0, False), (1, True)):
        if not alive[pid]:
            continue
        sprite_rows = res.player_ai if is_ai else res.players
        gy, gx = float(rpos[pid, 0]), float(rpos[pid, 1])
        cx, cy = gx * CELL, gy * CELL
        row = MOVE_TO_SPRITE_ROW.get(face[pid], 0)
        s = sprite_rows[row][anim_frame[pid]]
        sw, sh = s.get_size()
        blit_x = int(cx - sw / 2)
        # 锚点模型：切片底部贴格底（锚点 = 格底正中心；左/上/右可超出）。
        # 4×4 行走图 340px = 4×85 精确切分，脚底就在切片切线底部，无需偏移。
        blit_y = int(cy + CELL / 2 - sh)
        # 底线约束：帧底**永不越过地图底部** —— 角色贴底墙时中心格底
        # (gy+0.5)*CELL 会超过画布底（腿插入地面被裁，观感"腿超出底线"），
        # 钳到地图底 h*CELL，脚正好踩在最下行格底。
        blit_y = min(blit_y, h * CELL - sh)
        wudi = None
        wx, wy = blit_x, blit_y
        if getattr(sim, "invuln", None) is not None and \
                int(sim.invuln[0, pid]) > 0 and res.wudi_scaled is not None:
            wudi = res.wudi_scaled
            # 无敌光晕居中于**人物视觉主体质心**（不是帧左上角）：
            # 素材人物帧 85×85 里脚部是透明像素（脚底 = 帧底边），视觉主体悬在
            # 帧上部 —— 光晕若贴帧左上角会"罩在人物脚下"。质心画布坐标 =
            # (blit_x + body_center)，光晕中心对准它。fallback 无质心 → 旧行为
            # 叠帧左上角。
            if res.body_centers is not None:
                bcx, bcy = res.body_centers[row][anim_frame[pid]]
                ww, wh = wudi.get_size()
                wx = int(blit_x + bcx - ww / 2)
                wy = int(blit_y + bcy - wh / 2)
        chars.append((int(gy), blit_x, blit_y, s, wudi, wx, wy,
                      int(hp[pid]), cfg.max_hp))
        items.append((int(gy), blit_x, blit_y, s))

    items.sort(key=lambda it: it[0])
    for _, x, y, s in items:
        screen.blit(s, (x, y))

    # 无敌罩（加法混合，叠在角色上；UI 层最后画）—— 位置 wx/wy 已按人物
    # 视觉主体质心算好（罩住身体，不是贴脚底/帧左上角）
    for _, _, _, _, wudi, wx, wy, _, _ in chars:
        if wudi is not None:
            screen.blit(wudi, (wx, wy), special_flags=pygame.BLEND_ADD)

    # 血条统一最后画（UI 信息不被墙挡）
    # 水平对齐：右移一格宽（+CELL）后回移 12px —— 视觉主体偏右半格多，
    # 纯右移一格子偏过头（与 web/main.js 同一偏移 +48px）。
    for _, bx, by, _, _, _, _, hpv, mx in chars:
        seg_w, seg_h, gap = 5, 4, 1
        bar_w = mx * (seg_w + gap)
        color = (80, 220, 90) if hpv > mx / 3 else (240, 70, 70)
        for i in range(mx):
            rect = pygame.Rect(bx + CELL - 12 + i * (seg_w + gap),
                               by - 8, seg_w, seg_h)
            pygame.draw.rect(screen, color if i < hpv else (60, 60, 66), rect)

    # 成长 HUD（corridor 用）：右上角显示 泡数/威力/速度 当前值。
    # 预渲染缓存由主循环维护（值变化才重渲），这里直接 blit，避免每帧 SysFont。
    if hud_surf is not None:
        screen.blit(hud_surf, (w * CELL - hud_surf.get_width() - 8, 8))


def run_game(*, ckpt: str = "ckpt/duel_rw_ckpt.pt", ckpt_b: str | None = None,
             device: str = "cpu",
             size: int = GRID, map_mode: str = "open", scene: str = "比武",
             hz: int = 10, player_speed_mult: float = 1.0,
             open_growth_pct: float = 0.8, seed: int = 0, auto_ticks: int = 0,
             screenshot: str | None = None, bot_mode: bool = False,
             bombs_start: int | None = None, blast_start: int | None = None,
             speed_start: float | None = None,
             opp_bot: str | None = None, p0_bot: str | None = None,
             die_log: bool = False,
             recording: bool = False,
             human0: bool = True, human1: bool = False) -> None:
    """开一局对打。bot_mode=True 时玩家 0 也走 AI（机器人对机器人观战）；
    ckpt_b 给观战模式的 P1 用独立权重（None = P0/P1 同一个 ckpt）。
    obs_extra_enabled 跟随两档中更宽的通道数；窄档自动适配到同布局。

    `opp_bot` / `p0_bot`：把 AI 换成**规则 bot**（sim/bots.py 的
    random/greedy/astar），不读 checkpoint —— 启动器里可以直接选
    「寻路 AI」astar 跟模型对打/观战。选了 bot 时对应 ckpt 可以留空。

    返回 "menu"（对局内按 ESC，回启动器；pygame 保持存活）或 "quit"
    （Q / 关窗退出，已 pygame.quit）。
    除 bot_mode / 初始属性覆盖外，参数语义与 CLI 一致，launcher 直接复用。
    bombs_start/blast_start/speed_start 只对 corridor/ring 关生效（open 关
    的成长初始固定由 open_growth_pct 决定，与训练一致）。
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # P1 未指定（观战）：默认跟随 P0 的模型（launcher 的 P1 下拉跟随 P0 同款，
    # CLI 直开 --bot-mode 只给 --ckpt 时也成立）。
    if bot_mode and opp_bot is None and not ckpt_b and ckpt:
        ckpt_b = ckpt

    # 校验：每个非人类玩家位必须有对手（模型或规则 bot）。
    # human0/human1 表示该位是人类键盘玩家（launcher 双下拉合并后）。
    if not human0 and p0_bot is None and not ckpt:
        raise ValueError("玩家 0 既没给 ckpt 也没给 p0_bot（或选人类玩家）")
    if not human1 and opp_bot is None and not ckpt_b and not bot_mode:
        raise ValueError("玩家 1 既没给 ckpt_b 也没给 opp_bot（或选人类玩家）")

    extra_on = (bool(ckpt) and ckpt_channels(ckpt) > 7
                or (bool(ckpt_b) and ckpt_channels(ckpt_b) > 7)
                or recording)   # 录制时强制 14 通道扩展观测（与训练同构，
                                # P1 人类位 swap 不越界，且 BC 数据格式统一）
    if map_mode == "corridor":
        # corridor 关：初始 3.0 格/秒、泡数/威力 2，踩宝箱成长到上限
        # （7/7/速度倍率 2.1 = 6.3 格/秒）；3 分钟对局。
        # obs_extra_enabled 跟随 ckpt 通道数：7 通道旧档关扩展观测，14 通道新档开。
        cfg = SimConfig(height=size, width=size, n_players=2,
                        map_mode="corridor", speed=3.0, max_steps=1800,
                        growth_bombs_start=bombs_start or 2,
                        growth_blast_start=blast_start or 2,
                        growth_speed_start=speed_start or 1.0,
                        obs_extra_enabled=extra_on)
    elif map_mode == "ring":
        # 环岛：中间 7×7 永久墙山体 + 环带稀疏可炸墙 + 四角出生 + 宝箱 100% 成长
        cfg = SimConfig(height=size, width=size, n_players=2,
                        map_mode="corridor", speed=3.0, max_steps=1800,
                        ring_fraction=1.0,
                        growth_bombs_start=bombs_start or 2,
                        growth_blast_start=blast_start or 2,
                        growth_speed_start=speed_start or 1.0,
                        obs_extra_enabled=extra_on)
    else:
        # open 空场：**1800 tick（3 分钟）**，和训练一致 —— 之前默认 600 tick
        # 会在 60 秒超时自动重开，试玩看起来像"瞬移换地图"。
        # 用 map_mode=corridor + open_fraction=1.0 走混合地图的 open 子类：
        # 拿到与训练一致的成长初始（默认 80% 上限）与中线出生点 —— 直接
        # map_mode 缺省（= "open"）会走"纯 open 固定能力无成长"分支，能力
        # 只有 3/3/1.0，和训练里的 open 关（6/6/2.4）差太多，试玩体感不对。
        pct = min(1.0, max(0.0, open_growth_pct))
        cfg = SimConfig(height=size, width=size, n_players=2,
                        map_mode="corridor", max_steps=1800,
                        open_fraction=1.0,
                        open_growth_bombs=math.ceil(
                            SimConfig.growth_bombs_max * pct),
                        open_growth_blast=math.ceil(
                            SimConfig.growth_blast_max * pct),
                        open_growth_speed=round(
                            SimConfig.growth_speed_max * pct, 2),
                        obs_extra_enabled=extra_on)
    sim = BatchedSim(cfg, 1, device=device, seed=seed)
    # 玩家侧速度倍率（只影响玩家 0；AI 保持 3.0）。训练不设这个属性，
    # 行为与旧版逐位一致 —— 对打窗口里玩家跑得快一点，纯粹手感实验。
    mult = torch.ones(1, 2, device=device)
    mult[0, 0] = player_speed_mult
    sim.speed_mult = mult
    net = load_ai(ckpt, device) if ckpt else None

    def _prep(m):
        """把网络适配到当前环境布局（混合 7/14 通道对战时窄档补零），并置 eval。"""
        if tuple(m.obs_shape) != tuple(cfg.obs_shape):
            m = adapt_first_conv(m, tuple(cfg.obs_shape), arch=m.arch,
                                 n_players=2).to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    if net is not None:
        net = _prep(net)
    net_b = _prep(load_ai(ckpt_b, device)) if (ckpt_b and not human1) else None

    # 规则 bot 对手：绑定本局 sim（sim.reset_all 后仍有效，act 每次读最新状态）
    bot1 = make_bot(sim, opp_bot) if (opp_bot and not human1) else None
    bot0 = make_bot(sim, p0_bot) if (p0_bot and not human0) else None

    # LSTM 时序状态（按 pid 分开放，跨 tick 传递；对局结束 sim.step 返回
    # done 时清零，见主循环终局处理）。
    hidden_st: dict[int, object] = {0: None, 1: None}

    def _ai(pid: int, net_ref, bot_ref) -> torch.Tensor:
        """该玩家的决策：规则 bot 优先，否则网络（pid 决定视角）。

        **网络模型一律用 pid=0 视角**（训练时 learner 恒为 player 0，只有
        pid=0 视角被优化）。物理 P1 的模型靠**观测重排**（_swap_player_channels
        把自己搬到通道 0）+ pid=0 —— 即"玩家0、玩家1 都认为自己是玩家0"，
        对打多个 ckpt 时每个模型都用自己的训练视角，公平且行为正常。
        旧版 P1 模型用 pid=1，per-player extra 通道不置换 → 把对手的泡数
        当成自己的 → 行为乱。规则 bot 读 sim 原始状态，不走网络视角，任意位。

        **LSTM 例外**：局部特征（local 7×7 窗口 + rel/glob）以自己为中心生成，
        自带视角 —— 直接用本角色的特征三元组喂模型（模型恒 pid=0 视角 =
        "自己"，等价于 CNN 的重排），不需要 _swap_player_channels。
        """
        if bot_ref is not None:
            return bot_ref.act(obs, mm[:, pid], bm[:, pid], pid)
        if getattr(net_ref, "arch", "cnn") == "lstm":
            lf = local_view_features(sim.cfg, obs, sim.pos, sim.alive,
                                     sim.t, sim.fuse, sim.hp, only_p0=False)
            # lf 返回 (N,P,C,7,7)/(N,P,MAX_T,6)/(N,P,5)：按 pid 取本角色（去 P 维）
            feats = (lf[0][:, pid], lf[1][:, pid], lf[2][:, pid])
            with torch.no_grad():
                a, _, _, h = net_ref.act(feats, mm[:, pid], bm[:, pid], 0,
                                         hidden_st[pid])
            hidden_st[pid] = h
            return a
        if pid == 1 and net_ref is not None:
            # P1 网络模型：观测重排（自己→通道0）+ pid=0 视角；掩码用**物理 P1**
            # 的（mm[:,1] 已切好，物理位置决定可行动作）。不能走 ai_action ——
            # 它内部会按 pid 再切一次掩码（双重切片 → 形状错位 → 动作全乱）。
            with torch.no_grad():
                a, _, _ = net_ref.act(_swap_player_channels(obs),
                                      mm[:, 1], bm[:, 1], 0)
            return a
        return ai_action(net_ref, obs, mm, bm, device, pid)

    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        pass
    screen = pygame.display.set_mode((cfg.width * CELL, cfg.height * CELL + 48))
    n0 = ("人类" if human0 else
          (f"【{p0_bot}】" if p0_bot else os.path.basename(ckpt or "?")))
    n1 = ("人类" if human1 else
          (f"【{opp_bot}】" if opp_bot else os.path.basename(ckpt_b or ckpt)))
    title = f"QQT 对打：{n0} vs {n1}"
    pygame.display.set_caption(title)
    font = _load_cjk_font(24)
    small = _load_cjk_font(16)
    # 爆炸臂按**最大可能威力**切片（corridor 成长到 7 格；open 恒为 cfg.blast），
    # 渲染时以 blast 覆盖掩码为准 —— 实际爆多长画多长，晚档大威力也正确。
    res_blast = (max(cfg.blast, cfg.growth_blast_max)
                 if cfg.map_mode == "corridor" else cfg.blast)
    # 场景（默认比武）：砖块图 + BGM。缺省时砖块用色块兜底、BGM 静音
    res = Res(cell=CELL, blast=res_blast, scene=scene)
    if res.bgm_path and os.path.exists(os.path.join("res", "..", res.bgm_path)):
        try:
            pygame.mixer.music.load(res.bgm_path)
            pygame.mixer.music.play(-1)      # 循环播放场景 BGM
        except pygame.error:
            pass

    rng = random.Random(seed)
    TICK = 1.0 / hz

    # 人类轨迹录制：默认开（有人类玩家就录，L 键局内开关）→ BC 训练数据。
    # 一局一个 npz，重开/ESC/退出时落盘（见 finish_episode）。
    rec = Recorder() if (recording and (human0 or human1)) else None
    rec_on = rec is not None

    def finish_episode() -> None:
        if rec is None:
            return
        p = rec.finish()
        if p:
            print(f"[rec] 已录 {rec.ticks if False else ''}{p} "
                  f"(pid={rec._meta.get('pid')})")

    def new_round() -> None:
        sim.reset_all()
        if rec is not None:
            # 上一局落盘 + 新局开始（保持 rec_on 状态）
            if rec_on:
                finish_episode()
            pid = 1 if human1 and not human0 else 0
            rec.begin({
                "map": map_mode, "scene": scene, "pid": pid,
                "opp": "human" if (human0 and human1) else
                       ("human1" if human0 else
                        (f"bot:{opp_bot}" if opp_bot else
                         os.path.basename(ckpt_b or ckpt or "?"))),
                "ckpt": os.path.basename(ckpt or "?"),
                "seed": seed, "hz": hz, "bombs": int(sim.bombs_cap[0, pid]),
                "blast": int(sim.blast_cap[0, pid]),
                "speed": float(sim.spd_g[0, pid]),
                "max_steps": sim.cfg.max_steps,
            })

    new_round()
    # 静态层（背景+墙/砖）缓存：只在逻辑 tick / 换局后重建，渲染帧直接 blit
    static = build_static(res, sim)
    done = False
    result_msg = ""
    # 朝向持久化：face = 最后移动的方向（松手后保留，不再回落成"下"），
    # anim_frame = 行走动画帧；移动中推帧，静止停帧 0 但朝 face。
    face = {0: MOVE_DOWN, 1: MOVE_DOWN}
    anim_frame = {0: 0, 1: 0}
    # 状态栏 / 成长 HUD 文本缓存：内容只在 tick 边界变化，60Hz 渲染直接 blit，
    # 不再每帧 font.render（CJK 文本渲染 ~0.3-0.5ms/帧，后期掉帧源之一）。
    status_key, status_surf = None, None
    hud_key, hud_surf = None, None
    if sim.cfg.map_mode == "corridor":
        hud_font = _load_cjk_font(22)
        hud_font.set_bold(True)

    if screenshot:
        for _ in range(60):
            obs = sim.observe()
            mm, bm = sim.legal_mask()
            a0 = torch.tensor([[rng.randrange(5), rng.randrange(2)]], dtype=torch.long)
            a1 = _ai(1, net_b if net_b is not None else net, bot1)
            sim.step(torch.stack([a0.to(device), a1], dim=1), auto_reset=False)
        obs = sim.observe()
        screen.fill((20, 20, 24))
        draw_grid(screen, res, sim, obs, sim.pos[0].float(), face, anim_frame,
                  None, static=build_static(res, sim))
        pygame.image.save(screen, screenshot)
        print(f"[screenshot] saved {screenshot}")
        pygame.quit()
        return

    # ---------------- 主循环：渲染 60fps，逻辑固定 hz，位置插值 ----------------
    clock = pygame.time.Clock()
    # "最后按下的方向优先"：按住↑时轻点← 应切到←。用有序栈实现（末尾 = 最近按下），
    # 而不是按字典序选 —— 否则 ↑↓ 永远压过 ←→，轻点切换根本切不过去。
    dir_stack: list[int] = []            # P0 动作值 0..3，最近按下的在末尾
    dir_stack1: list[int] = []           # P1（WASD）
    # 方向锁存：KEYDOWN 只触发一次（pygame 是边缘触发），而方向 60Hz 采样 ——
    # 锁存 = 事件置位、采样消费：只要 KEYDOWN 被处理过，下一次采样必然读到这个方向。
    latch: set[int] = set()
    latch1: set[int] = set()
    player_move = MOVE_IDLE              # 每渲染帧（60Hz）刷新的 P0 方向
    player_move1 = MOVE_IDLE             # P1 方向
    ai_move = MOVE_IDLE                  # AI 最近一次 tick 的移动方向（动画用）
    ai_move0 = MOVE_IDLE                 # 观战模式玩家 0 的移动方向（动画用）
    pending_bomb = False
    pending_bomb1 = False
    accumulator = 0.0
    prev_ms = pygame.time.get_ticks()
    prev_pos = sim.pos[0].float().clone()
    cur_pos = sim.pos[0].float().clone()
    obs = sim.observe()
    explosion = None            # 最近一次 (blast, trig)
    explosion_t = 0.0           # 爆炸发生时刻，0.6s 后淡出
    last_boom = 0.0
    last_place = 0.0

    # 音效"以人类玩家为监听者"：所有事件只对人类播、位置相对人类做左右声道
    # pan + 距离音量；纯 AI 的行为（双方都非人类）完全静音。
    human_pids = [p for p, h in ((0, human0), (1, human1)) if h]

    def _play_at(snd, br: float, bc: float) -> None:
        """在网格坐标 (br, bc) 发声：相对最近人类玩家 pan（左右声道）+ 音量。"""
        if snd is None or not human_pids:
            return
        pid = human_pids[0]                     # 监听者 = 第一个人类玩家
        hx = float(sim.pos[0, pid, 1]) * CELL + CELL / 2
        hy = float(sim.pos[0, pid, 0]) * CELL + CELL / 2
        x = bc * CELL + CELL / 2
        y = br * CELL + CELL / 2
        dx, dy = x - hx, y - hy
        dist = math.hypot(dx, dy)
        maxd = math.hypot(GRID * CELL, GRID * CELL)
        vol = max(0.08, 1.0 - 0.9 * min(1.0, dist / maxd))   # 越近越响
        pan = max(-1.0, min(1.0, dx / (GRID * CELL / 2)))    # -1 偏左 / +1 偏右
        ch = snd.play()
        if ch is not None:
            ch.set_volume(vol * (1.0 - 0.8 * max(0.0, pan)),
                          vol * (1.0 - 0.8 * max(0.0, -pan)))

    while True:
        dt_s = (pygame.time.get_ticks() - prev_ms) / 1000.0
        prev_ms = pygame.time.get_ticks()
        accumulator += min(dt_s, 0.25)

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                finish_episode()
                pygame.quit()
                return "quit"
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                finish_episode()
                pygame.quit()
                return "quit"
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                # 结束当前对局：run_game 返回 "menu"（CLI 直开由入口收尾退出）。
                finish_episode()
                return "menu"
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                new_round()
                static = build_static(res, sim)
                done = False
                result_msg = ""
                explosion = None
            if e.type == pygame.KEYDOWN and e.key in DIR_KEYS:
                mv = DIR_KEYS[e.key]
                latch.add(mv)                    # 锁存：这个 tick 必走一格
                if mv not in dir_stack:
                    dir_stack.append(mv)
            elif e.type == pygame.KEYUP and e.key in DIR_KEYS:
                mv = DIR_KEYS[e.key]
                if mv in dir_stack:
                    dir_stack.remove(mv)
            # 放泡：空格 = P0（人类时）；P0 是 AI 而 P1 是人类（单人玩 P1 位）→
            # 空格给 P1 放炮。双人（P0/P1 都人类）时空格=P0、回车=P1，互不干扰。
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                if human0:
                    pending_bomb = True
                elif human1:
                    pending_bomb1 = True
            # P1（第二人类玩家）：方向键移动 + 回车放泡（P0 已让出方向键）
            if e.type == pygame.KEYDOWN and e.key in DIR_KEYS1:
                mv = DIR_KEYS1[e.key]
                latch1.add(mv)
                if mv not in dir_stack1:
                    dir_stack1.append(mv)
            elif e.type == pygame.KEYUP and e.key in DIR_KEYS1:
                mv = DIR_KEYS1[e.key]
                if mv in dir_stack1:
                    dir_stack1.remove(mv)
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN,
                                                      pygame.K_KP_ENTER):
                pending_bomb1 = True
            # 录制开关（L 键）：随时切，切掉后本局不落盘
            if e.type == pygame.KEYDOWN and e.key == pygame.K_l and rec is not None:
                rec_on = not rec_on
                print(f"[rec] 录制{'开' if rec_on else '关'}"
                      f"（L 键切换；{'当前局继续' if rec_on else '本局不落盘'}）")

        # 玩家输入 60Hz 采样（每渲染帧）：方向 = dir_stack 最新（事件+物理合并）。
        # AI 决策仍 10Hz（tick 循环），但**玩家方向每个渲染帧刷新** —— 输入采样
        # 60Hz、输出 60fps，手速不丢。latch 保证轻按（KEYDOWN 被 60Hz 采样到）
        # 也会进 dir_stack，不会漏。
        phys = pygame.key.get_pressed()
        active = set(latch)
        for mv in dir_stack:
            if phys[MV_TO_KEY[mv]]:
                active.add(mv)
        dir_stack = [mv for mv in dir_stack if mv in active]   # 清理已松开
        for mv in active:
            if mv not in dir_stack:
                dir_stack.append(mv)                           # 物理按住但事件没记到
        latch.clear()
        active1 = set(latch1)
        for mv in dir_stack1:
            if phys[MV_TO_KEY1[mv]]:
                active1.add(mv)
        dir_stack1 = [mv for mv in dir_stack1 if mv in active1]
        for mv in active1:
            if mv not in dir_stack1:
                dir_stack1.append(mv)
        latch1.clear()
        if human0:
            player_move = dir_stack[-1] if dir_stack else MOVE_IDLE
            # 玩家移动：**帧级实时**（60Hz 按键采样 → 当帧移动），不走 10Hz tick。
            # 轻点（按 1 帧 ≈ 16ms）走 ~0.06 格、按住每帧都走 —— 微操距离与按键
            # 时长真正成正比。只动玩家 0 的坐标，AI/泡泡/爆炸仍按 10Hz tick 走。
            if player_move != MOVE_IDLE:
                eff = _auto_turn(sim, player_move)     # 逐帧自动转向（贴墙/泡拐弯）
                _player_frame_move(sim, eff, min(dt_s, TICK))
                player_move = eff
        if human1:
            player_move1 = dir_stack1[-1] if dir_stack1 else MOVE_IDLE
            # P1 帧级移动：只动玩家 1 的坐标（与 P0 同款微操手感）
            if player_move1 != MOVE_IDLE:
                eff1 = _auto_turn1(sim, player_move1)
                _player_frame_move1(sim, eff1, min(dt_s, TICK))
                player_move1 = eff1

        # 每渲染帧最多跑 1 个逻辑 tick，剩余时间累积到下一帧 —— 渲染恒稳、
        # 输入在每个 tick 边界即时生效，不会因多 tick 积压在一帧里造成卡顿停顿。
        # （10Hz 下每 6 渲染帧才轮到一个 tick，一帧跑 1 个完全够；帧率低于
        #   10fps 的极端情况会自动降逻辑节奏，不会把操作憋出"半天不响应"。）
        if accumulator >= TICK:
            accumulator -= TICK
            obs = sim.observe()
            mm, bm = sim.legal_mask()
            # a0：P0 人类 → IDLE+放泡（帧级移动已完成）；否则 AI/规则 bot
            if human0:
                a0 = torch.tensor([[MOVE_IDLE, 1 if pending_bomb else 0]],
                                  dtype=torch.long)
                if pending_bomb:
                    _play_at(res.snd_place,
                             float(sim.pos[0, 0, 0]), float(sim.pos[0, 0, 1]))
                pending_bomb = False
            elif net is not None or bot0 is not None:
                a0 = _ai(0, net, bot0)
            else:
                a0 = torch.tensor([[MOVE_IDLE, 0]], dtype=torch.long, device=device)
            # a1：P1 人类 → IDLE+放泡（方向键+回车 帧级移动已完成）；否则 AI/规则 bot
            if human1:
                a1 = torch.tensor([[MOVE_IDLE, 1 if pending_bomb1 else 0]],
                                  dtype=torch.long)
                if pending_bomb1:
                    _play_at(res.snd_place,
                             float(sim.pos[0, 1, 0]), float(sim.pos[0, 1, 1]))
                pending_bomb1 = False
            else:
                a1 = _ai(1, net_b if net_b is not None else net, bot1)
            prev_pos = cur_pos.clone()
            hp0 = sim.hp[0].clone()                       # step 前血量（掉血/死亡判断）
            crate0 = sim.crate[0].clone()                 # step 前宝箱（吃到判断）
            owner_snap = sim.owner.clone()                # step 前泡归属（爆炸源判定）
            rew, done_any, info = sim.step(torch.stack([a0.to(device), a1], dim=1),
                                           auto_reset=False)
            # 轨迹录制：人类玩家视角（pid 1 时 obs 已 swap 成 player0）+ **实际执行**
            # 动作（帧级移动聚合后的方向 = player_move/player_move1，含自动拐弯；
            # 放泡 = a0/a1 的 bomb 位）+ 人类该 tick 奖励（step 后才有）。
            # 对局结束（done 已置）后停录 —— 超时/磨平局不会无限累积拖垮渲染。
            if rec is not None and rec_on and not done:
                if human0:
                    r_obs = obs                    # P0 = 物理 player0，直接共享 obs
                    r_act = torch.tensor([[player_move if player_move != MOVE_IDLE
                                           else MOVE_IDLE,
                                           1 if a0[0, 1] else 0]], dtype=torch.long)
                else:
                    r_obs = _swap_player_channels(obs)   # P1 模型视角
                    r_act = torch.tensor([[player_move1 if player_move1 != MOVE_IDLE
                                           else MOVE_IDLE,
                                           1 if a1[0, 1] else 0]], dtype=torch.long)
                rew_h = rew[0, 0] if human0 else rew[0, 1]
                rec.add(r_obs.to(device), r_act, rew_h, done_any[0])
            cur_pos = sim.pos[0].float().clone()
            # 死亡诊断（--die-log）：AI/玩家死亡时刻打印精确死因。
            # **爆炸源 owner 判定**：step 返回时爆炸已清场（owner→-1），"死时在场
            # 泡数"会误读成 0 —— 必须用 step 前的 owner 快照 + info["trig"]（本
            # tick 引爆的泡）查出"引爆泡是谁的"，才能判断是不是自杀。
            if die_log and bool(info["died"][0].any()):
                t_now = int(sim.t[0])
                trig_owner = owner_snap[0][info["trig"][0]].tolist()
                for pid_, tag_ in ((0, "玩家0"), (1, "AI")):
                    if bool(info["died"][0, pid_]):
                        r_ = int(sim.pos[0, pid_, 0]); c_ = int(sim.pos[0, pid_, 1])
                        print(f"[die] tick={t_now} {tag_}死于 ({r_},{c_}) "
                              f"hp={int(sim.hp[0, pid_])} "
                              f"引爆泡owner={trig_owner} "
                              f"({'自杀' if pid_ in trig_owner else '他杀'})")
                print(f"[die] 玩家0 hp={int(sim.hp[0, 0])} pos={sim.pos[0, 0].tolist()} "
                      f"AI hp={int(sim.hp[0, 1])} pos={sim.pos[0, 1].tolist()} "
                      f"双方在场泡数="
                      f"{int(((sim.owner[0] == 0) & (sim.fuse[0] > 0)).sum())}/"
                      f"{int(((sim.owner[0] == 1) & (sim.fuse[0] > 0)).sum())}")
            # 静态层重建：墙/砖只在爆炸后变化，10Hz 重渲一次足够（60fps 渲染
            # 直接 blit 缓存 —— 之前每渲染帧跑 169 格循环是后期掉帧主源）
            static = build_static(res, sim)
            # 音效（按本 tick 事件触发，全部以人类玩家为监听者）：
            #   爆炸 → 事件中心取本 tick 火焰覆盖的质心，相对人类 pan+音量；
            #   放泡/吃道具/掉血/死亡 → 只发生在人类玩家身上才播（AI 的
            #   行为不出声），位置即人类所在格。
            if info["blast"].any():
                explosion = (info["blast"][0].bool(), info["trig"][0].bool())
                explosion_t = pygame.time.get_ticks() / 1000.0
                if pygame.time.get_ticks() / 1000.0 - last_boom > 0.35:
                    bmask = info["blast"][0].bool()
                    ys, xs = bmask.nonzero(as_tuple=True)
                    br = float(ys.float().mean().item()) if ys.numel() else 0.0
                    bc = float(xs.float().mean().item()) if xs.numel() else 0.0
                    _play_at(res.snd_boom, br, bc)
                    last_boom = pygame.time.get_ticks() / 1000.0
            # 吃道具音效：**只有人类玩家吃到**才播（AI 吃的不播）。
            # 判定 = 人类移动后的中心格：step 前是宝箱、step 后不是。
            # （不能用 crate0[0] —— 那是宝箱图第 0 行，不是"玩家 0 的宝箱"。）
            if res.snd_pickup:
                for pid in human_pids:
                    p_cell = (int(sim.pos[0, pid, 0]), int(sim.pos[0, pid, 1]))
                    if bool(crate0[p_cell]) and not bool(
                            sim.crate[0, p_cell[0], p_cell[1]]):
                        _play_at(res.snd_pickup,
                                 float(sim.pos[0, pid, 0]),
                                 float(sim.pos[0, pid, 1]))
                        break
            # 掉血音效：**只有人类玩家**掉血才播，位置 = 该人类所在格
            if res.snd_hurt:
                for pid in human_pids:
                    if int(hp0[pid]) > int(sim.hp[0, pid]):
                        _play_at(res.snd_hurt,
                                 float(sim.pos[0, pid, 0]),
                                 float(sim.pos[0, pid, 1]))
                        break
            # 死亡：**只有人类玩家**死亡才播，位置 = 该人类所在格
            if res.snd_die:
                for pid in human_pids:
                    if bool(info["died"][0, pid]):
                        _play_at(res.snd_die,
                                 float(sim.pos[0, pid, 0]),
                                 float(sim.pos[0, pid, 1]))
                        break
            # 行走动画帧移到 60Hz 渲染段（不跟 tick），这里记录双方实际移动方向
            # （观战模式玩家 0 也走 AI —— 之前 a0 不更新，玩家 0 永远是"一张
            # 图平移"、朝向也不转）。
            ai_move = int(a1[0, 0])
            ai_move0 = int(a0[0, 0])   # P0 非人类时记录动作方向（动画/朝向用）
            if bool(done_any[0]) and not done:
                done = True
                r0 = float(rew[0, 0])
                result_msg = "你赢了!" if r0 > 0.5 else ("你输了" if r0 < -0.5 else "平局")
                # 对局结束：LSTM 时序记忆清零（下局从零开始）
                hidden_st[0] = None
                hidden_st[1] = None
            if auto_ticks > 0:
                auto_ticks -= 1
                if auto_ticks == 0:
                    finish_episode()
                    print("[smoke] 冒烟完成：sim + AI 对打路径正常")
                    pygame.quit()
                    return "quit"
        # 爆炸 0.6s 后淡出
        if explosion is not None and pygame.time.get_ticks() / 1000.0 - explosion_t > 0.6:
            explosion = None

        alpha = min(1.0, accumulator / TICK)
        rpos = prev_pos + (cur_pos - prev_pos) * alpha
        # 人类玩家不插值：位置已由 60Hz 帧级移动实时写入 sim.pos，直接渲染 ——
        # 轻点/微操的每帧位移都如实显示。AI 仍走 tick 插值平滑（10Hz 步进）。
        if human0:
            rpos[0] = sim.pos[0, 0]
        if human1:
            rpos[1] = sim.pos[0, 1]

        # 动画 60Hz 时间驱动：朝向与走路帧**不跟 tick**（否则要等下一个 100ms
        # 才换朝向/帧，正是"动画有等待"的观感来源）。人类玩家用帧级实际执行
        # 方向（含自动拐弯）；AI 用 tick 动作方向。
        now = pygame.time.get_ticks() / 1000.0
        if human0:
            if player_move != MOVE_IDLE:
                face[0] = player_move
                anim_frame[0] = int(now * WALK_HZ) % 4    # 走路帧按真实时间
            else:
                anim_frame[0] = 0                         # 静止停第 0 帧，朝向保留
        else:
            if ai_move0 != MOVE_IDLE:
                face[0] = ai_move0
            anim_frame[0] = int(now * WALK_HZ) % 4 if ai_move0 != MOVE_IDLE else 0
        if human1:
            if player_move1 != MOVE_IDLE:
                face[1] = player_move1
                anim_frame[1] = int(now * WALK_HZ) % 4
            else:
                anim_frame[1] = 0
        else:
            if ai_move != MOVE_IDLE:
                face[1] = ai_move
            anim_frame[1] = int(now * WALK_HZ) % 4 if ai_move != MOVE_IDLE else 0

        screen.fill((20, 20, 24))
        draw_grid(screen, res, sim, obs, rpos, face, anim_frame, explosion,
                  static=static, status_surf=status_surf, hud_surf=hud_surf)
        glyphs = "".join(DIR_GLYPH[mv] for mv in dir_stack)
        glyphs1 = "".join(DIR_GLYPH1[mv] for mv in dir_stack1)
        input_txt = (f"P:{glyphs or '—'}  P1:{glyphs1 or '—'}"
                     if (human0 and human1)
                     else (f"P1:{glyphs1 or '—'}" if human1
                           else f"P:{glyphs or '—'}"))
        # 倒计时：从 max_steps 倒着显示剩余秒数；到 0 自动开新局。
        # （超时那 tick sim 已置 done=True，所以这里**只看 sec_left**，
        #   不看 done —— 否则 done 挡住永远不重开。）
        sec_left = max(0, int(sim.cfg.max_steps - sim.t[0]) // sim.cfg.tick_hz)
        if sec_left <= 0:
            new_round()
            static = build_static(res, sim)
            done = False
            result_msg = ""
            explosion = None
            sec_left = int(sim.cfg.max_steps) // sim.cfg.tick_hz
        growth_txt = ""
        if sim.cfg.map_mode == "corridor":
            nb = int(sim.bombs_cap[0, 0]); nblast = int(sim.blast_cap[0, 0])
            spd = float(sim.spd_g[0, 0])
            bricks_left = int(sim.brick.sum())
            growth_txt = (f"  P0:泡×{nb} 威×{nblast} 速×{spd:.2f}"
                          f"  余砖{bricks_left}")
        status = (f"⏱{sec_left:>3}s  {input_txt}{growth_txt}   |   "
                  + (f"[终局] {result_msg}" if done else
                     "点击窗口→P方向键/空格 P1-WASD/F R重开 ESC返回 Q退出"))
        # 文本只在内容变化时重渲（≈10Hz），渲染帧直接 blit 缓存
        if status != status_key:
            status_surf = font.render(status, True, (220, 220, 220))
            status_key = status
        if sim.cfg.map_mode == "corridor":
            hud_text = f"泡泡×{nb}  威力×{nblast}  速度×{spd:.1f}"
            if hud_text != hud_key:
                hud_surf = hud_font.render(hud_text, True, (255, 255, 255))
                hud_surf.set_alpha(200)
                hud_key = hud_text
        screen.blit(status_surf, (8, cfg.height * CELL + 8))
        pygame.display.flip()
        # 帧率显示在窗口标题：卡不卡一眼可见（clock.get_fps 是最近几帧的平滑值）
        pygame.display.set_caption(
            f"{title}   FPS={clock.get_fps():.0f}")
        clock.tick(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/duel_rw_ckpt.pt")
    ap.add_argument("--device", default="cpu",
                    help="默认 cpu：对打每 tick 只有 ~3.5ms，响应即时；"
                         "mps 每 tick 约 17ms 的同步延迟会让操作感觉迟钝")
    ap.add_argument("--size", type=int, default=GRID)
    ap.add_argument("--map-mode", default="open",
                    choices=["open", "corridor", "ring"],
                    help="open = 纯空场；corridor = 左右可炸墙 + 顶部永久墙 + "
                         "宝箱成长；ring = 中间 7×7 永久墙山体 + 环带稀疏可炸墙"
                         " + 宝箱 100%%（四角出生）")
    ap.add_argument("--scene", default="比武",
                    help="场景（res/scenes.json）：比武/沙漠/雪地/… → 砖块图 + BGM")
    ap.add_argument("--hz", type=int, default=10, help="逻辑 tick 率（默认 10 = 训练一致）")
    ap.add_argument("--player-speed-mult", type=float, default=1.0,
                    help="玩家侧的移动速度倍率（AI 恒为 1.0）。默认 1.0 = 全局同速；"
                         "试手速可调 1.3 之类")
    ap.add_argument("--open-growth-pct", type=float, default=0.8,
                    help="open 图初始成长 = 上限百分比（默认 0.8 = 80%%，与训练一致："
                         "泡数/威力 6、速度倍率 1.68；可调 0.9 = 90%%：泡数/威力 7、"
                         "速度倍率 1.89，速度上限已封顶 2.1 不再有 2.7）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-ticks", type=int, default=0,
                    help=">0 时不开交互，自动跑 N tick 冒烟（无窗口）")
    ap.add_argument("--screenshot", default=None,
                    help="推进 60 tick 后渲染一帧存 PNG 退出（无窗口，调 UI 用）")
    ap.add_argument("--bot-mode", action="store_true",
                    help="机器人对机器人观战：玩家 0 也走 AI 决策（不读键盘）")
    ap.add_argument("--opp-bot", default=None,
                    choices=["random", "greedy", "idle", "astar", "hunter"],
                    help="把 AI 对手换成规则 bot（sim/bots.py），不读 checkpoint。"
                         "如 --opp-bot astar = 危险度融合寻路 AI；--opp-bot idle = 静止靶")
    ap.add_argument("--p0-bot", default=None,
                    choices=["random", "greedy", "idle", "astar", "hunter"],
                    help="观战模式玩家 0 用规则 bot（默认读 --ckpt）")
    ap.add_argument("--die-log", action="store_true",
                    help="死亡时刻打印精确死因（死时自己泡数/双方血量/位置）—— 排查自杀")
    ap.add_argument("--recording", action="store_true",
                    help="录制人类玩家轨迹（10Hz，obs+action → recordings/*.npz，"
                         "BC 训练数据）；局内 L 键开关")
    ap.add_argument("--ckpt-b", default=None,
                    help="观战模式玩家 1 的 checkpoint（默认跟随 --ckpt）")
    ap.add_argument("--human0", action="store_true",
                    help="玩家 0 是人类键盘玩家（WASD+空格，默认 True）")
    ap.add_argument("--human1", action="store_true",
                    help="玩家 1 是人类键盘玩家（方向键+回车）—— 双人同屏对打")
    args = ap.parse_args()

    outcome = run_game(ckpt=args.ckpt, ckpt_b=args.ckpt_b, device=args.device,
                       size=args.size,
                       map_mode=args.map_mode, scene=args.scene, hz=args.hz,
                       player_speed_mult=args.player_speed_mult,
                       open_growth_pct=args.open_growth_pct, seed=args.seed,
                       auto_ticks=args.auto_ticks, screenshot=args.screenshot,
                       bot_mode=args.bot_mode, opp_bot=args.opp_bot,
                       p0_bot=args.p0_bot, die_log=args.die_log,
                       recording=args.recording,
                       human0=args.human0 or not args.bot_mode,
                       human1=args.human1)
    # CLI 直开（无启动器）时 ESC 返回菜单无意义 → 直接退出
    if outcome == "menu":
        pygame.quit()
        sys.exit(0)


def entry() -> None:
    """无参数 → 提示走浏览器版启动器（web/，仓库当前展示入口）；
    挂了任何 CLI 参数 → 直进对局。"""
    if len(sys.argv) == 1:
        print("对打入口已迁移到浏览器版启动器：", file=sys.stderr)
        print("  一条命令:  bash scripts/serve_web.sh   →  http://localhost:8080", file=sys.stderr)
        print("  （自动增量导出 ckpt → web/models 后开服；端口可传参）", file=sys.stderr)
        print("  挂 CLI 参数（--ckpt/--opp-bot/--bot-mode …）可直开 Python 对局，见 --help", file=sys.stderr)
        sys.exit(1)
    else:
        main()


if __name__ == "__main__":
    entry()
