"""加载 res/ 素材：角色精灵、炸弹呼吸动画、爆炸切片、背景、音效。

精灵图布局（用户确认）：`角色4×4精灵图.png` = 4 行方向 × 4 列行走帧，
行序从上到下 = **下、左、右、上**。每帧 85×85，运行时缩到格大小。

爆炸图：`爆炸中心.png` 40×40；四个方向臂图是 520×40 / 40×520（13 格）。
爆炸是"按最大爆炸范围画的"，必须按 blast 格数从中心端切片
（向右/向下取中心端起始段，向左/向上取远离中心端的一段）。
炸弹：bomb1..bomb6 六帧，按时间循环形成"上下呼吸"。
"""

from __future__ import annotations

import os

import numpy as np
import pygame

from sim.config import MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(os.path.dirname(HERE), "res")

# 动作编码 → 精灵行（行序：下/左/右/上）
MOVE_TO_SPRITE_ROW = {
    MOVE_DOWN: 0,   # row0 = 下
    MOVE_LEFT: 1,   # row1 = 左
    MOVE_RIGHT: 2,  # row2 = 右
    MOVE_UP: 3,     # row3 = 上
}


def _load(name: str) -> pygame.Surface:
    return pygame.image.load(os.path.join(RES_DIR, name)).convert_alpha()


def _content_box(surf: pygame.Surface):
    """非透明内容的包围盒 (x0,y0,x1,y1)，全透明返回 None。"""
    import numpy as np
    arr = pygame.surfarray.pixels_alpha(surf)
    ys, xs = (arr > 100).nonzero()
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _body_center(surf: pygame.Surface) -> tuple[int, int]:
    """人物帧的视觉主体质心（相对帧左上角）。

    人物图 85×85 帧里脚部往往有十几像素透明（脚底 = 帧底边贴格底），
    视觉主体悬在帧上部 —— 无敌光晕若贴帧左上角会"罩在人物脚下"。
    用非透明像素质心做光晕的居中锚点，光晕正好罩住人物身体。
    """
    import numpy as np
    arr = pygame.surfarray.pixels_alpha(surf)
    ys, xs = (arr > 40).nonzero()
    if len(xs) == 0:
        w, h = surf.get_size()
        return (w // 2, h // 2)
    return (int(xs.mean()), int(ys.mean()))


class Res:
    """一次性加载全部素材；缺失的素材自动降级为占位色块，不崩溃。"""

    def __init__(self, cell: int, blast: int, scene: str = "比武") -> None:
        self.cell = cell
        self.blast = blast
        self.s = cell / 40.0          # 缩放因子：素材原生 40px/格 → 渲染格大小
        self.scene = scene
        self.players = self._load_player(cell)
        self.player_ai = self._tint_red(self.players)
        # 每帧人物的视觉主体质心（相对帧左上角，见 _body_center）；fallback
        # 色块人形没有 → None。无敌光晕按它居中（罩住人物，不是贴脚底）。
        self.body_centers: list[list[tuple[int, int]]] | None = None
        self._fill_body_centers()
        self.bombs = self._load_bombs(cell)
        self.props = self._load_props(cell)     # 道具图：威力/泡泡数量/鞋子
        self.explo_center, self.explo_arms = self._load_explosion(blast, cell)
        # 场景资源（纯 UI，不进训练）：砖块图 + 背景图 + BGM。
        # 缺失时回退 None → duel 用色块/默认背景兜底。
        self.wall_tile, self.brick_tile, self.ground_tile, \
            self.bgm_path, self.bg_path = self._load_scene(cell)
        self.bg = self._load_bg()
        # 无敌罩（res/无敌.PNG / wudi.PNG）：自带 alpha 渐变的透明光罩。
        # 加载时已预乘（rgb *= alpha/255），渲染走 BLEND_ADD —— 透明区不加、
        # 光晕随 alpha 平滑衰减，与参考引擎 additive（dst + src·a）一致。
        self.wudi, self.wudi_scaled = self._load_wudi()
        self.snd_place = self._load_sound("放炮.wav")
        self.snd_boom = self._load_sound("爆炸.wav")
        self.snd_pickup = self._load_sound("吃道具音效.wav")     # 吃宝箱
        self.snd_hurt = self._load_sound("生命损失音效.wav")      # 掉血
        self.snd_die = self._load_sound("角色消失音效.wav")       # 死亡（血归 0）

    def _load_wudi(self) -> tuple[pygame.Surface | None, pygame.Surface | None]:
        """无敌罩贴图（90×89）：自带 alpha 渐变的透明光罩，加法混合叠加。

        素材是旧式「蓝底色键 + alpha」的**直通(straight) alpha** 图：透明像素
        RGB 是纯蓝 (0,0,255)，其余像素是"全亮"的黄色发光体（RGB 不随 alpha
        缩放），alpha 单独控制透明度。pygame 的 BLEND_ADD 是 dst+src（完全
        忽略 alpha）——直接把直通 RGB 全量相加，半透明外围也会全强度加进
        背景（光晕边缘生硬、过曝发白）；而参考引擎（Flash/Unity 等
        additive）的公式是 dst + src·alpha（预乘加法），发光随透明度平滑衰减。
        这里在**加载时预乘**对齐引擎：
          1) 透明像素 RGB 清黑 —— 预乘后透明区贡献 0，BLEND_ADD 不变，
             既保留加法发光（不加法会很丑）又不会露出蓝底方块；
          2) rgb = rgb * alpha/255 —— 半透明像素的发光贡献随 alpha 线性
             衰减，光晕从中心到边缘柔和淡出，与引擎 dst+src·a 一致；
          3) 在预乘空间 smoothscale（透明区 RGB=0，插值不渗色），省掉
             无敌期每帧一次缩放。
        渲染时照旧 BLEND_ADD：核心（alpha=255）全强度、外围柔和、透明区不变。
        返回 (原图, 缩放版)。
        """
        surf = None
        for name in ("无敌.PNG", "wudi.PNG"):
            try:
                surf = _load(name)
                break
            except (FileNotFoundError, pygame.error):
                continue
        if surf is None:
            return None, None
        try:
            rgb = pygame.surfarray.pixels3d(surf)
            alpha = pygame.surfarray.pixels_alpha(surf)
            rgb[alpha == 0] = 0                     # 蓝底清黑（防渗色 + 防蓝块）
            premul = (rgb.astype(np.uint16) * alpha[..., None] + 127) // 255
            rgb[...] = premul.astype(np.uint8)      # 预乘：贡献 = rgb·a/255
            del rgb, alpha
        except pygame.error:
            pass                       # 无 alpha 通道的图不用清
        target = int(round(85 * self.s))    # 与角色帧缩放一致（CELL=60 → 128）
        scaled = pygame.transform.smoothscale(surf, (target, target))
        return surf, scaled

    def _load_scene(self, cell: int) -> tuple:
        """从 res/scenes.json 读场景配置：砖块图 + 背景音乐路径。

        砖块图是立式（原生 40×54 左右），等比缩放到格子：宽度 = cell，
        高度按比例。None 表示该场景没配 tile（duel 用色块兜底）。
        """
        import json
        cfg_path = os.path.join(RES_DIR, "scenes.json")
        try:
            with open(cfg_path) as f:
                scenes = json.load(f)["scenes"]
            sc = scenes.get(self.scene, {})
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None, None, None, None, None
        tiles = sc.get("tiles", {})

        def load_one(rel: str) -> pygame.Surface | None:
            p = os.path.join(os.path.dirname(RES_DIR), rel)
            try:
                img = pygame.image.load(p).convert_alpha()
            except (FileNotFoundError, pygame.error):
                return None
            w0, h0 = img.get_size()
            h1 = max(1, round(h0 * cell / w0))        # 等比缩放（宽 = cell）
            return pygame.transform.smoothscale(img, (cell, h1))

        def load_tile(key: str):
            """单个图 → Surface|None；列表（如 brick 变体）→ list[Surface]|None。"""
            rel = tiles.get(key)
            if not rel:
                return None
            if isinstance(rel, list):
                loaded = [s for s in (load_one(r) for r in rel) if s is not None]
                return loaded or None
            return load_one(rel)

        return (load_tile("wall"), load_tile("brick"), load_tile("ground"),
                sc.get("bgm"), sc.get("bg"))

    # ---------------- 角色：4 行方向 × 4 列行走帧 ----------------
    # **保持原尺寸 85x85**（不缩放、不裁切）—— 人物可以远大于 40px 格子。
    # 渲染时（duel.py）：帧底边 = 中心格底边（角色中心下方半格），
    # 帧水平中心 = 角色中心（站格中心时 42.5 与 20 同竖线）。

    def _load_player(self, cell: int) -> list[list[pygame.Surface]]:
        try:
            sheet = _load("角色4×4精灵图.png")
        except FileNotFoundError:
            return self._fallback_player(cell)
        fw, fh = sheet.get_width() // 4, sheet.get_height() // 4   # 85x85 原尺寸
        target = int(round(85 * self.s))             # 1.5 倍画布 → 128
        rows: list[list[pygame.Surface]] = []
        for r in range(4):
            rows.append([
                pygame.transform.smoothscale(
                    sheet.subsurface((c * fw, r * fh, fw, fh)), (target, target))
                for c in range(4)
            ])
        return rows

    def _fallback_player(self, cell: int) -> list[list[pygame.Surface]]:
        """素材缺失时的占位：色块圆头人形，保证游戏仍可玩。"""
        import random
        rng = random.Random(0)
        rows = []
        for r in range(4):
            rows.append([])
            for _ in range(4):
                s = pygame.Surface((cell, cell), pygame.SRCALPHA)
                body = (255, 170, 60) if r in (0, 3) else (240, 70, 70)
                pygame.draw.circle(s, (255, 224, 189), (cell // 2, cell // 3), cell // 4)
                pygame.draw.rect(s, body, (cell // 4, cell // 2, cell // 2, cell // 3))
                rows[r].append(s)
        return rows

    def _fill_body_centers(self) -> None:
        """算每帧人物的视觉主体质心（玩家 + AI 同构，共用一份）。

        素材人物帧（85×85）脚部有大量透明像素（脚底 = 帧底边贴格底），
        视觉主体悬在帧上部 —— 无敌光晕按质心居中才罩得住人物身体。
        fallback 色块人形没有 → body_centers = None。
        """
        if not self.players or not self.players[0]:
            self.body_centers = None
            return
        self.body_centers = [
            [_body_center(f) for f in row] for row in self.players]

    @staticmethod
    def _tint_red(players: list[list[pygame.Surface]]) -> list[list[pygame.Surface]]:
        """敌人（AI）整体染成红调：保留明暗，色相偏红 —— 一眼区分双方。"""
        out = []
        for r in range(4):
            out.append([])
            for f in players[r]:
                t = f.copy()
                arr = pygame.surfarray.pixels3d(t)          # (w,h,3) 就地引用
                gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
                arr[..., 0] = np.clip(gray * 1.3, 0, 255).astype(np.uint8)
                arr[..., 1] = (gray * 0.3).astype(np.uint8)
                arr[..., 2] = (gray * 0.3).astype(np.uint8)
                del arr
                out[r].append(t)
        return out

    # ---------------- 炸弹：同一张图，渲染时垂直浮动做"呼吸" ----------------
    # bomb1..6 是 6 种**不同样式**的泡泡（不是动画帧）。呼吸动画 = 用一张
    # 固定的泡泡图（bomb1），在格内上下小幅浮动。加载成单张 surface。

    def _load_bombs(self, cell: int) -> list[pygame.Surface]:
        try:
            img = _load("bomb1.png")
        except FileNotFoundError:
            img = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(img, (70, 150, 235), (20, 20), 13)
        # 素材原生 40px/格，按画布缩放 1.5 倍到格子大小（60×60，和爆炸中心一致）。
        # **不缩小** —— 和人物一样保持大图，摆放逻辑（duel.py）也是人物同款：
        # 水平中心对格子中心、底边对格子底线；图比格大时左右各溢出一半。
        size = (cell, cell)
        return [pygame.transform.smoothscale(img, size)]

    # ---------------- 道具图（宝箱用）：威力 / 泡泡数量 / 鞋子 ----------------
    # 三张 40×40 原生素材，按画布缩放 1.5 倍到格子大小（60×60，和炸弹一致）。
    # 宝箱格渲染时轮流展示（威力→数量→鞋子），上下浮动（呼吸），
    # 对应开箱可能升的三个属性（威力/泡数/速度）。

    def _load_props(self, cell: int) -> list[pygame.Surface]:
        size = (cell, cell)
        out = []
        for name in ("威力道具.png", "泡泡数量道具.png", "鞋子道具.png"):
            try:
                img = _load(name)
            except FileNotFoundError:
                img = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.rect(img, (200, 150, 60), (8, 8, 24, 24), 2)
            out.append(pygame.transform.smoothscale(img, size))
        return out

    # ---------------- 爆炸：中心 + 四方向臂（按 blast 切片） ----------------
    # 素材本身是 40px/格：**不缩放**，格子大小 = 爆炸中心图尺寸（1:1）。
    # 臂图（520 宽/高 = 13 格）的**炸弹边缘端**：向左=最左边、向右=最右边、
    # 向上=最上边、向下=最下边。从该端切 blast 格宽（砍掉另一端）。
    # 拼接（duel.py）：贴中心的那格用保留段的"贴中心端"段 ——
    # 向左爆炸素材图右边贴中心格左边、向右左边贴中心右边、上/下同理。

    def _load_explosion(self, blast: int, cell: int) -> tuple[pygame.Surface, dict]:
        try:
            center = pygame.transform.smoothscale(
                _load("爆炸中心.png"), (cell, cell))
        except FileNotFoundError:
            center = pygame.transform.smoothscale(
                self._fallback_cell((255, 220, 120)), (cell, cell))
        arms: dict[tuple[int, int], pygame.Surface] = {}
        length = blast * 40                       # 素材原生 40px/格
        for (drow, dcol), name, horiz in (
            ((-1, 0), "向上爆炸.png", False),
            ((1, 0), "向下爆炸.png", False),
            ((0, -1), "向左爆炸.png", True),
            ((0, 1), "向右爆炸.png", True),
        ):
            try:
                img = _load(name)
            except FileNotFoundError:
                img = self._fallback_cell((255, 200, 80))
            if horiz:
                w, h = img.get_width(), img.get_height()
                x = 0 if dcol < 0 else w - length   # 向左保留最左 / 向右保留最右
                sub = img.subsurface((x, 0, length, h))
                arms[(drow, dcol)] = pygame.transform.smoothscale(
                    sub, (int(length * self.s), cell))
            else:
                w, h = img.get_width(), img.get_height()
                y = 0 if drow < 0 else h - length   # 向上保留最上 / 向下保留最下
                sub = img.subsurface((0, y, w, length))
                arms[(drow, dcol)] = pygame.transform.smoothscale(
                    sub, (cell, int(length * self.s)))
        return center, arms

    @staticmethod
    def _fallback_cell(color) -> pygame.Surface:
        s = pygame.Surface((40, 40), pygame.SRCALPHA)
        s.fill(color)
        return s

    # ---------------- 背景 ----------------

    def _load_bg(self) -> pygame.Surface | None:
        """背景：优先场景整图（scenes.json 的 bg，如比武 bw.png）；
        缺失回退旧 res/bg1.png；都没有返回 None（duel 用深色填充）。

        **不缩放**：背景原图（600×520 = 15×13 个 40×40 格子）里带着地面格线，
        是玩家判断位置的参考；缩放（尤其非等比）会把格子拉变形。原图直接
        返回，铺满由 build_static 平铺完成（见 play/duel.py）。
        """
        if self.bg_path:
            p = os.path.join(os.path.dirname(RES_DIR), self.bg_path)
            try:
                return pygame.image.load(p).convert()
            except (FileNotFoundError, pygame.error):
                pass
        try:
            return _load("bg1.png")
        except FileNotFoundError:
            return None

    # ---------------- 音效（可缺失） ----------------

    @staticmethod
    def _load_sound(name: str):
        try:
            return pygame.mixer.Sound(os.path.join(RES_DIR, name))
        except (pygame.error, FileNotFoundError):
            return None
