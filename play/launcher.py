"""对局启动器：不带命令行参数时的图形入口（pygame 菜单页）。

启动器页面可配：
    - 对战模式：玩家 vs AI（你操控 player 0）/ AI vs AI 观战（双方都走 AI 决策）
    - 地图 / 场景（同一个下拉，条目多时滚轮滚动）：
        前 3 项是**地图结构**（open 纯空场 / corridor 左右可炸墙+宝箱成长 /
        ring 中央 7×7 永久墙山体环岛）；分隔线以下是 res/scenes.json 的
        **全部场景皮肤**（背景图 + BGM + 砖块贴图，共 11 个）—— 选地图结构
        不动场景，选场景不动地图结构，两者独立生效、同时保留。
    - 初始属性：open 用「成长百分比」（与训练一致的 80% 默认：泡6/威6/速1.68）；
      corridor/ring 用 泡数 / 威力 / 速度倍率 三档
      （1..7 / 1..7 / 1.00..2.10，踩宝箱继续成长到上限）
    - checkpoint：扫描 ckpt/ 目录全部模型，标签 = 「日期 步数M-架构 地图 文件名」
      （按修改时间倒序，最新训练排最前；如 "08-03 18:09  150M-mlp corridor  duel_5x.pt"）
    - 开始对局 → 复用 play.duel.run_game（含危险图渐变 + 连锁联动渲染）；
      对局内 ESC 回到本启动器（Q / 关窗才退出程序）

用法：
    python -m play.launcher        # 图形启动器（无参数默认入口）
    python -m play.duel ...        # 挂完整 CLI 参数则跳过启动器，直进对局
"""

from __future__ import annotations

import math
import os
import sys

# 无窗口场景（冒烟）用 dummy 驱动，交互模式用 cocoa。必须在 import pygame
# 之前决定（duel.py 同款逻辑，_load 的 setdefault 以这里为准）。
_headless = ("--smoke" in sys.argv or "--auto-ticks" in sys.argv
             or bool(os.environ.get("SDL_VIDEODRIVER_DUMMY")))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy" if _headless else "cocoa")

import pygame  # noqa: E402

import torch  # noqa: E402

from .duel import GRID, _load_cjk_font, run_game  # noqa: E402
from .replay import list_recordings  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJ, "ckpt")
SCENES_JSON = os.path.join(PROJ, "res", "scenes.json")
REC_DIR = os.path.join(PROJ, "recordings")


def rec_meta(path: str) -> tuple[int, int, dict]:
    """录像基础信息：(tick 数, pid, meta dict)。读失败返回 (0, 0, {})。"""
    import ast
    import numpy as np
    try:
        d = np.load(path, allow_pickle=True)
        T = int(d["obs"].shape[0])
        pid = int(d["pid"])
        try:
            meta = ast.literal_eval(str(d["meta"][0]))
        except Exception:
            meta = {}
        return T, pid, meta
    except Exception:
        return 0, 0, {}

# ------------------------------------------------------------------ 数据

def list_scenes() -> list[str]:
    """res/scenes.json 的场景名列表（缺文件时兜底默认列表）。"""
    import json
    try:
        with open(SCENES_JSON) as f:
            return list(json.load(f)["scenes"].keys())
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return ["比武", "中国城", "功夫", "夺宝", "抢包子",
                "水面", "沙漠", "矿洞", "野外", "雪地", "英雄"]


def list_ckpts() -> list[tuple[str, str]]:
    """(标签, 路径) 列表，按修改时间倒序。

    标签 = 「日期 步数M-架构 地图 文件名」，如
        "08-03 18:09  150M-mlp corridor  duel_5x.pt"
    步数/架构/地图读 ckpt 元信息（save_ckpt 写入的 format_version 2 字段）；
    旧档无这些字段时回退 "??-cnn"。23 个文件全量加载约 0.1s。
    """
    import datetime
    items = []
    if not os.path.isdir(CKPT_DIR):
        return items
    for f in sorted(os.listdir(CKPT_DIR)):
        if not f.endswith(".pt"):
            continue
        p = os.path.join(CKPT_DIR, f)
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue                      # 半截/损坏档：跳过不阻塞列表
        mtime = datetime.datetime.fromtimestamp(
            os.path.getmtime(p)).strftime("%m-%d %H:%M")
        step = ck.get("global_step")
        step_s = f"{int(step) // 1_000_000}M" if step else "??"
        arch = ck.get("arch", "cnn")
        mm = (ck.get("args") or {}).get("map_mode", "")
        tag = f"{mtime}  {step_s}-{arch}" + (f" {mm}" if mm else "")
        items.append((f"{tag}  {f}", p))
    items.sort(key=lambda t: os.path.getmtime(t[1]), reverse=True)
    return items


def short_ckpt_label(path: str) -> str:
    """紧凑标签：「文件名 步数M 日期」—— 观战双 ckpt 下拉（两列窄框放不下
    富标签，且文件名放最前面，选档时一眼认出哪个模型）。"""
    import datetime
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        step = ck.get("global_step")
        step_s = f"{int(step) // 1_000_000}M" if step else "??"
    except Exception:
        step_s = "??"
    mtime = datetime.datetime.fromtimestamp(
        os.path.getmtime(path)).strftime("%m-%d")
    return f"{os.path.basename(path)}  {step_s}  {mtime}"


# ---- 规则 bot 选项（sim/bots.py）：插在 ckpt 下拉最前面，可直接选它当 AI/对手 ----
# kind 与 label 一一对应：("human", "") = 人类键盘玩家；("bot", name) = 规则 bot；
# ("sep", "") = 分隔线；("ckpt", path) = 模型权重。下拉选中的下标直接映射到决策来源。
HUMAN_OPTION = ("human", "")
HUMAN_LABEL = "【人类玩家】键盘操作"
BOT_OPTIONS = [("astar", "【规则】寻路 AI astar"),
               ("hunter", "【规则】纯进攻寻路 hunter"),
               ("greedy", "【规则】贪心 greedy"),
               ("idle", "【规则】静止靶 idle"),
               ("random", "【规则】随机 random")]
BOT_SEPARATOR = "───── 模型 checkpoint ─────"


def ckpt_option_rows(short: bool = False) -> tuple[list[str], list]:
    """(labels, kinds)：人类玩家 + 规则 bot + 分隔线 + 全部 checkpoint。

    short=True 时 ckpt 用紧凑标签（P1 窄下拉）。
    """
    labels, kinds = [HUMAN_LABEL], [HUMAN_OPTION]
    for name, label in BOT_OPTIONS:
        labels.append(label)
        kinds.append(("bot", name))
    labels.append(BOT_SEPARATOR)
    kinds.append(("sep", ""))
    for label, path in list_ckpts():
        labels.append(label if not short else short_ckpt_label(path))
        kinds.append(("ckpt", path))
    return labels, kinds


# ------------------------------------------------------------------ 控件


class Dropdown:
    """下拉框：rect 是闭合态框体；点击展开 item 列表，点中收起并取值。

    展开列表条目多于一屏（max_visible）时支持鼠标滚轮滚动，右侧画滚动条。
    滚轮滚动**同步选中**（滚到哪选到哪，分隔线自动跳过）—— 用户"滚到
    ring/水面"即生效，不必再点一下条目；当前选中项带 ✓ 标记。
    列表绘制单独走 draw_items（Launcher 在**最后**统一调用）—— 保证展开的
    条目永远盖在其余 UI 上层，不会被后面画的标签/步进器/按钮压住（Z 序修复）。
    以 "─" 开头的条目是分隔线（不可选中）。
    """

    def __init__(self, rect: pygame.Rect, options: list[str],
                 font, max_visible: int = 8, default: int = 0):
        self.rect = rect
        self.options = options
        self.font = font
        self.max_visible = max_visible
        self.sel = min(default, len(options) - 1) if options else -1
        self.open = False
        self.scroll = 0              # 展开列表顶部显示的是第几个条目

    @property
    def value(self) -> str | None:
        return self.options[self.sel] if 0 <= self.sel < len(self.options) else None

    def _range(self) -> tuple[int, int]:
        n = len(self.options)
        top = min(self.scroll, max(0, n - self.max_visible))
        return top, min(n, top + self.max_visible)

    def _item_rects(self) -> tuple[int, list[pygame.Rect]]:
        top, bot = self._range()
        return (top, [pygame.Rect(self.rect.x, self.rect.bottom + i * self.rect.h,
                                  self.rect.w, self.rect.h)
                      for i in range(bot - top)])

    def handle(self, e, mouse) -> bool:
        """返回 True = 事件被消费。展开/收起、选条目、滚轮滚动。"""
        if e.type != pygame.MOUSEBUTTONDOWN:
            return False
        if e.button == 1:
            if self.rect.collidepoint(mouse):
                self.open = not self.open
                if self.open:              # 重开时保证当前选中项在可见范围内
                    top, bot = self._range()
                    if not (top <= self.sel < bot):
                        self.scroll = max(
                            0, min(self.sel, len(self.options) - self.max_visible))
                return True
            if self.open:
                top, rects = self._item_rects()
                for i, r in enumerate(rects):
                    if r.collidepoint(mouse):
                        idx = top + i
                        if self.options[idx].startswith("─"):
                            return True    # 分隔线：不选中，保持展开
                        self.sel = idx
                        self.open = False
                        return True
                self.open = False          # 点空白收起（消费，不穿透到下层控件）
                return True
            return False
        if self.open and e.button in (4, 5):
            # 滚轮：只在展开列表区域生效（button 4 = 上滚，5 = 下滚）；
            # 滚动同时把选中项带到新的顶部条目（跳过分隔线）——
            # "滚到哪选到哪"，闭合框与"当前："实时跟随，开始即用。
            top, rects = self._item_rects()
            if rects:
                area = pygame.Rect(self.rect.x, self.rect.y, self.rect.w,
                                   (1 + len(rects)) * self.rect.h)
                if area.collidepoint(mouse):
                    n = len(self.options)
                    lo = max(0, n - self.max_visible)
                    if e.button == 4:
                        self.scroll = max(0, self.scroll - 1)
                    else:
                        self.scroll = min(lo, self.scroll + 1)
                    top, _ = self._item_rects()
                    idx = top
                    while idx < n and self.options[idx].startswith("─"):
                        idx += 1               # 顶部是分隔线 → 落到其后首个可选项
                    if idx < n:
                        self.sel = idx
                    return True
        return False

    def _fit(self, text: str, max_w: int) -> str:
        """按可用宽度截断文本，超宽加 "…"（长 ckpt 标签不会溢出框底）。"""
        if self.font.size(text)[0] <= max_w:
            return text
        t = text
        while len(t) > 1 and self.font.size(t + "…")[0] > max_w:
            t = t[:-1]
        return t + "…"

    def draw_closed(self, screen):
        # 闭合框：深底 + 边框 + 当前值 + 右侧 ▾（值超宽自动截断）
        pygame.draw.rect(screen, (40, 38, 48), self.rect, border_radius=6)
        pygame.draw.rect(screen, (120, 118, 132), self.rect, 2, border_radius=6)
        txt = self.value or "—"
        fit = self._fit(txt, self.rect.w - 34)
        screen.blit(self.font.render(fit, True, (235, 235, 235)),
                    (self.rect.x + 10, self.rect.y + 4))
        screen.blit(self.font.render("▾", True, (160, 160, 170)),
                    (self.rect.right - 24, self.rect.y + 4))

    def hover_index(self, mouse: tuple[int, int]) -> int | None:
        """鼠标悬停的展开条目下标（未展开/不在列表上 → None），预览用。"""
        if not self.open:
            return None
        top, rects = self._item_rects()
        for i, r in enumerate(rects):
            if r.collidepoint(mouse):
                return top + i
        return None

    def draw_items(self, screen, hover: tuple[int, int]):
        """展开条目列表：单独调用、Launcher 最后绘制 → 盖在所有 UI 上层。"""
        if not self.open:
            return
        top, rects = self._item_rects()
        n = len(self.options)
        check_w = self.font.size("✓ ")[0]
        for i, r in enumerate(rects):
            item = self.options[top + i]
            if item.startswith("─"):         # 分隔线：暗底 + 居中灰字
                pygame.draw.rect(screen, (30, 28, 38), r)
                lab = self.font.render(item.strip("─ "), True, (120, 120, 130))
                screen.blit(lab, (r.x + (r.w - lab.get_width()) // 2, r.y + 4))
                continue
            on = r.collidepoint(hover)
            sel = (top + i) == self.sel
            bg = (60, 58, 72) if on else ((44, 70, 52) if sel else (36, 34, 44))
            pygame.draw.rect(screen, bg, r)
            pygame.draw.rect(screen, (120, 190, 140) if sel else (90, 88, 100),
                             r, 1 if not sel else 2)
            if sel:
                screen.blit(self.font.render("✓", True, (140, 230, 160)),
                            (r.x + 8, r.y + 4))
            fit = self._fit(item, r.w - 34 - check_w)
            screen.blit(self.font.render(fit, True, (230, 230, 230)),
                        (r.x + 10 + check_w, r.y + 4))
        if n > self.max_visible:             # 右侧滚动条（可滚动的下拉才有）
            track = pygame.Rect(self.rect.right - 7, self.rect.bottom + 2,
                                5, len(rects) * self.rect.h - 4)
            pygame.draw.rect(screen, (70, 68, 82), track, border_radius=2)
            thumb_h = max(14, int(track.h * self.max_visible / n))
            lo = n - self.max_visible
            thumb_y = track.y + (track.h - thumb_h) * top // max(1, lo)
            pygame.draw.rect(screen, (150, 150, 165),
                             (track.x, thumb_y, track.w, thumb_h), border_radius=2)


def stepper_rects(x: int, y: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
    """(值框, 减号钮, 加号钮) —— 一行 泡数/威力/速度 的加减步进器。"""
    return (pygame.Rect(x, y, 44, 28),
            pygame.Rect(x + 50, y, 26, 28),
            pygame.Rect(x + 82, y, 26, 28))


# ------------------------------------------------------------------ 主菜单


class Launcher:
    def __init__(self):
        self.scenes = list_scenes()
        self.ckpts = list_ckpts()                 # 最新训练排最前
        self.map = "open"
        self.scene = "比武"
        # 初始属性：open 用成长百分比；corridor/ring 用三档（1..7 / 1..7 / 1.00..2.10）
        self.open_pct = 0.8
        self.bombs_start = 2
        self.blast_start = 2
        self.speed_start = 1.0
        # P0 / P1 两个下拉都可选：人类键盘 / 规则 bot / 模型 —— 不再有"对战模式"
        # 切换（人类 vs AI = P0 选人类 + P1 选 AI；观战 = 两边都选 AI；双人 = 都选人类）。
        ckpt_labels0, self.ckpt_kinds0 = ckpt_option_rows(short=False)
        ckpt_labels1, self.ckpt_kinds1 = ckpt_option_rows(short=True)

        def _default_idx(kinds: list) -> int:
            """默认选最新的**对打档**（跳过人类/bot 项；dodge_ 前缀 = 纯生存特训档，
            只躲不攻，行为异常，不让它当默认；dodge 档仍在列表里可手动选来对比）。"""
            for i, (k, v) in enumerate(kinds):
                if k == "ckpt" and not os.path.basename(v).startswith("dodge_"):
                    return i
            return 0 if kinds else -1

        self._ckpt_sel = _default_idx(self.ckpt_kinds0)
        self.ckpt = ""          # P0 模型权重路径（选人类/bot 时为空）
        self.ckpt_b = ""        # P1 模型权重路径
        self.opp_bot = None     # P0 是规则 bot 时的名字
        self.p1_bot = None      # P1 是规则 bot 时的名字
        self.p0_human = False   # P0 是人类键盘玩家
        self.p1_human = False   # P1 是人类键盘玩家

        self.font_t = _load_cjk_font(30)
        self.font_l = _load_cjk_font(22)
        self.font_s = _load_cjk_font(18)
        self.font_t.set_bold(True)

        self.W, self.H = W, H = 780, 700
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("QQT 对打 · 启动器")
        # 地图/场景合并下拉：前 3 = 地图结构，分隔线后 = 全部场景皮肤（滚轮滚动）
        map_options = ["open（纯空场）", "corridor（砖墙成长）", "ring（中央山体环岛）",
                       "───── 场景皮肤 ─────"] + self.scenes
        # 宽框到 710：长标签（如 ckpt 日期+步数+架构+地图+文件名）不超出框底，
        # draw_closed/_fit 还会对超宽文本截断加 "…" 兜底。
        self.dd_map = Dropdown(pygame.Rect(170, 104, 540, 32),
                               map_options, self.font_l,
                               max_visible=8, default=0)
        # 双档常驻：玩家 0 / 玩家 1 各一个下拉，都可选 人类/规则bot/模型。
        self.dd_ckpt = Dropdown(pygame.Rect(170, 262, 540, 32),
                                ckpt_labels0 or ["（无 ckpt）"], self.font_l,
                                max_visible=9,
                                default=self._ckpt_sel if ckpt_labels0 else 0)
        self.dd_ckpt_b = Dropdown(
            pygame.Rect(170, 322, 540, 32),
            ckpt_labels1 or ["（无 ckpt）"], self.font_s, max_visible=7,
            default=self._ckpt_sel if ckpt_labels1 else 0)
        # 录像回放：recordings/*.npz 列表（最新在前）+ 播放按钮
        self.recs = list_recordings(REC_DIR)
        self._rec_meta = [rec_meta(p) for p in self.recs]
        rec_labels = [self._rec_label(i) for i in range(len(self.recs))] \
            or ["（无录像，先打几局）"]
        self._replay_sel = 0        # 选中的录像下标（-1 = 未选）
        self.dd_replay = Dropdown(self.REPLAY_DD, rec_labels, self.font_s,
                                  max_visible=7, default=0)

    # ----- 布局坐标 -----
    START_BTN = pygame.Rect(240, 600, 300, 46)
    REPLAY_BTN = pygame.Rect(575, 470, 170, 32)
    REPLAY_DD = pygame.Rect(170, 470, 405, 32)

    def _rec_label(self, i: int) -> str:
        """录像下拉标签：文件名 + 长度(tick) + pid + 对手（紧凑）。"""
        name = os.path.basename(self.recs[i])
        T, pid, meta = self._rec_meta[i]
        if T <= 0:
            return f"{name}  (读取失败)"
        t_s = f"{T // 10}.{T % 10}s"
        opp = str(meta.get("opp", "?"))[:12]
        return f"{name}  {t_s}·pid{pid}·{opp}"

    def _apply_sel(self) -> None:
        """地图/场景/ckpt 下拉的当前选中 → self.map / self.scene / 决策来源。

        主档（dd_ckpt：玩家模式的对手 / 观战的玩家 0）与观战 P1（dd_ckpt_b）
        各解析成 (bot 名, ckpt 路径)：
        - 规则 bot 条目（astar/greedy/random）→ 对应 bot 名，ckpt 路径置空；
        - 模型条目 → 实际要加载的权重路径（start() 之前同步，否则永远加载
          __init__ 的默认档 —— 修复"选哪个模型都一样"）；
        - 分隔线 → 保持上一选择（滚轮扫过不误选）。
        """
        v = self.dd_map.value or ""
        if v.startswith("open"):
            self.map = "open"
        elif v.startswith("corridor"):
            self.map = "corridor"
        elif v.startswith("ring"):
            self.map = "ring"
        elif not v.startswith("─"):
            self.scene = v
        self._resolve_slot(0)
        self._resolve_slot(1)

    def _resolve_slot(self, slot: int) -> None:
        """把 slot 0（P0）/ 1（P1）的选中项解析成 人类 / bot / ckpt 来源。"""
        dd = self.dd_ckpt if slot == 0 else self.dd_ckpt_b
        kinds = self.ckpt_kinds0 if slot == 0 else self.ckpt_kinds1
        if not kinds or not (0 <= dd.sel < len(kinds)):
            return
        k, v = kinds[dd.sel]
        if k == "sep":
            return                      # 分隔线：保持上一选择
        if slot == 0:
            self.p0_human = (k == "human")
            self.opp_bot = v if k == "bot" else None
            self.ckpt = "" if k != "ckpt" else v
        else:
            self.p1_human = (k == "human")
            self.p1_bot = v if k == "bot" else None
            self.ckpt_b = "" if k != "ckpt" else v

    # ----- 事件 -----
    def handle(self, e, mouse: tuple[int, int] | None = None) -> bool:
        """返回 True = 开始对局。mouse 可选（默认实时取），测试可注入。"""
        if mouse is None:
            mouse = pygame.mouse.get_pos()
        if e.type == pygame.QUIT:          # 点窗口红叉 → 退出进程（修复"关不掉"）
            pygame.quit()
            sys.exit(0)
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE):
            pygame.quit()
            sys.exit(0)
        if e.type != pygame.MOUSEBUTTONDOWN:
            return False
        if e.button == 1 and self.REPLAY_BTN.collidepoint(mouse):
            # 回放按钮：选中录像 → 打开 Replay 播放器（pygame 复用，ESC 回来）
            self.dd_map.open = False
            self.dd_ckpt.open = False
            self.dd_ckpt_b.open = False
            self.dd_replay.open = False
            if self.recs:
                from .replay import Replay
                Replay(self.recs[self._replay_sel]).run()
            return False
        if e.button == 1 and self.START_BTN.collidepoint(mouse):
            # 开始对局优先：下拉开着也能直接点开始（先把列表收起）
            self.dd_map.open = False
            self.dd_ckpt.open = False
            self.dd_ckpt_b.open = False
            return True
        # 下拉：一次只展开一个；地图/场景改动立即同步 self.map/self.scene
        if self.dd_map.handle(e, mouse):
            self.dd_ckpt.open = False
            self.dd_ckpt_b.open = False
            self._apply_sel()
            return False
        if self.dd_ckpt.handle(e, mouse):
            self.dd_map.open = False
            self.dd_ckpt_b.open = False
            self._apply_sel()
            return False
        if self.dd_ckpt_b.handle(e, mouse):
            self.dd_map.open = False
            self.dd_ckpt.open = False
            self._apply_sel()
            return False
        if self.dd_replay.handle(e, mouse):
            self.dd_map.open = False
            self.dd_ckpt.open = False
            self.dd_ckpt_b.open = False
            if 0 <= self.dd_replay.sel < len(self.recs):
                self._replay_sel = self.dd_replay.sel
            return False
        if e.button != 1:
            return False
        # 初始属性步进器
        if self.map == "open":
            # 成长百分比：− 5% / ＋ 5%（0.50..1.00）
            minus, plus = pygame.Rect(420, 222, 28, 28), pygame.Rect(456, 222, 28, 28)
            if minus.collidepoint(mouse):
                self.open_pct = max(0.50, round(self.open_pct - 0.05, 2))
            elif plus.collidepoint(mouse):
                self.open_pct = min(1.00, round(self.open_pct + 0.05, 2))
        else:
            for x, attr, lo, hi, step in (
                    (170, "bombs_start", 1, 7, 1),
                    (370, "blast_start", 1, 7, 1),
                    (570, "speed_start", 1.00, 2.10, 0.05)):
                vr, minus, plus = stepper_rects(x + 58, 222)
                if minus.collidepoint(mouse):
                    setattr(self, attr,
                            max(lo, round(getattr(self, attr) - step, 2)))
                elif plus.collidepoint(mouse):
                    setattr(self, attr,
                            min(hi, round(getattr(self, attr) + step, 2)))
        if self.START_BTN.collidepoint(mouse):
            return True
        return False

    # ----- 绘制 -----
    def draw(self):
        s = self.screen
        s.fill((28, 26, 34))
        hover = pygame.mouse.get_pos()

        s.blit(self.font_t.render("QQT 对打 · 启动器", True, (240, 240, 240)),
               (24, 16))

        # 地图 / 场景（合并下拉：地图结构 + 全部场景皮肤）
        s.blit(self.font_l.render("地图 / 场景", True, (180, 180, 190)), (24, 109))
        self.dd_map.draw_closed(s)
        s.blit(self.font_s.render(f"当前：地图 {self.map} · 场景 {self.scene}",
                                  True, (200, 200, 210)), (170, 140))

        # 初始属性（地图相关）
        s.blit(self.font_l.render("初始属性", True, (180, 180, 190)), (24, 197))
        if self.map == "open":
            pct = int(round(self.open_pct * 100))
            nb, nb2 = math.ceil(7 * self.open_pct), math.ceil(7 * self.open_pct)
            spd = round(2.1 * self.open_pct, 2)
            s.blit(self.font_l.render(f"成长 {pct}%", True, (240, 240, 240)),
                   (170, 224))
            pygame.draw.rect(s, (40, 38, 48), (360, 222, 52, 28), border_radius=6)
            s.blit(self.font_l.render(f"{pct}%", True, (235, 235, 235)),
                   (368, 224))
            for x, ch in ((420, "−"), (456, "＋")):
                pygame.draw.rect(s, (60, 58, 72), (x, 222, 28, 28),
                                 border_radius=6)
                s.blit(self.font_l.render(ch, True, (230, 230, 230)),
                       (x + 6, 221))
            s.blit(self.font_s.render(
                f"→ 泡×{nb} 威×{nb2} 速×{spd:.2f}", True, (200, 200, 210)),
                (500, 226))
        else:
            for (x, label, val, fmt) in (
                    (170, "泡数", self.bombs_start, "{:d}"),
                    (370, "威力", self.blast_start, "{:d}"),
                    (570, "速度", self.speed_start, "{:.2f}")):
                s.blit(self.font_l.render(label, True, (240, 240, 240)),
                       (x, 226))
                vr, minus, plus = stepper_rects(x + 58, 222)
                pygame.draw.rect(s, (40, 38, 48), vr, border_radius=6)
                s.blit(self.font_l.render(fmt.format(val), True, (235, 235, 235)),
                       (vr.x + 8, 224))
                for b, ch in ((minus, "−"), (plus, "＋")):
                    pygame.draw.rect(s, (60, 58, 72), b, border_radius=6)
                    s.blit(self.font_l.render(ch, True, (230, 230, 230)),
                           (b.x + 5, 221))
            s.blit(self.font_s.render("踩宝箱继续成长到 7/7/2.10",
                                      True, (200, 200, 210)), (170, 260))

        # 玩家 0 / 玩家 1：两个下拉常驻，都可选 人类键盘 / 规则 bot / 模型
        s.blit(self.font_l.render("玩家 0 (P)", True, (180, 180, 190)), (24, 267))
        self.dd_ckpt.draw_closed(s)
        s.blit(self.font_l.render("玩家 1 (P1)", True, (180, 180, 190)), (24, 327))
        self.dd_ckpt_b.draw_closed(s)

        # 录像回放：列表下拉 + 播放按钮（hover 显示基础信息：双方/长度/成长）
        s.blit(self.font_l.render("录像回放", True, (180, 180, 190)), (24, 475))
        self.dd_replay.draw_closed(s)
        rep_lbl = os.path.basename(self.recs[self._replay_sel]) if self.recs else "（无录像）"
        pygame.draw.rect(s, (60, 130, 170), self.REPLAY_BTN, border_radius=8)
        pygame.draw.rect(s, (140, 200, 230), self.REPLAY_BTN, 2, border_radius=8)
        s.blit(self.font_l.render("▶ 回放", True, (240, 248, 255)),
               (self.REPLAY_BTN.x + 48, self.REPLAY_BTN.y + 5))
        # hover 基础信息 tooltip：下拉框上（闭合=当前选中项，展开=悬停项）
        if self.recs:
            idx = self._replay_sel
            if self.dd_replay.open:
                top, rects = self.dd_replay._item_rects()
                for i, r in enumerate(rects):
                    if r.collidepoint(hover):
                        idx = top + i
                        break
            if self.REPLAY_DD.collidepoint(hover) or self.dd_replay.open:
                T, pid, meta = self._rec_meta[idx]
                if T > 0:
                    opp = str(meta.get("opp", "?"))
                    m_ = str(meta.get("map", "?"))
                    b, z, sp = (meta.get("bombs", "?"), meta.get("blast", "?"),
                                meta.get("speed", "?"))
                    tip = (f"{T // 10}.{T % 10}s · 自己(pid{pid}) vs {opp} · "
                           f"地图 {m_} · 成长 {b}/{z}/{sp}")
                else:
                    tip = "（读取失败）"
                tb = pygame.Surface((520, 26), pygame.SRCALPHA)
                tb.fill((20, 20, 26, 210))
                s.blit(tb, (24, 636))
                s.blit(self.font_s.render(tip, True, (230, 230, 240)), (30, 640))

        # 开始按钮
        pygame.draw.rect(s, (60, 160, 90), self.START_BTN, border_radius=10)
        pygame.draw.rect(s, (140, 220, 160), self.START_BTN, 2, border_radius=10)
        s.blit(self.font_t.render("开始对局", True, (255, 255, 255)),
               (self.START_BTN.x + 90, self.START_BTN.y + 6))

        s.blit(self.font_s.render(
            "P0：WASD+空格  P1：方向键+回车  R重开  L录制开关  ESC返回  Q退出",
            True, (150, 150, 160)), (24, 664))

        # 展开的下拉条目最后画 → 永远在最上层（不被标签/步进器/开始按钮压住）
        dds = [self.dd_map, self.dd_ckpt, self.dd_ckpt_b, self.dd_replay]
        for dd in dds:
            dd.draw_items(s, hover)

    # ----- 启动 -----
    def start(self):
        self._apply_sel()
        # 人类键盘玩家（P0/P1 任一/都选人类）→ 传 human0/human1 标志；
        # 规则 bot → p0_bot（P0 的）/ opp_bot（P1 的，duel 语义）；
        # 模型 → ckpt（P0）/ ckpt_b（P1）。
        # **opp_bot 只表达 P1 的来源**（p1_bot 或 None）—— 旧版把 P0 的 opp_bot
        # 传给了 P1，导致"P0 选 idle + P1 选模型"时 P1 被错配成 idle bot（不动）。
        kwargs = dict(ckpt=self.ckpt, device="cpu", size=GRID,
                      map_mode=self.map, scene=self.scene, hz=10,
                      seed=int.from_bytes(os.urandom(4), "big"),
                      opp_bot=self.p1_bot, p0_bot=self.opp_bot,
                      ckpt_b=self.ckpt_b, human0=self.p0_human,
                      human1=self.p1_human,
                      recording=True)   # 人类轨迹录制默认开（L 键局内开关）→ BC 数据
        if self.map == "open":
            kwargs["open_growth_pct"] = self.open_pct
        else:
            kwargs["bombs_start"] = self.bombs_start
            kwargs["blast_start"] = self.blast_start
            kwargs["speed_start"] = self.speed_start
        names = [("人类" if self.p0_human else
                  (f"【{self.opp_bot}】" if self.opp_bot
                   else os.path.basename(self.ckpt or "?"))),
                 ("人类" if self.p1_human else
                  (f"【{self.p1_bot}】" if self.p1_bot
                   else os.path.basename(self.ckpt_b or "?")))]
        print(f"[launcher] P0={names[0]} vs P1={names[1]} "
              f"地图={self.map} 场景={self.scene}")
        outcome = run_game(**kwargs)
        # 对局内按 ESC → 回启动器（恢复菜单窗口继续跑）；Q / 关窗 → 退出程序
        if outcome == "menu":
            try:
                pygame.mixer.music.stop()
            except pygame.error:
                pass
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("QQT 对打 · 启动器")
            return
        pygame.quit()
        sys.exit(0)


def run_launcher() -> None:
    """图形菜单主循环。--smoke 时渲染数帧即退（无窗口冒烟用）。"""
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        pass
    launcher = Launcher()
    clock = pygame.time.Clock()
    frames = 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:      # 关窗兜底：直接干净退出
                pygame.quit()
                return
            if launcher.handle(e):
                # start() 内部：对局 ESC → "menu" → 重建菜单窗口后正常返回，
                # 循环继续回菜单；Q/关窗 → sys.exit(0)，走不到这里。
                # 不能 return —— 否则 ESC 回来闪一帧启动器就退出进程。
                launcher.start()
        launcher.draw()
        pygame.display.flip()
        clock.tick(60)
        frames += 1
        if "--smoke" in sys.argv and frames >= 6:
            print("[launcher] 冒烟完成：启动器页面渲染正常")
            pygame.quit()
            return


def main() -> None:
    run_launcher()


if __name__ == "__main__":
    main()
