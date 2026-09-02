#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ堂关卡富化: 墙体统计 / 宝箱爆率 / 音乐 / 底图 / 元件PNG 编码
================================================================
对 levels_qqt/*.pt 做**兼容性扩展**(只加新键, 不动 wall/brick/meta/cfg):

新增键:
  meta.brick_count      可炸毁墙体数 W (每个被炸的墙 -> 一个宝箱)
  meta.crate_rate       墙体爆率 p = min(1, TARGET/W): 每个墙炸开变宝箱的概率
  meta.crate_needed     需求宝箱数(默认 60 = 2人x150% 单人满属性 = 300%)
  meta.crate_coverage   实际覆盖率 p*W/TARGET (墙体不够时 < 100%)
  meta.crate_attr       宝箱开出三属性的概率 + 每次增量:
                        {bombs: +1, blast: +1, speed: +0.15} 各 1/3
  meta.growth           {starts, maxs, per_player_units, target_units}
  music                音乐文件 (res/wall/<主题>/<music>.ogg, 参考 mapDesc.py)
  background           底图文件 (res/wall/<主题>/<bg>.png)
  sprites              {元素ID: "res/mapElem/<城市>/elem<N>_stand.png"} (本图用到的全部元件)
  sprite_count         元件PNG数量

宝箱数值设定(可 CLI 覆盖):
  泡数: 初始2 -> 上限10 (+1/次)       威力: 初始2 -> 上限8 (+1/次)
  速度: 初始1.2 -> 上限2.1 (+0.15/次, 即 (2.1-1.2)/0.15 = 6 次)
  单人满属性 = 8+6+6 = 20 单位;  地图目标 = 2人 x 150% = 300% = 60 单位
  墙体爆率 p = 60/W;  W<60 时 p=1(墙体不够, 无法改变单箱增量)

用法:
    python3 qqt_level_enrich.py --levels qqt-gpu-sim-copy/levels_qqt
        [--bombs-max 10 --blast-max 8 --speed-max 2.1 --speed-start 1.2 ...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch

# ---------- 模拟器数值上限(与 sim/config.py 对齐, 可 CLI 覆盖) ----------
DEFAULTS = dict(
    bombs_start=2, bombs_max=10,
    blast_start=2, blast_max=8,
    speed_start=1.3, speed_max=2.1, speed_step=0.8 / 7,  # 7档: (2.1-1.3)/7=0.1143
    target_mult=3.0,          # 300% 单人满属性
)

SIM_ROOT = "qqt-gpu-sim-copy"
MAPDESC_PATH = "mapDesc.py"
ELEM_PNG_SRC = "geno-extracted/QQTang5.2_Beta1Build1/data/object/mapElem"
ELEM_IMG_SRC = "qqt/qqt_map_editor_fin-main/mapElem"

# QQ堂城市 -> 模拟器 res/wall 主题目录
THEME_DIR = {    1: "沙漠", 2: "雪地", 3: "中国城", 4: "矿洞", 5: "水面", 6: "野外",
    7: "比武", 8: "抢包子", 9: "功夫", 10: "夺宝", 11: "比武",
    12: "英雄", 13: "比武", 14: "比武", 15: "比武", 16: "比武",
    17: "比武", 18: "比武", 19: "比武",
}
# 主题 -> 底图文件 (模拟器 res/wall/<主题>/ 里的背景图)
THEME_BG = {
    "沙漠": "shamo.png", "雪地": "xd.png", "中国城": "town.png",
    "矿洞": "kd.png", "水面": "sm.png", "野外": "yw.PNG",
    "抢包子": "qbz.png", "比武": "bw.png", "夺宝": "db.png",
    "功夫": "gf.PNG", "英雄": "",
}

# ---- 地图级覆盖（个别图按实际美术修正主题/音乐/背景/分类）----
MAP_OVERRIDES = {
    # 比赛02：mode 是比武，但地面/墙体全是沙漠元素 -> 归入沙漠
    "contest02_8.map": {
        "theme": "沙漠", "music": "res/wall/沙漠/desert.ogg",
        "background": "res/wall/沙漠/shamo.png", "category": "沙漠",
    },
}


def load_mapdesc(path: str) -> dict:
    """mapDesc.py (GBK): mapfile -> {name,id,theme,music,level,players}"""
    raw = Path(path).read_bytes().decode("gbk", errors="replace")
    out = {}
    for m in re.finditer(
        r"\((\d+),\s*'([^']*)',\s*'([^']*)',\s*(\d+),\s*(\d+),\s*(\d+),"
        r"\s*'([^']*)',\s*'([^']*)',\s*'([^']*)',\s*(\d+)",
        raw,
    ):
        id_, theme, name, w, h, pl, mf, imgf, music, lvl = m.groups()
        out[mf.strip()] = {"id": int(id_), "theme": theme, "name": name,
                           "players": int(pl), "img": imgf.strip(),
                           "music": music.strip(), "level": int(lvl)}
    return out


def theme_of(eid: int) -> str:
    return THEME_DIR.get(eid // 1000, "比武")


# mapDesc 主题名(与 mapElem 城市目录一致) -> 城市号
THEME_CITY = {v: k for k, v in {
    1: "desert", 2: "snow", 3: "town", 4: "mine", 5: "water", 6: "field",
    7: "bomb", 8: "bun", 9: "pig", 10: "treasure", 11: "match",
    12: "sculpture", 13: "machine", 14: "box", 15: "practice",
    16: "exploration", 17: "common", 18: "pve", 19: "tank",
}.items()}


def theme_from_md(md: dict, d: dict) -> str:
    """优先用 mapDesc 的 theme 字段 -> 模拟器主题目录; 兜底取地图第一个城市"""
    t = md.get("theme", "")
    city = THEME_CITY.get(t)
    if city:
        return THEME_DIR.get(city, "比武")
    return theme_of(_first_city(d))


def resolve_music(mapdesc_music: str, theme: str) -> tuple[str, bool]:
    """把 mapDesc 的音乐名对到模拟器 res/wall/<主题>/*.ogg; 找不到则原样记录"""
    if not mapdesc_music:
        return "", False
    d = os.path.join(SIM_ROOT, "res", "wall", theme)
    if os.path.isdir(d):
        low = mapdesc_music.lower()
        for f in os.listdir(d):
            if f.lower() == low:
                return os.path.join("res", "wall", theme, f), True
    return os.path.join("res", "wall", theme, mapdesc_music), False


def ensure_sprite_png(eid: int, out_root: str) -> str | None:
    """把元件 eid 的原件图复制为 res/mapElem/<城市>/elem<N>_stand.png, 返回相对路径"""
    city = eid // 1000
    n = eid % 1000
    city_dir = {1: "desert", 2: "snow", 3: "town", 4: "mine", 5: "water",
                6: "field", 7: "bomb", 8: "bun", 9: "pig", 10: "treasure",
                11: "match", 12: "sculpture", 13: "machine", 14: "box",
                15: "practice", 16: "exploration", 17: "common",
                18: "pve", 19: "tank"}.get(city)
    if city_dir is None:
        return None
    rel_dir = os.path.join("res", "mapElem", city_dir)
    out_dir = os.path.join(out_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    rel = os.path.join(rel_dir, f"elem{n}_stand.png")
    out = os.path.join(out_root, rel)
    if os.path.exists(out):
        return rel
    # 候选: 客户端 png/gif -> 编辑器 img
    for stem in (f"elem{n}_stand", f"elem{n}_trigger", f"elem{n}_die"):
        for ext in (".png", ".gif"):
            src = os.path.join(ELEM_PNG_SRC, city_dir, stem + ext)
            if os.path.exists(src):
                if ext == ".png":
                    shutil.copy2(src, out)
                else:  # gif -> png 首帧
                    from PIL import Image
                    Image.open(src).convert("RGBA").save(out)
                return rel
        src_img = os.path.join(ELEM_IMG_SRC, city_dir, stem + ".img")
        if os.path.exists(src_img):
            import qqfdimg2png as qf
            ver, nf, nd, xo, yo, wo, ho, frames = qf.parse_qqfdimg(Path(src_img).read_bytes())
            from PIL import Image as PImage
            w, h, px = frames[0]
            im = PImage.new("RGBA", (w, h)); im.putdata(px)
            im.save(out)
            return rel
    return None


def compute_crate(meta: dict, cfg: dict) -> dict:
    """宝箱爆率与三属性分配。

    普通宝箱 +1 档；超级宝箱(超级威力/泡泡/速度 三个整体) = 普通爆率的 10% 且 +4 档。
    保持 300% 总成长: p_n*1 + 0.1*p_n*4 = 1.4*p_n = target/W
        -> 普通爆率 p_n = target/(1.4W), 超级爆率 p_s = 0.1*p_n
        -> 总爆率 crate_rate = 1.1*p_n = 1.1*target/(1.4W) = 11*target/(14W)
    """
    W = meta["brick_count"]
    per_player = ((cfg["bombs_max"] - cfg["bombs_start"])
                  + (cfg["blast_max"] - cfg["blast_start"])
                  + round((cfg["speed_max"] - cfg["speed_start"]) / cfg["speed_step"]))
    target = round(cfg["target_mult"] * per_player)
    if W:
        total_rate = min(1.0, 11 * target / (14 * W))   # 总宝箱爆率(普通+超级)
        super_frac = 1 / 11                              # 超级占宝箱比例 = 10%/1.1
        if total_rate >= 1.0:                            # 砖不足: 每格都出宝箱
            pn, ps = 10 / 11, 1 / 11
        else:
            pn, ps = total_rate * 10 / 11, total_rate / 11
        super_expect = ps * W
        normal_expect = pn * W
        empty_expect = max(0.0, W - (pn + ps) * W)
        coverage = min(1.0, (pn + 4 * ps) * W / target)
    else:
        total_rate, super_frac = 0.0, 0.0
        super_expect = normal_expect = empty_expect = 0.0
        coverage = 0.0
    return {
        "brick_count": W,
        "crate_rate": round(total_rate, 6),          # 总宝箱(普通+超级)爆率
        "crate_super_fraction": round(super_frac, 6),  # 宝箱中超级占比 (1/11)
        "crate_needed": target,
        "crate_coverage": round(coverage, 6),
        "crate_expect": {
            "super": round(super_expect, 3),
            "normal": round(normal_expect, 3),
            "empty": round(empty_expect, 3),
        },
        "crate_attr": {
            "bombs": {"prob": 1 / 3, "add": 1, "super_add": 4},
            "blast": {"prob": 1 / 3, "add": 1, "super_add": 4},
            "speed": {"prob": 1 / 3, "add": cfg["speed_step"],
                      "super_add": 4 * cfg["speed_step"]},
        },
        "growth": {
            "bombs": [cfg["bombs_start"], cfg["bombs_max"]],
            "blast": [cfg["blast_start"], cfg["blast_max"]],
            "speed": [cfg["speed_start"], cfg["speed_max"], cfg["speed_step"]],
            "per_player_units": per_player,
            "target_units": target,
            "target_desc": f"{cfg['target_mult']:.1f}x 单人满属性 "
                           f"(= 2人 x {cfg['target_mult']/2:.1%} 满属性)",
        },
    }


def enrich(path: Path, mapdesc: dict, cfg: dict) -> dict | None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    src = d["source"]
    md = mapdesc.get(src, {})
    theme = theme_from_md(md, d)

    # 比武图实际可操控空间更小: 起点为 3，但威力上限统一为 8，配置标准化进地图文件
    mcfg = dict(cfg)
    if "比武" in d.get("game_mode", ""):
        mcfg["bombs_start"], mcfg["bombs_max"] = 3, 7
        mcfg["blast_start"], mcfg["blast_max"] = 3, 8
    d["bombs_max"] = mcfg["bombs_max"]
    d["blast_max"] = mcfg["blast_max"]

    crate = compute_crate(d["meta"], mcfg)
    d["meta"].update(crate)

    music, music_ok = resolve_music(md.get("music", ""), theme)
    if not music:  # mapDesc 无音乐条目 -> 用主题目录里的默认音乐
        td = os.path.join(SIM_ROOT, "res", "wall", theme)
        if os.path.isdir(td):
            for f in sorted(os.listdir(td)):
                if f.lower().endswith(".ogg"):
                    music, music_ok = os.path.join("res", "wall", theme, f), True
                    break
    bg = THEME_BG.get(theme, "")
    d["music"] = music if music_ok else (music or "")
    d["music_in_sim"] = music_ok
    d["background"] = os.path.join("res", "wall", theme, bg) if bg else ""

    # 地图级覆盖：个别图按实际美术修正主题/音乐/背景/分类
    ov = MAP_OVERRIDES.get(src, {})
    if ov.get("theme"):
        theme = ov["theme"]
        bg = THEME_BG.get(theme, "")
        d["background"] = os.path.join("res", "wall", theme, bg) if bg else d["background"]
    if ov.get("music"):
        d["music"] = ov["music"]
        d["music_in_sim"] = True
    if ov.get("category"):
        d["category"] = ov["category"]

    # 元件: 本图用到的全部元素 -> PNG 相对路径
    sprites = {}
    for layer in d["layers_raw"]:
        for row in layer:
            for v in row:
                eid = abs(v)
                if eid and eid not in sprites:
                    rel = ensure_sprite_png(eid, SIM_ROOT)
                    if rel:
                        sprites[eid] = rel
    d["sprites"] = {str(k): v for k, v in sorted(sprites.items())}
    d["sprite_count"] = len(sprites)
    torch.save(d, path)
    return d


def _first_city(d) -> int:
    for layer in d["layers_raw"]:
        for row in layer:
            for v in row:
                if v:
                    return abs(v) // 1000
    return 3


def add_initial_stats(d: dict, is_empty: bool = False) -> None:
    """按规则写初始参数 (兼容扩展键 initial_stats / initial_stats_reason):
       - 空场景: 参考代码 open_growth_* = 泡3/威3/速0.84
       - 比武类 或 砖不足300%(覆盖率<100%): 泡3/威3/速1.2 (按比武配置)
       - 其他: 泡2/威2/速1.2
    """
    if is_empty or d["meta"].get("kind") == "qqt_empty_scene":
        d["initial_stats"] = {"bombs": 8, "blast": 6, "speed": 1.68}
        d["initial_stats_reason"] = ("空场景(open): 源码注释'上限的80%' -> "
                                     "round(0.8x10)=8泡 / round(0.8x7)=6威 / 0.8x2.1=1.68速 "
                                     "(config 实际 learner 起点为 3/3/0.84, 注释矛盾)")
    elif "比武" in d.get("game_mode", "") or d["meta"].get("crate_coverage", 0) < 1.0:
        d["initial_stats"] = {"bombs": 3, "blast": 3, "speed": 1.3}
        d["initial_stats_reason"] = ("比武类" if "比武" in d.get("game_mode", "")
                                     else "砖不足300%按比武配置")
    else:
        d["initial_stats"] = {"bombs": 2, "blast": 2, "speed": 1.3}
        d["initial_stats_reason"] = "普通"


def make_empty_level(cfg: dict) -> dict:
    """空场景(open) 15x13: 无墙无砖, 全地板; 初始参数 = 上限的 80%
    (torch_sim.py reset_ 注释"open 关成长初始 = 上限的 80%";
     实际 config learner 起点 open_growth_*=3/3/0.84, 注释自相矛盾,
     按用户定 80%: 泡 round(0.8*10)=8, 威 round(0.8*7)=6, 速 0.8*2.1=1.68)
    + 中心十字宝箱布局(与 _open_geometry/_place_open_cross_crates 一致):
      行带 {cy-1,cy} 全宽 ∪ 列带 {cx-1,cx} 全高, 扣除出生点及四邻。
    """
    h, w = 13, 15
    cy, cx = (h - 1) // 2, (w - 1) // 2          # 6, 7
    spawns = [[6, 5], [6, 9]]                    # open 整宽中线均分 (参考 _open_spawns)
    excl = set()
    for (sr, sc) in spawns:
        excl.add((sr, sc))
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < h and 0 <= nc < w:
                excl.add((nr, nc))
    cross = set()
    for cc in range(w):
        for rr in (cy - 1, cy):
            if (rr, cc) not in excl:
                cross.add((rr, cc))
    for rr in range(h):
        for cc in (cx - 1, cx):
            if (rr, cc) not in excl:
                cross.add((rr, cc))
    cross = sorted(cross)

    zeros = lambda: torch.zeros((h, w), dtype=torch.bool)
    d = {
        "wall": zeros(), "brick": zeros(), "pushable": zeros(),
        "overhead": zeros(), "cover": zeros(), "bush": zeros(),
        "ground": torch.ones((h, w), dtype=torch.bool),
        "spawns": spawns,
        "items": [],
        "game_mode": "open(空场景)",
        "map_name": "空场景",
        "qqt_id": 0,
        "source": "empty_scene",
        "version": 0,
        "speed_max": 2.3,               # 空场景最大速度单独设 2.3
        "layers_raw": [[[0] * w for _ in range(h)] for _ in range(3)],
        "music": "res/wall/中国城/town.ogg",
        "music_in_sim": True,
        "background": "res/wall/中国城/town.png",
        "sprites": {}, "sprite_count": 0,
        "initial_crates": [list(p) for p in cross],   # 开局中心十字宝箱 (地上直接捡)
        "crate_cross": {"cy": cy, "cx": cx, "rows": [cy - 1, cy], "cols": [cx - 1, cx]},
        "meta": {
            "kind": "qqt_empty_scene",
            "brick_count": 0, "crate_rate": 0.0, "crate_needed": cfg["crate_needed"],
            "crate_coverage": 0.0,
            "crate_attr": {
                "bombs": {"prob": 1 / 3, "add": 1},
                "blast": {"prob": 1 / 3, "add": 1},
                "speed": {"prob": 1 / 3, "add": cfg["speed_step"]},
            },
            "growth": cfg["growth"],
        },
        "cfg": {"height": h, "width": w, "n_players": 2,
                "corridor_width": 0, "top_wall_rows": 0},
    }
    add_initial_stats(d, is_empty=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="qqt-gpu-sim-copy/levels_qqt")
    ap.add_argument("--add-empty", action="store_true",
                    help="生成空场景关卡 level_XXXX.pt")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float if isinstance(v, float) else int, default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    cfg["crate_needed"] = round(cfg["target_mult"] * (
        (cfg["bombs_max"] - cfg["bombs_start"])
        + (cfg["blast_max"] - cfg["blast_start"])
        + round((cfg["speed_max"] - cfg["speed_start"]) / cfg["speed_step"])))
    cfg["growth"] = {
        "bombs": [cfg["bombs_start"], cfg["bombs_max"]],
        "blast": [cfg["blast_start"], cfg["blast_max"]],
        "speed": [cfg["speed_start"], cfg["speed_max"], cfg["speed_step"]],
        "per_player_units": ((cfg["bombs_max"] - cfg["bombs_start"])
                             + (cfg["blast_max"] - cfg["blast_start"])
                             + round((cfg["speed_max"] - cfg["speed_start"])
                                     / cfg["speed_step"])),
        "target_units": cfg["crate_needed"],
        "target_desc": f"{cfg['target_mult']:.1f}x 单人满属性",
    }

    mapdesc = load_mapdesc(MAPDESC_PATH)
    files = sorted(glob.glob(os.path.join(args.levels, "level_*.pt")))
    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d["meta"].get("kind") == "qqt_empty_scene":
            add_initial_stats(d)          # 空场景: 保持其音乐/底图, 只确保初始参数
            d["speed_max"] = 2.3          # 空场景最大速度 = 2.3
            d["music"] = "res/wall/中国城/town.ogg"
            d["music_in_sim"] = True
            d["background"] = "res/wall/中国城/town.png"
        else:
            d = enrich(Path(f), mapdesc, cfg)
            add_initial_stats(d)
        torch.save(d, f)   # 初始参数写回磁盘
        rows.append(d)
    if args.add_empty:
        has_empty = any(d["meta"].get("kind") == "qqt_empty_scene" for d in rows)
        if not has_empty:
            nxt = len(files)
            d = make_empty_level(cfg)
            torch.save(d, os.path.join(args.levels, f"level_{nxt:04d}.pt"))
            rows.append(d)
            print(f"已生成空场景关卡: level_{nxt:04d}.pt")
        else:
            print("空场景已存在, 跳过生成")
    print(f"已富化 {len(rows)} 个关卡 -> {args.levels}/ (仅新增键, 兼容)")

    # 更新清单
    csv_path = os.path.join(args.levels, "qqt_levels_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "map_name", "game_mode", "qqt_id", "h", "w",
                    "n_players", "wall", "brick", "overhead", "pushable",
                    "crate_rate", "crate_coverage", "music", "background",
                    "sprites", "init_bombs", "init_blast", "init_speed"])
        for i, d in enumerate(rows):
            src = d["source"]
            is_ = d.get("initial_stats", {})
            w.writerow([f"level_{i:04d}.pt", src, d.get("map_name", ""),
                        d["game_mode"], d.get("qqt_id", ""),
                        d["cfg"]["height"], d["cfg"]["width"],
                        d["cfg"]["n_players"], int(d["wall"].sum()),
                        int(d["brick"].sum()), int(d["overhead"].sum()),
                        int(d["pushable"].sum()),
                        d["meta"]["crate_rate"], d["meta"]["crate_coverage"],
                        d.get("music", ""), d.get("background", ""),
                        d["sprite_count"],
                        is_.get("bombs", ""), is_.get("blast", ""),
                        is_.get("speed", "")])

    # 统计: 爆率范围 / 墙体不够的地图
    low = [d for d in rows if d["meta"]["crate_coverage"] < 1.0]
    rates = sorted(d["meta"]["crate_rate"] for d in rows)
    print(f"爆率范围: {rates[0]:.4f} ~ {rates[-1]:.4f}")
    print(f"墙体不够(覆盖率<100%): {len(low)} 张")
    for d in low:
        print(f"  {d['source']}: 砖 {d['meta']['brick_count']} < 60, "
              f"覆盖率 {d['meta']['crate_coverage']:.1%}")


if __name__ == "__main__":
    main()
