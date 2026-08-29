#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 web 端资源: levels.json + elements.json + 贴图/背景/音乐
============================================================
从 levels_qqt/*.pt + qqt_element_manifest.csv + res/ 导出浏览器可加载的数据:

  web/assets/maps/levels.json     241 关 (wall/brick/overhead/pushable/ground/layers/
                                     spawns/initial_stats/crate_rate/music/bg/category)
  web/assets/maps/elements.json   元素属性表 {eid: {w,h,xo,yo,file}}
  web/assets/mapElem/<城市>/...   元件 PNG (res/mapElem 全集)
  web/assets/bg/<主题>.png         主题底图 (res/wall/<主题>/<bg>)
  web/assets/music/<名>.ogg        音乐 (241 关引用去重)

用法:
  ./.venv/bin/python export_web.py
"""

import csv
import json
import os
import shutil
from pathlib import Path

import torch

import qqfdimg2png as qf
try:
    from PIL import Image as PImage
except ImportError:
    raise SystemExit("缺少 pillow: 用 PYTHONPATH=<pylibs312目录> 运行, 或 pip install --target ./pylibs312 pillow")

ROOT = Path(__file__).resolve().parent
LEVELS = ROOT / "levels_qqt"
WEB = ROOT / "web" / "assets"
OUT_LEVELS = WEB / "maps"
OUT_ELEM = WEB / "mapElem"
OUT_BG = WEB / "bg"
OUT_MUSIC = WEB / "music"

# 地图文件 -> (分类, 排序键) : 二级菜单第一级=模式分类
def category_of(mode: str) -> str:
    if "比武" in mode:
        return "比武"
    if "夺宝" in mode:
        return "夺宝"
    if "推箱子" in mode:
        return "推箱子"
    if "空场景" in mode:
        return "空场景"
    return "普通竞技"


def load_elements(csv_path: Path) -> dict:
    elems = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            eid = int(r["element_id"])
            src = r["png_client"] or r["img_editor"]
            base = Path(src).stem          # elemN_stand / elemN_trigger / elemN_die
            elems[eid] = {
                "w": int(r["w"]), "h": int(r["h"]),
                "xo": int(r["xOffset"]), "yo": int(r["yOffset"]),
                "file": f"assets/mapElem/{r['city_name']}/{base}.png",
            }
    return elems


def repair_multicell_footprints(levels: list[dict], elems: dict) -> int:
    """多格元件占位补碰撞(与 scripts/repair_multicell_footprints.js 同规则)。

    超过 1×1 的元件不可通行半格：layer1 正值(元件原点)落在 wall/brick 上时，
    整个 w×h footprint 补成 wall/brick。生成管线内联执行 —— 重新生成
    levels.json 不再依赖手动跑修复脚本(漏跑会把多格元件占位回退成可通行)。
    返回修补格数。
    """
    changed = 0
    for lv in levels:
        w, h = lv["w"], lv["h"]
        layers = lv.get("layers") or []
        if len(layers) < 2 or not layers[1]:
            continue
        layer = layers[1]
        for r in range(h):
            for c in range(w):
                raw = layer[r * w + c]
                if not raw or raw < 0:
                    continue
                elem = elems.get(raw) or elems.get(str(raw))
                if not elem or (elem["w"] <= 1 and elem["h"] <= 1):
                    continue
                origin = r * w + c
                is_wall = bool(lv["wall"][origin])
                is_brick = bool(lv["brick"][origin])
                if not is_wall and not is_brick:
                    continue          # 可通行原点：结构不挡路，不补
                for dr in range(elem["h"]):
                    for dc in range(elem["w"]):
                        rr, cc = r + dr, c + dc
                        if rr >= h or cc >= w:
                            continue
                        i = rr * w + cc
                        if is_wall:
                            if not lv["wall"][i]:
                                lv["wall"][i] = 1
                                changed += 1
                        elif not lv["brick"][i]:
                            lv["brick"][i] = 1
                            changed += 1
    return changed


def ensure_die_png(eid: int, city_dir: str, out_root: Path) -> str | None:
    """把元件被炸毁的中间态图(_die)复制/转换为 PNG:
       res/mapElem/<城市>/elemN_die.png 优先, 其次 geno 的 gif/编辑器 img 转换。
    """
    rel_dir = OUT_ELEM / city_dir
    rel = f"assets/mapElem/{city_dir}/elem{eid % 1000}_die.png"
    out = OUT_ELEM / city_dir / f"elem{eid % 1000}_die.png"
    if out.exists():
        return rel
    # 1) res/mapElem 已有 die png
    p = ROOT / "res" / "mapElem" / city_dir / f"elem{eid % 1000}_die.png"
    if p.exists():
        shutil.copy2(p, out)
        return rel
    # 2) geno-extracted gif -> png 首帧
    gen = Path("/Users/a1-6/Documents/test/geno-extracted/QQTang5.2_Beta1Build1/data/object/mapElem")
    g = gen / city_dir / f"elem{eid % 1000}_die.gif"
    if g.exists():
        from PIL import Image
        Image.open(g).convert("RGBA").save(out)
        return rel
    # 3) 编辑器 img -> png (QQFDIMG)
    im = Path("/Users/a1-6/Documents/test/qqt/qqt_map_editor_fin-main/mapElem") / city_dir / f"elem{eid % 1000}_die.img"
    if im.exists():
        sys.path.insert(0, str(ROOT))
        import qqfdimg2png as qf
        from PIL import Image as PImage
        ver, nf, nd, xo, yo, wo, ho, frames = qf.parse_qqfdimg(im.read_bytes())
        w, h, px = frames[0]
        img = PImage.new("RGBA", (w, h)); img.putdata(px)
        img.save(out)
        return rel
    return None


def main():
    OUT_LEVELS.mkdir(parents=True, exist_ok=True)
    OUT_ELEM.mkdir(parents=True, exist_ok=True)
    OUT_BG.mkdir(parents=True, exist_ok=True)
    OUT_MUSIC.mkdir(parents=True, exist_ok=True)

    # 元素表: 只保留地图实际引用且有 PNG 的元素 (渲染按需取)
    elems = load_elements(ROOT / "qqt_element_manifest.csv")
    used_ids = set()
    for pt in LEVELS.glob("level_*.pt"):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        for layer in d["layers_raw"]:
            for row in layer:
                for v in row:
                    if v:
                        used_ids.add(abs(v))
    elems = {eid: e for eid, e in elems.items()
             if eid in used_ids and (OUT_ELEM / e["file"].split("mapElem/", 1)[1]).exists()}
    # 给有 _die 帧的元件补 "die" 字段（炸毁中间态）
    n_die = 0
    for eid, e in elems.items():
        d = ensure_die_png(eid, e["file"].split("/")[-2], OUT_ELEM)
        if d:
            e["die"] = d
            n_die += 1
    (OUT_LEVELS / "elements.json").write_text(
        json.dumps(elems, ensure_ascii=False), encoding="utf-8")
    print(f"有 _die 炸毁帧的元素: {n_die}")

    # 复制元件 PNG (res/mapElem)
    src_elem = ROOT / "res" / "mapElem"
    n_png = 0
    if src_elem.is_dir():
        for city in src_elem.iterdir():
            if not city.is_dir():
                continue
            dst = OUT_ELEM / city.name
            dst.mkdir(exist_ok=True)
            for f in city.glob("*.png"):
                shutil.copy2(f, dst / f.name)
                n_png += 1

    # 复制背景 + 音乐 (按关卡引用, 去重)
    bg_done, music_done = set(), set()
    levels = []
    for pt in sorted(LEVELS.glob("level_*.pt")):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        h, w = int(d["wall"].shape[0]), int(d["wall"].shape[1])
        flat = lambda t: [int(v) for v in (t.flatten().tolist() if hasattr(t, "flatten") else
                                            (v for row in t for v in row))]
        # 可推箱原点: pushable 为1 且 layers[1] 为正值(元素原点)的格
        def _push_origins(dd, ww, hh):
            out = []
            layers1 = dd.get("layers_raw", [None, None])[1]
            if layers1 is None: return out
            push_flat = dd["pushable"].flatten().tolist() if hasattr(dd["pushable"], "flatten") else dd["pushable"]
            for r in range(hh):
                for c in range(ww):
                    v = layers1[r][c] if hasattr(layers1[r], "__getitem__") else layers1[r * ww + c]
                    if push_flat[r * ww + c] and v and v > 0:
                        out.append((r, c, int(v)))
            return out
        bg = d.get("background", "")
        if bg:
            src = ROOT / bg
            if src.exists():
                name = f"{src.parent.name}.png"
                if name not in bg_done:
                    shutil.copy2(src, OUT_BG / name)
                    bg_done.add(name)
                bg_rel = f"assets/bg/{name}"
            else:
                bg_rel = ""
        else:
            bg_rel = ""
        music = d.get("music", "")
        music_rel = ""
        if music:
            src = ROOT / music
            if src.exists():
                # GitHub Pages 文件名大小写敏感：关卡数据沿用原版的 water.OGG/
                # M15.OGG，mac 本地大小写不敏感能播，线上 fetch 404 → 静默无声。
                # 引用一律取磁盘上的真实文件名，保证部署后命中。
                actual = next((f.name for f in src.parent.iterdir()
                               if f.name.lower() == src.name.lower()), None)
                name = actual or src.name
                if name not in music_done:
                    shutil.copy2(src, OUT_MUSIC / name)
                    music_done.add(name)
                music_rel = f"assets/music/{name}"
        layers_flat = [flat(t) for t in d["layers_raw"]]
        bush_flat = flat(d["bush"]) if "bush" in d else [0] * (w * h)
        # 老的 levels_qqt 导出可能保留了 6003 图层但 bush 布尔层为空；
        # 6003 是野外灌木，必须在共享 JSON 中恢复为可通行、可炸状态。
        if layers_flat and len(layers_flat[0]) == w * h:
            bush_flat = [int(bool(v) or abs(layers_flat[0][i]) == 6003)
                         for i, v in enumerate(bush_flat)]
        levels.append({
            "id": int(pt.stem.split("_")[1]),
            "source": d["source"], "name": d.get("map_name", ""),
            "mode": d["game_mode"],
            "category": d.get("category") or category_of(d["game_mode"]),
            "theme": Path(d.get("background", "")).parent.name if d.get("background") else "",
            "w": w, "h": h,
            "wall": flat(d["wall"]), "brick": flat(d["brick"]),
            "overhead": flat(d["overhead"]), "pushable": flat(d["pushable"]),
            "cover": flat(d["cover"]) if "cover" in d else [0] * (w * h),
            "bush": bush_flat,
            "ground": flat(d["ground"]),
            "layers": layers_flat,
            "spawns": [[int(y), int(x)] for y, x in d["spawns"]],
            "initial_stats": d["initial_stats"],
            "crate_rate": d["meta"]["crate_rate"],
            "bombs_max": d.get("bombs_max", 10),
            "blast_max": d.get("blast_max", 7),
            "speed_max": d.get("speed_max", 2.1),
            "crate_super_fraction": d["meta"].get("crate_super_fraction", 0),
            "crate_expect": d["meta"].get("crate_expect", {}),
            "crate_coverage": d["meta"]["crate_coverage"],
            "initial_crates": [[int(y), int(x)] for y, x in d.get("initial_crates", [])],
            "music": music_rel, "bg": bg_rel,
            # 可推箱(origin格): [r, c, w, h] — 原点 = layers[1] 正值的可推格
            # elems 键是 int；此前误用 str(eid) 永远查空 → 多格可推箱足迹
            # 被压成 1×1，非原点格全部可穿行。
            "push_boxes": [[int(r), int(c), int(elems.get(eid, {}).get("w", 1)),
                            int(elems.get(eid, {}).get("h", 1))]
                           for (r, c, eid) in _push_origins(d, w, h)],
        })
    n_fix = repair_multicell_footprints(levels, elems)
    print(f"多格元件占位补碰撞 {n_fix} 格")
    (OUT_LEVELS / "levels.json").write_text(
        json.dumps(levels, ensure_ascii=False), encoding="utf-8")

    # ---- 地图缩略图: 官方预览图 map/<源>.img -> web/assets/maps/thumb/<源>.png ----
    # 空场景用主题底图当缩略图
    THUMB = WEB / "maps" / "thumb"
    THUMB.mkdir(parents=True, exist_ok=True)
    SRC_MAP = Path("/Users/a1-6/Documents/test/QQTang_extract/map")
    n_thumb = 0
    for lv in levels:
        stem = lv["source"][:-4] if lv["source"].endswith(".map") else lv["source"]
        out = THUMB / (stem + ".png")
        if lv["source"] == "empty_scene":
            # 空场景: 用主题底图
            if lv["bg"]:
                bgp = WEB / lv["bg"].split("assets/", 1)[1]
                if bgp.exists():
                    im = PImage.open(bgp).convert("RGBA")
                    im.thumbnail((360, 360))
                    im.save(out)
                    lv["thumb"] = f"assets/maps/thumb/{stem}.png"
            continue
        img_p = SRC_MAP / (stem + ".img")
        if not img_p.exists() or out.exists():
            if out.exists():
                lv["thumb"] = f"assets/maps/thumb/{stem}.png"
            continue
        try:
            ver, nf, nd, xo, yo, wo, ho, frames = qf.parse_qqfdimg(img_p.read_bytes())
            w, h, px = frames[0]
            im = PImage.new("RGBA", (w, h)); im.putdata(px)
            im.save(out)
            lv["thumb"] = f"assets/maps/thumb/{stem}.png"
            n_thumb += 1
        except Exception as e:
            print(f"  [thumb fail] {lv['source']}: {e}")
    # 官方"随机地图"缩略图 (rand.img)
    rand_p = SRC_MAP / "rand.img"
    if rand_p.exists() and not (THUMB / "rand.png").exists():
        try:
            ver, nf, nd, xo, yo, wo, ho, frames = qf.parse_qqfdimg(rand_p.read_bytes())
            w, h, px = frames[0]
            im = PImage.new("RGBA", (w, h)); im.putdata(px)
            im.save(THUMB / "rand.png")
        except Exception as e:
            print(f"  [rand thumb fail]: {e}")
    (OUT_LEVELS / "levels.json").write_text(
        json.dumps(levels, ensure_ascii=False), encoding="utf-8")
    print(f"地图缩略图 {n_thumb} 张 -> {THUMB}")

    # 分类清单
    cats = {}
    for lv in levels:
        cats.setdefault(lv["category"], []).append(lv["id"])
    (OUT_LEVELS / "categories.json").write_text(
        json.dumps(cats, ensure_ascii=False), encoding="utf-8")

    print(f"导出 {len(levels)} 关 -> {OUT_LEVELS / 'levels.json'}")
    print(f"元素 {len(elems)} 个 -> elements.json")
    print(f"元件PNG {n_png} 张 -> {OUT_ELEM}")
    print(f"背景 {len(bg_done)} 张 -> {OUT_BG} ({sorted(bg_done)})")
    print(f"音乐 {len(music_done)} 首 -> {OUT_MUSIC} ({sorted(music_done)})")
    print("分类:", {k: len(v) for k, v in cats.items()})


if __name__ == "__main__":
    main()
