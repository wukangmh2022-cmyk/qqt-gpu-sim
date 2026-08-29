#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ堂 原件图(元素贴图) 映射表 + 真实地图合成渲染
================================================
1) 生成 qqt_element_manifest.csv:
   全部 773 个地图元素 -> 原件图文件 (mapElem/<城市>/elem<N>_stand.png|.img)
   + 元素属性(w/h/偏移/生命/special/属性位), 两种素材源的存在标记。

2) 用原件图合成真实地图 (编辑器官方绘制公式):
   draw_x = x*40 - elem.xOffset,  draw_y = y*40 - elem.yOffset   (一格=40px)
   绘制顺序: 地板L2 -> 网格L1 -> 顶层L0(拱门/装饰最后盖在最上)。

用法:
    PYTHONPATH=./pylibs python3 qqt_render_sprites.py --manifest
    PYTHONPATH=./pylibs python3 qqt_render_sprites.py --map QQTang_extract/map/town10_8.map --out qqt_composite
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
from pathlib import Path

from PIL import Image

import qqt_map_parser as qp

CELL = 40  # QQ堂一格 = 40x40 px (编辑器 kGridPixels)

PROP_PATH = "qqt/qqt_map_editor_fin-main/mapElem.prop"
EDITOR_MAPELEM = "qqt/qqt_map_editor_fin-main/mapElem"          # .img 源
CLIENT_MAPELEM = "geno-extracted/QQTang5.2_Beta1Build1/data/object/mapElem"  # .png/.gif 源

CITY_DIR = {
    1: "desert", 2: "snow", 3: "town", 4: "mine", 5: "water", 6: "field",
    7: "bomb", 8: "bun", 9: "pig", 10: "treasure", 11: "match",
    12: "sculpture", 13: "machine", 14: "box", 15: "practice",
    16: "exploration", 17: "common", 18: "pve", 19: "tank",
}


def load_prop(path: str) -> dict[int, dict]:
    data = Path(path).read_bytes()
    ver, n = struct.unpack_from("<ii", data, 0)
    off = 8
    elems = {}
    for _ in range(n):
        (eid,) = struct.unpack_from("<i", data, off); off += 4
        w, h, xo, yo = struct.unpack_from("<hhhh", data, off); off += 8
        life, level, special = struct.unpack_from("<iii", data, off); off += 12
        attrs = list(struct.unpack_from(f"<{w * h}I", data, off)); off += 4 * w * h
        elems[eid] = {"w": w, "h": h, "xo": xo, "yo": yo,
                      "life": life, "special": special, "attrs": attrs}
    return elems


def sprite_path(eid: int, base: str) -> str | None:
    """元素ID -> 原件图文件路径 (stand优先, 其次 trigger/die; png 优先, 其次 gif/img)"""
    city = eid // 1000
    n = eid % 1000
    d = CITY_DIR.get(city)
    if d is None:
        return None
    d = os.path.join(base, d)
    for stem in (f"elem{n}_stand", f"elem{n}_trigger", f"elem{n}_die"):
        for ext in (".png", ".gif", ".img"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def build_manifest(elems: dict, out_csv: str) -> None:
    rows = []
    for eid in sorted(elems):
        e = elems[eid]
        img = sprite_path(eid, EDITOR_MAPELEM)
        png = sprite_path(eid, CLIENT_MAPELEM)
        rows.append({
            "element_id": eid,
            "city": eid // 1000,
            "city_name": CITY_DIR.get(eid // 1000, ""),
            "elem_no": eid % 1000,
            "w": e["w"], "h": e["h"],
            "xOffset": e["xo"], "yOffset": e["yo"],
            "life": e["life"], "special": hex(e["special"]),
            "attrs": ",".join(str(a) for a in sorted(set(e["attrs"]))),
            "img_editor": img or "",
            "png_client": png or "",
        })
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    have = sum(1 for r in rows if r["png_client"] or r["img_editor"])
    print(f"元素总数 {len(rows)}, 找到原件图 {have} 个 -> {out_csv}")


def load_sprite(eid: int) -> Image.Image | None:
    """加载原件图(RGBA), png/gif/img 均可"""
    p = sprite_path(eid, CLIENT_MAPELEM) or sprite_path(eid, EDITOR_MAPELEM)
    if not p:
        return None
    try:
        if p.endswith(".img"):
            # QQFDIMG 转 RGBA (调用 qqfdimg2png 的解析)
            import qqfdimg2png as qf
            ver, nf, nd, xo, yo, wo, ho, frames = qf.parse_qqfdimg(Path(p).read_bytes())
            w, h, px = frames[0]
            img = Image.new("RGBA", (w, h))
            img.putdata(px)
            return img
        im = Image.open(p)
        im.load()
        return im.convert("RGBA")
    except Exception:
        return None


def composite_map(src: str, elems: dict, out_dir: Path) -> Path | None:
    m = qp.parse_map(Path(src).read_bytes())
    h, w = m.height, m.width
    canvas = Image.new("RGBA", (w * CELL, h * CELL), (0, 0, 0, 0))
    drawn = 0
    missing = set()
    for k in (2, 1, 0):          # 地板 -> 网格 -> 顶层
        for y in range(h):
            for x in range(w):
                v = m.layers[k][y][x]
                if v <= 0:       # 只画原点格(多格元素负值延续不重复画)
                    continue
                eid = abs(v)
                e = elems.get(eid)
                if e is None:
                    missing.add(eid)
                    continue
                img = load_sprite(eid)
                if img is None:
                    missing.add(eid)
                    continue
                # 编辑器公式: draw at (x*40 - xo, y*40 - yo)
                dx = x * CELL - e["xo"]
                dy = y * CELL - e["yo"]
                canvas.alpha_composite(img, (dx, dy))
                drawn += 1
    out = out_dir / (Path(src).stem + "_sprite.png")
    canvas.convert("RGB").save(out)
    if missing:
        print(f"  {Path(src).name}: 缺贴图元素 {sorted(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="生成元素->原件图映射表")
    ap.add_argument("--map", action="append", help="用原件图合成某张 .map")
    ap.add_argument("--out", default="qqt_composite")
    args = ap.parse_args()

    elems = load_prop(PROP_PATH)
    if args.manifest:
        build_manifest(elems, "qqt_element_manifest.csv")

    if args.map:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in args.map:
            p = composite_map(src, elems, out_dir)
            if p:
                print(f"  合成完成: {p}")


if __name__ == "__main__":
    main()
