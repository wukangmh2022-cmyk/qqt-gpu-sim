#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ堂原版地图 -> 本项目(QQT GPU Sim)关卡格式转换器
=================================================
把 QQ堂原版客户端 .map 二进制转成 sim/levelgen.py 的 level_XXXX.pt 格式，
并**兼容性扩展**新字段，保留原版特有的地图信息：

原有字段(加载器兼容, 一个不动):
    wall    (H,W) bool  永久墙(不可通行, 不可炸)
    brick   (H,W) bool  可炸墙(不可通行, 可被爆炸摧毁)
    meta    dict        元信息
    cfg     dict        高度/宽度/人数 等 (key 与原 save_levels 一致)

新增字段(可选, 旧代码读不到就忽略):
    overhead (H,W) bool  头顶可通行装饰: 能走但画在人物头上的图块
                         (中国城红色拱门 3005、沙漠装饰 1008/1010、
                          水面荷叶、野外花草、功夫/夺宝/英雄装饰、
                          推箱子道具格、包子铺/基地内部可走格 等)
    pushable (H,W) bool  可推动的箱子 (推箱子模式: 元素 special 带 0x2 位,
                         如 14100/14105/14110/14111/14112/14113/14115)
    ground   (H,W) bool  地板层图块(恒可通行)
    spawns   [[y,x],..]  出生点 (队伍0+队伍1, 来自地图数据)
    items    [{id,min,max,rate},..]  道具掉落表
    game_mode str        游戏模式名 (普通竞技/比武/夺宝/推箱子/...)
    qqt_id   int         原版地图ID (mapDesc.py)
    map_name str         官方中文名 (如 沙漠01/中国城01/比武01)
    source   str         源 .map 文件名
    version  int         QQ堂地图文件版本 (3/4)
    layers_raw [[[..],..],..]  原始三层元素ID(完整保真, 可追溯)

通行性判定(由原版数据实证):
    - 地板层(L2) 恒可通行;
    - 顶层(L0)/网格层(L1) 按该格元素属性: attr==0 阻挡, 非0 可通行
      (属性来自 mapElem.prop, 多格元素按负值延续定位原点取逐格属性);
    - 可炸(brick) = 地图自带的"道具生成点"(可炸毁元素点, 最权威);
    - 永久墙(wall) = 阻挡 且 不可炸。

用法:
    python3 qqt_to_levels.py --in qqt_selected --out ../qqt-gpu-sim-copy/levels_qqt
    python3 qqt_to_levels.py --in qqt_selected --in QQTang_extract/map/box01_8.map ...
    python3 qqt_to_levels.py --ascii levels_qqt/level_0000.pt   # 文本渲染某关
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import struct
import sys
from pathlib import Path

import torch

import qqt_map_parser as qp

# ---------------------------------------------------------------- 常量

PROP_PATH = "qqt/qqt_map_editor_fin-main/mapElem.prop"
MAPDESC_PATH = "QQTang_extract/map/mapDesc.py"
GAME_MODES = qp.GAME_MODES  # 复用

# 推箱子模式的"可推动箱子"元素判定: city==14 且 special 带 0x2 位 (实证)
PUSHABLE_SPECIAL_BIT = 0x2

# ---- 可进入的遮挡元件（可通行，角色走进后隐藏 + 果冻动画）----
# cover: 房子类 —— 可通行、**不可炸**（当前无；沙漠城堡 1007 是实心障碍，不是可进入的房子）
# bush : 灌木类 —— 可通行、**可炸毁**（如野外绿色躲猫猫灌木 elem3 = 6003）
COVER_ELEMENTS = set()
BUSH_ELEMENTS = {6003}                  # 野外绿色躲猫猫灌木


# ---------------------------------------------------------------- 元素属性表

def load_elem_prop(path: str) -> dict[int, dict]:
    """解析 mapElem.prop: id -> {w,h,life,special,attrs[]}"""
    data = Path(path).read_bytes()
    ver, n = struct.unpack_from("<ii", data, 0)
    off = 8
    elems = {}
    for _ in range(n):
        (eid,) = struct.unpack_from("<i", data, off); off += 4
        w, h, xo, yo = struct.unpack_from("<hhhh", data, off); off += 8
        life, level, special = struct.unpack_from("<iii", data, off); off += 12
        attrs = list(struct.unpack_from(f"<{w * h}I", data, off)); off += 4 * w * h
        elems[eid] = {"w": w, "h": h, "life": life, "special": special, "attrs": attrs}
    assert off == len(data), f"mapElem.prop 解析偏移 {off} != 文件长 {len(data)}"
    return elems


def load_map_names(path: str) -> dict[str, tuple[str, int]]:
    """解析 mapDesc.py (GBK): mapfile -> (官方名, ID)"""
    raw = Path(path).read_bytes().decode("gbk", errors="replace")
    names = {}
    for m in re.finditer(
        r"\((\d+),\s*'([^']*)',\s*'([^']*)',\s*\d+,\s*\d+,\s*\d+,\s*'([^']*)',\s*'([^']*)'",
        raw,
    ):
        id_, theme, name, mf, _imgf = m.groups()
        names[mf.strip()] = (name.replace("\\n", "/"), int(id_))
    return names


# ---------------------------------------------------------------- 格子分类

def _resolve_origin(m: qp.QQTMap, layer, y, x, elems) -> tuple[int, int, dict]:
    """多格元素负值延续 -> 返回 (origin_y, origin_x, 元素属性)。"""
    v = m.layers[layer][y][x]
    eid = abs(v)
    e = elems.get(eid)
    if v > 0 or e is None:
        return y, x, e
    # 负值: 在 w×h 邻域内找正号原点
    for i in range(e["h"]):
        for j in range(e["w"]):
            oy, ox = y - i, x - j
            if 0 <= oy < m.height and 0 <= ox < m.width:
                if m.layers[layer][oy][ox] == eid:
                    return oy, ox, e
    return y, x, e  # 找不到原点(异常), 退化为自身


def classify_map(m: qp.QQTMap, elems: dict, unknown_log: set) -> dict:
    """把一张 QQ堂地图分类成各布尔掩码 + 元信息。"""
    h, w = m.height, m.width
    wall = [[False] * w for _ in range(h)]
    brick = [[False] * w for _ in range(h)]
    overhead = [[False] * w for _ in range(h)]
    ground = [[False] * w for _ in range(h)]
    pushable = [[False] * w for _ in range(h)]
    cover = [[False] * w for _ in range(h)]    # 房子: 可通行 + 不可炸
    bush = [[False] * w for _ in range(h)]     # 灌木: 可通行 + 可炸毁

    item_points = set(m.points[0]) if m.points else set()

    for y in range(h):
        for x in range(w):
            # 最顶层非空元素
            top_k = None
            for k in (0, 1, 2):
                if m.layers[k][y][x]:
                    top_k = k
                    break
            if top_k is None:
                continue
            oy, ox, e = _resolve_origin(m, top_k, y, x, elems)
            if e is None:
                eid = abs(m.layers[top_k][y][x])
                if eid not in unknown_log:
                    unknown_log.add(eid)
                    print(f"  [warn] 元素 {eid} 不在 mapElem.prop, 按层默认处理")
                if top_k == 2:
                    ground[y][x] = True
                elif top_k == 0:
                    overhead[y][x] = True   # 顶层默认可通行装饰
                # L1 未知元素默认阻挡(墙)
                continue

            # ---- 遮挡元件优先：房子(cover, 可通行不可炸) / 灌木(bush, 可通行可炸) ----
            v_now = m.layers[top_k][y][x]
            eid_abs = abs(v_now)
            if eid_abs in COVER_ELEMENTS or eid_abs in BUSH_ELEMENTS:
                is_bush = eid_abs in BUSH_ELEMENTS
                for ly2 in range(e["h"]):
                    for lx2 in range(e["w"]):
                        cy, cx = oy + ly2, ox + lx2
                        if not (0 <= cy < h and 0 <= cx < w):
                            continue
                        if is_bush:
                            bush[cy][cx] = True
                            overhead[cy][cx] = False
                        else:
                            cover[cy][cx] = True
                continue

            if top_k == 2:
                # 地板层: 恒可通行
                ground[y][x] = True
                continue

            # L0/L1: 取该格属性(多格元素按局部坐标)
            lx, ly = x - ox, y - oy
            attr = e["attrs"][ly * e["w"] + lx] if e["attrs"] else 0
            blocked = (attr == 0)
            v_now = m.layers[top_k][y][x]

            is_item = (y, x) in item_points
            if is_item:
                brick[y][x] = True                      # 可炸(权威)
            elif blocked:
                wall[y][x] = True                       # 永久墙
            else:
                if top_k == 0:
                    overhead[y][x] = True               # 头顶可通行装饰
                else:
                    # L1 可通行(矿洞发光地/探险桥等) —— 归入 ground 视觉层
                    ground[y][x] = True

            # 推箱子: 可推动箱子 (city==14 且 special 带 0x2 位)
            if (m.game_mode == 8
                    and (e["special"] & PUSHABLE_SPECIAL_BIT)
                    and abs(v_now) // 1000 == 14):
                pushable[y][x] = True

            # ---- 多格元件足迹展开 ----
            # 原版地图对多格元件(城堡/球门/比武墙等)只在**原点格**存正 ID，
            # 其余格为 0（不是负延续）—— 必须按元件的 w×h 展开，否则只有原点
            # 一格被判为障碍。展开只填"没有自己元素(L0/L1 全空)"的格，避免覆盖
            # 其他独立元件/砖。
            if v_now > 0 and (e["w"] > 1 or e["h"] > 1):
                for ly2 in range(e["h"]):
                    for lx2 in range(e["w"]):
                        cy, cx = oy + ly2, ox + lx2
                        if not (0 <= cy < h and 0 <= cx < w):
                            continue
                        if cy == y and cx == x:
                            continue
                        has_own = any(m.layers[kk][cy][cx] for kk in (0, 1))
                        if has_own:
                            continue
                        a2 = e["attrs"][ly2 * e["w"] + lx2] if e["attrs"] else 0
                        if (cy, cx) in item_points:
                            brick[cy][cx] = True
                        elif a2 == 0:
                            wall[cy][cx] = True
                        else:
                            if top_k == 0:
                                overhead[cy][cx] = True
                            else:
                                ground[cy][cx] = True
                        if (m.game_mode == 8
                                and (e["special"] & PUSHABLE_SPECIAL_BIT)
                                and abs(v_now) // 1000 == 14):
                            pushable[cy][cx] = True

    # 出生点强制可通行(原版出生点四邻已清空, 双保险)
    spawns = [list(p) for grp in m.points[1:3] for p in grp] if len(m.points) >= 3 else []
    for (y, x) in spawns:
        wall[y][x] = brick[y][x] = False

    return {
        "wall": wall, "brick": brick, "overhead": overhead,
        "ground": ground, "pushable": pushable,
        "cover": cover, "bush": bush,
        "spawns": spawns,
        "item_points": sorted(item_points),
    }


# ---------------------------------------------------------------- 输出

def to_bool_tensor(grid: list) -> torch.Tensor:
    return torch.tensor(grid, dtype=torch.bool)


def convert_map(src: Path, elems: dict, names: dict, unknown_log: set,
                out_dir: Path, index: int) -> dict:
    data = src.read_bytes()
    m = qp.parse_map(data)
    c = classify_map(m, elems, unknown_log)

    h, w = m.height, m.width
    map_name, qqt_id = names.get(src.name, ("", 0))
    mode_name = GAME_MODES.get(m.game_mode, f"mode{m.game_mode}")

    level = {
        # ---- 原有字段(兼容) ----
        "wall": to_bool_tensor(c["wall"]),
        "brick": to_bool_tensor(c["brick"]),
        "meta": {
            "kind": "qqt_original",
            "qqt_version": m.version,
            "wall_count": sum(map(sum, c["wall"])),
            "brick_count": sum(map(sum, c["brick"])),
            "cover_count": sum(map(sum, c["cover"])),
            "bush_count": sum(map(sum, c["bush"])),
            "overhead_count": sum(map(sum, c["overhead"])),
            "pushable_count": sum(map(sum, c["pushable"])),
        },
        "cfg": {
            "height": h, "width": w, "n_players": m.max_players,
            "corridor_width": 0, "top_wall_rows": 0,
        },
        # ---- 新增字段(兼容扩展) ----
        "overhead": to_bool_tensor(c["overhead"]),
        "pushable": to_bool_tensor(c["pushable"]),
        "cover": to_bool_tensor(c["cover"]),
        "bush": to_bool_tensor(c["bush"]),
        "ground": to_bool_tensor(c["ground"]),
        "spawns": c["spawns"],
        "items": m.drops,
        "game_mode": mode_name,
        "qqt_id": qqt_id,
        "map_name": map_name,
        "source": src.name,
        "version": m.version,
        "layers_raw": [[[int(v) for v in row] for row in layer] for layer in m.layers],
    }

    fn = f"level_{index:04d}.pt"
    torch.save(level, out_dir / fn)
    return {"file": fn, "source": src.name, "map_name": map_name,
            "game_mode": mode_name, "qqt_id": qqt_id, "h": h, "w": w,
            "n_players": m.max_players,
            "wall": level["meta"]["wall_count"], "brick": level["meta"]["brick_count"],
            "cover": level["meta"]["cover_count"], "bush": level["meta"]["bush_count"],
            "overhead": level["meta"]["overhead_count"],
            "pushable": level["meta"]["pushable_count"]}


def render_ascii(level: dict) -> str:
    """文本渲染: #=永久墙 B=可炸 P=可推 ^=头顶装饰 .=空地"""
    h, w = level["wall"].shape
    lines = []
    for y in range(h):
        row = []
        for x in range(w):
            if level["brick"][y, x]:
                row.append("P" if level["pushable"][y, x] else "B")
            elif level["wall"][y, x]:
                row.append("#")
            elif level["overhead"][y, x]:
                row.append("^")
            elif level["ground"][y, x]:
                row.append(".")
            else:
                row.append(" ")
        lines.append("  " + " ".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="QQ堂 .map -> 本模拟器关卡格式转换")
    ap.add_argument("--in", dest="ins", action="append", required=True,
                    help="输入目录或 .map 文件(可多次)")
    ap.add_argument("--out", default="levels_qqt", help="输出目录")
    ap.add_argument("--prop", default=PROP_PATH, help="mapElem.prop 路径")
    ap.add_argument("--mapdesc", default=MAPDESC_PATH, help="mapDesc.py 路径")
    ap.add_argument("--ascii", metavar="PT", help="文本渲染某个已转换的 .pt")
    args = ap.parse_args()

    if args.ascii:
        d = torch.load(args.ascii, map_location="cpu", weights_only=False)
        print(f"{args.ascii}  {d.get('map_name','')} {d.get('game_mode','')} "
              f"{d['wall'].shape[1]}x{d['wall'].shape[0]}")
        print(render_ascii(d))
        return

    elems = load_elem_prop(args.prop)
    names = load_map_names(args.mapdesc)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for item in args.ins:
        p = Path(item)
        if p.is_dir():
            files += sorted(p.glob("*.map"))
        elif p.is_file():
            files.append(p)
    files = [f for f in files if f.suffix.lower() == ".map"]
    if not files:
        print("没有找到 .map 文件"); sys.exit(1)

    unknown_log: set = set()
    rows = []
    for i, f in enumerate(files):
        try:
            rows.append(convert_map(f, elems, names, unknown_log, out_dir, i))
        except Exception as e:
            print(f"[error] {f.name}: {e}")

    import csv
    with open(out_dir / "qqt_levels_manifest.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "map_name", "game_mode", "qqt_id",
                    "h", "w", "n_players", "wall", "brick", "overhead", "pushable"])
        for r in rows:
            w.writerow([r["file"], r["source"], r["map_name"], r["game_mode"],
                        r["qqt_id"], r["h"], r["w"], r["n_players"],
                        r["wall"], r["brick"], r["overhead"], r["pushable"]])

    (out_dir / "LEVELS.md").write_text(
        f"# QQ堂原版地图转换关卡（{len(rows)} 关）\n\n"
        "来源: QQ堂 5.2 原版客户端 .map, 经 qqt_to_levels.py 转换。\n"
        "兼容性: 保留原 levelgen 的 wall/brick/meta/cfg 四键, 新增 overhead/pushable/\n"
        "ground/spawns/items/game_mode/qqt_id/map_name/source/version/layers_raw。\n"
        "字段说明见转换器文件头注释。\n", encoding="utf-8")

    print(f"已转换 {len(rows)} 张地图 -> {out_dir}/")
    print(f"{'level':<14}{'源文件':<18}{'官方名':<12}{'模式':<8}{'尺寸':<8}{'墙':>4}{'砖':>4}{'头顶':>4}{'可推':>4}")
    for r in rows[:240]:
        print(f"{r['file']:<14}{r['source']:<18}{r['map_name']:<12}"
              f"{r['game_mode']:<8}{str(r['w'])+'x'+str(r['h']):<8}"
              f"{r['wall']:>4}{r['brick']:>4}{r['overhead']:>4}{r['pushable']:>4}")
    if len(rows) > 240:
        print(f"... 共 {len(rows)} 张(余下见清单)")
    if unknown_log:
        print("未知元素(按层默认处理):", sorted(unknown_log)[:20])


if __name__ == "__main__":
    main()
