#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQ堂 (QQTang / QQT) 地图文件解析器
==================================
基于两份独立逆向实现交叉验证的二进制格式:
  * brangpd/qqt_map_editor_fin   (C++, 官方地图编辑器格式 v3/v4)
  * Kizzzy/lib47-qqt             (Java, 原版客户端 .map 解析)

用法:
  python3 qqt_map_parser.py scan <目录> [--json out.json]
       扫描目录下所有 .map, 输出每个地图的 版本/模式/人数/格子宽高
  python3 qqt_map_parser.py dump <xxx.map> [--out 前缀] [--json]
       完整解析单个地图, 输出:
         - 前缀.map.json   结构化完整数据(所有层/道具/出生点/特殊点)
         - 前缀.layers.txt 可读的三层格子矩阵 (L0顶/L1网格/L2地板)
         - 前缀.legend.txt 出现的元素ID及含义
  python3 qqt_map_parser.py roundtrip <xxx.map>
       重新写回二进制并逐字节比对, 验证解析器正确性
  python3 qqt_map_parser.py render <xxx.map> [--out 目录] [--scale 12]
       把格子数据渲染成 PNG (灰=地面, 蓝=墙/砖, 红=顶层元素)
"""

import json
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------- 常量表

GAME_MODES = {
    1: "普通(normal)", 2: "足球(bomb)", 3: "抢包山(bun)", 4: "比武(match)",
    5: "夺宝(treasure)", 6: "英雄传说(sculpture)", 7: "机械世界(machine)",
    8: "推箱子(box)", 9: "练习(practice)", 10: "探险(exploration)",
    11: "通用(common)", 12: "PVE", 13: "糖客战(tank)",
}

CITIES = {
    1: "沙漠", 2: "雪地", 3: "中国城", 4: "矿洞", 5: "水面", 6: "野外",
    7: "足球", 8: "抢包山", 9: "功夫", 10: "夺宝", 11: "比武",
    12: "英雄传说", 13: "机械世界", 14: "推箱子", 15: "练习",
    16: "探险", 17: "通用", 18: "PVE", 19: "糖客战",
}

# 版本3(探险模式开发之前) 地图长宽固定
V3_FIXED_W, V3_FIXED_H = 15, 13

LAYER_NAMES = ["L0顶(可放置物)", "L1网格(隐藏物)", "L2地板(地面)"]
POINT_NAMES = ["道具生成点", "队伍0出生点", "队伍1出生点", "特殊元素点"]


# ---------------------------------------------------------------- 解析

class QQTMap:
    def __init__(self):
        self.version = 0
        self.game_mode = 0
        self.max_players = 0
        self.width = 0
        self.height = 0
        self.layers = []          # 3 个 [h][w] 的 int32 矩阵
        self.drops = []           # [{id,min,max,rate}]
        self.points = []          # 4 个 [(y,x),...] 列表

    def element_at(self, layer, y, x):
        return self.layers[layer][y][x]

    def city_of(self, v):
        return v // 1000 if v > 0 else 0

    def id_in_city(self, v):
        return v % 1000 if v > 0 else 0

    def to_dict(self):
        return {
            "version": self.version,
            "gameMode": self.game_mode,
            "gameModeName": GAME_MODES.get(self.game_mode, "未知"),
            "maxPlayers": self.max_players,
            "width": self.width,
            "height": self.height,
            "grid": f"{self.width}x{self.height} (宽x高, 格)",
            "layers": [
                [[int(v) for v in row] for row in layer]
                for layer in self.layers
            ],
            "drops": self.drops,
            "points": [
                [{"y": y, "x": x} for (y, x) in group]
                for group in self.points
            ],
        }


def _read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"需要 {n} 字节, 实际 {len(b)} 字节 (文件可能被截断)")
    return b


def parse_map(data: bytes) -> QQTMap:
    f = __import__("io").BytesIO(data)
    m = QQTMap()

    (m.version,) = struct.unpack("<i", _read_exact(f, 4))
    (m.game_mode,) = struct.unpack("<i", _read_exact(f, 4))
    (m.max_players,) = struct.unpack("<i", _read_exact(f, 4))

    if m.version not in (3, 4):
        raise ValueError(f"不支持的地图版本 {m.version} (已知 3, 4)")

    if m.version == 4:
        (m.width,) = struct.unpack("<i", _read_exact(f, 4))
        (m.height,) = struct.unpack("<i", _read_exact(f, 4))
    else:
        m.width, m.height = V3_FIXED_W, V3_FIXED_H

    if m.width <= 0 or m.height <= 0 or m.width > 4096 or m.height > 4096:
        raise ValueError(f"非法地图尺寸 {m.width}x{m.height}")

    # 3 层, 每层 h*w 个 int32 元素ID
    for _k in range(3):
        layer = []
        for _y in range(m.height):
            row = list(struct.unpack(f"<{m.width}i", _read_exact(f, m.width * 4)))
            layer.append(row)
        m.layers.append(layer)

    # 道具类型及其掉落概率
    (n_item,) = struct.unpack("<i", _read_exact(f, 4))
    for _ in range(n_item):
        iid, lo, hi = struct.unpack("<iii", _read_exact(f, 12))
        (rate,) = struct.unpack("<f", _read_exact(f, 4))
        m.drops.append({"id": iid, "min": lo, "max": hi, "rate": rate})

    # 4 组点: 道具生成点 / 队伍0出生点 / 队伍1出生点 / 特殊元素点
    for _ in range(4):
        (n,) = struct.unpack("<i", _read_exact(f, 4))
        pts = []
        for _ in range(n):
            y, x = struct.unpack("<hh", _read_exact(f, 4))
            pts.append((y, x))
        m.points.append(pts)

    return m


def serialize_map(m: QQTMap) -> bytes:
    """把解析结果原样写回二进制(roundtrip 校验用)"""
    out = bytearray()
    out += struct.pack("<iii", m.version, m.game_mode, m.max_players)
    if m.version == 4:
        out += struct.pack("<ii", m.width, m.height)
    for k in range(3):
        for row in m.layers[k]:
            out += struct.pack(f"<{len(row)}i", *row)
    out += struct.pack("<i", len(m.drops))
    for d in m.drops:
        out += struct.pack("<iiif", d["id"], d["min"], d["max"], d["rate"])
    for group in m.points:
        out += struct.pack("<i", len(group))
        for (y, x) in group:
            out += struct.pack("<hh", y, x)
    return bytes(out)


# ---------------------------------------------------------------- 输出

def legend_of(m: QQTMap):
    legend = {}
    for k in range(3):
        for row in m.layers[k]:
            for v in row:
                if v != 0 and v not in legend:
                    city = m.city_of(v)
                    legend[v] = (
                        f"{CITIES.get(city, '未知')}城 elem{abs(v) % 1000}"
                        + (f" (city={city})" if city not in CITIES else "")
                    )
    return legend


def fmt_matrix(m: QQTMap, layer_idx: int, width=6):
    layer = m.layers[layer_idx]
    lines = []
    for row in layer:
        lines.append(" ".join(f"{v:{width}d}" if v else "." * width
                              for v in row))
    return "\n".join(lines)


def dump_map(m: QQTMap) -> str:
    out = []
    out.append(f"版本        : {m.version}")
    out.append(f"游戏模式    : {m.game_mode} = {GAME_MODES.get(m.game_mode, '未知')}")
    out.append(f"最大人数    : {m.max_players}")
    out.append(f"格子宽高    : {m.width} x {m.height} (宽 x 高, 单位:格)")
    out.append(f"总格数      : {m.width * m.height}")
    for k in range(3):
        out.append(f"\n[第{k}层 {LAYER_NAMES[k]}] {m.height}行 x {m.width}列:")
        out.append(fmt_matrix(m, k))
    if m.drops:
        out.append("\n[道具掉落表] id  min  max  rate")
        for d in m.drops:
            out.append(f"  {d['id']:6d} {d['min']:4d} {d['max']:4d} {d['rate']:.4f}")
    for i, group in enumerate(m.points):
        if group:
            out.append(f"\n[{POINT_NAMES[i]}] {len(group)}个: " +
                       ", ".join(f"({y},{x})" for (y, x) in group))
    leg = legend_of(m)
    if leg:
        out.append("\n[元素图例] 出现过的元素ID:")
        for v in sorted(leg):
            out.append(f"  {v:6d} -> {leg[v]}")
    return "\n".join(out)


# ---------------------------------------------------------------- CLI

def _png_write(path, pixels, w, h):
    """pixels: list of (r,g,b) rows -> 纯标准库 PNG"""
    import zlib
    raw = b""
    for row in pixels:
        raw += b"\x00" + b"".join(bytes(px) for px in row)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def cmd_render(path: Path, out_dir: Path, scale: int):
    """把地图渲染成 PNG: 灰=地面, 蓝=网格层元素(墙/砖), 红=顶层元素"""
    m = parse_map(path.read_bytes())
    palette = {}
    for k, rows in enumerate(m.layers):
        for row in rows:
            for v in row:
                if v and v not in palette:
                    city = m.city_of(v)
                    if k == 2:      # 地面: 按城市淡色
                        palette[v] = (140 + (city * 37) % 90,
                                      140 + (city * 61) % 90,
                                      150 + (city * 23) % 70)
                    elif k == 1:    # 网格: 蓝绿砖
                        palette[v] = (50 + (v * 53) % 120,
                                      90 + (v * 29) % 110,
                                      180 + (v * 41) % 60)
                    else:           # 顶层: 红/紫
                        palette[v] = (200 + (v * 17) % 50,
                                      40 + (v * 31) % 60,
                                      60 + (v * 47) % 120)
    W, H = m.width * scale, m.height * scale
    img = []
    for y in range(m.height):
        for _ in range(scale):
            row = []
            for x in range(m.width):
                c = (205, 200, 190)          # 默认空地面
                for k in (2, 1, 0):          # 地面->网格->顶层 覆盖
                    v = m.layers[k][y][x]
                    if v:
                        c = palette.get(v, (0, 0, 0))
                        break
                row += [c] * scale
            img.append(row)
    _png_write(out_dir / (path.stem + ".png"), img, W, H)
    print(f"已渲染: {out_dir / (path.stem + '.png')} ({W}x{H} px, 每格 {scale}px)")


def cmd_scan(root: Path, json_out):
    maps = sorted(root.rglob("*.map"))
    rows = []
    for p in maps:
        try:
            m = parse_map(p.read_bytes())
            rows.append({
                "file": str(p),
                "version": m.version,
                "gameMode": m.game_mode,
                "gameModeName": GAME_MODES.get(m.game_mode, "未知"),
                "maxPlayers": m.max_players,
                "width": m.width,
                "height": m.height,
            })
        except Exception as e:
            rows.append({"file": str(p), "error": str(e)})
    print(f"{'文件':<56} {'版本':>4} {'模式':>4} {'人数':>4} {'格子(宽x高)':>12}")
    print("-" * 90)
    for r in rows:
        if "error" in r:
            print(f"{r['file']:<56} 解析失败: {r['error']}")
        else:
            print(f"{r['file']:<56} {r['version']:>4} "
                  f"{r['gameModeName']:>4} {r['maxPlayers']:>4} "
                  f"{str(r['width'])+'x'+str(r['height']):>12}")
    if json_out:
        json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n已写出: {json_out}")


def cmd_dump(path: Path, prefix, want_json: bool):
    m = parse_map(path.read_bytes())
    base = prefix or str(path.with_suffix(""))
    Path(base + ".layers.txt").write_text(dump_map(m), encoding="utf-8")
    if want_json:
        Path(base + ".map.json").write_text(
            json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(dump_map(m))
    print(f"\n>>> 文本输出: {base}.layers.txt")
    if want_json:
        print(f">>> JSON输出: {base}.map.json")


def cmd_roundtrip(path: Path):
    data = path.read_bytes()
    m = parse_map(data)
    back = serialize_map(m)
    ok = data == back
    print(f"roundtrip {'OK  逐字节一致' if ok else 'FAIL 不一致!'}")
    if not ok:
        for i, (a, b) in enumerate(zip(data, back)):
            if a != b:
                print(f"首个差异 @0x{i:06x}: 原={a:#04x} 重写={b:#04x}")
                break
        print(f"原文件 {len(data)} 字节, 重写 {len(back)} 字节")
    return ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, target = sys.argv[1], Path(sys.argv[2])
    rest = sys.argv[3:]

    if cmd == "scan":
        cmd_scan(target, Path(rest[rest.index("--json") + 1]) if "--json" in rest else None)
    elif cmd == "dump":
        prefix = rest[rest.index("--out") + 1] if "--out" in rest else None
        cmd_dump(target, prefix, "--json" in rest)
    elif cmd == "render":
        out = Path(rest[rest.index("--out") + 1]) if "--out" in rest else target.parent
        scale = int(rest[rest.index("--scale") + 1]) if "--scale" in rest else 12
        cmd_render(target, out, scale)
    elif cmd == "roundtrip":
        sys.exit(0 if cmd_roundtrip(target) else 2)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
