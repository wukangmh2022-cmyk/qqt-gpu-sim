#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QQFDIMG (.img) -> PNG 转换器
============================
QQ堂客户端图片资源格式 (文件头 "QQF\\x1aDIMG"), 版本0=RGB565+Alpha, 版本1=ARGB32。
地图预览图位于客户端 map/ 目录: 如 desert01_4.img

用法:
  python3 qqfdimg2png.py <xxx.img> [--out dir] [--frame N]
  可一次性转换目录: python3 qqfdimg2png.py <目录>
"""

import struct
import sys
import zlib
from pathlib import Path


def parse_qqfdimg(data: bytes):
    if data[:8] != b"QQF\x1aDIMG":
        raise ValueError("不是 QQFDIMG 文件 (缺 QQF\\x1aDIMG 头)")
    off = 8
    (version,) = struct.unpack_from("<h", data, off); off += 2
    off += 6  # 2 + 4 未知字段
    n_frames, n_dir, x_off, y_off, w_orig, h_orig = struct.unpack_from(
        "<IIiiii", data, off); off += 24
    frames = []
    for _ in range(n_dir * (n_frames // n_dir if n_frames >= n_dir else 1)):
        off += 4
        fx, fy = struct.unpack_from("<ii", data, off); off += 8
        off += 4
        w, h = struct.unpack_from("<ii", data, off); off += 8
        off += 4
        if version == 0:
            cnt = w * h * 3
            buf = data[off:off + cnt]; off += cnt
            px = []
            for i in range(w * h):
                rgb565 = buf[i * 2] | (buf[i * 2 + 1] << 8)
                r = ((rgb565 >> 11) & 0x1F) * 255 // 31
                g = ((rgb565 >> 5) & 0x3F) * 255 // 63
                b = (rgb565 & 0x1F) * 255 // 31
                a = buf[cnt - w * h + i] * 255 // 32
                px.append((r, g, b, a))
            frames.append((w, h, px))
        elif version == 1:
            cnt = w * h * 4
            buf = data[off:off + cnt]; off += cnt
            px = []
            for i in range(w * h):
                b, g, r, a = buf[i * 4:i * 4 + 4]
                px.append((r, g, b, a))
            frames.append((w, h, px))
        else:
            raise ValueError(f"不支持的 QQFDIMG 版本 {version}")
    return version, n_frames, n_dir, x_off, y_off, w_orig, h_orig, frames


def _png(path: Path, w: int, h: int, px):
    raw = b""
    for y in range(h):
        raw += b"\x00"
        for x in range(w):
            r, g, b, a = px[y * w + x]
            raw += bytes((r, g, b, a))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def convert(src: Path, out_dir: Path, frame: int = 0):
    ver, nf, nd, xo, yo, wo, ho, frames = parse_qqfdimg(src.read_bytes())
    if not frames:
        raise ValueError(f"{src.name}: 无帧")
    frame = min(frame, len(frames) - 1)
    w, h, px = frames[frame]
    out = out_dir / (src.stem + f"_f{frame}.png")
    _png(out, w, h, px)
    print(f"{src.name}: v{ver} 帧{frame}/{len(frames)} {w}x{h} (原尺寸{wo}x{ho}, 偏移{xo},{yo}) -> {out.name}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = Path(sys.argv[1])
    args = sys.argv[2:]
    out_dir = Path(args[args.index("--out") + 1]) if "--out" in args else Path(".")
    frame = int(args[args.index("--frame") + 1]) if "--frame" in args else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        for f in sorted(target.glob("*.img")):
            try:
                convert(f, out_dir, frame)
            except Exception as e:
                print(f"{f.name}: 失败 - {e}")
    else:
        convert(target, out_dir, frame)


if __name__ == "__main__":
    main()
