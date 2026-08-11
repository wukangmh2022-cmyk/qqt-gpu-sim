#!/usr/bin/env python3
"""按设计约定给 res/scenes.json 补全 tiles.wall（不可炸毁墙元件图）。

设计约定：场景素材目录 res/wall/<场景名>/ 里，**除 z*.png（可炸毁砖块）和
底图（bg）外，剩下的图片全是不可炸毁墙的元件图**。本脚本扫描目录自动收集
为 tiles.wall 列表（每格渲染时确定性随机选一张，同 brick 变体机制）。

只补 wall，不碰 bg/brick/bgm；无目录或元件图的场景跳过。用法：
    python3 deploy/gen_wall_tiles.py
"""
import json
import os
import sys

SCENES = "res/scenes.json"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    path = os.path.join(ROOT, SCENES)
    with open(path) as f:
        cfg = json.load(f)
    changed = 0
    for name, sc in cfg["scenes"].items():
        d = os.path.join(ROOT, "res", "wall", name)
        if not os.path.isdir(d):
            print(f"[skip] {name}: 无目录 res/wall/{name}/")
            continue
        bg = os.path.basename(sc.get("bg") or "")
        imgs = sorted(
            f for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and not f.lower().startswith("z")
            and f != bg
            and not f.startswith(".")
        )
        if not imgs:
            print(f"[warn] {name}: 未找到不可炸毁墙元件图（目录: {os.listdir(d)}）")
            continue
        tiles = sc.setdefault("tiles", {})
        wall = [f"res/wall/{name}/{f}" for f in imgs]
        if tiles.get("wall") != wall:
            tiles["wall"] = wall
            changed += 1
        print(f"[ok]   {name}: wall = {len(imgs)} 张 {imgs}")
    if changed:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=1, ensure_ascii=False)
        print(f"已更新 {SCENES}（{changed} 个场景变动）")
    else:
        print("无变动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
