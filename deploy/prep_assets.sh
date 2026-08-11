#!/bin/bash
# 素材搬运：把 res/ 里 launcher 用的原版图像素材复制到 web/assets/，供浏览器版渲染。
#
# 只搬图像 + 音效，**不搬 BGM ogg**（背景音乐 v1 不做，ogg 是每个场景 ~700KB 的大头）。
# 同时生成 web/assets/scenes.json（路径改写为相对 assets/ 的地址），页面按它加载场景。
#
# 用法： bash deploy/prep_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RES=res
OUT=web/assets

mkdir -p "$OUT/scenes" "$OUT/snd"

# ---- 通用素材（角色皮肤 / 炸弹 / 爆炸 / 道具 / 无敌罩）----
# 角色行走图：海王子（默认）+ 小虾米 + 角色c（敌人）+ 火影
for f in 角色4×4精灵图.png 角色b4×4.png 角色c4×4.png 角色火影4x4.png \
         bomb1.png 爆炸中心.png 向上爆炸.png 向下爆炸.png \
         向左爆炸.png 向右爆炸.png 威力道具.png 泡泡数量道具.png 鞋子道具.png 无敌.PNG; do
  cp "$RES/$f" "$OUT/$f"
done

# ---- 音效（可缺失）----
for f in 放炮.wav 爆炸.wav 吃道具音效.wav 生命损失音效.wav 角色消失音效.wav; do
  [ -f "$RES/$f" ] && cp "$RES/$f" "$OUT/snd/"
done

# ---- 场景（bg + 砖块/墙贴图 + BGM，只搬 scenes.json 里引用的文件）----
# scenes.json 的路径是相对工程根的（如 res/wall/比武/bw.png），直接使用。
python3 - "$OUT" <<'PY'
import json, os, shutil, sys
out = sys.argv[1]
cfg = json.load(open("res/scenes.json"))
scene_out = {}
for name, sc in cfg["scenes"].items():
    # 统一成 路径字符串列表（bg/wall 单个路径，brick 多个变体）
    groups = {"bg": [sc["bg"]] if sc.get("bg") else []}
    tiles = sc.get("tiles", {})
    for key in ("brick", "wall", "ground"):
        v = tiles.get(key)
        if isinstance(v, list):
            groups[key] = v
        elif v:
            groups[key] = [v]
        else:
            groups[key] = []
    # BGM（ogg，懒加载不阻塞启动；体积大头但只有启用时取）
    bgm = sc.get("bgm")
    if bgm:
        dst = os.path.join(out, "scenes", f"{name}_{os.path.basename(bgm)}")
        if os.path.exists(bgm) and not os.path.exists(dst):
            shutil.copy(bgm, dst)
    # 复制全部引用文件（按场景名前缀存放，避免 z1.png 等跨场景同名冲突）
    def dst_name(rel):
        return f"{name}_{os.path.basename(rel)}"
    for rels in groups.values():
        for rel in rels:
            if not rel:
                continue
            dst = os.path.join(out, "scenes", dst_name(rel))
            if os.path.exists(rel) and not os.path.exists(dst):
                shutil.copy(rel, dst)
    # 场景无背景图 → 用 bg1.png 兜底（与 launcher 一致）
    if groups["bg"]:
        bg_file = "scenes/" + dst_name(groups["bg"][0])
    else:
        bg_file = "bg1.png"
        if not os.path.exists(os.path.join(out, "bg1.png")):
            shutil.copy("res/bg1.png", os.path.join(out, "bg1.png"))
    scene_out[name] = {
        "bg": bg_file,
        "bgm": ("scenes/" + f"{name}_{os.path.basename(bgm)}" if bgm else None),
        "brick": ["scenes/" + dst_name(t) for t in groups["brick"]],
        "wall": ("scenes/" + dst_name(groups["wall"][0])
                 if groups["wall"] else None),
    }
json.dump(scene_out, open(os.path.join(out, "scenes.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"scenes.json: {len(scene_out)} 个场景")
PY

echo "assets 已就绪 → $OUT"
du -sh "$OUT"
