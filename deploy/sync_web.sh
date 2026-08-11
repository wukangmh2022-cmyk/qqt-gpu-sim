#!/bin/bash
# 部署工具链最后一环：把 web/ 构建产物同步到公开游戏仓库并推送。
#
# 背景：qqt-gpu-sim 是私有仓库（免费套餐的 GitHub Pages 只支持公开仓库），
# 所以游戏页面部署在独立的公开仓库 qqt-gpu-web（Pages 服务其 main 分支根目录）。
# 本脚本把 web/ 内容整体覆盖到 qqt-gpu-web 的工作区并 push —— 每次 push 都会
# 触发 Pages 自动重建，游戏立即更新。
#
# 完整工具链：
#   .venv/bin/python deploy/export_ckpt.py --verify   # ckpt/ → web/models/*.json
#   bash deploy/prep_assets.sh                        # res/ → web/assets/
#   bash deploy/sync_web.sh                           # web/ → qqt-gpu-web → Pages 上线
#
# 用法： bash deploy/sync_web.sh [提交说明]
set -euo pipefail
cd "$(dirname "$0")/.."

GAME_REPO="wukangmh2022-cmyk/qqt-gpu-web"
WORK="${SYNC_WORK_DIR:-/tmp/qqt-gpu-web-sync}"
MSG="${1:-同步 web 构建产物 → Pages 自动部署}"

if [ ! -d "$WORK/.git" ]; then
  echo "[sync] clone $GAME_REPO → $WORK"
  gh repo clone "$GAME_REPO" "$WORK"
fi

echo "[sync] 同步 web/ → $WORK"
# 清掉旧文件（含已删除的），再整体拷贝
( cd "$WORK" && git rm -rq --cached . 2>/dev/null || true )
find "$WORK" -mindepth 1 -not -path "$WORK/.git*" -exec rm -rf {} + 2>/dev/null || true
cp -R web/. "$WORK/"
# 保留一份构建说明（README 已在游戏仓库里，同步时补一下链接文件）
if [ ! -f "$WORK/README.md" ]; then
  cp /dev/null /dev/null  # 占位：README 由游戏仓库自己维护
fi

cd "$WORK"
git add -A
if git diff --cached --quiet; then
  echo "[sync] 无变更，跳过"
else
  git -c user.name="1-6" -c user.email="a1-6@users.noreply.github.com" \
    commit -q -m "$MSG"
  git push -q origin main
  echo "[sync] 已推送 → https://github.com/$GAME_REPO"
  echo "[sync] Pages 自动重建中：https://wukangmh2022-cmyk.github.io/qqt-gpu-web/"
fi
