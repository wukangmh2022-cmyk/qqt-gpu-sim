#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== 等待当前 git 进程完成 ==="
while pgrep -f "git-remote-https" >/dev/null 2>&1; do
    sleep 5
done
echo "=== 当前 git 进程已结束 ==="

git status

if [ -f web/models/params_it00000400.onnx ] && ! git ls-files --error-unmatch web/models/params_it00000400.onnx >/dev/null 2>&1; then
    echo "=== 正在提交与推送 Batch 3: it400 终局冠军模型 ==="
    git add web/models/params_it00000400*
    git commit -m "feat(web): deploy it400 champion model"
    git push origin main
    echo "=== Batch 3 推送成功 ==="
fi

if [ -f web/models/params_it00000034.onnx ] && ! git ls-files --error-unmatch web/models/params_it00000034.onnx >/dev/null 2>&1; then
    echo "=== 正在提交与推送 Batch 4: it34 狂放炮对比模型 ==="
    git add web/models/params_it00000034*
    git commit -m "feat(web): deploy it34 crazy bomb baseline model"
    git push origin main
    echo "=== Batch 4 推送成功 ==="
fi

if [ -f web/models/params_aggr_it00000210.onnx ] && ! git ls-files --error-unmatch web/models/params_aggr_it00000210.onnx >/dev/null 2>&1; then
    echo "=== 正在提交与推送 Batch 5: aggr it210 破拆模型 + index.json ==="
    git add web/models/params_aggr_it00000210* web/models/index.json
    git commit -m "feat(web): deploy aggr it210 19.7B step model and updated index"
    git push origin main
    echo "=== Batch 5 推送成功 ==="
fi

echo "🎉 所有模型及 index.json 均已全量部署至 GitHub 远端！"
