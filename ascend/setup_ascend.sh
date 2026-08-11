#!/bin/bash
# Ascend 910B 一键环境安装（triton-ascend + CANN + torch_npu）
# 用法：bash ascend/setup_ascend.sh
set -e

echo "=== 1. 检查 CANN 工具链 ==="
if [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    echo "❌ 未找到 CANN。请先安装：
    - 驱动/firmware（Ascend HDK）
    - CANN Toolkit 9.1.0（https://www.hiascend.com/software/cann）
    安装后确认 /usr/local/Ascend/ascend-toolkit/set_env.sh 存在"
    exit 1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "✅ CANN 工具链 OK"

echo "=== 2. Python 版本检查（推荐 3.11）==="
python3 --version

echo "=== 3. 安装 triton-ascend + torch_npu ==="
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
# torch_npu（CANN 配套版本 2.7.1.post8，如已装跳过）
pip install torch-npu==2.7.1.post8 2>/dev/null || echo "torch_npu 可能已装或需按 CANN 版本匹配"

echo "=== 4. 验证 ==="
python3 - << 'EOF'
import torch
try:
    import torch_npu  # noqa: F401
    print("torch", torch.__version__, "| torch_npu OK | NPU:", torch.npu.device_count())
except ImportError as e:
    print("torch_npu import 失败:", e)
import triton
import triton.language as tl
print("triton:", triton.__version__)
try:
    import triton_ascend  # noqa: F401
    print("triton-ascend 插件 OK")
except ImportError:
    print("（triton_ascend 插件未直接 import——部分版本经 triton 后端自动加载）")
EOF

echo "=== 完成。下一步：python3 ascend/verify_ascend.py（kernel 对拍）==="
