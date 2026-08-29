#!/bin/bash
# 新 notebook 一键部署：解代码 + 装 optax 依赖 + 环境自检
# 用法：把 dcu_deploy/ 整个目录拷到 notebook（如 /root/private_data/），然后
#   cd /root/private_data/dcu_deploy && bash setup_notebook.sh
# 幂等：重复运行安全（代码覆盖、optax 版本不符才重装）。
set -e
cd "$(dirname "$0")"

echo "=== [1/3] 解压代码到 /root/private_data/qqt-gpu-sim ==="
mkdir -p /root/private_data/qqt-gpu-sim
tar xzf jaxbomb.tgz -C /root/private_data/qqt-gpu-sim/
grep -q ppo_update_gradsync /root/private_data/qqt-gpu-sim/jax_bomb/jax_train.py \
  && grep -q ckpt_local_dir /root/private_data/qqt-gpu-sim/jax_bomb/multicard_train.py \
  && echo "代码 OK（含 ppo_update_gradsync + ckpt_local_dir）" \
  || { echo "代码版本不对"; exit 1; }

echo "=== [2/3] 安装 optax 依赖（--no-index --no-deps，不碰 numpy/jax）==="
# 版本必须与 wheels 一致（预装的老版本可能 API 不兼容）；不一致才重装
REQ_OPTAX="0.2.8"; REQ_CHEX="0.1.92"
if python3 -c "import optax, chex; assert optax.__version__ == '$REQ_OPTAX', (optax.__version__, 'needs $REQ_OPTAX'); assert chex.__version__ == '$REQ_CHEX', (chex.__version__, 'needs $REQ_CHEX')" 2>/dev/null; then
  echo "optax $REQ_OPTAX / chex $REQ_CHEX 已就绪，跳过安装"
else
  echo "安装 wheels: optax $REQ_OPTAX / chex $REQ_CHEX"
  pip install --no-index --find-links=wheels --no-deps --force-reinstall \
    optax chex dm-tree toolz wrapt etils typing_extensions absl-py attrs
fi

echo "=== [3/3] 环境自检 ==="
source /opt/dtk/env.sh 2>/dev/null || true
for mpi in /usr/mpi/gcc/openmpi-*/lib/libmpi.so*; do
  [ -f "$mpi" ] && export LD_PRELOAD="$mpi" && break
done
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
python3 - <<'PY'
import jax, optax
print("jax", jax.__version__, "optax", optax.__version__)
devs = jax.devices()
print("devices:", [str(d) for d in devs])
print("local_device_count:", jax.local_device_count())
if jax.local_device_count() != 2:
    print("⚠️⚠️  警告：本机只看到 %d 个 DCU（生产 10 机×2 卡要求每机 2 卡）。" %
          jax.local_device_count())
    print("         如果这是测试节点可忽略；生产节点请在创建 notebook 时把设备数选为 2，")
    print("         否则 launch_10nodes.sh 的自检会拒绝启动。")
PY
echo "=== 部署完成 ==="
echo "启动训练由本地 launch_10nodes.sh 统一编排（本脚本只需跑一次）。"
