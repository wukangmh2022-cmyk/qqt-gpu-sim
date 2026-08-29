# BW-1（SCNet hx1hdnormal 分区）实测笔记 — 2026-08-16

## 1. 环境搭建（一次性）

平台不预装 PyTorch：DCU 版 torch 由 DAS 软件栈以 wheel 形式提供，需从光源下载。

```bash
# 下载 DTK 26.04 适配的 torch 2.5.1（Python 3.12, x86_64）
mkdir -p ~/das_torch && cd ~/das_torch
curl -skL -o torch.whl \
  "https://download.sourcefind.cn:65024/directlink/4/pytorch/dtk2604-dcc2604-tlinux44-torch251-py312/torch-2.5.1%2Bdas.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl"

python3.12 -m venv ~/das_torch/venv
~/das_torch/venv/bin/pip install --no-deps ~/das_torch/torch.whl
~/das_torch/venv/bin/pip install "numpy==1.26.4" sympy==1.13.1 \
  typing_extensions filelock fsspec jinja2 networkx

# torch 2.5.1 链接 libmsgpackc.so.2；msgpack-c 6.x 改名 libmsgpack-c.so.2，
# 编译一份旧名符号链接（C ABI 兼容）
curl -skL -o msgpack-c.tar.gz \
  https://github.com/msgpack/msgpack-c/archive/refs/tags/c-6.1.0.tar.gz
tar xzf msgpack-c.tar.gz && cd msgpack-c-c-6.1.0
cmake -S . -B build -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF
cmake --build build && mkdir -p ~/das_torch/lib
cp build/libmsgpack-c.so.2 ~/das_torch/lib/
ln -sf libmsgpack-c.so.2 ~/das_torch/lib/libmsgpackc.so.2

# env.sh（Slurm 作业里 source）
cat > ~/das_torch/env.sh <<'EOF'
module load compiler/dtk/26.04
export LD_LIBRARY_PATH=$HOME/das_torch/lib:$LD_LIBRARY_PATH
export PATH=$HOME/das_torch/venv/bin:$PATH
EOF
```

注意：`sim/torch_sim.py` 的 `torch.compile(backend="npu")` 在 DCU 上会走 try/except 退化到未编译版（位级一致，仅不加速）；bench 脚本里禁用 triton（`_ts._HAS_TRITON = False`）避免 host 指针误入设备。

## 2. 配额实测（sacctmgr + srun 验证）

| 资源 | 限制 | 实测 |
|---|---|---|
| CPU | 每作业最多 16 核 | `--cpus-per-task=64` 被拒（AssocGrpCpuLimit），16 通过 |
| DCU | 每作业 1 卡 | `--gres=dcu:1` 通过 |
| 并行作业 | ≤10 运行 | 可并行跑多个单卡实验 |

**结论**：单卡全 DCU 布局完全在配额内；8 卡分布式需提配额。

## 3. 瓶颈实测（bench_breakdown.py，完整训练口径）

### 3.1 分段（N=8192，16 核）

| 分量 | CPU sim | DCU sim（推荐） |
|---|---|---|
| sim | 137.4 ms/tick | **27.6 ms/tick** |
| transfer (H2D obs) | 3.3 ms/tick | 0.07 ms/tick |
| DCU policy (5×前向) | 10.3 ms/tick | 10.6 ms/tick |
| DCU ppo_update | 16.1 ms/tick | 15.5 ms/tick |

### 3.2 完整训练 SPS（bench_hybrid.py：collect + ppo_update）

| 配置 | N=8192 | N=16384 |
|---|---|---|
| **全 DCU**（sim+policy+update 在 DCU） | **156.9k** | **249.0k** |
| CPU 混合（sim 在 16 核 CPU） | 43.3k | — |

### 3.3 结论

1. **瓶颈是 Simulator 本身，不是 Actor-Critic 更新**：sim 是 policy 的 ~10 倍、update 的 ~11 倍。
2. **没有带宽问题**：N=16384 时 transfer 仅 2.6 ms/tick（5%），同步时间很小。
3. **藏掉通信后瓶颈 = sim 计算**：DCU 上 27.6 ms/tick 且几乎不随 N 涨（GPU 并行吃满）；CPU 上随 N 线性涨、慢 5 倍——本仓库的"单大 batch 向量化" Simulator 设计天生吃 GPU，**CPU 混合模式在 BW-1 上不划算**。
4. overlap（sim/transfer/policy 异步流水线）理论上可再推高 30-40%（296k @ N=8192），但 transfer 本身只有 5%，收益主要来自 sim 与 policy 的重叠，属可选优化。

## 4. 使用

```bash
# 推荐（全 DCU，配额内最优）：
sbatch start_bw1.sh
# 等价手动命令：
python -m train.train --backend torch --device cuda --arch mlp --single-stage \
  --map-mode corridor --open-fraction 0.5 --num-envs 16384 \
  --rollout-steps 128 --minibatches 4 ...
```

代码侧混合设备改动（`--train-device` / `--sim-device`，见 sim/dev.py 与 train/train.py）保留为可选开关：`--train-device cuda --sim-device cpu` 可跑 CPU 混合实验，但实测较慢，默认不启用。

## 5. rays 火焰传播距离缓冲优化（2026-08-16）

**问题**：`rays()` 原按 blast 档分组迭代 `for b in range(1, b_max+1)`，每档 b 跑 4×b 次
pad shift（总 4×Σb；b_max=7 时 112 次/调用），空档（无 seed 的档位）也照付。成长玩法
档位 3~7 混合（实测每 tick 3~5 个不同档），空档浪费是大头 —— corridor+open_f=1.0
（带宝箱/成长）step 88.3ms，其中 rays 占大头。

**改动**（sim/blast.py）：改成与 danger 阶段 B 同结构的**单张量距离缓冲传播**——
fd=剩余传播距离（seed 格 = 自己的 blast 档），每方向统一 b_max 步、每步递减、
耗尽即停。pad 数从 4×Σb 降到 4×b_max（b_max=7 → 28）。档位混合照样只跑一趟，
不用均匀性假设。dtype 用 int8（blast ≤ growth_blast_max=7，int8 与 bool 同速，
int64 pad 慢 5 倍）。

**等价性**：每颗炮的传播距离固定、覆盖先于挡火、与档位传播顺序无关 —— 与 danger
阶段 B 同一套逐炮参考 PASS 的论证。验证：83 组 rays 逐位对照 + 400 组 resolve 连锁 +
全 step 120 tick 进程隔离逐位对照 + 项目测试 120 通过，全部 PASS。

**DCU 实测（hx1hdnormal，N=8192，cuda）**：
- corridor+open_f=1.0（带宝箱/成长，用户实际训练目标）：step 88.3ms → **56.4ms**
  （92.8k → **145.3k env-steps/s，+57%**）
- 纯 open（start_bw1.sh 现配置）：30.0ms → 29.7ms（无回归，略升）

本地 CPU 参考：isolated rays 8.8 → 4.7 ms/call（+47%）；corridor+open step 98.8 → 82.4ms。
