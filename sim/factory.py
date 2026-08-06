"""backend 工厂：统一的 `make_sim`，让训练脚本不关心跑在哪个 runtime 上。"""

from __future__ import annotations

import torch

from .config import SimConfig
from .torch_sim import BatchedSim


def make_sim(
    cfg: SimConfig,
    num_envs: int,
    backend: str = "auto",
    device: str | None = None,
    seed: int = 0,
):
    """backend ∈ {"auto", "torch", "cuda"}。

    - torch：纯张量参考实现，cpu / mps / cuda 都能跑，Mac 上开发用这个。
    - cuda：自定义 kernel，需要 nvcc，首次调用会 JIT 编译（约 30~60s）。
    - auto：有 CUDA 卡且能编译就用 cuda，否则退回 torch。
    """
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "torch"

    if backend == "torch":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return BatchedSim(cfg, num_envs, device=dev, seed=seed)

    if backend == "cuda":
        from .cuda.wrapper import CudaSim

        return CudaSim(cfg, num_envs, seed=seed)

    raise ValueError(f"未知 backend: {backend}")
