"""backend 工厂：统一的 `make_sim`，让训练脚本不关心跑在哪个 runtime 上。"""

from __future__ import annotations

import torch

from .config import SimConfig
from .dev import resolve_device
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
    - auto：使用当前完整的 torch Simulator，并把显式 device 交给设备解析器。
    """
    if backend == "auto":
        # An explicit device is authoritative.  The legacy CUDA kernel does not
        # implement the current extended SimConfig contract, so the reference
        # torch backend is the reliable auto choice for both CPU and DCU.
        backend = "torch"

    if backend == "torch":
        dev = resolve_device(device) if device is not None else resolve_device()
        return BatchedSim(cfg, num_envs, device=dev, seed=seed)

    if backend == "cuda":
        if device is not None and resolve_device(device).type != "cuda":
            raise ValueError("backend=cuda requires a CUDA-compatible device")
        from .cuda.wrapper import CudaSim

        if cfg.n_channels != 2 * cfg.n_players + 3:
            raise RuntimeError(
                "backend=cuda does not support extended observation channels; "
                "use backend=torch"
            )
        return CudaSim(cfg, num_envs, seed=seed)

    raise ValueError(f"未知 backend: {backend}")
