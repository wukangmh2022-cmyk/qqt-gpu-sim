"""设备探测与运行时能力检查。"""

from __future__ import annotations

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def available(device: str | torch.device) -> bool:
    """Return whether an explicitly requested device can be used now."""
    dev = torch.device(device)
    if dev.type == "cpu":
        return dev.index in (None, 0)
    if dev.type == "mps":
        return dev.index in (None, 0) and bool(torch.backends.mps.is_available())
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            return False
        return dev.index is None or 0 <= dev.index < torch.cuda.device_count()
    if dev.type == "npu":
        if not _npu_available():
            return False
        count = getattr(torch.npu, "device_count", lambda: 0)()
        return dev.index is None or 0 <= dev.index < count
    return False


def resolve_device(spec: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or choose the best available accelerator.

    Vendor DCUs exposed through the PyTorch CUDA API intentionally use
    ``cuda`` here; no vendor package name is hard-coded into the project.
    """
    if spec is not None:
        dev = torch.device(spec)
        if not available(dev):
            raise RuntimeError(f"requested device {dev} is not available")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _npu_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def synchronize(device: str | torch.device) -> None:
    """Synchronize one supported accelerator, if its runtime exposes it."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "npu" and _npu_available():
        torch.npu.synchronize()


def pick_device() -> str:
    """Backward-compatible string form of :func:`resolve_device`."""
    return str(resolve_device())
