"""跨后端设备选择（MPS / CUDA / Ascend NPU / CPU）。

verify 系列脚本统一走这里：910B 上若不识别 NPU 会 fallback 到 cpu，而
torch_sim 在装有 triton 的环境会自动启用 triton kernel（triton-ascend 在
NPU 上执行），CPU 张量的 host 指针被当作 NPU 地址 → aivec 矢量核异常
（aic error mask 0x6500020bd00028c，pc 落在用户态地址，见 HANDOFF_20260810）。
"""
import torch


def pick_device() -> str:
    """按 mps → cuda → npu:0 → cpu 优先级挑设备名。"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu:0"
    except (ImportError, AttributeError):
        pass
    return "cpu"
