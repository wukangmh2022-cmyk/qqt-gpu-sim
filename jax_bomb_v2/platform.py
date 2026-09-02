"""Platform layer: the ONLY file that touches device / precision specifics.

JAX programs are otherwise hardware-agnostic; this module centralizes what
differs across platforms (CUDA vs ROCm vs CPU) so switching the backend means
touching only this file (plus whatever env vars the runtime needs, e.g.
LD_PRELOAD on DCU — those live in shell scripts, not code).

Usage:
    from .platform import setup_platform, device_summary
    devs = setup_platform(matmul_precision="tensorfloat32")
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Matmul precision per platform:
#   tensorfloat32 — ~2x faster on Ampere+ / CDNA+ / Hygon DCU, negligible
#                   accuracy loss for training (same choice as Average Joe).
#   None (default) — full fp32, slower.
_MATMUL_PRECISION = "tensorfloat32"


def setup_platform(matmul_precision: str | None = _MATMUL_PRECISION,
                   x64: bool = False) -> list:
    """Apply platform-wide JAX config; return available devices.

    Raise RuntimeError if no accelerator is visible (JAX falls back to CPU
    silently otherwise, which makes end-to-end benches meaningless).
    """
    jax.config.update("jax_enable_x64", x64)
    if matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
    devs = jax.devices()
    accel = [d for d in devs if d.platform not in ("cpu",)]
    if not accel:
        raise RuntimeError(
            "no accelerator device visible to JAX (devices=%r); "
            "on DCU run inside sbatch with --gres=dcu:1 and source ~/jax_env.sh" % (devs,))
    return devs


def device_summary(devs: list) -> str:
    kinds = sorted({d.platform for d in devs})
    return f"platforms={kinds} n={len(devs)} first={devs[0]!r}"
