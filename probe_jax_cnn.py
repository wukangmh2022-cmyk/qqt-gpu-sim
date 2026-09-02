"""Locate why cnn arch is ~170x slower than mlp in the JAX end-to-end bench.

Round 2 (GAP version): conv kernels benchmark fast in isolation but the
end-to-end CNN is still ~153 s/iter. Need to know:
  A. GAP cnn full fwd / fwd+bwd wall time (should be ~2 ms if conv+mean+small FC)
  B. matmul dimension sensitivity (64/5184 inner dim) — is the 2D GEMM the issue?
  C. conv in isolation again (sanity)
  D. 4D einsum (attention-like) vs 2D matmul — DCU 4D fallback suspicion
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W
from jax_bomb.jax_net import count_params, init_cnn, init_mlp, cnn_forward, mlp_forward


def timed(name, f, *args, iters=5):
    out = f(*args)
    jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / iters * 1000
    print(f"{name:<52s} {ms:9.2f} ms", flush=True)
    return out


def main():
    print(f"devices: {jax.devices()}", flush=True)
    n = 4096
    key = jrandom.PRNGKey(0)
    x = jrandom.normal(key, (n, 7, H, W))

    p_cnn = init_cnn(key, 7, H, W, ch1=32, ch2=64)
    p_mlp = init_mlp(key, 7, H, W, hidden=256)
    print(f"cnn params={count_params(p_cnn):,}  mlp params={count_params(p_mlp):,}", flush=True)

    # ---- A: GAP cnn full fwd / fwd+bwd ----
    @jax.jit
    def cnn_fwd(p, z):
        return cnn_forward(p, z)
    timed("cnn(GAP) full fwd 4096", cnn_fwd, p_cnn, x)

    @jax.jit
    def cnn_bwd(p, z):
        return jax.grad(lambda q, o: cnn_forward(q, o)[2].sum())(p, z)
    timed("cnn(GAP) full fwd+bwd", cnn_bwd, p_cnn, x)

    @jax.jit
    def mlp_fwd(p, z):
        return mlp_forward(p, z)
    timed("mlp full fwd 4096", mlp_fwd, p_mlp, x)

    @jax.jit
    def mlp_bwd(p, z):
        return jax.grad(lambda q, o: mlp_forward(q, o)[2].sum())(p, z)
    timed("mlp full fwd+bwd", mlp_bwd, p_mlp, x)

    # ---- B: matmul dimension sensitivity ----
    @jax.jit
    def mm(a, b):
        return a @ b
    a64 = jrandom.normal(key, (n, 64));    b64 = jrandom.normal(key, (64, 256))
    a5184 = jrandom.normal(key, (n, 5184)); b5184 = jrandom.normal(key, (5184, 256))
    a1024 = jrandom.normal(key, (n, 1024)); b1024 = jrandom.normal(key, (1024, 256))
    timed("matmul (4096,64)@(64,256)", mm, a64, b64)
    timed("matmul (4096,1024)@(1024,256)", mm, a1024, b1024)
    timed("matmul (4096,5184)@(5184,256)", mm, a5184, b5184)

    # ---- C: conv in isolation (re-check) ----
    xh = jnp.transpose(x, (0, 2, 3, 1))

    @jax.jit
    def conv1(a, w):
        return jax.lax.conv_general_dilated(
            a, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    w1 = p_cnn["w1"]
    timed("conv1 fwd (4096,13,13,7) 3x3->32", conv1, xh, w1)

    # ---- D: 4D einsum (attention-style) vs 2D matmul ----
    q4 = jrandom.normal(key, (n, 4, 169, 48)); k4 = jrandom.normal(key, (n, 4, 169, 48))

    @jax.jit
    def e4(a, b):
        return jnp.einsum("...htd,...hTd->...htT", a, b)
    timed("einsum 4D (4096,4,169,48)x(4096,4,169,48)", e4, q4, k4)

    # same flops as 2D matmul
    a2 = jrandom.normal(key, (n * 4, 169, 48)); b2 = jrandom.normal(key, (n * 4, 48, 169))

    @jax.jit
    def e2(a, b):
        return jnp.einsum("...td,...dT->...tT", a, b)
    timed("einsum 3D (16384,169,48)x(16384,48,169)", e2, a2, b2)


if __name__ == "__main__":
    main()
