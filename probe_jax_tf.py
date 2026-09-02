"""Transformer-on-DCU probe: is it compile time or per-iter runtime?

Disambiguates:
  A. attention einsum (4096,4,169,192) — big 169-token attention
  B. patch-ified attention (25 tokens, Average Joe style) vs 169 tokens
  C. full transformer fwd fwd+bwd wall time
  D. plain matmul control
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W
from jax_bomb.jax_net import count_params, init_transformer, transformer_forward


def timed(name, f, *args, iters=5, compile_ms=None):
    t0 = time.time()
    out = f(*args)
    jax.block_until_ready(out)
    compile_ms = (time.time() - t0) * 1000
    t0 = time.time()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / iters * 1000
    print(f"{name:<46s} run {ms:9.2f} ms   (first incl. compile {compile_ms:9.2f} ms)",
          flush=True)
    return out


def main():
    print(f"devices: {jax.devices()}", flush=True)
    n = 4096
    key = jrandom.PRNGKey(0)
    x = jrandom.normal(key, (n, 7, H, W))

    # ---- A: big attention einsum (patch=1, 169 tokens) ----
    q = jrandom.normal(key, (n, 4, H * W, 48))
    k = jrandom.normal(key, (n, 4, H * W, 48))
    v = jrandom.normal(key, (n, 4, H * W, 48))

    @jax.jit
    def attn169(q, k, v):
        s = jnp.einsum("...htd,...hTd->...htT", q, k) / 6.93
        w = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("...htT,...hTd->...htd", w, v)
    timed("attention 169 tok (4096,4,169,48)", attn169, q, k, v)

    # ---- B: small attention (patch 3x3 -> 25 tokens) ----
    T = 25
    qs = jrandom.normal(key, (n, 4, T, 48))
    ks = jrandom.normal(key, (n, 4, T, 48))
    vs = jrandom.normal(key, (n, 4, T, 48))

    @jax.jit
    def attn25(q, k, v):
        s = jnp.einsum("...htd,...hTd->...htT", q, k) / 6.93
        w = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("...htT,...hTd->...htd", w, v)
    timed("attention 25 tok (4096,4,25,48)", attn25, qs, ks, vs)

    # ---- C: full transformer fwd / fwd+bwd ----
    p = init_transformer(key, 7, H, W, embed=192, depth=4)
    print(f"transformer params={count_params(p):,}", flush=True)

    @jax.jit
    def tf_fwd(pp, z):
        return transformer_forward(pp, z)
    timed("transformer fwd (embed=192 d=4)", tf_fwd, p, x)

    @jax.jit
    def tf_bwd(pp, z):
        return jax.grad(lambda qq, o: transformer_forward(qq, o)[2].sum())(pp, z)
    timed("transformer fwd+bwd", tf_bwd, p, x)

    # ---- D: matmul control ----
    @jax.jit
    def mm(a, b):
        return a @ b
    a = jrandom.normal(key, (n, 1024))
    b = jrandom.normal(key, (1024, 256))
    timed("matmul 4096x1024 @ 1024x256", mm, a, b)


if __name__ == "__main__":
    main()
