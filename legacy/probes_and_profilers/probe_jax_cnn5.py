"""Round 5: bf16 / expression-shape workarounds for slow conv+relu fusion on DCU.

Baseline facts:
  pure conv chain          1.1 ms
  relu(conv+b) chain       96.8 ms   <- pathological
  single relu(conv+b)       7.4 ms

Try:
  A. bf16 conv chain + relu (Average Joe casts obs/weights to bf16)
  B. fp32 conv chain + jnp.maximum(z+b, 0)  (different exprsn shape)
  C. fp32 conv chain + separate add then relu (no fusion temptation)
  D. bf16 single relu(conv+b)
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W


def timed(name, f, *args, iters=5):
    out = f(*args)
    jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / iters * 1000
    print(f"{name:<46s} {ms:9.2f} ms", flush=True)
    return out


def main():
    print(f"devices: {jax.devices()}", flush=True)
    n = 4096
    key = jrandom.PRNGKey(0)
    conv = jax.lax.conv_general_dilated

    # fp32 weights (2-stage chain: 7->32->64)
    w1 = jrandom.normal(key, (3, 3, 7, 32)); b1 = jrandom.normal(key, (32,))
    w2 = jrandom.normal(key, (3, 3, 32, 64)); b2 = jrandom.normal(key, (64,))

    # bf16 versions
    w1b = w1.astype(jnp.bfloat16); b1b = b1.astype(jnp.bfloat16)
    w2b = w2.astype(jnp.bfloat16); b2b = b2.astype(jnp.bfloat16)

    xh = jnp.transpose(jrandom.normal(key, (n, 7, H, W)), (0, 2, 3, 1))       # fp32
    xh16 = xh.astype(jnp.bfloat16)
    dn = ("NHWC", "HWIO", "NHWC")

    @jax.jit
    def a(x):
        z = jax.nn.relu(conv(x, w1b, (1, 1), "VALID", dimension_numbers=dn) + b1b)
        z = jax.nn.relu(conv(z, w2b, (1, 1), "VALID", dimension_numbers=dn) + b2b)
        return z
    timed("A: bf16 relu(conv+b) chain", a, xh16)

    @jax.jit
    def b(x):
        z = jnp.maximum(conv(x, w1, (1, 1), "VALID", dimension_numbers=dn) + b1, 0)
        z = jnp.maximum(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2, 0)
        return z
    timed("B: fp32 maximum(conv+b,0) chain", b, xh)

    @jax.jit
    def c(x):
        z = conv(x, w1, (1, 1), "VALID", dimension_numbers=dn) + b1
        z = jax.nn.relu(z)
        z = conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2
        z = jax.nn.relu(z)
        return z
    timed("C: fp32 separate add then relu", c, xh)

    @jax.jit
    def d(x):
        z = jax.nn.relu(conv(x, w1b, (1, 1), "VALID", dimension_numbers=dn) + b1b)
        return z
    timed("D: bf16 single relu(conv+b)", d, xh16)

    @jax.jit
    def e(x):  # fp32 chain reference (the slow one)
        z = jax.nn.relu(conv(x, w1, (1, 1), "VALID", dimension_numbers=dn) + b1)
        z = jax.nn.relu(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2)
        return z
    timed("E: fp32 relu(conv+b) chain (ref)", e, xh)

    @jax.jit
    def f(x):  # bf16 pure conv chain (no relu) for isolation
        z = conv(x, w1b, (1, 1), "VALID", dimension_numbers=dn)
        z = conv(z, w2b, (1, 1), "VALID", dimension_numbers=dn)
        return z
    timed("F: bf16 pure conv chain", f, xh16)


if __name__ == "__main__":
    main()
