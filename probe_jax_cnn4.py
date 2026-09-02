"""Round 4: which conv+bias+relu formulation is fast on DCU?

Found: conv chain 1.1 ms, but relu(conv+b) chain 96.8 ms (NHWC).
Test alternatives:
  A. NHWC relu(conv+b)            — the slow baseline
  B. NCHW relu(conv+b)            — MIOpen-native layout
  C. NCHW via jax.nn.conv + b
  D. NHWC relu(conv) without bias (isolate bias-add fusion)
  E. NCHW relu(conv) without bias
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
    x = jrandom.normal(key, (n, 7, H, W))               # NCHW
    xh = jnp.transpose(x, (0, 2, 3, 1))                 # NHWC
    w1h = jrandom.normal(key, (3, 3, 7, 32))            # HWIO
    b1 = jrandom.normal(key, (32,))
    w1c = jrandom.normal(key, (32, 7, 3, 3))            # OIHW
    conv = jax.lax.conv_general_dilated

    @jax.jit
    def a(x):
        z = jax.nn.relu(conv(x, w1h, (1, 1), "VALID",
                            dimension_numbers=("NHWC", "HWIO", "NHWC")) + b1)
        return z
    timed("A: NHWC relu(conv+b)", a, xh)

    @jax.jit
    def b(x):
        z = jax.nn.relu(conv(x, w1c, (1, 1), "VALID",
                            dimension_numbers=("NCHW", "OIHW", "NCHW"))
                        + b1.reshape(1, -1, 1, 1))
        return z
    timed("B: NCHW relu(conv+b)", b, x)

    @jax.jit
    def d(x):
        z = jax.nn.relu(conv(x, w1h, (1, 1), "VALID",
                            dimension_numbers=("NHWC", "HWIO", "NHWC")))
        return z
    timed("D: NHWC relu(conv) no bias", d, xh)

    @jax.jit
    def e(x):
        z = jax.nn.relu(conv(x, w1c, (1, 1), "VALID",
                            dimension_numbers=("NCHW", "OIHW", "NCHW")))
        return z
    timed("E: NCHW relu(conv) no bias", e, x)

    @jax.jit
    def f(x):
        z = jax.nn.relu(conv(x, w1h, (1, 1), "VALID",
                            dimension_numbers=("NHWC", "HWIO", "NHWC")) + b1)
        return z
    timed("F: NHWC conv+b+relu single (echo A)", f, xh)


if __name__ == "__main__":
    main()
