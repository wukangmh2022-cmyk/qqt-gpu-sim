"""Round 3: which segment of the CNN forward graph is slow on DCU?

cnn(GAP) full fwd = 97 ms but every op in isolation is fast:
  transpose 0.12 ms, conv1 0.42 ms, conv2 ~1 ms, matmul <0.32 ms, mean trivial
Break the graph into segments and time each.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W
from jax_bomb.jax_net import init_cnn


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
    x = jrandom.normal(key, (n, 7, H, W))
    xh = jnp.transpose(x, (0, 2, 3, 1))
    p = init_cnn(key, 7, H, W, ch1=32, ch2=64)
    w1, b1 = p["w1"], p["b1"]
    w2, b2 = p["w2"], p["b2"]
    w3, b3 = p["w3"], p["b3"]
    conv = jax.lax.conv_general_dilated
    dn = ("NHWC", "HWIO", "NHWC")

    @jax.jit
    def seg_conv1(a):
        return conv(a, w1, (1, 1), "VALID", dimension_numbers=dn)
    timed("A: conv1 only (NHWC in)", seg_conv1, xh)

    @jax.jit
    def seg_conv12(a):
        z = conv(a, w1, (1, 1), "VALID", dimension_numbers=dn)
        z = conv(z, w2, (1, 1), "VALID", dimension_numbers=dn)
        return z
    timed("B: conv1->conv2 chain", seg_conv12, xh)

    @jax.jit
    def seg_relu12(a):
        z = jax.nn.relu(conv(a, w1, (1, 1), "VALID", dimension_numbers=dn) + b1)
        z = jax.nn.relu(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2)
        return z
    timed("C: relu(conv1+b1)->relu(conv2+b2)", seg_relu12, xh)

    @jax.jit
    def seg_tran12(a):
        z = jnp.transpose(a, (0, 2, 3, 1))
        z = jax.nn.relu(conv(z, w1, (1, 1), "VALID", dimension_numbers=dn) + b1)
        z = jax.nn.relu(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2)
        return z
    timed("D: transpose + conv1 + conv2 (full cnn body)", seg_tran12, x)

    @jax.jit
    def seg_tran12_gap(a):
        z = jnp.transpose(a, (0, 2, 3, 1))
        z = jax.nn.relu(conv(z, w1, (1, 1), "VALID", dimension_numbers=dn) + b1)
        z = jax.nn.relu(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2)
        z = z.mean((1, 2))
        return z
    timed("E: + GAP mean", seg_tran12_gap, x)

    @jax.jit
    def seg_full(a):
        z = jnp.transpose(a, (0, 2, 3, 1))
        z = jax.nn.relu(conv(z, w1, (1, 1), "VALID", dimension_numbers=dn) + b1)
        z = jax.nn.relu(conv(z, w2, (1, 1), "VALID", dimension_numbers=dn) + b2)
        z = z.mean((1, 2))
        z = jax.nn.relu(z @ w3 + b3)
        return z
    timed("F: full cnn body incl small FC", seg_full, x)


if __name__ == "__main__":
    main()
