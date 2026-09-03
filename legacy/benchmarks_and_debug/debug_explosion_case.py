"""调试：dump case 87 的 while vs matrix 差异。"""
import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (H, W, BLAST, FUSE, MAX_BOMBS,
                              _resolve_explosions, _resolve_explosions_matrix)

N = 1024
base = jax.random.PRNGKey(12345)
keys = jax.random.split(base, 8)
n_bombs = jax.random.randint(keys[0], (N,), 8, MAX_BOMBS * 2 + 1)
maxn = MAX_BOMBS * 2
perm = jax.vmap(lambda k: jax.random.choice(k, H * W, (maxn,), replace=False))(
    jax.random.split(keys[1], N))
pmask = jnp.arange(maxn)[None, :] < n_bombs[:, None]
pfull = jnp.where(pmask, perm, 0)
fr = jax.random.randint(keys[2], (N, maxn), 0, FUSE + 1)
or_ = jax.random.randint(keys[3], (N, maxn), 0, 2)
br_ = jax.random.randint(keys[4], (N, maxn), 1, BLAST + 1)
idx = pfull
r_idx = idx // W
c_idx = idx % W
fuse2d = jnp.zeros((N, H, W), jnp.int32)
owner2d = jnp.full((N, H, W), -1, jnp.int32)
bb2d = jnp.zeros((N, H, W), jnp.int32)
fuse2d = fuse2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, fr, 0))
owner2d = owner2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, or_, -1))
bb2d = bb2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, br_, 0))
fuse2d = fuse2d.at[jnp.arange(N), r_idx[:, 0], c_idx[:, 0]].set(0)
ci = jnp.arange(N) // (N // 4)
k5, k6 = jax.random.split(keys[5], 2)
wall = jax.random.uniform(k5, (N, H, W)) < 0.15
brick = jax.random.uniform(k6, (N, H, W)) < 0.2
wall = jnp.where(ci[:, None, None] < 1, False, wall)
brick = jnp.where(ci[:, None, None] < 2, False, brick)
brick = jnp.where(ci[:, None, None] >= 3, jax.random.uniform(k6, (N, H, W)) < 0.45, brick)
fuse2d = jnp.where(wall | brick, 0, fuse2d)
owner2d = jnp.where(wall | brick, -1, owner2d)
bb2d = jnp.where(wall | brick, 0, bb2d)
free = ~(wall | brick)
idx0 = jnp.argmax(free.reshape(N, -1), axis=1)
r0, c0 = idx0 // W, idx0 % W
fuse2d = fuse2d.at[jnp.arange(N), r0, c0].set(0)
owner2d = owner2d.at[jnp.arange(N), r0, c0].set(0)
bb2d = bb2d.at[jnp.arange(N), r0, c0].set(BLAST)

i = 87
f, o, b, wl, br = (fuse2d[i], owner2d[i], bb2d[i], wall[i], brick[i])
c1, t1 = _resolve_explosions(f, o, b, wl, br, 8)
c2, t2 = _resolve_explosions_matrix(f, o, b, wl, br, 8)
print("nbombs:", int((f > 0).sum()))
print("while covered:", int(c1.sum()), "matrix covered:", int(c2.sum()))
print("while trig:", int(t1.sum()), "matrix trig:", int(t2.sum()))
print("== diff covered cells (while=1 matrix=0):")
for r, c in jnp.argwhere(c1 & ~c2):
    print(f"  ({r},{c}) fuse={f[r,c]} owner={o[r,c]} blast={b[r,c]} wall={wl[r,c]} brick={br[r,c]}")
print("== diff triggered:")
for r, c in jnp.argwhere(t1 != t2):
    print(f"  ({r},{c}) fuse={f[r,c]} owner={o[r,c]} blast={b[r,c]} while={t1[r,c]} matrix={t2[r,c]}")
print("== 所有泡:")
for r, c in jnp.argwhere(f > 0):
    print(f"  ({r},{c}) fuse={f[r,c]} owner={o[r,c]} blast={b[r,c]} trig_while={t1[r,c]} trig_matrix={t2[r,c]}")
