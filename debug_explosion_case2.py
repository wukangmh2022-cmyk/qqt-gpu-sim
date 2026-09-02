"""dump case 87 矩阵版中间量：order/mask/burst/reach 关键行。"""
import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (H, W, BLAST, FUSE, MAX_BOMBS, _reach_dp)

# 从 verify 脚本完整构造（复制生成逻辑），只取 case 87
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
trig = (f == 0) & (o >= 0)
live = f > 0
bomb_flat = (live | trig).reshape(-1)
print("n trig (fuse==0&owner>=0):", int(trig.sum()), " n live:", int(live.sum()))
K = MAX_BOMBS * 2
_, order = jax.lax.top_k(bomb_flat.astype(jnp.int32), K)
mask = bomb_flat[order]
print("n mask True in top20:", int(mask.sum()))
# (11,12)=155, (10,6)=136
for cell in [(11, 12), (7, 12), (10, 6)]:
    fi = cell[0] * W + cell[1]
    pos = jnp.argwhere(order == fi)
    in_mask = mask[pos] if len(pos) else "NOT IN TOP20"
    print(f"cell {cell} idx={fi} bomb_flat={bomb_flat[fi]} order_pos={pos} mask={in_mask}")
# reach 检查: (10,6) 为什么爆? 谁覆盖它?
print("== (10,6) 被谁覆盖 ==")
cr, cc = order // W, order % W
up, down, left, right = _reach_dp(wl | br | live)
for j in range(K):
    if not mask[j]:
        continue
    rj, cj = int(cr[j]), int(cc[j])
    if rj == 10 and cj == 6:
        continue
    dist = abs(rj - 10) + abs(cj - 6)
    aligned = (rj == 10) or (cj == 6)
    in_blast = dist > 0 and dist <= int(b[j].reshape(-1)[order[j]])
    print(f"  from ({rj},{cj}) b={int(b.reshape(-1)[order[j]])} dist={dist} aligned={aligned} in_blast={in_blast}")
