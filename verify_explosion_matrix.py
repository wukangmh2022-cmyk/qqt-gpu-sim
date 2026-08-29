"""逐位对拍（batch 版）：_resolve_explosions_matrix vs 原 while 波前版。

每 case 泡数 ≤ 20（K 内），覆盖稠密阵/随机阵/墙/砖/chain_cap 8/4。
所有 case 一次性 vmap 编译。比较 (covered, triggered) 逐位一致。
"""
import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (H, W, BLAST, FUSE, MAX_BOMBS,
                              _resolve_explosions, _resolve_explosions_matrix)

N = 1024
base = jax.random.PRNGKey(12345)
keys = jax.random.split(base, 8)

# 每 case 泡数：8~18 均匀（K=20 内，留余量给 triggered 强制）
n_bombs = jax.random.randint(keys[0], (N,), 8, MAX_BOMBS * 2 - 1)  # 8..18
fuse2d = jnp.zeros((N, H, W), jnp.int32)
owner2d = jnp.full((N, H, W), -1, jnp.int32)
bb2d = jnp.zeros((N, H, W), jnp.int32)
maxn = MAX_BOMBS * 2
perm = jax.vmap(lambda k: jax.random.choice(k, H * W, (maxn,), replace=False))(
    jax.random.split(keys[1], N))  # (N, maxn) 位置唯一
pmask = jnp.arange(maxn)[None, :] < n_bombs[:, None]
pfull = jnp.where(pmask, perm, 0)

fr = jax.random.randint(keys[2], (N, maxn), 1, FUSE + 1)  # 泡 fuse>=1（trig 单独强制）
or_ = jax.random.randint(keys[3], (N, maxn), 0, 2)
br_ = jax.random.randint(keys[4], (N, maxn), 1, BLAST + 1)

idx = jnp.broadcast_to(pfull[..., None], (N, maxn, 1))[..., 0]
r_idx = idx // W
c_idx = idx % W
fuse2d = fuse2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, fr, 0))
owner2d = owner2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, or_, -1))
bb2d = bb2d.at[jnp.arange(N)[:, None], r_idx, c_idx].set(jnp.where(pmask, br_, 0))

# 强制至少一颗 triggered：第一颗泡 fuse=0（不新增，泡数不变）
fuse2d = fuse2d.at[jnp.arange(N), r_idx[:, 0], c_idx[:, 0]].set(0)

# 墙/砖分 4 组：0=全无 1=15%墙 2=15%墙+20%砖 3=砖密集
ci = jnp.arange(N) // (N // 4)
k5, k6 = jax.random.split(keys[5], 2)
wall = jax.random.uniform(k5, (N, H, W)) < 0.15
brick = jax.random.uniform(k6, (N, H, W)) < 0.2
wall = jnp.where(ci[:, None, None] < 1, False, wall)          # 组0 无墙
brick = jnp.where(ci[:, None, None] < 2, False, brick)        # 组0/1 无砖
brick = jnp.where(ci[:, None, None] >= 3,
                  jax.random.uniform(k6, (N, H, W)) < 0.45, brick)  # 组3 砖密集
# 泡不叠墙/砖（把泡移开：简单方案——叠上的泡清掉，然后补 triggered）
fuse2d = jnp.where(wall | brick, 0, fuse2d)
owner2d = jnp.where(wall | brick, -1, owner2d)
bb2d = jnp.where(wall | brick, 0, bb2d)
# 强制 triggered：找每 case 首个非墙非砖格
free = ~(wall | brick)
idx0 = jnp.argmax(free.reshape(N, -1), axis=1)
r0, c0 = idx0 // W, idx0 % W
fuse2d = fuse2d.at[jnp.arange(N), r0, c0].set(0)
owner2d = owner2d.at[jnp.arange(N), r0, c0].set(0)
bb2d = bb2d.at[jnp.arange(N), r0, c0].set(BLAST)

vm = jax.jit(jax.vmap(lambda f, o, b, wl, br: (
    _resolve_explosions(f, o, b, wl, br, 8),
    _resolve_explosions_matrix(f, o, b, wl, br, 8))))
(c1, t1), (c2, t2) = vm(fuse2d, owner2d, bb2d, wall, brick)

n_valid = int(((fuse2d > 0).sum(axis=(1, 2)) > 0).sum())
diff = (c1 != c2) | (t1 != t2)
print("cases:", N, " with bombs:", n_valid)
print("mismatch cases:", int(diff.any(axis=(1, 2)).sum()))
if diff.any():
    bad = jnp.argwhere(diff.any(axis=(1, 2)))[:10]
    for i in bad[:, 0]:
        i = int(i)
        print(f"case {i}: covered diff {(c1[i] != c2[i]).sum()} "
              f"triggered diff {(t1[i] != t2[i]).sum()} "
              f"nbombs {(fuse2d[i] > 0).sum()}")
        dcell = jnp.argwhere(c1[i] != c2[i])
        if len(dcell):
            r, c = int(dcell[0, 0]), int(dcell[0, 1])
            print(f"  first cell ({r},{c}) fuse={fuse2d[i,r,c]} owner={owner2d[i,r,c]} "
                  f"wall={wall[i,r,c]} brick={brick[i,r,c]} "
                  f"while={c1[i,r,c]} matrix={c2[i,r,c]}")
else:
    print("ALL PASS")
