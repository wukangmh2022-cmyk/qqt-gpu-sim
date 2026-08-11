"""jax_sim.py —— S1a：jax 版 rays + resolve（爆炸链），纯函数风格。

语义与 sim/blast.py 的 rays / resolve_explosions 对齐（固定轮，无 early_exit），
XLA 编译成融合图（消除逐 shift kernel 的 launch）。
"""
import jax
import jax.numpy as jnp

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _shift(x, dr, dc, fill):
    """result[i,j] = x[i-dr, j-dc]，越界补 fill（与 blast._shift 同语义）。

    slice+concat 实现（不用 jnp.pad —— 通用 pad kernel 在 HIP 后端慢 60x+）。
    """
    h, w = x.shape[1], x.shape[2]
    if dr == -1:      # 上移：取 x[1:]，底部补 fill
        r = jnp.concatenate([x[:, 1:], jnp.full((x.shape[0], 1, w), fill, x.dtype)], 1)
    elif dr == 1:     # 下移：顶部补 fill
        r = jnp.concatenate([jnp.full((x.shape[0], 1, w), fill, x.dtype), x[:, :-1]], 1)
    else:
        r = x
    if dc == -1:      # 左移：取 r[:, :, 1:]，右侧补 fill
        r = jnp.concatenate([r[:, :, 1:], jnp.full((x.shape[0], h, 1), fill, r.dtype)], 2)
    elif dc == 1:     # 右移：左侧补 fill
        r = jnp.concatenate([jnp.full((x.shape[0], h, 1), fill, r.dtype), r[:, :, :-1]], 2)
    return r


def rays(src, wall, bombed, brick, blast, b_max):
    """从 sources 出发的十字火焰覆盖（与 blast.rays 逐位一致，固定 b_max）。"""
    not_wall = ~wall
    not_solid = ~(bombed | brick)
    seed = src & not_wall & ~brick
    covered = seed
    for b in range(1, b_max + 1):
        src_b = seed & (blast == b)
        for dr, dc in DIRS:
            front = src_b
            for _ in range(b):
                front = _shift(front, dr, dc, False) & not_wall
                covered = covered | front
                front = front & not_solid
    return covered


def resolve(fuse, owner, wall, bomb_blast, brick, max_chain=16, b_max=7):
    """爆炸与连锁。返回 (covered, triggered)。固定轮（无同步），与固定轮 torch 一致。"""
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0

    def cover(newly):
        return rays(newly, wall, live, brick, bomb_blast, b_max)

    covered = cover(triggered)

    def body(i, carry):
        cov, trig = carry
        newly = live & cov & ~trig
        cov = cov | cover(newly)
        trig = trig | newly
        return cov, trig

    covered, triggered = jax.lax.fori_loop(0, max_chain - 1, body,
                                           (covered, triggered))
    return covered, triggered


def step_fuse(fuse, bomb_blast, owner, placed, cell_flat, alive, bomb,
              bombs_cap, blast_cap, wall, brick, cfg):
    """（S1a 暂不实现完整 step —— 放泡/移动在 S1b/S1c）"""
    raise NotImplementedError
