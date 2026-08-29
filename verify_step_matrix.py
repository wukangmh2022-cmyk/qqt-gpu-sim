"""step 级端到端对拍：while 版 vs 矩阵版爆炸连锁，完整 BombState 逐位一致。

同 seed 序列跑 250 tick（随机动作 + 掉血/回收/拾取全链路），比较全部
BombState 字段。monkeypatch 切换 jax_env._resolve_explosions。
"""
import jax
import jax.numpy as jnp
import jax_bomb.jax_env as E


def run_traj(seed, n, ticks, use_matrix):
    if use_matrix:
        E._resolve_explosions = E._resolve_explosions_matrix
    else:
        E._resolve_explosions = E._resolve_explosions.__wrapped__ if hasattr(
            E._resolve_explosions, "__wrapped__") else E._resolve_explosions
    # 保存原版引用
    import types
    # 直接构造两个闭包引用：矩阵版固定，原版从模块 dict 取
    return None


# 更简单：直接保存两版引用，step 内部通过模块属性查找
ORIG = E._resolve_explosions
MTRX = E._resolve_explosions_matrix


def run_traj(seed, n, ticks, variant):
    """variant: 0=while 1=matrix。返回字段字典列表快照（最后状态）。"""
    E._resolve_explosions = MTRX if variant else ORIG
    k = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(k)
    states = E.init_batch(key, n)
    stepv = jax.jit(jax.vmap(lambda s, a, kk: E.step(s, a, kk)))
    for t in range(ticks):
        # 随机动作
        key, ka, ks = jax.random.split(key, 3)
        acts = jax.random.randint(ka, (n, 2, 2), 0, 5)  # 方向 0-4
        bombs = (jax.random.uniform(ks, (n, 2)) < 0.3).astype(jnp.int32)
        acts = acts.at[..., 1].set(bombs)
        keys = jax.random.split(jax.random.fold_in(key, t), n)
        states, done = stepv(states, acts, keys)
    return states


# 多 seed 多 config 对拍
fails = 0
for seed in (0, 1, 2, 7, 42):
    for n in (32, 128):
        for ticks in (80, 250):
            s0 = run_traj(seed, n, ticks, 0)
            s1 = run_traj(seed, n, ticks, 1)
            diffs = []
            for fld in s0._fields:
                a, b = getattr(s0, fld), getattr(s1, fld)
                if a.dtype == jnp.bool_ or a.dtype in (jnp.int32, jnp.int64):
                    neq = int((a != b).sum())
                else:
                    neq = int((jnp.abs(a - b) > 1e-5).sum())
                if neq:
                    diffs.append(f"{fld}:{neq}")
            if diffs:
                fails += 1
                print(f"FAIL seed={seed} n={n} ticks={ticks} " + " ".join(diffs))
if fails == 0:
    print("STEP-LEVEL ALL PASS (5 seeds x 2 n x 2 ticks)")
else:
    print(f"{fails} FAILURES")
