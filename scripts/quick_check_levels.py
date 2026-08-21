"""关卡模式快速校验：241 张标准化地图在 jax_env 里的正确性（~2 分钟）。

1) 采样正确性：出生点 P0≠P1 且在可通行格；初始属性 == 关卡数据；
   预置宝箱 == 关卡数据（逐格）；采样能覆盖大量关卡
2) 权重：--level-weights "240=0.2" 时空场景关占比 ≈ 20%
3) 物理：随机动作 6 tick 不穿墙、不越界（复用快速校验模式）
4) 专项：第 240 关（空场景）= 8/6/1.68 + 44 预置宝箱

用法：cd qqt-gpu-sim && python3 quick_check_levels.py levels.json
"""
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W, MAX_STEPS, init_batch, step
from jax_bomb import levels

levels_path = sys.argv[1] if len(sys.argv) > 1 else "levels.json"
ls = levels.set_active(levels_path, weights="240=0.2")
fails = []
n = 4096


def check(name, bad, n_total):
    b = int(np.asarray(jnp.sum(bad)))
    print(f"  [{'PASS' if b == 0 else 'FAIL'}] {name}: {b}/{n_total} 违例")
    if b:
        fails.append(name)


key = jrandom.PRNGKey(11)
states = init_batch(key, n)
key = jrandom.split(key)[0]

# 1) 出生点 / 属性 / 宝箱逐关核对
lids = np.asarray(states.level_id)
seen = np.unique(lids)
print(f"  采样覆盖关卡: {len(seen)}/{ls.L}（4096 局，权重 240=0.2 下其余均分）")
if len(seen) < 200:
    fails.append(f"关卡覆盖不足 {len(seen)}")

p0, p1 = states.pos[:, 0], states.pos[:, 1]
c0 = p0.astype(jnp.int32); c1 = p1.astype(jnp.int32)
same = (c0[:, 0] == c1[:, 0]) & (c0[:, 1] == c1[:, 1])
check("出生点 P0 != P1", same, n)
blocked = (states.wall | states.brick).reshape(n, -1)
b0 = blocked[jnp.arange(n), c0[:, 0] * W + c0[:, 1]]
b1 = blocked[jnp.arange(n), c1[:, 0] * W + c1[:, 1]]
check("出生点不在墙/砖上", (b0 | b1) & states.alive[:, 0] & states.alive[:, 1], n)
oob = ((p0[:, 0] < 0) | (p0[:, 0] > H) | (p0[:, 1] < 0) | (p0[:, 1] > W)
       | (p1[:, 0] < 0) | (p1[:, 0] > H) | (p1[:, 1] < 0) | (p1[:, 1] > W))
check("出生点不越界", oob, n)

# 初始属性 == 关卡数据（lo 栈按 level_id 对照）
exp_b = ls.lo[:, 0][lids]; exp_z = ls.lo[:, 1][lids]; exp_s = ls.lo[:, 2][lids]
check("初始泡数==关卡数据", (states.bombs_cap[:, 0] != exp_b) | (states.bombs_cap[:, 1] != exp_b), n)
check("初始威力==关卡数据", (states.blast_cap[:, 0] != exp_z) | (states.blast_cap[:, 1] != exp_z), n)
check("初始速度==关卡数据", (states.spd_g[:, 0] != exp_s) | (states.spd_g[:, 1] != exp_s), n)

# 预置宝箱 == 关卡数据（逐格）
exp_crate = ls.crate[lids]
check("预置宝箱逐格==关卡数据", ~(((states.crate > 0) == exp_crate) & (states.rec_crate == exp_crate)), n * H * W)
check("预置宝箱全部必升(rec)", ~(states.rec_crate == ls.crate[lids]), n * H * W)
check("预置宝箱编码=7(问号随机)", ~((states.crate == 7) | ~exp_crate), n * H * W)

# pushable 可推墙（预留）：加载正确 + 必 ⊆ brick（可推墙必须是障碍物）
raw = json.load(open(levels_path))
pu_raw = np.asarray([np.asarray(l.get("pushable", []), bool).sum() for l in raw])
check("每关 pushable 数 == JSON", np.asarray(ls.pushable.sum(axis=(1, 2))) != pu_raw, ls.L)
check("可推墙必在砖上(障碍物)", (ls.pushable & ~ls.brick).any(axis=(1, 2)), ls.L)

# 2) 权重：240 关占比
frac240 = np.mean(lids == 240)
print(f"  空场景(240) 实际占比 = {frac240:.3f}（期望 ≈0.20）")
if not 0.15 <= frac240 <= 0.26:
    fails.append(f"240 权重偏离 {frac240:.3f}")

# 3) 随机动作 6 tick：不穿墙 / 不越界
states = init_batch(key, n)
for t in range(6):
    key, k0 = jrandom.split(key)
    acts = jrandom.randint(k0, (n, 2, 2), 0, 5)
    keys = jrandom.split(key, n)
    states, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(states, acts, keys)
    for me in (0, 1):
        p = states.pos[:, me]
        cy = jnp.clip(p[:, 0].astype(jnp.int32), 0, H - 1)
        cx = jnp.clip(p[:, 1].astype(jnp.int32), 0, W - 1)
        blocked_cell = (states.wall | states.brick).reshape(
            n, -1)[jnp.arange(n), cy * W + cx]
        oob = ((p[:, 0] < 0) | (p[:, 0] > H) | (p[:, 1] < 0) | (p[:, 1] > W))
        check(f"t{t} 玩家{me} 不穿墙/不越界",
              (blocked_cell & states.alive[:, me]) | oob, n)

# 4) 专项：240 空场景关（用初始 state，避免 6 tick 后属性已被宝箱改变）
s0 = init_batch(jrandom.PRNGKey(99), 512)
l0 = np.asarray(s0.level_id)
i240 = np.where(l0 == 240)[0]
if len(i240) > 0:
    b8 = np.all(np.asarray(s0.bombs_cap[i240]) == 8)
    z6 = np.all(np.asarray(s0.blast_cap[i240]) == 6)
    sp = np.all(np.asarray(s0.spd_g[i240]) == 1.68)
    print(f"  240关 属性 8/6/1.68: bombs={b8} blast={z6} speed={sp}")
    if not (b8 and z6 and sp):
        fails.append("240 关属性不符")
    print(f"  240关 预置宝箱数 = {int(np.asarray(ls.crate[240]).sum())}（期望 44）")
    if int(np.asarray(ls.crate[240]).sum()) != 44:
        fails.append("240 关宝箱数不符")
else:
    print("  [SKIP] 本批 512 局未抽到 240 关")
    fails.append("未抽到 240 关")

print(f"\n=== 关卡模式结果: {'全部通过' if not fails else 'FAIL: ' + str(fails)} ===")
