"""宝箱语义快速校验：与 web/sim.js 逐项对照（CPU 可跑，~2 分钟）。

JS 基准（web/sim.js:336, 394-396）：
  1) 炸砖→生箱：在炸砖瞬间按本关 crate_rate 掷爆率（<=0/缺失 → 1.0）
  2) 拾取：踩到宝箱必升（rng() < 1.0），三属性均匀
  3) 预置宝箱 / 掉血回收箱：同样必升
4b 落点：掉血回收箱（rec）会污染"炸砖成箱"统计，计数时排除 rec。

用法：cd qqt-gpu-sim && python3 scripts/quick_check_crate_semantics.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import (H, W, MAX_STEPS, init_batch, step,
                              GROWTH_BOMBS_MAX, GROWTH_BLAST_MAX,
                              GROWTH_SPEED_MAX, GROWTH_SPEED_STEP, CRATE_PROB)
from jax_bomb import levels

levels_path = sys.argv[1] if len(sys.argv) > 1 else "web/assets/maps/levels.json"
fails = []
n = 4096
IDLE, PLANT = 4, 1          # 移动编码 4=停；动作 [move, bomb]


def check(name, bad, n_total, tol=1e-9):
    b = int(np.asarray(jnp.sum(bad)))
    ok = b == 0 if tol == 0 else b <= tol * n_total
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {b}/{n_total} 违例"
          + ("" if ok else f"（容差 {tol}）"))
    if not ok:
        fails.append(name)


_step_vmapped = jax.jit(jax.vmap(lambda s, a, kk: step(s, a, kk, return_info=True)))

def emp_rate(states, ticks=35, plant=True):
    """双方不动、p0 首 tick 放泡一次（引信 30 自然引爆）后全停；
    返回 (destroyed, created) 累计。created 排除 rec（掉血回收箱）。
    注意：不能每 tick 按放泡——放泡判定在爆炸结算前（fuse<=0 即重放），
    原地连按会无限刷新引信（JS sim.js:291 同款顺序，行为一致）。"""
    destroyed = 0
    created = 0
    n_env = states.pos.shape[0]
    a_plant = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, PLANT]), (n_env, 2)),
                         jnp.broadcast_to(jnp.array([IDLE, 0]), (n_env, 2))],
                        axis=1)
    a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (n_env, 2))] * 2,
                       axis=1)
    for t in range(ticks):
        acts = a_plant if (plant and t == 0) else a_idle
        b0, c0, r0 = states.brick, states.crate, states.rec_crate
        keys = jrandom.split(jrandom.PRNGKey(1234 + t), n_env)
        states, _d, info = _step_vmapped(states, acts, keys)
        destroyed += int(np.asarray((b0 & ~states.brick).sum()))
        created += int(np.asarray((((states.crate > 0) & (c0 == 0))
                                   & ~states.rec_crate).sum()))
    return destroyed, created


# ---------- A. levels.py 的 rate 钳制 == JS 公式 ----------
print("[A] crate_rate 钳制（JS: >0 ? rate : 1.0）")
import json
raw = json.load(open(levels_path))
ls = levels.set_active(levels_path, weights="4=1.0")
bad = []
for lv in raw:
    i = int(lv["id"])
    js_rate = float(lv.get("crate_rate", 0) or 0)
    js_rate = js_rate if js_rate > 0 else 1.0
    if abs(float(np.asarray(ls.rate[i])) - js_rate) > 1e-6:
        bad.append(i)
check("全部 241 关 rate == JS 公式", np.asarray(bad).size, len(raw))
check("rate 恒 > 0（无 0 爆率死关）", ls.rate <= 0, ls.L)
l240 = next(i for i, lv in enumerate(raw) if lv.get("crate_rate") == 0)
print(f"  钳制命中: id={l240} crate_rate=0 → rate={float(np.asarray(ls.rate[l240]))}")

# ---------- B. 炸砖爆率（level 4: rate=0.6） ----------
print("[B] 炸砖→生箱 统计爆率（level 4）")
ls = levels.set_active(levels_path, weights="4=1.0")
r4 = float(np.asarray(ls.rate[4]))
print(f"  rate[4] = {r4}（levels.json 实测）")
states = init_batch(jrandom.PRNGKey(7), n)
assert np.all(np.asarray(states.level_id) == 4)
dest, cre = emp_rate(states)
rate_emp = cre / dest if dest else float("nan")
print(f"  destroyed={dest} created={cre} → 实测 {rate_emp:.4f}（期望 0.6）")
check(f"level4 炸砖爆率 ≈ {r4:.3f}（样本 {dest}）", abs(rate_emp - r4) > 0.03, 1, 0.5)
check("level4 样本充足", dest < 1000, 1, 0.5)      # dest 不足 1000 → FAIL

# ---------- C. 拾取必升（level 4，craft 宝箱在脚下） ----------
print("[C] 拾取必升（level 4，预置/回收两种 rec 标记）")
states = init_batch(jrandom.PRNGKey(8), n)
pos = states.pos
cy = pos[:, :, 0].astype(jnp.int32); cx = pos[:, :, 1].astype(jnp.int32)
rows = jnp.broadcast_to(jnp.arange(n)[:, None], (n, 2))
for tag, set_rec in (("普通道具", False), ("回收道具(rec)", True)):
    # craft 三种编码（1=泡 2=威 3=速）+ 问号 7 各 1/4，验证必升 + 种类对应
    rows_e = jnp.broadcast_to(jnp.arange(n)[:, None], (n, 2))
    code = jnp.repeat(jnp.array([1, 2, 3, 7], jnp.int8), n // 4)  # (n,) 每 4 env 一码
    code2 = jnp.stack([code, code], axis=1)              # 两玩家同码
    crate = states.crate.at[rows_e, cy, cx].set(code2)
    rec = states.rec_crate.at[rows_e, cy, cx].set(True) if set_rec else states.rec_crate
    st = states._replace(crate=crate, rec_crate=rec)
    b0, z0, s0 = st.bombs_cap, st.blast_cap, st.spd_g
    a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (n, 2))] * 2, axis=1)
    keys = jrandom.split(jrandom.PRNGKey(9), n)
    st2, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
    db = np.asarray(st2.bombs_cap - b0); dz = np.asarray(st2.blast_cap - z0)
    ds = np.asarray((st2.spd_g - s0) / GROWTH_SPEED_STEP)
    grew = (db + dz + ds) > 0.5
    check(f"{tag}: 必升（{n * 2} 个脚下道具全成长）", ~grew, n * 2)
    # 种类对应（仅非问号格子）：码1→泡+1，码2→威+1，码3→速+1
    cc = np.asarray(code2)
    non_q = cc != 7
    for m, arr, name in ((1, db, "泡"), (2, dz, "威"), (3, ds, "速")):
        bad = (np.abs(arr - (cc == m).astype(int)) > 0.5) & non_q
        check(f"{tag}: 码{m}({name}) 精确对应 +1", bad.sum(), non_q.sum())
    q = cc == 7
    check(f"{tag}: 问号(7) 随机拾取恰好一种", (db + dz + ds)[q] < 0.5, q.sum())
    over = (db > 1) | (dz > 1) | (ds > 1)
    check(f"{tag}: 普通道具单次只升一层", over, n * 2)

# ---------- D. rate=1.0 关：炸砖必成箱 ----------
print("[D] rate 高关（id=5）炸砖爆率")
ls = levels.set_active(levels_path, weights="5=1.0")
r5 = float(np.asarray(ls.rate[5]))
print(f"  rate[5] = {r5}（levels.json 实测）")
states = init_batch(jrandom.PRNGKey(10), n)
assert np.all(np.asarray(states.level_id) == 5)
dest, cre = emp_rate(states)
rate_emp = cre / dest if dest else float("nan")
print(f"  destroyed={dest} created={cre} → 实测 {rate_emp:.4f}（期望 {r5:.3f}）")
check(f"level5 炸砖爆率 ≈ {r5:.3f}（样本 {dest}）", abs(rate_emp - r5) > 0.03, 1, 0.5)

# ---------- E. 过程式回归：拾取必升 + corridor 爆率 = CRATE_PROB ----------
print("[E] 过程式路径回归")
levels.clear()
states = init_batch(jrandom.PRNGKey(11), n)
pos = states.pos
cy = pos[:, :, 0].astype(jnp.int32); cx = pos[:, :, 1].astype(jnp.int32)
crate = states.crate.at[rows, cy, cx].set(True)
st = states._replace(crate=crate)
b0, z0, s0 = st.bombs_cap, st.blast_cap, st.spd_g
a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (n, 2))] * 2, axis=1)
keys = jrandom.split(jrandom.PRNGKey(12), n)
st2, _d, _i = jax.vmap(
    lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
db = np.asarray(st2.bombs_cap - b0); dz = np.asarray(st2.blast_cap - z0)
ds = np.asarray((st2.spd_g - s0) / GROWTH_SPEED_STEP)
check("过程式: 拾取必升", (db + dz + ds) < 0.5, n * 2)

# corridor 炸砖爆率（过程式 corridor 关，爆率 CRATE_PROB）
states = init_batch(jrandom.PRNGKey(13), 4 * n)
dest, cre = emp_rate(states)
rate_emp = cre / dest if dest else float("nan")
print(f"  过程式 corridor: destroyed={dest} created={cre} → {rate_emp:.4f}（期望 {CRATE_PROB}）")
check(f"corridor 炸砖爆率 ≈ {CRATE_PROB}（样本 {dest}）",
      abs(rate_emp - CRATE_PROB) > 0.03, 1, 0.5)
check("corridor 样本充足", dest < 1000, 1, 0.5)
levels.clear()

# ---------- F. 超级道具 + 每关成长上限（与 Web sim.js:442-448 一致） ----------
print("[F] 超级道具占比（super_fraction）+ +4档 + 每关上限")
ls = levels.set_active(levels_path, weights="4=1.0")
sf4 = float(np.asarray(ls.super_f[4]))
print(f"  level4 super_fraction = {sf4:.4f}")
states = init_batch(jrandom.PRNGKey(14), n)
dest, cre = emp_rate(states)
# 统计掉落中的超级占比
st = states
nd = n
a_plant = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, PLANT]), (nd, 2)),
                     jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))], axis=1)
a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))] * 2, axis=1)
tot_super = tot_drop = 0
for t in range(40):
    acts = a_plant if t == 0 else a_idle
    c0 = st.crate
    keys = jrandom.split(jrandom.PRNGKey(15), nd)
    st, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(st, acts, keys)
    newk = np.asarray(st.crate)[np.asarray((st.crate > 0) & (c0 == 0))]
    tot_drop += len(newk)
    tot_super += int(((newk >= 4) & (newk <= 6)).sum())
if tot_drop:
    frac = tot_super / tot_drop
    print(f"  掉落 {tot_drop}，超级 {tot_super} → {frac:.4f}（期望 {sf4:.4f}）")
    check(f"超级道具占比 ≈ {sf4:.3f}", abs(frac - sf4) > 0.03, 1, 0.5)
# +5 档：craft 码 4（超级泡）→ 泡 +5（clamp 到 bombs_max=10，初始 2 → 7）
states = init_batch(jrandom.PRNGKey(16), 64)
cy = states.pos[:, :, 0].astype(jnp.int32); cx = states.pos[:, :, 1].astype(jnp.int32)
rows = jnp.broadcast_to(jnp.arange(64)[:, None], (64, 2))
st = states._replace(crate=states.crate.at[rows, cy, cx].set(4))
b0 = st.bombs_cap
a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (64, 2))] * 2, axis=1)
keys = jrandom.split(jrandom.PRNGKey(17), 64)
st2, _d, _i = jax.vmap(
    lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
db = np.asarray(st2.bombs_cap - b0)
check("码4(超级泡) +5 档", np.abs(db - 5) > 0.5, 64 * 2)
# 每关上限：反复拾取码1（泡）直到 clamp 到 bombs_max（level4 = 10）
states = init_batch(jrandom.PRNGKey(18), 64)
caps = np.asarray(ls.caps[4])
st = states._replace(bombs_cap=jnp.full((64, 2), 9.5, jnp.float32))
for _ in range(5):
    cy = st.pos[:, :, 0].astype(jnp.int32); cx = st.pos[:, :, 1].astype(jnp.int32)
    st = st._replace(crate=st.crate.at[rows, cy, cx].set(1))
    st, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
print(f"  上限测试：5 次码1 后 bombs_cap = {float(np.asarray(st.bombs_cap[0, 0]))}（level4 bombs_max={caps[0]}）")
check("成长 clamp 到每关 bombs_max", np.asarray(st.bombs_cap > caps[0]).sum(), 64 * 2)
levels.clear()

print(f"\n=== 宝箱语义结果: {'全部通过' if not fails else 'FAIL: ' + str(fails)} ===")
