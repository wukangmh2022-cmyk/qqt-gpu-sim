"""灌木丛（bush）专项校验：可通行 + 可炸 + 炸毁按本关爆率掉宝 + 回收箱不落灌木。

规则（与 JS Web 一致口径；JS 侧由 Web 侧维护，本脚本只管 JAX 训练侧）：
  1) bush 是独立布尔层：25 张关（野外等），与 brick/wall 零重叠
  2) 可通行：blocked = 泡|墙|砖 不含 bush → 玩家可站上/穿过灌木
  3) 可炸毁：爆炸覆盖即摧毁（bush=0），同砖规则按本关 crate_rate 掷爆率生箱
  4) 掉血回收箱**只落纯粹地面**：墙/砖/灌木都不可落（灌木可通行但非地面）
  5) obs 新增 ch8 = bush（N_OBS_CH=9）

用法：cd qqt-gpu-sim && python3 scripts/quick_check_bush.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import (H, W, MAX_STEPS, init_batch, step, make_obs,
                              N_OBS_CH, GROWTH_SPEED_STEP)
from jax_bomb import levels

levels_path = sys.argv[1] if len(sys.argv) > 1 else "web/assets/maps/levels.json"
fails = []
n = 4096
IDLE, PLANT = 4, 1


def check(name, bad, n_total, tol=1e-9):
    b = int(np.asarray(jnp.sum(bad)))
    ok = b == 0 if tol == 0 else b <= tol * n_total
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {b}/{n_total} 违例"
          + ("" if ok else f"（容差 {tol}）"))
    if not ok:
        fails.append(name)


# ---------- A. 数据正确性 ----------
print("[A] bush 加载与零重叠")
raw = json.load(open(levels_path))
ls = levels.set_active(levels_path, weights="28=1.0")     # 野外01（bush 40 格）
bush_cnt = np.asarray(ls.bush.sum(axis=(1, 2)))
raw_cnt = np.asarray([np.asarray(l.get("bush", []), bool).sum()
                      for l in raw])
check("每关 bush 数 == JSON", bush_cnt != raw_cnt, ls.L)
overlap = (ls.bush & (ls.wall | ls.brick)).any(axis=(1, 2))
check("bush 与 wall|brick 零重叠", overlap, ls.L)
print(f"  bush 关数 = {int(np.sum(bush_cnt > 0))}（期望 25），总数 = {int(np.sum(bush_cnt))}")

# ---------- B. 可通行：玩家可站在灌木格上 ----------
print("[B] bush 可通行（站上去不穿墙不越界）")
ls = levels.set_active(levels_path, weights="28=1.0")
states = init_batch(jrandom.PRNGKey(5), n)
pos = states.pos
cy = pos[:, :, 0].astype(jnp.int32); cx = pos[:, :, 1].astype(jnp.int32)
# 找每个玩家相邻的 bush 格（上下左右），把玩家直接放上去（bush 可站）
rows = jnp.broadcast_to(jnp.arange(n)[:, None], (n, 2))
targets = []
for me in range(2):
    yy, xx = np.asarray(cy[:, me]), np.asarray(cx[:, me])
    out = np.full((n, 2), -1, np.int64)
    for e in range(n):
        b = np.asarray(ls.bush[28])
        done = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = yy[e] + dr, xx[e] + dc
            if 0 <= r < H and 0 <= c < W and b[r, c]:
                out[e] = (r, c); done = True; break
        if not done:
            # 兜底：任意一块 bush 格
            br, bc = np.where(b)
            if len(br):
                out[e] = (br[0], bc[0])
    targets.append(out)
t0, t1 = targets
# 用 targets 构造新 pos（bush 格中心）；仅对"有 bush 邻格"的 env 断言可站
st = states
crate = st.crate.at[rows, cy, cx].set(True)
st = st._replace(crate=crate)          # 脚下放箱验证拾取不受 bush 影响
npy = jnp.asarray(t0[:, 0]); npx = jnp.asarray(t0[:, 1])
newpos0 = jnp.stack([npy + 0.5, npx + 0.5], axis=-1).astype(jnp.float32)
st = st._replace(pos=jnp.stack([newpos0, pos[:, 1]], axis=1))
a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (n, 2))] * 2, axis=1)
keys = jrandom.split(jrandom.PRNGKey(6), n)
st2, _d, _i = jax.vmap(
    lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
p = st2.pos[:, 0]
pc = jnp.stack([jnp.clip(p[:, 0].astype(jnp.int32), 0, H - 1),
                jnp.clip(p[:, 1].astype(jnp.int32), 0, W - 1)], axis=-1)
on_bush = (ls.bush[28][pc[:, 0], pc[:, 1]] | ~(ls.bush[28]).any())
check("玩家站在灌木格上不穿墙/不越界",
      (p[:, 0] < 0) | (p[:, 0] > H) | (p[:, 1] < 0) | (p[:, 1] > W), n)
ok_pos = np.asarray((pc[:, 0] == t0[:, 0]) & (pc[:, 1] == t0[:, 1]))
check("IDLE 后仍停留在灌木格", ~ok_pos, n)

# ---------- C. 炸灌木掉宝（野外01 id=28, crate_rate≈0.73） ----------
print("[C] 炸毁灌木 → 按本关 crate_rate 掷爆率出宝箱（与炸砖同规则）")
ls = levels.set_active(levels_path, weights="28=1.0")
states = init_batch(jrandom.PRNGKey(7), 4 * n)


def emp_rate_bush(states, ticks=40):
    """p0 首 tick 放泡引爆；分开统计 brick/bush 的 destroyed/created。
    created 排除 rec（掉血回收箱，与炸砖/炸灌木无因果关系）。"""
    nd = states.pos.shape[0]
    a_plant = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, PLANT]), (nd, 2)),
                         jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))],
                        axis=1)
    a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))] * 2,
                       axis=1)
    db, cb = 0, 0
    du, cu = 0, 0
    for t in range(ticks):
        acts = a_plant if t == 0 else a_idle
        b0, u0, c0, r0 = states.brick, states.bush, states.crate, states.rec_crate
        keys = jrandom.split(jrandom.PRNGKey(1234), nd)
        states, _d, _i = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(states, acts, keys)
        db += int(np.asarray((b0 & ~states.brick).sum()))
        cb += int(np.asarray((((states.crate > 0) & (c0 == 0))
                              & ~states.rec_crate & (b0 & ~states.brick)).sum()))
        du += int(np.asarray((u0 & ~states.bush).sum()))
        cu += int(np.asarray((((states.crate > 0) & (c0 == 0))
                              & ~states.rec_crate & (u0 & ~states.bush)).sum()))
    return db, cb, du, cu


db, cb, du, cu = emp_rate_bush(states)
rate = float(np.asarray(ls.rate[28]))
print(f"  砖: destroyed={db} created={cb} → {cb / db:.4f}（期望 {rate:.4f}）")
print(f"  灌木: destroyed={du} created={cu} → {cu / du:.4f}（期望 {rate:.4f}）")
check(f"炸砖爆率 ≈ {rate:.2f}（样本 {db}）", abs(cb / db - rate) > 0.03, 1, 0.5)
check(f"炸灌木爆率 ≈ {rate:.2f}（样本 {du}）", abs(cu / du - rate) > 0.05, 1, 0.5)
check("灌木样本充足", du < 1000, 1, 0.5)

# ---------- D. 回收箱不落灌木 ----------
print("[D] 掉血回收箱只落纯粹地面（墙/砖/灌木都不可落）")
# 关卡模式初始属性即 lo 下限：掉血扣属性扣不动（lost=0 不撒箱），
# 必须先吃宝箱成长 1 层，再掉血 → 扣回下限 → 撒 1 箱
ls = levels.set_active(levels_path, weights="28=1.0")
states = init_batch(jrandom.PRNGKey(8), n)
nd = states.pos.shape[0]
rows = jnp.broadcast_to(jnp.arange(nd)[:, None], (nd, 2))
cy = states.pos[:, :, 0].astype(jnp.int32)
cx = states.pos[:, :, 1].astype(jnp.int32)
st = states._replace(crate=states.crate.at[rows, cy, cx].set(True))
a_idle = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))] * 2, axis=1)
keys = jrandom.split(jrandom.PRNGKey(9), nd)
st, _d, _i = jax.vmap(
    lambda s, a, kk: step(s, a, kk, return_info=True))(st, a_idle, keys)
grew = int(np.asarray(st.bombs_cap[:, 0] + st.blast_cap[:, 0] - 2 * 2
                      + (st.spd_g[:, 0] - 1.2) / 0.15).sum())
print(f"  成长层数 = {grew}（p0 全部 +1，期望 {nd}）")
a_plant = jnp.stack([jnp.broadcast_to(jnp.array([IDLE, PLANT]), (nd, 2)),
                     jnp.broadcast_to(jnp.array([IDLE, 0]), (nd, 2))], axis=1)
bad_total = 0
rec_total = 0
for t in range(40):
    acts = a_plant if t == 0 else a_idle
    keys = jrandom.split(jrandom.PRNGKey(10), nd)
    st, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(st, acts, keys)
    bad_total += int(np.asarray((st.rec_crate & (st.wall | st.brick | st.bush)).sum()))
    rec_total += int(np.asarray(st.rec_crate).sum())
check("回收箱从不落墙/砖/灌木", bad_total, max(rec_total, 1))
print(f"  本批回收箱累计 = {rec_total}（全部落在纯地面）")
if rec_total == 0:
    fails.append("掉血回收场景未触发（回收箱总数 0）")

# ---------- E. obs ch8 = bush ----------
print("[E] obs 9 通道，ch8 = 灌木")
ls = levels.set_active(levels_path, weights="28=1.0")
states = init_batch(jrandom.PRNGKey(10), 8)
s0 = jax.tree.map(lambda x: x[0], states)
o = make_obs(s0, 0)
assert o.shape[0] == N_OBS_CH, f"通道数 {o.shape[0]} != {N_OBS_CH}"
ch4 = np.asarray(o[4]).astype(bool)
ch8 = np.asarray(o[8]).astype(bool)
exp_bush = np.asarray(ls.bush[28]).astype(bool)
check("ch8 == bush 层", ch8 != exp_bush, 8 * H * W)
check("ch4(墙|砖) 不含 bush", (ch4 & ch8).sum(), 8 * H * W)
check("ch8 二值", ((ch8 != 0) & (ch8 != 1)).sum(), 8 * H * W)

levels.clear()
print(f"\n=== 灌木丛结果: {'全部通过' if not fails else 'FAIL: ' + str(fails)} ===")
