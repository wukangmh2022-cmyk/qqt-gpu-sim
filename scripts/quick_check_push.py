"""推箱子机制快速校验（CPU 可跑，~30s）：对齐 Web sim.js 推箱语义。

验证（对齐 Web sim.js:373-410 语义）：
  1) 持续推 3 tick（PUSH_TIME=0.3）→ 箱子移一格（brick/pushable 同步平移）
  2) 目标格有砖 → 推不动（计时清零，箱子不动）
  3) 箱子被爆炸覆盖 → 整箱消失（brick/pushable 同时清除）
  4) 中断后计时**保留**（Web pushT 挂在箱子上，玩家走开不清零；
     推 2 + 停 2 + 推 1 → 累计 0.3 照样移动；只有被挡/被炸才清零）

用法：cd qqt-gpu-sim && .venv/bin/python scripts/quick_check_push.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb import levels
from jax_bomb.jax_env import (H, W, init_batch, step, PUSH_TIME,
                              BombState)

MOVE_RIGHT = 3
IDLE = 4

# 用 id=1（比赛02，26 个 1×1 可推箱）作为测试图
levels.set_active('web/assets/maps/levels.json',
                  weights='1=1.0,0=0.0,2=0.0,3=0.0,4=0.0')
n = 4
key = jrandom.PRNGKey(7)
states = init_batch(key, n)          # 全部 env 都是 id=1 图
fails = []


def check(name, cond, detail):
    ok = bool(np.asarray(cond).all())
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


def env0(states):
    """取 env 0 的字段（vmap 后取第一个）。"""
    return jax.tree.map(lambda x: x[0], states)


# ---- 找一个可推箱格 + 它的左侧空地，把 P0 贴过去 ----
s0 = env0(states)
pb = np.asarray(s0.pushable)
boxes = np.argwhere(pb)
assert len(boxes) > 0, "测试图没有可推箱？"
# 找"左侧空地 + 右侧空地"的箱子（推右：P0 在左、箱子目标格在右）
cand = None
for (rr, cc) in boxes:
    if cc - 1 >= 0 and cc + 1 < W and not pb[rr, cc - 1] and not pb[rr, cc + 1]:
        cand = (rr, cc)
        break
assert cand is not None, "没有左右都空的箱子，换测试图"
r, c = cand
left = c - 1
s_target = (r, c + 1)               # 箱子要去的格

# 手动放置：P0 在 (r, left)，P1 丢远角（不干扰）
def place(env, pr0, pc0):
    # P0 贴箱：x = pc0 + 0.56（前缘格 floor(x+radius) 才落到箱子格）
    env = env._replace(pos=env.pos.at[0].set(jnp.array([pr0 + 0.5, pc0 + 0.56])))
    env = env._replace(pos=env.pos.at[1].set(jnp.array([12.5, 14.5])))
    return env


# ---- 测试 1：连续推 3 tick，箱子右移一格 ----
print("== 测试 1：连续推 3 tick → 箱子右移一格 ==")
st = place(env0(states), r, left)
a_r = jnp.array([[MOVE_RIGHT, 0], [IDLE, 0]], jnp.int32)
s1 = st
for _ in range(3):
    s1, done, info = step(s1, a_r, jrandom.PRNGKey(1), auto_reset=False,
                          return_info=True)
check("箱子原格 brick 已清", not bool(s1.brick[r, left + 1]),
      f"({r},{left+1}) brick={bool(s1.brick[r, left+1])}")
check("箱子目标格 brick 已占", bool(s1.brick[r, c + 1]),
      f"({r},{c+1}) brick={bool(s1.brick[r, c+1])}")
check("pushable 同步移动", bool(s1.pushable[r, c + 1])
      and not bool(s1.pushable[r, c]),
      f"pushable ({r},{c})={bool(s1.pushable[r,c])} ({r},{c+1})={bool(s1.pushable[r,c+1])}")

# ---- 测试 2：目标格有砖 → 推不动 ----
print("== 测试 2：目标格有砖 → 推不动 ==")
st = place(env0(states), r, left)
# 在箱子目标格放一块砖
st = st._replace(brick=st.brick.at[r, c + 1].set(True))
s2 = st
for _ in range(3):
    s2, _ = step(s2, a_r, jrandom.PRNGKey(2), auto_reset=False)
check("箱子原地不动", bool(s2.brick[r, c]) and bool(s2.pushable[r, c])
      and bool(s2.brick[r, c + 1]),
      f"箱子仍占 ({r},{c})，挡路砖 ({r},{c+1}) 还在")

# ---- 测试 3：爆炸清箱 ----
print("== 测试 3：箱子被炸 → 消失 ==")
st = place(env0(states), r, left)
# 直接在箱子正下方一格 (r+1,c) 埋泡（引信 1 tick 后爆，blast 半径 ≥1 覆盖箱子格）；
# 两名玩家停远角避免受伤干扰
st = st._replace(pos=st.pos.at[0].set(jnp.array([11.5, 1.5])))
st = st._replace(pos=st.pos.at[1].set(jnp.array([11.5, 13.5])))
st = st._replace(fuse=st.fuse.at[r + 1, c].set(1),
                 owner=st.owner.at[r + 1, c].set(0),
                 bomb_blast=st.bomb_blast.at[r + 1, c].set(1))
a_idle = jnp.array([[IDLE, 0], [IDLE, 0]], jnp.int32)
s3, _ = step(st, a_idle, jrandom.PRNGKey(3), auto_reset=False)
check("箱子 brick 被炸清", not bool(s3.brick[r, c]),
      f"({r},{c}) brick={bool(s3.brick[r,c])}")
check("箱子 pushable 被炸清", not bool(s3.pushable[r, c]),
      f"({r},{c}) pushable={bool(s3.pushable[r,c])}")

# ---- 测试 4：中断推动计时保留（Web: pushT 挂箱上，走开不清零）----
print("== 测试 4：中断后计时保留（推2+停2+推1 → 移动）==")
st = place(env0(states), r, left)
a_stop = jnp.array([[IDLE, 0], [IDLE, 0]], jnp.int32)
s4 = st
s4, _ = step(s4, a_r, jrandom.PRNGKey(4), auto_reset=False)     # 推 1（0.1）
s4, _ = step(s4, a_r, jrandom.PRNGKey(5), auto_reset=False)     # 推 2（0.2）
for _ in range(2):
    s4, _ = step(s4, a_stop, jrandom.PRNGKey(6), auto_reset=False)  # 走开（保留 0.2）
s4, _ = step(s4, a_r, jrandom.PRNGKey(7), auto_reset=False)     # 再推 1（0.3 → 移动）
check("中断后累计推动成功", bool(s4.brick[r, c + 1])
      and not bool(s4.brick[r, c]),
      f"箱子到 ({r},{c+1})")

print("----")
if fails:
    print(f"FAIL: {fails}")
    sys.exit(1)
print("ALL PASS")
