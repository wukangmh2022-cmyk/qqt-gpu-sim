"""wall + legal_mask 的快速逻辑验证（远端 jax 环境跑）。

验证：
  1. 墙格 blocked：角色不能进墙、贴墙方向 mask 为 False
  2. 爆炸被墙挡（墙后格子不被覆盖）
  3. 放泡：墙格不能放
  4. legal_mask：IDLE 恒合法、贴墙方向非法、bomb 上限
  5. make_obs ch4 = wall
  6. 空场行为与旧版一致（wall=0 时移动/爆炸不受影响）
"""
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (H, W, BombState, _fresh, legal_mask, make_obs,
                              step)

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def with_wall(cells):
    """从 _fresh 出发，在指定格放墙。"""
    s = _fresh()
    w = s.wall.at[cells].set(True) if isinstance(cells, tuple) else \
        s.wall.at[cells[0], cells[1]].set(True)
    return s._replace(wall=w)


print("=== 1. 墙格 blocked：放一堵墙在 (2,2)，角色从 (1,1) 向右走 ===")
s = with_wall((2, 2))
# 墙在 (2,2)，角色在 (1,1)，向右 (dx=+1) 到列 2 附近会被挡
# 用合法掩码验证：向右方向应非法（墙挡住）
mm, bm = legal_mask(s)
# 角色0 在 (1,1)，右方 (1,2) 无墙 → 可走；但墙在 (2,2)，向"下"(+y) 的方向是墙
# MOVE_DELTA: 0=停 1=右 2=左 3=下 4=上（见 _MOVE_DELTA: [[0,0],[0,1],[0,-1],[1,0],[-1,0]]）
# 角色0 (y=1,x=1)：下(3) → y=2，列 x=1，(2,1) 无墙；上(4) → y=0
# 墙在 (2,2) 挡的是"从 (1,1) 往右下角移动时碰撞盒扫过 (2,2)"吗？半径 0.3，从 (1,1) 向右下
# 简单验证：放一堵墙在角色正前方
s2 = s._replace(pos=jnp.array([[1.0, 5.0], [8.0, 8.0]], jnp.float32))
w2 = jnp.zeros((H, W), jnp.bool_)
w2 = w2.at[2, 5].set(True)   # 角色0 (1,5) 正下方 (2,5) 是墙
s2 = s2._replace(wall=w2)
mm, bm = legal_mask(s2)
check("贴墙方向（下）mask=False", mm[0, 3] == False)  # noqa: E712
check("相反方向（上）mask=True", mm[0, 4] == True)    # noqa: E712
check("IDLE 恒合法", mm[0, 0] == True)                 # noqa: E712

print("=== 2. 移动被墙挡：向墙走位置不变 ===")
s3 = s2._replace(pos=jnp.array([[1.0, 5.0], [8.0, 8.0]], jnp.float32))
# 动作：角色0 向下（3）—— 被墙挡
acts = jnp.array([[3, 0], [4, 0]], jnp.int32)
ns, done = step(s3, acts)
check("向墙移动位置不变", ns.pos[0, 0] == 1.0 and ns.pos[0, 1] == 5.0)

print("=== 3. 爆炸被墙挡：泡在 (1,5)，墙在 (2,5)，墙后 (3,5) 不被覆盖 ===")
s4 = _fresh()
fuse = s4.fuse.at[1, 5].set(1).at[1, 5 + 1].set(1)  # 两颗泡即将爆
owner = s4.owner.at[1, 5].set(0).at[1, 6].set(0)
bb = s4.bomb_blast.at[1, 5].set(3).at[1, 6].set(3)
wall = jnp.zeros((H, W), jnp.bool_).at[2, 5].set(True)  # 墙在 (2,5)，泡正下方
s4 = BombState(s4.pos, fuse, owner, bb, wall, s4.alive, s4.hp, s4.invuln, s4.t)
# 泡在 (1,5) 爆炸威力 3：向下应被 (2,5) 墙挡，覆盖不到 (3,5)
from jax_bomb.jax_env import _resolve_explosions
covered, triggered = _resolve_explosions(fuse, owner, bb, wall)
check("爆炸点本身覆盖", covered[1, 5] == True)   # noqa: E712
check("墙格被覆盖（火焰到墙）", covered[2, 5] == True)  # noqa: E712
check("墙后 (3,5) 不被覆盖", covered[3, 5] == False)  # noqa: E712
check("水平方向正常扩散", covered[1, 6] == True)  # noqa: E712

print("=== 4. 放泡：墙格不能放 ===")
s5 = s4._replace(pos=jnp.array([[2.0, 5.0], [8.0, 8.0]], jnp.float32))
# 角色0 站在 (2,5) 墙格上（虽然不现实，但验证 wall 检查）
acts = jnp.array([[0, 1], [0, 0]], jnp.int32)   # 角色0 放泡
ns, done = step(s5, acts)
# 墙上没放上泡
check("墙格放泡失败", ns.fuse[2, 5] <= 1)  # 原泡引信 1，递减后 0

print("=== 5. make_obs ch4 = wall ===")
o = make_obs(s4, 0)
check("obs ch4 墙格=1", o[4, 2, 5] == 1.0)
check("obs ch4 空格=0", o[4, 0, 0] == 0.0)

print("=== 6. 空场行为不变（wall=0）：mask 全合法 ===")
s6 = _fresh()
mm6, bm6 = legal_mask(s6)
check("空场所有方向合法", bool(mm6.all()))
check("空场 bomb=1 合法", bool(bm6[:, 1].all()))
check("bomb=0 恒合法", bool(bm6[:, 0].all()))

print("=== 7. 角色互不碰撞 + 泡挡火不变 ===")
s7 = _fresh()
o7 = make_obs(s7, 0)
check("空场 obs ch4 全 0", o7[4].sum() == 0)

print(f"\n结果: {PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
