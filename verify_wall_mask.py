"""wall + legal_mask 的快速逻辑验证（远端 jax 环境跑）。

方向编码与 torch 一致：0=上 1=下 2=左 3=右 4=停(IDLE)。
验证：
  1. 移动与方向编码：上/下/左/右/停 五方向位移正确
  2. 墙 blocked：向墙移动贴墙停（撞墙前停下，不穿墙）
  3. 贴死方向 mask=False（"按了也一格都动不了"才屏蔽）；IDLE 恒合法
  4. 空场 mask 全合法（含 IDLE 与 bomb=0）
  5. 爆炸被墙挡：墙格覆盖、墙后不覆盖（经 step 引信递减触发）
  6. 放泡：墙格不能放
  7. make_obs ch4 = wall
  8. bomb mask：bomb=0 恒合法；死亡角色 move/bomb 整行放开
"""
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
from jax_bomb.jax_env import (BLAST, FUSE, H, W, MAX_BOMBS, MAX_HP, BombState,
                              _fresh, legal_mask, make_obs, step)

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
    w = s.wall.at[cells].set(True)
    return s._replace(wall=w)


def place_at(pos, s):
    """把玩家 0 放到 pos，玩家 1 放到远处角落避免干扰。"""
    return s._replace(pos=jnp.array([pos, [H - 2.0, W - 2.0]], jnp.float32))


print("=== 1. 方向编码：上/下/左/右/停 ===")
s = _fresh()
s = place_at([5.0, 5.0], s)
act_idle = jnp.array([[4, 0], [4, 0]], jnp.int32)      # 停
ns, _ = step(s, act_idle)
check("IDLE 位置不变", ns.pos[0, 0] == 5.0 and ns.pos[0, 1] == 5.0)
act_up = jnp.array([[0, 0], [4, 0]], jnp.int32)        # 上
ns, _ = step(s, act_up)
check("上: y 减小", ns.pos[0, 0] < 5.0)
act_down = jnp.array([[1, 0], [4, 0]], jnp.int32)      # 下
ns, _ = step(s, act_down)
check("下: y 增大", ns.pos[0, 0] > 5.0)
act_left = jnp.array([[2, 0], [4, 0]], jnp.int32)      # 左
ns, _ = step(s, act_left)
check("左: x 减小", ns.pos[0, 1] < 5.0)
act_right = jnp.array([[3, 0], [4, 0]], jnp.int32)     # 右
ns, _ = step(s, act_right)
check("右: x 增大", ns.pos[0, 1] > 5.0)

print("=== 2. 墙 blocked：向墙移动贴墙停（不穿墙） ===")
# 墙在 (6,5)，角色 (5,5) 正下方（y+1 行）整行放墙保证贴停判定干净
s = with_wall((6, 5))
s = place_at([5.0, 5.0], s)
ns, _ = step(s, jnp.array([[1, 0], [4, 0]], jnp.int32))   # 下
check("向墙移动不穿墙（y < 墙行 6 上缘 6-0.3）", ns.pos[0, 0] < 6.0 - 0.3 + 1e-3)
check("撞墙滑移生效（y 前进了 > 0）", ns.pos[0, 0] > 5.0)

print("=== 3. 贴死方向 mask=False / 相对方向 True / IDLE 恒合法 ===")
# 角色先向下撞墙（墙在 y=6 行）滑到贴死位置，再探测：贴死方向 mask=False
s = with_wall((6, 5))
s = place_at([5.0, 6.0], s)
s, _ = step(s, jnp.array([[1, 0], [4, 0]], jnp.int32))   # 向下撞墙 → 停到 6-0.3-EPS
mm, bm = legal_mask(s)
check("贴死方向（下=1）mask=False", mm[0, 1] == False)  # noqa: E712
check("相反方向（上=0）mask=True", mm[0, 0] == True)    # noqa: E712
check("IDLE(4) 恒合法", mm[0, 4] == True)               # noqa: E712

print("=== 4. 空场 mask 全合法（IDLE 在内） ===")
s6 = _fresh()
mm6, bm6 = legal_mask(s6)
check("空场所有方向合法", bool(mm6.all()))
check("空场 bomb=1 合法", bool(bm6[:, 1].all()))
check("bomb=0 恒合法", bool(bm6[:, 0].all()))

print("=== 5. 爆炸被墙挡（经 step 触发：fuse=1 递减到 0 爆炸） ===")
# 泡在 (5,5) 威力 7，墙在 (5,6)，玩家1 在 (5,7) 墙后 —— 不应被烧
s = _fresh()
fuse = s.fuse.at[5, 5].set(1)
owner = s.owner.at[5, 5].set(0)
bb = s.bomb_blast.at[5, 5].set(7)
wall = jnp.zeros((H, W), jnp.bool_).at[5, 6].set(True)
s = BombState(s.pos, fuse, owner, bb, wall, s.alive, s.hp, s.invuln, s.t)
s = s._replace(pos=jnp.array([[8.0, 8.0], [5.0, 7.0]], jnp.float32))
ns, _ = step(s, jnp.array([[4, 0], [4, 0]], jnp.int32))   # 都停，泡爆炸
check("墙后玩家不掉血", ns.hp[1] == MAX_HP)
# 对照：无墙时墙后位置被覆盖 → 掉血
s2 = _fresh()
fuse2 = s2.fuse.at[5, 5].set(1)
owner2 = s2.owner.at[5, 5].set(0)
bb2 = s2.bomb_blast.at[5, 5].set(7)
s2 = BombState(s2.pos, fuse2, owner2, bb2, jnp.zeros((H, W), jnp.bool_),
               s2.alive, s2.hp, s2.invuln, s2.t)
s2 = s2._replace(pos=jnp.array([[8.0, 8.0], [5.0, 7.0]], jnp.float32))
ns2, _ = step(s2, jnp.array([[4, 0], [4, 0]], jnp.int32))
check("无墙对照：墙后位置被烧掉血", ns2.hp[1] < MAX_HP)

print("=== 6. 放泡：墙格不能放 ===")
s = with_wall((5, 5))
s = place_at([5.0, 5.0], s)
ns, _ = step(s, jnp.array([[4, 1], [4, 0]], jnp.int32))   # 玩家0 脚下是墙，放泡
check("墙格放泡失败（fuse 保持 0）", ns.fuse[5, 5] == 0)
s2 = _fresh()
s2 = place_at([5.0, 5.0], s2)
ns2, _ = step(s2, jnp.array([[4, 1], [4, 0]], jnp.int32))
check("空地放泡成功（fuse=FUSE）", ns2.fuse[5, 5] == FUSE)

print("=== 7. make_obs ch4 = wall ===")
o = make_obs(with_wall((2, 5)), 0)
check("obs ch4 墙格=1", o[4, 2, 5] == 1.0)
check("obs ch4 空格=0", o[4, 0, 0] == 0.0)

print("=== 8. 死亡角色 move/bomb 整行放开 ===")
s = _fresh()
s = s._replace(alive=jnp.array([True, False], jnp.bool_))
mm8, bm8 = legal_mask(s)
check("死亡玩家 move 全合法", bool(mm8[1].all()))
check("死亡玩家 bomb 全合法", bool(bm8[1].all()))

print(f"\n结果: {PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
