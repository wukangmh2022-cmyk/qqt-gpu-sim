"""训练前快速校验（~2 分钟）：行走坐标 + 输入 patch 格子信息在移动中的正确性。

1) 格通道逐格精确：obs ch4 == wall|brick、ch7 == crate、ch6 == t/MAX_STEPS
   （逐格逐 tick；corridor 关含 brick，ch4 语义就是 墙|砖）
2) 行走坐标：ch0/ch2 splat 峰值与 states.pos 偏差 ≤ 1 格（存活玩家；
   ch0=自己、ch2=对手，按视角交叉核对）
3) 碰撞：移动后存活玩家所在格不可为墙/砖（AABB 中心不进阻塞格），pos 不越界
4) 不穿墙：脚本化朝已知墙连续右移，位移被墙拦停（x 终值 < 墙格 x）

任一 FAIL 建议先修再开训。用法：cd qqt-gpu-sim && python3 quick_check_obs_move.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import (H, W, MAX_STEPS, init_batch, step)
from jax_bomb.jax_train import both_perspectives

key = jrandom.PRNGKey(7)
n = 2048
states = init_batch(key, n)
key = jrandom.split(key)[0]
fails = []
total_checks = 0


def check(name, violation_mask, n_total):
    global total_checks
    bad = int(np.asarray(jnp.sum(violation_mask)))
    total_checks += 1
    print(f"  [{'PASS' if bad == 0 else 'FAIL'}] {name}: {bad}/{n_total} 违例")
    if bad != 0:
        fails.append(name)


T = 6
for t in range(T):
    obs = both_perspectives(states)              # (2N, C, H, W)
    o0, o1 = obs[:n], obs[n:]
    pos0, pos1 = states.pos[:, 0], states.pos[:, 1]
    alive0, alive1 = states.alive[:, 0], states.alive[:, 1]

    # 1) 格通道逐格精确（ch4 语义 = 墙|砖，与 blocking 一致）
    solid = (states.wall | states.brick).astype(jnp.float32)
    check(f"t{t} ch4==wall|brick 逐格",
          ~((o0[:, 4] == solid) & (o1[:, 4] == solid)),
          2 * n * H * W)
    crate_f = (states.crate > 0).astype(jnp.float32)   # ch7 = 道具存在性（int8 编码 >0）
    check(f"t{t} ch7==crate存在性 逐格",
          ~((o0[:, 7] == crate_f) & (o1[:, 7] == crate_f)),
          2 * n * H * W)
    exp6 = jnp.full((n, H, W), float(t) / MAX_STEPS, jnp.float32)
    check(f"t{t} ch6==t/MAX 逐格", ~((o0[:, 6] == exp6) & (o1[:, 6] == exp6)),
          2 * n * H * W)

    # 2) splat 峰值 ≈ pos（±1 格；ch0=自己视角、ch2=对手视角，交叉核对）
    def peak(ch):
        return jnp.unravel_index(jnp.argmax(ch.reshape(n, -1), axis=1), (H, W))
    py, px = peak(o0[:, 0])
    check(f"t{t} 视角0 ch0(自己) 峰值==pos0(±1)",
          ((jnp.abs(py - pos0[:, 0]) > 1) | (jnp.abs(px - pos0[:, 1]) > 1))
          & alive0, n)
    py, px = peak(o0[:, 2])
    check(f"t{t} 视角0 ch2(对手) 峰值==pos1(±1)",
          ((jnp.abs(py - pos1[:, 0]) > 1) | (jnp.abs(px - pos1[:, 1]) > 1))
          & alive1, n)
    py, px = peak(o1[:, 0])
    check(f"t{t} 视角1 ch0(自己) 峰值==pos1(±1)",
          ((jnp.abs(py - pos1[:, 0]) > 1) | (jnp.abs(px - pos1[:, 1]) > 1))
          & alive1, n)
    py, px = peak(o1[:, 2])
    check(f"t{t} 视角1 ch2(对手) 峰值==pos0(±1)",
          ((jnp.abs(py - pos0[:, 0]) > 1) | (jnp.abs(px - pos0[:, 1]) > 1))
          & alive0, n)

    # 3) 随机动作步进 → 碰撞/越界检查（actions 逐 env (2,2)）
    key, k0 = jrandom.split(key)
    acts = jrandom.randint(k0, (n, 2, 2), 0, 5)
    keys = jrandom.split(key, n)
    new_states, _done, _info = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(states, acts, keys)
    for me in (0, 1):
        p = new_states.pos[:, me]
        cy = jnp.clip(p[:, 0].astype(jnp.int32), 0, H - 1)
        cx = jnp.clip(p[:, 1].astype(jnp.int32), 0, W - 1)
        blocked_cell = (new_states.wall | new_states.brick).reshape(
            n, -1)[jnp.arange(n), cy * W + cx]
        oob = ((p[:, 0] < 0) | (p[:, 0] > H) | (p[:, 1] < 0) | (p[:, 1] > W))
        check(f"t{t} 玩家{me} 中心不在墙内/不越界",
              (blocked_cell & new_states.alive[:, me]) | oob, n)
    states = new_states

# 4) 脚本化撞墙：先随机走 3 tick 让玩家散开，再找玩家0右侧相邻格是墙/砖、
#    本身所在格开放的 env，向右移 5 tick —— 位移必须被墙拦停（x 终值 < 墙格 x）
key, k0 = jrandom.split(key)
states = init_batch(key, n)
key = jrandom.split(key)[0]
for _ in range(3):
    k0, key = jrandom.split(key)
    acts = jrandom.randint(k0, (n, 2, 2), 0, 5)
    keys = jrandom.split(key, n)
    states, _d, _i = jax.vmap(
        lambda s, a, kk: step(s, a, kk, return_info=True))(states, acts, keys)
pos = states.pos[:, 0]
cy = pos[:, 0].astype(jnp.int32)
cx = pos[:, 1].astype(jnp.int32)
flat = (states.wall | states.brick).reshape(n, -1)
right_wall = (flat[jnp.arange(n), cy * W + jnp.clip(cx + 1, 0, W - 1)]
              & (cx < W - 1) & states.alive[:, 0])
cand = np.asarray(jnp.where(right_wall)[0])[:512]
if len(cand) == 0:
    print("  [SKIP] 4) 无右侧邻墙的候选 env，跳过撞墙脚本检查")
else:
    wall_x = np.asarray(cx + 1)[cand]
    x_before = np.asarray(pos[:, 1])[cand]
    for _ in range(5):
        keys = jrandom.split(key, n)
        acts = jnp.stack([jnp.stack([jnp.full((n,), 3, jnp.int32),
                                     jnp.zeros((n,), jnp.int32)], axis=-1),
                          jnp.stack([jnp.zeros((n,), jnp.int32),
                                     jnp.zeros((n,), jnp.int32)], axis=-1)],
                         axis=1)
        states, _d, _i = jax.vmap(
            lambda s, a, kk: step(s, a, kk, return_info=True))(
            states, acts, keys)
    x0 = np.asarray(states.pos[:, 0, 1])[cand]
    pen = np.sum(x0 >= wall_x)          # 终值 ≥ 墙格 x = 穿墙
    moved = np.sum(x0 > x_before + 1e-3)   # 有实际位移才算有效检查
    print(f"  [{'PASS' if pen == 0 else 'FAIL'}] 4) 右移5tick 不穿墙 "
          f"({len(cand)} 个候选 {moved} 有位移，{pen} 穿墙)")
    if pen:
        fails.append("不穿墙")

# 5) 防穿炮（中心路径硬约束）：放泡后能离开泡格，但不能踩回泡格中心
from jax_bomb.jax_env import _move_player
blocked_b = jnp.zeros((H, W), jnp.bool_).at[2, 2].set(True)
r_leave = _move_player(jnp.array([2.5, 2.5]), jnp.int32(2), jnp.bool_(True),
                       blocked_b, jnp.float32(1.4))   # 左移离开泡格
r_back = _move_player(jnp.array([2.5, 1.44]), jnp.int32(3), jnp.bool_(True),
                      blocked_b, jnp.float32(1.4))    # 右移想回泡格
r_free = _move_player(jnp.array([2.5, 1.44]), jnp.int32(3), jnp.bool_(True),
                      jnp.zeros((H, W), jnp.bool_), jnp.float32(1.4))
pen = 0
if not (float(np.asarray(r_leave[1])) < 2.0):
    pen += 1; print("  [FAIL] 放泡后能离开泡格")
if not (jnp.floor(r_back[1]) < 2):
    pen += 1; print("  [FAIL] 中心不能踩回泡格（穿炮）")
if not (float(np.asarray(r_free[1])) > 2.0):
    pen += 1; print("  [FAIL] 无泡时正常移动")
total_checks += 1
print(f"  [{'PASS' if pen == 0 else 'FAIL'}] 5) 防穿炮（放泡后能出/回不去/无泡正常）")
if pen:
    fails.append("防穿炮")

print(f"\n=== 结果: {'全部通过，可开训' if not fails else 'FAIL: ' + str(fails)} "
      f"（{total_checks} 项检查）===")
