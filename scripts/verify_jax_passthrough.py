#!/usr/bin/env python3
"""
纯 Python 复刻 JAX _impassable_pair / _resolve_axis / _move_player，
逐条验证 JS 扫出的 3600 组穿泡数据点。

JAX 参数（jax_env.py）：
  H=13, W=15, STEP=0.3, RADIUS=0.36, EPS=1e-4, MAX_SWEEP=3
  _MOVE_DELTA = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]

逻辑逐行对齐 jax_bomb/jax_env.py:629-733。
"""
import json, math

H = 13
W = 15
STEP = 0.3
RADIUS = 0.45 * 0.8   # 0.36
EPS = 1e-4
MAX_SWEEP = 3
MOVE_DELTA = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]


def _impassable_pair(blocked_flat, r0, c0, r1, c1, y, x):
    """逐行对齐 jax_env.py:629-647。返回 True=不可通行。"""
    h, w = H, W
    oob = (r0 < 0 or r0 >= h or c0 < 0 or c0 >= w or
           r1 < 0 or r1 >= h or c1 < 0 or c1 >= w)
    idx0 = max(0, min(r0, h-1)) * w + max(0, min(c0, w-1))
    idx1 = max(0, min(r1, h-1)) * w + max(0, min(c1, w-1))
    solid0 = blocked_flat[idx0]
    solid1 = blocked_flat[idx1]
    # ceil + 严格小于（左闭右开）—— 修复版
    r0c = math.floor(y - RADIUS)
    r1c = math.ceil(y + RADIUS)
    c0c = math.floor(x - RADIUS)
    c1c = math.ceil(x + RADIUS)
    in0 = (r0 >= r0c) and (r0 < r1c) and (c0 >= c0c) and (c0 < c1c)
    in1 = (r1 >= r0c) and (r1 < r1c) and (c1 >= c0c) and (c1 < c1c)
    return oob or (solid0 and not in0) or (solid1 and not in1)


def _resolve_axis(coord, delta, other, y, x, blocked_flat, vertical):
    """逐行对齐 jax_env.py:650-689。"""
    sgn = (delta > 0) - (delta < 0)  # jnp.sign 标量版
    if sgn == 0:
        return coord
    old_lead = math.floor(coord - delta + sgn * RADIUS)
    new_lead = math.floor(coord + sgn * RADIUS)
    lo = min(old_lead, new_lead)
    hi = max(old_lead, new_lead)
    span0 = math.floor(other - RADIUS)
    span1 = math.floor(other + RADIUS)

    def hit_at(lead):
        if vertical:
            return _impassable_pair(blocked_flat, lead, span0, lead, span1, y, x)
        return _impassable_pair(blocked_flat, span0, lead, span1, lead, y, x)

    sweep = hi - lo + 1
    found = False
    first = 0
    # fori_loop(0, MAX_SWEEP, body, ...) —— 终点侧优先
    for i in range(MAX_SWEEP):
        lead = hi - i if sgn > 0 else lo + i
        if i < sweep:
            hit = hit_at(lead)
            if hit and not found:
                first = lead
            found = found or hit

    if found:
        if sgn > 0:
            stop_pos = float(first) - RADIUS - EPS
        else:
            stop_pos = float(first) + 1.0 + RADIUS + EPS
        return stop_pos
    return coord


def _move_player(y, x, move, alive, blocked_2d, spd=1.0):
    """逐行对齐 jax_env.py:692-733。返回 (ny, nx)。"""
    dy, dx = MOVE_DELTA[move]
    dy *= STEP * spd
    dx *= STEP * spd
    if not alive or move == 4:
        dy, dx = 0.0, 0.0

    blocked_flat = [blocked_2d[r * W + c] for r in range(H) for c in range(W)]

    ny = _resolve_axis(y + dy, dy, x, y, x, blocked_flat, True)

    # 中心路径硬约束 —— y 段
    start_r = max(0, min(H - 1, math.floor(y)))
    start_c = max(0, min(W - 1, math.floor(x)))
    y_lo = max(0, min(H - 1, math.floor(min(y, ny))))
    y_hi = max(0, min(H - 1, math.floor(max(y, ny))))
    # seg_y = blocked[y_lo..y_lo+MAX_SWEEP, start_c], 排除 rows != start_r & rows <= y_hi
    ok_y = True
    for i in range(MAX_SWEEP):
        r = y_lo + i
        if r <= y_hi and r != start_r:
            if blocked_2d[r * W + start_c]:
                ok_y = False
                break
    if not ok_y:
        ny = y

    nx = _resolve_axis(x + dx, dx, y, ny, x, blocked_flat, False)

    # 中心路径硬约束 —— x 段
    x_lo = max(0, min(W - 1, math.floor(min(x, nx))))
    x_hi = max(0, min(W - 1, math.floor(max(x, nx))))
    cy0 = max(0, min(H - 1, math.floor(ny)))
    ok_x = True
    for i in range(MAX_SWEEP):
        c = x_lo + i
        if c <= x_hi and not (c == start_c and cy0 == start_r):
            if blocked_2d[cy0 * W + c]:
                ok_x = False
                break
    if not ok_x:
        nx = x

    out_y = ny if dy != 0 else y
    out_x = nx if dx != 0 else x
    out_y = max(RADIUS, min(H - RADIUS, out_y))
    out_x = max(RADIUS, min(W - RADIUS, out_x))
    return out_y, out_x


def main():
    with open('/tmp/passthrough_cases.json') as f:
        data = json.load(f)

    cases = data['cases']
    bomb_r, bomb_c = data['bomb']

    # 构造 blocked（只有泡）
    blocked_2d = [0] * (H * W)
    blocked_2d[bomb_r * W + bomb_c] = 1

    print(f"JAX 参数: H={H} W={W} STEP={STEP} RADIUS={RADIUS} EPS={EPS} MAX_SWEEP={MAX_SWEEP}")
    print(f"泡位置: ({bomb_r},{bomb_c})")
    print(f"待验证穿泡案例: {len(cases)} 组")
    print()

    # 逐条验证
    jax_blocked = 0      # JAX 中心路径 block 了（=JAX 有修复，与 JS 一致）
    jax_passthrough = 0  # JAX 放行了（=JAX 穿泡，与 JS 不一致）
    mismatches = []      # JAX 终点入泡格但与 JS 的 resolveAxis 终点不同

    for c in cases:
        y, x = c['y'], c['x']
        move = c['dir']  # 0=up 1=down 2=left 3=right
        js_ny, js_nx = c['ny'], c['nx']

        jax_ny, jax_nx = _move_player(y, x, move, True, blocked_2d, 1.0)

        # JAX 是否移动了（中心路径没 block = 穿泡）
        jax_moved = abs(jax_ny - y) > 0.0001 or abs(jax_nx - x) > 0.0001
        jax_end_in_bomb = (math.floor(jax_ny) == bomb_r and
                           math.floor(jax_nx) == bomb_c)

        if jax_moved and jax_end_in_bomb:
            jax_passthrough += 1
            if len(mismatches) < 20:
                mismatches.append({
                    'y': y, 'x': x, 'dir': move,
                    'js_ny': js_ny, 'js_nx': js_nx,
                    'jax_ny': round(jax_ny, 6), 'jax_nx': round(jax_nx, 6),
                    'jax_moved': True, 'jax_end_in_bomb': True,
                })
        else:
            jax_blocked += 1

    print("=" * 60)
    print(f"JS 穿泡案例总数: {len(cases)}")
    print(f"  → JAX 也穿泡(放行): {jax_passthrough}")
    print(f"  → JAX 拦截(block): {jax_blocked}")
    print()
    if jax_passthrough > 0:
        print(f"⚠️  JAX 存在穿泡 bug！前 {len(mismatches)} 条对比:")
        print(f"    {'y':>8} {'x':>8} {'dir':>4}  {'js_ny':>10} {'js_nx':>10}  {'jax_ny':>10} {'jax_nx':>10}")
        for m in mismatches:
            dirs = ['up', 'down', 'left', 'right']
            print(f"    {m['y']:>8} {m['x']:>8} {dirs[m['dir']]:>4}  {m['js_ny']:>10} {m['js_nx']:>10}  {m['jax_ny']:>10} {m['jax_nx']:>10}")
    else:
        print("✅ JAX 全部拦截 —— JAX 中心路径硬约束生效，无穿泡 bug")


if __name__ == '__main__':
    main()
