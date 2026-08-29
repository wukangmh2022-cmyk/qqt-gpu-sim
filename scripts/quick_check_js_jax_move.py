#!/usr/bin/env python3
"""JS(web/sim.js) ↔ JAX(jax_env._resolve_axis) 移动碰撞逐位对拍。

背景：WebGL 碰撞盒半径已从 0.3 改大到 0.45（sim.js CFG.radius，盒 54x54px），
JAX/torch 仍为 0.3 —— 从未做过移动侧的双端验证（之前的双端验证只覆盖宝箱
语义）。本脚本用真实 JS 引擎（node 加载 web/sim.js 的 resolveAxis）与 JAX
_resolve_axis 在同一组确定性场景下对比：

  - 相同 radius 下结果必须逐位一致（公式同构：old/new_lead + span + 盒覆盖豁免）
  - radius 不同（0.3 vs 0.45）时统计不一致的用例数，量化偏差
  - 额外验证：JAX 跨格扫描（MAX_SWEEP，防穿墙）在 JS 步长范围内与 JS 两点
    检查完全一致；在 JAX 大步长（>1 格/tick）下能拦住 JS 两点检查漏掉的
    中间砖（这是 JAX 侧修复的跨格穿墙，JS 速度达不到该步长，无分歧场景）。

用法：PYTHONPATH=. .venv/bin/python scripts/quick_check_js_jax_move.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jax_bomb import jax_env

ROOT = Path(__file__).resolve().parent.parent
JS_FILE = ROOT / "web" / "sim.js"

H, W = jax_env.H, jax_env.W
N = H * W


# ---------------- 场景生成（确定性，覆盖边界/贴墙/斜切/盒覆盖豁免） ----------------

def _blocked_from_walls(wall_cells):
    b = np.zeros(N, np.uint8)
    for r, c in wall_cells:
        b[r * W + c] = 1
    return b


def gen_cases():
    """返回 (coord, delta, other, y, x, blocked_flat, vertical, 说明)。"""
    walls_pool = [
        [(3, 4)], [(5, 7)], [(0, 0)], [(H - 1, W - 1)],
        [(3, 4), (3, 5)], [(5, 7), (6, 7)], [(2, 2), (4, 4)],
        [(7, 3), (7, 4), (7, 5)],   # 横向墙
        [(3, 6), (4, 6), (5, 6)],   # 纵向墙
        [],                          # 空场
    ]
    # 位置：整数格心、0.5 格心、贴近格界、贴近边界
    # 注意：+rad(0.45) 恰为整数（如 0.55→1.0、1.55→2.0、12.55→13.0）是 JS
    #   ceil+严格小于 与 JAX floor+闭区间 的分歧点 —— 必须覆盖这类坐标
    ys = [3.5, 3.0, 3.2, 3.48, 3.51, 0.45, 0.5, H - 0.45, 5.0,
          1.55, 2.55, 4.55, 6.55, 0.05, 0.55, 1.05, 2.05]
    xs = [4.5, 4.0, 4.3, 4.49, 4.51, 0.45, 0.5, W - 0.45, 6.0,
          0.55, 1.55, 3.55, 5.55, 8.55, 10.55, 12.55, 13.55]
    # 步长：JS/JAX 统一现实范围（STEP=0.3 × spdG；大值仅压力测试跨格扫描）
    deltas = [0.3, 0.39, 0.48, 0.57, 0.63, 0.756, 1.059, 1.588]
    cases = []
    for walls in walls_pool:
        blocked = _blocked_from_walls(walls)
        for y in ys:
            for x in xs:
                for delta in deltas:
                    # 垂直：朝 -y（sgn<0）与 +y（sgn>0）两个方向
                    cases.append((y, delta, x, y, x, blocked, True,
                                  f"V {y:.2f},{x:.2f} d={delta} walls={walls}"))
                    cases.append((y, -delta, x, y, x, blocked, True,
                                  f"V- {y:.2f},{x:.2f} d={delta} walls={walls}"))
                    cases.append((x, delta, y, y, x, blocked, False,
                                  f"H {y:.2f},{x:.2f} d={delta} walls={walls}"))
                    cases.append((x, -delta, y, y, x, blocked, False,
                                  f"H- {y:.2f},{x:.2f} d={delta} walls={walls}"))
    return cases


# ---------------- JS 侧（node 加载真实 sim.js，分块避免输入过长） ----------------

def js_resolve(cases, radius, chunk=4000):
    """调 node 运行 web/sim.js 的 resolveAxis，返回逐用例结果。"""
    js_code = f"""
const {{ resolveAxis }} = require('{JS_FILE}');
let buf = '';
process.stdin.on('data', d => buf += d);
process.stdin.on('end', () => {{
  const cases = JSON.parse(buf);
  const out = cases.map(c => resolveAxis(...c));
  console.log(JSON.stringify(out));
}});
"""
    out_all = []
    for i in range(0, len(cases), chunk):
        rows = []
        for coord, delta, other, y, x, blocked, vertical, desc in cases[i:i + chunk]:
            arr = ",".join(str(int(v)) for v in blocked)
            rows.append(
                f"[{coord:.6f},{delta:.6f},{other:.6f},{y:.6f},{x:.6f},"
                f"[{arr}],{radius:.6f},{H},{W},{1 if vertical else 0}]"
            )
        payload = "[" + ",".join(rows) + "]"
        r = subprocess.run(["node", "-e", js_code], input=payload,
                           capture_output=True, text=True, check=True)
        out_all.extend(json.loads(r.stdout.strip().splitlines()[-1]))
    return out_all


# ---------------- JAX 侧（vmap 批处理；jit 在函数内定义 → 每次按当前 RADIUS trace） ----------------

def jax_resolve(cases, radius):
    old = jax_env.RADIUS
    jax_env.RADIUS = radius

    def batch(vertical):
        @jax.jit
        def f(coords, deltas, others, ys, xs, blocked):
            return jax.vmap(
                lambda coord, delta, other, y, x, b:
                jax_env._resolve_axis(coord, delta, other, y, x, b, vertical),
                in_axes=(0, 0, 0, 0, 0, 0),
            )(coords, deltas, others, ys, xs, blocked)
        return f

    try:
        coords = jnp.asarray([c[0] for c in cases], jnp.float32)
        deltas = jnp.asarray([c[1] for c in cases], jnp.float32)
        others = jnp.asarray([c[2] for c in cases], jnp.float32)
        ys = jnp.asarray([c[3] for c in cases], jnp.float32)
        xs = jnp.asarray([c[4] for c in cases], jnp.float32)
        blocked = jnp.asarray([c[5] for c in cases], jnp.bool_)
        verticals = np.asarray([c[6] for c in cases])
        out = np.zeros(len(cases), np.float32)
        for vertical, idx in ((True, np.where(verticals)[0]),
                              (False, np.where(~verticals)[0])):
            if len(idx):
                out[idx] = np.asarray(batch(vertical)(
                    coords[idx], deltas[idx], others[idx],
                    ys[idx], xs[idx], blocked[idx]))
        return out
    finally:
        jax_env.RADIUS = old


# ---------------- 主流程 ----------------

def main():
    cases = gen_cases()
    print(f"共 {len(cases)} 组确定性场景（9 位置 × 8 步长 × 4 方向 × 10 墙型）")

    # 1. 半径一致（0.45 = JS 现值）：JS 可达步长（≤0.63）内必须逐位一致
    reachable = [c for c in cases if abs(c[1]) <= 0.63]
    js45 = js_resolve(reachable, 0.45)
    jx45 = jax_resolve(reachable, 0.45)
    mism = [i for i, (a, b) in enumerate(zip(js45, jx45)) if abs(a - b) > 1e-6]
    print(f"\n[radius=0.45 双端一致(JS可达步长≤0.63)] JS={len(js45)} "
          f"JAX={len(jx45)} 不一致={len(mism)}/{len(reachable)}")
    for i in mism[:5]:
        print(f"  MISMATCH #{i}: JS={js45[i]} JAX={jx45[i]}  {reachable[i][7]}")

    # 2. 半径不一致（JS 0.45 vs JAX 0.3 = 现状）：量化偏差
    jx03 = jax_resolve(reachable, 0.3)
    mism_curr = [i for i, (a, b) in enumerate(zip(js45, jx03)) if abs(a - b) > 1e-6]
    print(f"[现状 JS 0.45 vs JAX 0.3] 不一致={len(mism_curr)}/{len(reachable)}")
    for i in mism_curr[:5]:
        print(f"  MISMATCH #{i}: JS={js45[i]} JAX={jx03[i]}  {reachable[i][7]}")

    # 3. 大步长（>0.63，JS 步长达不到）是 JAX 特有场景：JS 两点检查会漏中间
    #    砖（穿墙），JAX 全扫描必须拦住。不能逐位对比（JS 语义缺失），用
    #    构造用例验证 JAX 不穿墙 + JS 步长范围（≤0.63）内逐位一致。
    small = [c for c in cases if abs(c[1]) <= 0.63]
    js_s = js_resolve(small, 0.45)
    jx_s = jax_resolve(small, 0.45)
    m_s = [i for i, (a, b) in enumerate(zip(js_s, jx_s)) if abs(a - b) > 1e-6]
    print(f"\n[JS 步长范围(≤0.63) 半径0.45] 不一致={len(m_s)}/{len(small)}")

    # 大步长防穿墙（JAX 特性，JS 步长达不到）：中间单砖 + 2 格厚墙必须拦住
    tests = [
        # (起点y, 方向delta, x, 障碍格, 被挡行) —— 下冲/上冲都应停在障碍前
        (3.5, 1.588, 4.5, [(5, 4)], 5, "大步长下冲中间单砖"),
        (5.5, -1.588, 4.5, [(3, 4)], 3, "大步长上冲中间单砖"),
        (3.5, 1.588, 6.0, [(4, 6), (5, 6)], 4, "大步长下冲2格厚墙"),
        (6.5, -1.588, 6.0, [(4, 6), (5, 6)], 5, "大步长上冲2格厚墙"),
    ]
    jx_guard = 0
    for y0, delta, x0, walls, block_row, name in tests:
        b = _blocked_from_walls(walls)
        r = jax_resolve([(y0, delta, x0, y0, x0, b, True, name)], 0.45)
        # 下冲（delta>0）必须停在被挡行上方（< block_row-0.45）；
        # 上冲（delta<0）必须停在被挡行下方（> block_row+1+0.45）
        if delta > 0:
            ok = r[0] < block_row - 0.45 + 1e-3
        else:
            ok = r[0] > block_row + 1 + 0.45 - 1e-3
        if not ok:
            jx_guard += 1
        print(f"  {name}: 结果={r[0]:.4f} "
              f"{'OK' if ok else f'FAIL(期望 {"<"+str(block_row-0.45) if delta>0 else ">"+str(block_row+1.45)})'}")
    print(f"[大步长 JAX 防穿墙] 穿墙用例: {jx_guard}/{len(tests)}")

    ok = len(m_s) == 0 and jx_guard == 0
    print("\n结果:", "PASS — JS↔JAX 同半径 0.45 逐位一致（JS 步长范围内）"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
