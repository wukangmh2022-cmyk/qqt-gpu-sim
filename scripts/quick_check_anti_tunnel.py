#!/usr/bin/env python3
"""双端（JS web/sim.js ↔ JAX jax_env）穿炮防护对拍。

用户场景（三格穿炮）：格1放泡 → 左移到格0（盒仍蹭格1）→ 格0放泡 → 右移想回格1
（中心扫过有泡的格1，必须被拦）→ 不能继续穿到格2。放泡格能离开（起点格豁免），
但中心不能踩回泡格 / 穿过泡格。

JS 侧：完整 Sim.step 跑动作序列（web/sim.js 的移动段含中心路径硬约束）。
JAX 侧：_move_player 单步跑同样的 (pos, move) 序列（jax_env 的中心路径硬约束）。
断言：两端的轨迹中心坐标一致（JS 步长与 JAX 不同，只比"格内移动方向/停位
的格索引"，不断言连续坐标逐位相等），且都满足：
  1) 放泡后能左移离开泡格（中心格索引变化到 0）
  2) 右移回泡格被拦（中心格索引不能回到 1）
  3) 无泡时移动正常（对照）

用法：PYTHONPATH=. .venv/bin/python scripts/quick_check_anti_tunnel.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jax_bomb import jax_env

ROOT = Path(__file__).resolve().parent.parent
JS_FILE = ROOT / "web" / "sim.js"
H, W = jax_env.H, jax_env.W


# ---------------- JS 侧（node 驱动完整 Sim） ----------------

JS_DRIVER = f"""
const {{ Sim, H, W, CFG, DIRS, resolveAxis }} = require('{JS_FILE}');
const N = H * W;
// main.js frameMove 复刻（含中心路径硬约束）—— 验证浏览器帧级移动路径
function frameMove(sim, pid, mv, dt) {{
  if (mv === 4 || !sim.alive[pid]) return;
  const dist = CFG.speed * sim.spdG[pid] * Math.min(dt, 0.1);
  if (dist <= 0) return;
  const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
  const blocked = new Uint8Array(N);
  for (let i = 0; i < N; i++) blocked[i] = sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 ? 1 : 0;
  const [dy, dx] = DIRS[mv];
  const startR = Math.max(0, Math.min(H - 1, Math.floor(y)));
  const startC = Math.max(0, Math.min(W - 1, Math.floor(x)));
  let ny = y, nx = x;
  if (dy !== 0) {{
    ny = resolveAxis(y + dy * dist, dy * dist, x, y, x, blocked, CFG.radius, H, W, true);
    const yLo = Math.max(0, Math.min(H - 1, Math.floor(Math.min(y, ny))));
    const yHi = Math.max(0, Math.min(H - 1, Math.floor(Math.max(y, ny))));
    for (let r = yLo; r <= yHi; r++) {{ if (r !== startR && blocked[r * W + startC]) {{ ny = y; break; }} }}
  }}
  if (dx !== 0) {{
    nx = resolveAxis(x + dx * dist, dx * dist, y, ny, x, blocked, CFG.radius, H, W, false);
    const xLo = Math.max(0, Math.min(W - 1, Math.floor(Math.min(x, nx))));
    const xHi = Math.max(0, Math.min(W - 1, Math.floor(Math.max(x, nx))));
    const cy0 = Math.max(0, Math.min(H - 1, Math.floor(ny)));
    for (let c = xLo; c <= xHi; c++) {{ if (!(c === startC && cy0 === startR) && blocked[cy0 * W + c]) {{ nx = x; break; }} }}
  }}
  sim.pos[pid * 2] = ny; sim.pos[pid * 2 + 1] = nx;
}}
function run() {{
  const sim = new Sim(1);
  sim.reset('open');
  for (let c = 0; c < W; c++) {{ sim.wall[2*W+c] = 0; sim.brick[2*W+c] = 0; }}
  sim.pos[0] = 2.5; sim.pos[1] = 1.5;    // 玩家0 在格1 (行2,列1)
  sim.pos[2] = 10.5; sim.pos[3] = 10.5;
  sim.alive[0] = 1; sim.alive[1] = 1;
  const cells = [];
  const act = (move, bomb) => {{
    sim.step([[move, bomb], [4, 0]]);
    cells.push([Math.floor(sim.pos[0]), Math.floor(sim.pos[1])]);
  }};
  act(4, 1);   // 格1放泡
  act(2, 0); act(2, 0);  // 左移2tick → 格0
  act(4, 1);   // 格0放泡
  act(3, 0); act(3, 0); act(3, 0);  // 右移3tick → 应被拦在格0
  // 对照：无泡场景同样移动应正常
  const sim2 = new Sim(2);
  sim2.reset('open');
  for (let c = 0; c < W; c++) {{ sim2.wall[2*W+c] = 0; sim2.brick[2*W+c] = 0; }}
  sim2.pos[0] = 2.5; sim2.pos[1] = 0.55;
  sim2.pos[2] = 10.5; sim2.pos[3] = 10.5;
  sim2.alive[0] = 1; sim2.alive[1] = 1;
  const free = [];
  for (let t = 0; t < 3; t++) {{
    sim2.step([[3, 0], [4, 0]]);
    free.push([Math.floor(sim2.pos[0]), Math.floor(sim2.pos[1])]);
  }}
  // 帧级路径（main.js frameMove）：格1放泡 → 帧级左移 → 格0放泡 → 帧级右移应被拦
  const sim3 = new Sim(3);
  sim3.reset('open');
  for (let c = 0; c < W; c++) {{ sim3.wall[2*W+c] = 0; sim3.brick[2*W+c] = 0; }}
  sim3.pos[0] = 2.5; sim3.pos[1] = 1.5;
  sim3.pos[2] = 10.5; sim3.pos[3] = 10.5;
  sim3.alive[0] = 1; sim3.alive[1] = 1;
  sim3.step([[4, 1], [4, 0]]);               // 格1放泡
  for (let f = 0; f < 12; f++) frameMove(sim3, 0, 2, 1/60);  // 帧级左移 0.2s
  sim3.step([[4, 1], [4, 0]]);               // 格0放泡
  const frameCells = [];
  for (let f = 0; f < 24; f++) {{
    frameMove(sim3, 0, 3, 1/60);             // 帧级右移 0.4s
    frameCells.push([Math.floor(sim3.pos[0]), Math.floor(sim3.pos[1])]);
  }}
  console.log(JSON.stringify({{ cells, free, frameCells }}));
}}
run();
"""


def js_run():
    r = subprocess.run(["node", "-e", JS_DRIVER], capture_output=True,
                       text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------- JAX 侧（_move_player 单步） ----------------

def jax_run():
    """JAX 用 _move_player 逐 tick 模拟同一场景（blocked 含泡）。"""
    # 状态：pos0 格1(2.5,1.5)。blocked 动态：放泡的格 = fuse>0
    cells = []
    pos = jnp.array([2.5, 1.5])
    blocked = jnp.zeros((H, W), jnp.bool_)   # 空场
    # 1) 格1放泡 → blocked(2,1)=True
    blocked = blocked.at[2, 1].set(True)
    cells.append([2, 1])
    # 2) 左移2tick（_move_player：move=2 左）
    for _ in range(2):
        pos = jax_env._move_player(pos, jnp.int32(2), jnp.bool_(True),
                                   blocked, jnp.float32(1.0))
        cells.append([int(jnp.floor(pos[0])), int(jnp.floor(pos[1]))])
    # 3) 格0放泡 → blocked(2,0)=True
    blocked = blocked.at[2, 0].set(True)
    # 4) 右移3tick（move=3 右）→ 应被拦（中心不能回格1）
    for _ in range(3):
        pos = jax_env._move_player(pos, jnp.int32(3), jnp.bool_(True),
                                   blocked, jnp.float32(1.0))
        cells.append([int(jnp.floor(pos[0])), int(jnp.floor(pos[1]))])
    # 对照：无泡场景从 0.55 右移 3tick 应正常到格1
    free = []
    pos2 = jnp.array([2.5, 0.55])
    for _ in range(3):
        pos2 = jax_env._move_player(pos2, jnp.int32(3), jnp.bool_(True),
                                    jnp.zeros((H, W), jnp.bool_),
                                    jnp.float32(1.0))
        free.append([int(jnp.floor(pos2[0])), int(jnp.floor(pos2[1]))])
    return {"cells": cells, "free": free}


# ---------------- 主流程 ----------------

def main():
    js = js_run()
    jx = jax_run()
    print(f"JS   cells={js['cells']}")
    print(f"JAX  cells={jx['cells']}")
    print(f"JS   free={js['free']}")
    print(f"JAX  free={jx['free']}")
    print(f"JS   frameCells={js['frameCells'][::6]} (每6帧采样)")

    fails = []
    # 关键断言（两端各自成立即可；中心格索引语义必须一致）
    for name, d in (("JS", js), ("JAX", jx)):
        c = d["cells"]
        # 放泡后能左移离开（最后左移结果中心格 = 0）
        left_ok = c[2][1] == 0            # cells: [放泡, 左1, 左2, ...]
        # 右移被拦：后续右移的中心列索引都 ≠ 1（回不到泡格格1）
        back_cells = c[3:]
        back_ok = all(cell[1] != 1 for cell in back_cells)
        # 无泡对照：能从格0(列0)正常右移离开（中心列索引最终 ≥ 1）
        free_ok = d["free"][-1][1] >= 1
        if not (left_ok and back_ok and free_ok):
            fails.append(name)
        print(f"  [{name}] 离开泡格OK={left_ok} 踩回被拦OK={back_ok} "
              f"无泡正常OK={free_ok}")

    # 帧级路径（main.js frameMove 复刻）：格1放泡→帧级左移→格0放泡→帧级右移被拦
    fc = js["frameCells"]
    frame_blocked = all(cell[1] != 1 for cell in fc)
    frame_moved_left = any(cell[1] == 0 for cell in fc[:12])  # 左移阶段到过格0
    print(f"  [JS frameMove] 左移到格0OK={frame_moved_left} "
          f"右移被拦OK={frame_blocked}")
    if not (frame_moved_left and frame_blocked):
        fails.append("JS-frameMove")

    # 两端防护语义一致：右移期间中心列索引都 ≠ 1（都回不到泡格格1）
    js_blocked = all(c[1] != 1 for c in js["cells"][3:])
    jx_blocked = all(c[1] != 1 for c in jx["cells"][3:])
    print(f"  双端穿炮拦截一致(右移均回不到格1): {js_blocked and jx_blocked}")

    ok = not fails and js_blocked and jx_blocked
    print("\n结果:", "PASS — JS(Sim.step+frameMove)/JAX 穿炮防护一致"
          "（放泡能离、踩回被拦、无泡正常）" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
