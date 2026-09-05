#!/usr/bin/env node
'use strict';

const assert = require('assert');
const Q = require('./sim.js');

const sim = Object.create(Q.Sim.prototype);
const blocked = new Uint8Array(Q.N);
const dist = Q.CFG.stepLen;

// 直上受阻且两侧皆开阔（开阔地撞单障碍）：坚决不主动向障碍中心线归中，保留横向身位（保护半身位）。
// 当前 CFG.radius=0.42，测试起点 y=5.43, x=5.20
blocked[4 * Q.W + 5] = 1;
let [ny, nx] = sim._steer(5.43, 5.20, Q.MOVE_UP, blocked, dist);
assert(Math.abs(nx - 5.20) < 1e-9,
  `开阔地直行受阻时不应强行归中，必须保留横向坐标 5.20，实际 (${ny}, ${nx})`);

// 镜像场景：角色在格中心右侧 (5.80) 时同样不篡改坐标。
[ny, nx] = sim._steer(5.43, 5.80, Q.MOVE_UP, blocked, dist);
assert(Math.abs(nx - 5.80) < 1e-9,
  `开阔地直行受阻时不应强行归中，必须保留横向坐标 5.80，实际 (${ny}, ${nx})`);

// 目标格是可推箱时，直走受阻也必须原地顶箱，不能垂直侧滑。
blocked.fill(0);
blocked[5 * Q.W + 6] = 1;
sim.pushable = new Uint8Array(Q.N);
sim.pushable[5 * Q.W + 6] = 1;
[ny, nx] = sim._steer(5.5, 5.5, Q.MOVE_RIGHT, blocked, dist);
assert(Math.abs(ny - 5.5) < 1e-9 && nx >= 5.5 && nx < 6,
  `顶右侧箱子时只能水平贴近、不能上下侧滑，实际 (${ny}, ${nx})`);

// 向下一格的入口两侧是砖、碰撞盒仅擦住左砖：第一步应横向归中，下一步
// 能向下进入，不能把部分直走误判成成功后永远卡在边缘。
blocked.fill(0);
sim.pushable.fill(0);
blocked[6 * Q.W + 4] = 1;
blocked[6 * Q.W + 6] = 1;
let y = 5.5, x = 5.12;
[y, x] = sim._steer(y, x, Q.MOVE_DOWN, blocked, dist);
assert(x > 5.12 && Math.abs(y - 5.5) < 1e-9,
  `擦左砖时第一步应向右归中，实际 (${y}, ${x})`);
const yBefore = y;
[y, x] = sim._steer(y, x, Q.MOVE_DOWN, blocked, dist);
assert(y > yBefore + 0.25,
  `归中后下一步应能向下进入通道，实际 y=${y}`);

// -------------------------------------------------------------
// 回归测试：边界直走保留与高速极限环振荡免疫
// 场景 1：录像 t=55~65 处从 (1.0979, 5.7800) 高速 (dist=0.72) 向上直走，
// 应成功进入第 0 行并抵达上边界 y=0.4201，不能左右在 5.78 和 5.06 互跳。
blocked.fill(0);
const highSpeedDist = 0.72;
let py = 1.0979, px = 5.7800;
[py, px] = sim._steer(py, px, Q.MOVE_UP, blocked, highSpeedDist);
assert(Math.abs(py - 0.4201) < 1e-4, `高速向上应直达上边界 0.4201，实际 y=${py}`);
assert(Math.abs(px - 5.7800) < 1e-4, `直达边界过程中不应发生无谓侧滑，实际 x=${px}`);
// 连续继续按 UP，应平稳停在上边界且完全不发生左右振荡或侧滑
for (let step = 0; step < 5; step++) {
  [py, px] = sim._steer(py, px, Q.MOVE_UP, blocked, highSpeedDist);
  assert(Math.abs(py - 0.4201) < 1e-4, `抵住上边界时不应垂直下移，实际 y=${py}`);
  assert(Math.abs(px - 5.7800) < 1e-4, `抵住上边界时不应水平侧滑，实际 x=${px}`);
}

// 场景 2：录像 t=130~150 处从 (5.1914, 13.9244) 高速 (dist=0.72) 向右直走，
// 应成功进入第 14 列并抵达右边界 x=14.5799，不能上下在 5.19 和 5.91 互跳。
py = 5.1914; px = 13.9244;
[py, px] = sim._steer(py, px, Q.MOVE_RIGHT, blocked, highSpeedDist);
assert(Math.abs(px - 14.5799) < 1e-4, `高速向右应直达右边界 14.5799，实际 x=${px}`);
assert(Math.abs(py - 5.1914) < 1e-4, `直达边界过程中不应发生无谓侧滑，实际 y=${py}`);
for (let step = 0; step < 5; step++) {
  [py, px] = sim._steer(py, px, Q.MOVE_RIGHT, blocked, highSpeedDist);
  assert(Math.abs(px - 14.5799) < 1e-4, `抵住右边界时不应水平左移，实际 x=${px}`);
  assert(Math.abs(py - 5.1914) < 1e-4, `抵住右边界时不应垂直侧滑，实际 y=${py}`);
}

// 场景 3：开阔地单个障碍物（高速 0.72 撞向障碍物坚决不主动向中心线吸附，保持 5.20 原身位）
blocked[4 * Q.W + 5] = 1;
py = 5.43; px = 5.20;
[py, px] = sim._steer(py, px, Q.MOVE_UP, blocked, highSpeedDist);
assert(Math.abs(px - 5.20) < 1e-4, `开阔地单障碍坚决不主动向中线吸附，保持原身位 5.20，实际 x=${px}`);
// 下一步继续 UP，仍稳在 5.20，绝不发生侧滑
[py, px] = sim._steer(py, px, Q.MOVE_UP, blocked, highSpeedDist);
assert(Math.abs(px - 5.20) < 1e-4, `再次决策必须保持在 5.20，实际 x=${px}`);
blocked.fill(0);

// -------------------------------------------------------------
// 回归测试：legalMask 探针正确性（_tryMove 直走轴向探测，严禁侧滑假合法）
const simInst = new Q.Sim(1);
simInst.pos[0] = 0.4201; simInst.pos[1] = 5.5000;
let { mm } = simInst.legalMask();
assert.strictEqual(mm[0][Q.MOVE_UP], 0, '贴紧地图上边界时 MOVE_UP 必须被 mask 为 0 (非法)');

simInst.pos[0] = 1.0979; simInst.pos[1] = 5.5000;
({ mm } = simInst.legalMask());
assert.strictEqual(mm[0][Q.MOVE_UP], 1, '第 1 行向上可达第 0 行时 MOVE_UP 必须合法 (1)');

simInst.pos[0] = 5.5000; simInst.pos[1] = 14.5799;
({ mm } = simInst.legalMask());
assert.strictEqual(mm[0][Q.MOVE_RIGHT], 0, '贴紧地图右边界时 MOVE_RIGHT 必须被 mask 为 0 (非法)');

simInst.pos[0] = 5.5000; simInst.pos[1] = 13.9244;
({ mm } = simInst.legalMask());
assert.strictEqual(mm[0][Q.MOVE_RIGHT], 1, '第 13 列向右可达第 14 列时 MOVE_RIGHT 必须合法 (1)');

// -------------------------------------------------------------
// 回归测试：22debug 开阔地直走远端障碍不触发受阻侧滑与 5Hz 超调震荡
// 场景：从 (6.9001, 8.0999) 高速向左移动，目标列 7 为开阔通路，隔列 (6,6) 有泡。
// 直走迈进 0.6798 格（94.4% 步长）已成功跨入第 7 列，必须判定为有效直行，严禁清零并上下侧滑震荡。
blocked.fill(0);
blocked[6 * Q.W + 6] = 1; // (6,6) 处有障碍
let p22y = 6.9001, p22x = 8.0999;
[p22y, p22x] = sim._steer(p22y, p22x, Q.MOVE_LEFT, blocked, highSpeedDist);
assert(Math.abs(p22y - 6.9001) < 1e-4, `向左跨入通路列时绝不应发生垂直超调侧滑，实际 y=${p22y}`);
assert(p22x < 7.5, `应成功跨入第 7 列 (x < 7.5)，实际 x=${p22x}`);

console.log('mouse _steer & legalMask regression tests passed');
