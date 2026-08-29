#!/usr/bin/env node
'use strict';

const assert = require('assert');
const Q = require('./sim.js');

const sim = Object.create(Q.Sim.prototype);
const blocked = new Uint8Array(Q.N);
const dist = Q.CFG.stepLen;

// 直上被挡且两侧条件相同：角色在格中心左侧时应向右归中，不能固定左滑。
blocked[4 * Q.W + 5] = 1;
let [ny, nx] = sim._steer(5.36, 5.20, Q.MOVE_UP, blocked, dist);
assert(nx > 5.20 && Math.abs(ny - 5.36) < 1e-9,
  `上方受阻、偏左时应右滑，实际 (${ny}, ${nx})`);

// 镜像场景：角色在格中心右侧时应向左归中。
[ny, nx] = sim._steer(5.36, 5.80, Q.MOVE_UP, blocked, dist);
assert(nx < 5.80 && Math.abs(ny - 5.36) < 1e-9,
  `上方受阻、偏右时应左滑，实际 (${ny}, ${nx})`);

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

console.log('mouse _steer regression tests passed');
