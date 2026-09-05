#!/usr/bin/env node
'use strict';

const assert = require('assert');
const { Sim } = require('./sim.js');
const NukemanAI = require('./nukeman_ai.js');

console.log('--- 测试 1: 基础寻路与实例化 ---');
const ai = new NukemanAI();
const sim = new Sim(1);
sim.reset('open');
assert(ai !== null);
console.log('NukemanAI 实例化成功 ✔');

console.log('--- 测试 2: 死胡同放泡自杀拦截验证 ---');
// 构造死胡同场景：13x13 地图，(1, 1) 为三面环墙的死胡同末端
for (let i = 0; i < sim.wall.length; i++) {
  sim.wall[i] = 0; sim.brick[i] = 0; sim.fuse[i] = 0;
  sim.owner[i] = -1; sim.bombBlast[i] = 0; sim.blastLinger[i] = 0;
}
const W = 15, H = 13;
// 墙包围 (1, 1)：
// (0, 1) 墙, (1, 0) 墙, (2, 1) 墙；仅右侧 (1, 2) 连通，但 (1, 2) 是 1 格走廊
sim.wall[0 * W + 1] = 1;
sim.wall[1 * W + 0] = 1;
sim.wall[2 * W + 1] = 1;
// 在 (1, 3) 处也放墙，使得 (1, 1) 和 (1, 2) 是深度为 2 的全封闭死胡同！
sim.wall[0 * W + 2] = 1;
sim.wall[2 * W + 2] = 1;
sim.wall[1 * W + 3] = 1; // 堵死右侧出口

sim.pos[0] = 1.5; sim.pos[1] = 1.5; // P0 在 (1, 1)
sim.hp = [5, 5]; sim.alive = [true, true];

// 在当前死胡同中，假设威力为 2 的炸弹，如果放下去，整条走廊 (1, 1) 和 (1, 2) 全部在爆炸范围内，绝无生还可能！
const safeToDrop = ai.canSafelyPlaceBomb(sim, 1 * W + 1, 2, 3.0, 0);
assert.strictEqual(safeToDrop, false, '在封闭死胡同中坚决不应允许放泡！');
console.log('死胡同自杀前瞻拦截成功：canSafelyPlaceBomb 返回 false ✔');

console.log('--- 测试 3: 开阔地安全放泡验证 ---');
// 清空阻挡，变成开阔地
sim.wall[1 * W + 3] = 0;
sim.wall[0 * W + 2] = 0;
sim.wall[2 * W + 2] = 0;
const safeOpenDrop = ai.canSafelyPlaceBomb(sim, 1 * W + 1, 2, 3.0, 0);
assert.strictEqual(safeOpenDrop, true, '有开阔逃逸通路的场景应允许放泡！');
console.log('开阔地安全逃生探测成功：canSafelyPlaceBomb 返回 true ✔');

console.log('--- 测试 4: 时空危险窗避让验证 ---');
// 在 (1, 3) 放置一枚将在 400ms（4 tick）后爆炸的水泡
sim.fuse[1 * W + 3] = 4;
sim.bombBlast[1 * W + 3] = 2;
const danger = ai.buildDangerMap(sim, 0);
// 检查 (1, 2) 是否在 400ms 处落入危险窗（威力 2 会覆盖 (1, 2)）
assert.strictEqual(danger.hitTest(1 * W + 2, 400, 0), true, '(1, 2) 应在 400ms 处于爆炸危险窗');
// 检查 500ms 安全余量前瞻：在 0ms 时，如果打算在 100ms 走到 (1, 2)，因为 100 + 500 >= 400，应被安全余量拦截
assert.strictEqual(danger.hitTest(1 * W + 2, 100, 500), true, '500ms 安全余量应成功拦截危险格');
console.log('时空危险窗与 500ms 安全余量验证通过 ✔');

console.log('--- 测试 5: 与规则 Hunter 自动无头对局 100 Ticks ---');
const { HunterAI } = require('./sim.js');
const hunter = new HunterAI();
sim.reset('corridor');
let p0Actions = 0, p1Actions = 0;
for (let tick = 0; tick < 100 && sim.alive[0] && sim.alive[1]; tick++) {
  const a0 = ai.act(sim, 0);
  const a1 = hunter.act(sim, 1);
  assert(Array.isArray(a0) && a0.length === 2, 'P0 动作格式应为 [move, bomb]');
  assert(Array.isArray(a1) && a1.length === 2, 'P1 动作格式应为 [move, bomb]');
  if (a0[0] !== 4 || a0[1] !== 0) p0Actions++;
  if (a1[0] !== 4 || a1[1] !== 0) p1Actions++;
  sim.step([a0, a1]);
}
console.log(`100 Ticks 顺利完成，P0 活跃步数=${p0Actions}, P1 活跃步数=${p1Actions}, 存活状态: [${sim.alive[0]}, ${sim.alive[1]}] ✔`);

console.log('\n🎉 NukemanAI 单元测试全部通过！');
