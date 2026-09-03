#!/usr/bin/env node
// 爆炸余威回归测试：逻辑伤害是单 tick，后续 0.3s 允许进入火区受伤，
// 但 invuln 防止同一余威期间重复掉血；动画时长仍由 main.js 的 0.4s 控制。
'use strict';

const assert = require('assert');
const { Sim } = require('./sim.js');

const sim = new Sim(1);
sim.reset('open');
for (let i = 0; i < sim.wall.length; i++) {
  sim.wall[i] = 0; sim.brick[i] = 0; sim.fuse[i] = 0;
  sim.owner[i] = -1; sim.bombBlast[i] = 0; sim.blastLinger[i] = 0;
}
sim.pos[0] = 6.5; sim.pos[1] = 6.5;
sim.pos[2] = 10.5; sim.pos[3] = 10.5;
sim.hp = [5, 5]; sim.alive = [true, true]; sim.invuln = [0, 0];

const bomb = 6 * 15 + 6;
sim.fuse[bomb] = 1;
sim.owner[bomb] = 0;
sim.bombBlast[bomb] = 1;

// 爆炸开始：P0 只掉一次血，覆盖格余威计时=3。
let info = sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.hp[0], 4);
assert.strictEqual(sim.hp[1], 5);
assert.strictEqual(sim.blastLinger[bomb], 3);
assert.strictEqual(info.covered[bomb], 1);
assert.strictEqual(sim.dangerMap()[bomb], 1);

// P1 直接走入爆炸覆盖格：余威仍能造成一次伤害。
sim.pos[2] = 6.5; sim.pos[3] = 7.5;
sim.invuln[1] = 0;
info = sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.hp[1], 4);
assert.strictEqual(sim.blastLinger[bomb], 2);
assert.strictEqual(sim.dangerMap()[bomb], 1);

// 同一余威期间再次命中被 invuln 屏蔽。
info = sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.hp[1], 4);
assert.strictEqual(sim.blastLinger[bomb], 1);
assert.strictEqual(sim.dangerMap()[bomb], 1);

// 第三个后续 tick 仍在余威窗口内，但无敌保护继续挡住重复伤害。
info = sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.hp[1], 4);
assert.strictEqual(sim.blastLinger[bomb], 0);

// 余威消失后，即使手动清掉无敌也不再掉血。
sim.invuln[1] = 0;
info = sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.hp[1], 4);
assert.strictEqual(sim.dangerMap()[bomb], 0);

// 被炸砖在余威窗口内仍挡路：爆炸 tick 后连续 3 个 tick 保持碰撞，
// 余威结算结束后才允许下一 tick 穿过。
for (let i = 0; i < sim.wall.length; i++) {
  sim.wall[i] = 0; sim.brick[i] = 0; sim.fuse[i] = 0;
  sim.owner[i] = -1; sim.bombBlast[i] = 0;
  sim.blastLinger[i] = 0; sim.brickLinger[i] = 0;
}
sim.pos[0] = 6.5; sim.pos[1] = 6.5;
sim.pos[2] = 10.5; sim.pos[3] = 10.5;
sim.hp = [5, 5]; sim.alive = [true, true]; sim.invuln = [0, 0];
const brick = 6 * 15 + 7;
sim.brick[brick] = 1;
sim.fuse[bomb] = 1;
sim.owner[bomb] = 0;
sim.bombBlast[bomb] = 1;

sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.brick[brick], 1);
assert.strictEqual(sim.brickLinger[brick], 4);
sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.brick[brick], 1);
assert.strictEqual(sim.brickLinger[brick], 3);
sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.brick[brick], 1);
assert.strictEqual(sim.brickLinger[brick], 2);
sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.brick[brick], 1);
assert.strictEqual(sim.brickLinger[brick], 1);
sim.step([[4, 0], [4, 0]]);
assert.strictEqual(sim.brick[brick], 0);
assert.strictEqual(sim.brickLinger[brick], 0);
const xBefore = sim.pos[1];
sim.step([[3, 0], [4, 0]]);
assert(sim.pos[1] > xBefore, '砖残威消失后应允许通行');

console.log('blast linger test passed');
