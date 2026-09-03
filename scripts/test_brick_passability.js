// 验证砖块炸毁后 0.4s 开放通行
const assert = require('assert');
const QQT = require('../web/sim.js');
const { Sim, CFG, H, W, MOVE_IDLE, MOVE_RIGHT } = QQT;

console.log('🧪 正在测试砖块炸毁后 0.4s 开放通行时序...');

const mapData = {
  h: 13, w: 15,
  wall: new Array(13 * 15).fill(0),
  brick: new Array(13 * 15).fill(0),
  bush: new Array(13 * 15).fill(0),
  spawns: [[3.5, 0.5], [10.5, 10.5]]
};
const sim = new Sim(mapData, 42);
// 在 (3, 1) 放一块砖
sim.brick[3 * W + 1] = 1;

// 泡放在 (2, 1)，威力 2，fuse=1（下一拍引爆，覆盖 (3, 1) 的砖）
sim.bombBlast[2 * W + 1] = 2;
sim.fuse[2 * W + 1] = 1;
sim.owner[2 * W + 1] = 1;

// Tick 0: 引爆，波及砖
sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
assert.strictEqual(sim.brick[3 * W + 1], 1, 'Tick 0: 砖虽被炸，但残骸应阻挡，brick 仍为 1');
assert.strictEqual(sim.brickLinger[3 * W + 1], 4, 'Tick 0: brickLinger 初始应为 4 tick (0.4s)');

// Tick 1 (100ms)
sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
assert.strictEqual(sim.brick[3 * W + 1], 1, 'Tick 1: 0.1s 仍阻挡');
assert.strictEqual(sim.brickLinger[3 * W + 1], 3, 'Tick 1: brickLinger 递减为 3');

// Tick 2 (200ms)
sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
assert.strictEqual(sim.brick[3 * W + 1], 1, 'Tick 2: 0.2s 仍阻挡');
assert.strictEqual(sim.brickLinger[3 * W + 1], 2, 'Tick 2: brickLinger 递减为 2');

// Tick 3 (300ms, 伤害余威已截止，但砖残骸仍阻挡)
sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
assert.strictEqual(sim.brick[3 * W + 1], 1, 'Tick 3: 0.3s 仍阻挡');
assert.strictEqual(sim.brickLinger[3 * W + 1], 1, 'Tick 3: brickLinger 递减为 1');

// Tick 4 (400ms = 0.4s, 砖残骸消散，正式开放通行！)
sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
assert.strictEqual(sim.brick[3 * W + 1], 0, 'Tick 4 (0.4s): 砖残骸倒计时归零，正式开放通行，brick 为 0！');
assert.strictEqual(sim.brickLinger[3 * W + 1], 0, 'Tick 4: brickLinger 为 0');

console.log('🎉 验证通过：砖块炸毁后严格在 0.4s（第 4 tick）开放通行！');
