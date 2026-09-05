/**
 * TimeAStarAI 专项测试套件：
 * 1. 基础寻路与实例化
 * 2. 死胡同放泡自杀前瞻拦截
 * 3. 掩体逃生与安全放泡
 * 4. 时空危险窗与 500ms 安全余量
 * 5. 连锁老炮引爆时刻预测
 * 6. 与规则 Hunter 自动无头对局（防连续堆泡与防自爆）
 */
'use strict';

const assert = require('assert');
const QQT = require('./sim.js');
const TimeAStarAI = require('./time_astar_ai.js');

console.log('--- 测试 1: 基础寻路与实例化 ---');
const ai = new TimeAStarAI();
assert(ai != null, '实例化失败');
const sim1 = new QQT.Sim({ level: 'empty_scene' });
const W = sim1.W || 15;
const danger1 = ai.buildDangerMap(sim1, 0);
const res1 = ai.search(sim1, danger1, 1 * W + 1, 1 * W + 5, 3.0, 0);
assert(res1 != null && res1.path.length === 5, '直线寻路失败');
assert(res1.arrivalTimes[0] === 0, '起点到达时刻应为 0');
assert(res1.arrivalTimes[4] > 0, '终点到达时刻应为正数');
console.log('TimeAStarAI 实例化与基础时空寻路成功 ✔');

console.log('--- 测试 2: 死胡同放泡自杀拦截验证 ---');
const simTrap = new QQT.Sim({ level: 'empty_scene' });
simTrap.wall.fill(1);
simTrap.wall[1 * W + 1] = 0;
simTrap.wall[1 * W + 2] = 0;
simTrap.wall[1 * W + 3] = 0;
simTrap.pos[2] = 1.5; simTrap.pos[3] = 1.5; // P1 处于死胡同尽头 (1,1)
simTrap.pos[0] = 1.5; simTrap.pos[1] = 3.5; // P0 在 (1,3)
simTrap.blastCap[1] = 3;

const canDropTrap = ai.canSafelyPlaceBomb(simTrap, 1 * W + 1, 3, 3.0, 0);
assert.strictEqual(canDropTrap, false, '死胡同放泡必死，必须拦截！');
console.log('死胡同自杀前瞻拦截成功：canSafelyPlaceBomb 返回 false ✔');

console.log('--- 测试 3: 掩体逃生与安全放泡验证 ---');
// 在 (1,2) 侧面开一个掩体 (2,2)
simTrap.wall[2 * W + 2] = 0;
const canDropWithNook = ai.canSafelyPlaceBomb(simTrap, 1 * W + 1, 3, 3.0, 0);
assert.strictEqual(canDropWithNook, true, '有侧向掩体时应允许放泡！');
console.log('掩体逃生与安全放泡验证成功：canSafelyPlaceBomb 返回 true ✔');

console.log('--- 测试 4: 时空危险窗避让验证 ---');
const dangerTest = new TimeAStarAI();
const dangerMap = dangerTest.buildDangerMap(sim1, 0);
// 手动在 (3, 3) 注入一个 1000ms ~ 1250ms 的危险窗
dangerMap.addWindow(3 * W + 3, 1000, 1250);
// 800ms 到达：离 1000ms 不足 500ms 安全余量 -> 拒绝
assert.strictEqual(dangerMap.hitTest(3 * W + 3, 800, 500), true);
// 400ms 到达：离 1000ms 有 600ms 余量 -> 允许
assert.strictEqual(dangerMap.hitTest(3 * W + 3, 400, 500), false);
// 1300ms 到达：火已熄灭 -> 允许
assert.strictEqual(dangerMap.hitTest(3 * W + 3, 1300, 500), false);
console.log('时空危险窗与 500ms 安全余量验证通过 ✔');

console.log('--- 测试 5: 连锁老炮引爆时刻预测 ---');
const simChain = new QQT.Sim({ level: 'empty_scene' });
// 在 (5, 5) 放一个老泡（即将引爆，fuse=10 即 1000ms，blast=3）
simChain.fuse[5 * W + 5] = 10;
simChain.bombBlast[5 * W + 5] = 3;
// 在 (5, 7) 放一个新泡（原本 fuse=30 即 3000ms，blast=2）
simChain.fuse[5 * W + 7] = 30;
simChain.bombBlast[5 * W + 7] = 2;

const chainDanger = ai.buildDangerMap(simChain, 0);
// 新泡位于老泡十字射程内，新泡的爆炸时间应被提前更新为 1000ms
const nextStart = chainDanger.nextDangerStart(5 * W + 7, 0);
assert.strictEqual(nextStart, 1000, `新泡应被连锁提前引爆于 1000ms，实际: ${nextStart}`);
console.log('连锁老炮引爆时刻预测验证通过 ✔');

console.log('--- 测试 6: 与规则 Hunter 自动无头对局 100 Ticks ---');
const { HunterAI } = require('./sim.js');
const hunter = new HunterAI();
const matchSim = new QQT.Sim({ level: 'empty_scene' });

let p1DropCount = 0;
let consecutiveDrops = 0;
let maxConsecutiveDrops = 0;

for (let t = 0; t < 100; t++) {
  const a0 = hunter.act(matchSim, 0);
  const a1 = ai.act(matchSim, 1);

  if (a1[1] === 1) {
    p1DropCount++;
    consecutiveDrops++;
    if (consecutiveDrops > maxConsecutiveDrops) maxConsecutiveDrops = consecutiveDrops;
  } else {
    consecutiveDrops = 0;
  }

  matchSim.step([a0, a1]);

  if (matchSim.done) {
    console.log(`对局在 tick ${t} 结束，胜者: P${matchSim.winner}`);
    break;
  }
}

console.log(`100 Ticks 完成: P1 放泡总数=${p1DropCount}, 最大连续放泡=${maxConsecutiveDrops}`);
assert(maxConsecutiveDrops <= 1, `严禁每 tick 连续无脑放泡自杀！实际最大连续: ${maxConsecutiveDrops}`);
console.log('P1 动作节制合理，无连续自残放泡行为 ✔');

console.log('\n🎉 TimeAStarAI 单元测试全部通过！');
