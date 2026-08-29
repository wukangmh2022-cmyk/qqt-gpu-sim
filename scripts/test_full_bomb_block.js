'use strict';
const path = require('path');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const { Sim, CFG, H, W } = QQT;

const rad = CFG.radius;
console.log(`RADIUS=${rad}, stepLen=${CFG.stepLen}, H=${H}, W=${W}`);

const level = {
  w: W, h: H,
  wall: new Uint8Array(H * W),
  brick: new Uint8Array(H * W),
  spawns: [[6.5, 6.5], [6.5, 8.5]],
  initial_stats: { bombs: 2, blast: 2, speed: 1.0 },
  source: 'test', name: 'test',
};

function testScenario(y, x, moveDir, bombR, bombC, desc) {
  const sim = new Sim(42);
  sim.reset(level);
  sim.pos[0] = y; sim.pos[1] = x;
  sim.pos[2] = 10.5; sim.pos[3] = 10.5;
  const bi = bombR * W + bombC;
  sim.fuse[bi] = CFG.fuse; sim.owner[bi] = 0; sim.bombBlast[bi] = CFG.blast;
  const oldY = sim.pos[0], oldX = sim.pos[1];
  sim.step([[moveDir, 0], [4, 0]]);
  const newY = sim.pos[0], newX = sim.pos[1];
  const moved = Math.abs(newY - oldY) > 0.001 || Math.abs(newX - oldX) > 0.001;
  const enteredBombCell = Math.floor(newY) === bombR && Math.floor(newX) === bombC;
  console.log(`  ${desc}`);
  console.log(`    起点=(${oldY.toFixed(2)},${oldX.toFixed(2)}) 泡=(${bombR},${bombC}) dir=${['up','down','left','right','idle'][moveDir]}`);
  console.log(`    终点=(${newY.toFixed(4)},${newX.toFixed(4)}) moved=${moved} inBombCell=${enteredBombCell}`);
  if (enteredBombCell && moved) console.log(`    ⚠️ 穿泡！`);
  console.log('');
  return enteredBombCell && moved;
}

const UP=0,DOWN=1,LEFT=2,RIGHT=3;
let pt = 0;
if (testScenario(5.5, 7.5, DOWN, 6, 7, '泡上方1格 向下')) pt++;
if (testScenario(7.5, 7.5, UP, 6, 7, '泡下方1格 向上')) pt++;
if (testScenario(6.5, 6.5, RIGHT, 6, 7, '泡左方1格 向右')) pt++;
if (testScenario(6.5, 8.5, LEFT, 6, 7, '泡右方1格 向左')) pt++;
if (testScenario(6.5, 7.5, DOWN, 6, 7, '泡格中心 向下(脚下有泡)')) pt++;
if (testScenario(6.5, 7.5, UP, 6, 7, '泡格中心 向上(脚下有泡)')) pt++;
if (testScenario(5.8, 7.5, DOWN, 6, 7, '泡上方0.7格 向下')) pt++;
if (testScenario(5.2, 7.5, DOWN, 6, 7, '泡上方1.3格 向下')) pt++;
if (testScenario(5.5, 6.5, DOWN, 6, 7, '对角 向下')) pt++;
if (testScenario(5.5, 6.5, RIGHT, 6, 7, '对角 向右')) pt++;

console.log(`穿泡总计: ${pt}/10`);
