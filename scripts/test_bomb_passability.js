#!/usr/bin/env node
/* JAX vs JS 泡泡障碍物通行性对拍：
 * 对每个连续坐标 (y, x) 在 13×15 网格上，在玩家下方放一个泡，
 * 然后测 4 个方向的 resolveAxis 通行结果是否一致。
 *
 * 用法: node scripts/test_bomb_passability.js
 * 依赖: 先跑 JAX 侧生成参考值（下方 Python 脚本），再与本 JS 对比。
 */
'use strict';
const path = require('path');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const { Sim, CFG, EPS, H, W } = QQT;

const rad = CFG.radius;  // 0.36
const stepLen = CFG.stepLen;

// 构造一个空场 + 在 (br, bc) 放一个泡
function makeBlockedWithBomb(br, bc) {
  const blocked = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) blocked[i] = 0;
  blocked[br * W + bc] = 1;  // 泡 = blocked
  return blocked;
}

// 测某个坐标向某方向能否移动（用 resolveAxis）
function testDirection(y, x, dy, dx, blocked) {
  const dist = dy !== 0 ? dy * stepLen : dx * stepLen;
  if (dy !== 0) {
    const ny = QQT.resolveAxis(y + dy * stepLen, dist, x, y, x, blocked, rad, H, W, true);
    return { canMove: Math.abs(ny - y) > 0.001, newPos: ny };
  } else {
    const nx = QQT.resolveAxis(x + dx * stepLen, dist, y, y, x, blocked, rad, H, W, false);
    return { canMove: Math.abs(nx - x) > 0.001, newPos: nx };
  }
}

const DIRS = [[-1, 0, 'up'], [1, 0, 'down'], [0, -1, 'left'], [0, 1, 'right']];
let mismatches = 0;
let total = 0;

// 在格 (6, 7) 放泡，测试玩家在不同坐标下 4 方向的通行性
const bombR = 6, bombC = 7;
const blocked = makeBlockedWithBomb(bombR, bombC);

console.log(`泡泡位置: (${bombR}, ${bombC})  RADIUS=${rad}  STEP=${stepLen}`);
console.log(`网格: ${H}×${W}`);
console.log('');

// 测试玩家中心在泡格附近的各种坐标
const testCoords = [];
// 泡格中心 = (6.5, 7.5)
// 测试从泡格四周逐渐远离的坐标
for (let r = bombR - 2; r <= bombR + 2; r++) {
  for (let c = bombC - 2; c <= bombC + 2; c++) {
    // 格中心 + 各种亚格偏移
    for (let frac = 0; frac <= 10; frac++) {
      const off = frac / 10;  // 0.0 ~ 1.0
      testCoords.push([r + off, c + 0.5]);
      testCoords.push([r + 0.5, c + off]);
    }
  }
}

console.log(`测试 ${testCoords.length} 个坐标 × 4 方向 = ${testCoords.length * 4} 次通行性检查`);
console.log('');

const results = [];
for (const [y, x] of testCoords) {
  if (y < rad || y > H - rad || x < rad || x > W - rad) continue;
  for (const [dy, dx, name] of DIRS) {
    const res = testDirection(y, x, dy, dx, blocked);
    results.push({ y, x, dir: name, canMove: res.canMove, newPos: res.newPos });
    total++;
  }
}

// 输出关键场景：玩家在泡格正上方/正下方/正左/正右
console.log('=== 关键场景（玩家中心在泡格四邻中心）===');
const keyScenarios = [
  { y: 5.5, x: 7.5, desc: '玩家在泡上方一格中心(5.5,7.5) 向下(进入泡格)' },
  { y: 7.5, x: 7.5, desc: '玩家在泡下方一格中心(7.5,7.5) 向上(进入泡格)' },
  { y: 6.5, x: 6.5, desc: '玩家在泡左方一格中心(6.5,6.5) 向右(进入泡格)' },
  { y: 6.5, x: 8.5, desc: '玩家在泡右方一格中心(6.5,8.5) 向左(进入泡格)' },
  { y: 6.5, x: 7.5, desc: '玩家在泡格中心(6.5,7.5) 向下(脚下有泡)' },
  { y: 6.5, x: 7.5, desc: '玩家在泡格中心(6.5,7.5) 向上(脚下有泡)' },
];

for (const sc of keyScenarios) {
  console.log(`\n  ${sc.desc}`);
  for (const [dy, dx, name] of DIRS) {
    const res = testDirection(sc.y, sc.x, dy, dx, blocked);
    const moveStr = res.canMove ? `✓ 能走 → ${name === 'up' || name === 'down' ? res.newPos.toFixed(4) : res.newPos.toFixed(4)}` : '✗ 不能走';
    console.log(`    ${name}: ${moveStr}`);
  }
}

// 输出穿泡场景（能走到泡格中心的情况）
console.log('\n=== 穿泡检测（能从泡格外移动到泡格内的场景）===');
let passThrough = 0;
for (const r of results) {
  // 如果向某个方向移动后，新位置落入了泡格 (6,7) 的范围
  const newPosInBombCell = (r.dir === 'up' || r.dir === 'down')
    ? Math.floor(r.newPos) === bombR
    : Math.floor(r.newPos) === bombC;
  if (r.canMove && newPosInBombCell) {
    passThrough++;
    if (passThrough <= 20) {
      console.log(`  坐标(${r.y.toFixed(2)},${r.x.toFixed(2)}) 方向=${r.dir} → 新位置=${r.newPos.toFixed(4)} [进入泡格!]`);
    }
  }
}
console.log(`\n穿泡场景总计: ${passThrough}/${total}`);

// 输出 JSON 供 JAX 侧对比
const fs = require('fs');
fs.writeFileSync('/tmp/js_bomb_passability.json', JSON.stringify({
  radius: rad,
  stepLen: stepLen,
  bomb: [bombR, bombC],
  grid: [H, W],
  results: results.map(r => ({
    y: parseFloat(r.y.toFixed(6)),
    x: parseFloat(r.x.toFixed(6)),
    dir: r.dir,
    canMove: r.canMove,
    newPos: parseFloat(r.newPos.toFixed(6))
  }))
}, null, 1));
console.log('\nJS 结果已写: /tmp/js_bomb_passability.json');
