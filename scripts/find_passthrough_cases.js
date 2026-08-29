'use strict';
// 系统性扫描：找出所有"resolveAxis 放行、但中心路径约束会 block"的穿泡数据点。
// 输出 JSON 数组供 JAX 逐条复验。
const path = require('path');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const { CFG, H, W } = QQT;
const rad = CFG.radius;

// 泡放在 (6,7)，空场无墙无砖
const BOMB_R = 6, BOMB_C = 7;

// 构造 blocked 数组（只有泡）
function makeBlocked() {
  const b = new Uint8Array(H * W);
  b[BOMB_R * W + BOMB_C] = 1;
  return b;
}

// 中心路径检查（从 sim.js 原始修复逻辑复制，独立计算）
// 返回 true = 中心路径会 block（修复生效）；false = 中心路径放行
function centerPathBlocks(y, x, ny, nx, dy, dx, blocked) {
  const startR = Math.max(0, Math.min(H - 1, Math.floor(y)));
  const startC = Math.max(0, Math.min(W - 1, Math.floor(x)));
  if (dy !== 0) {
    // y 段：列 = startC，行 yLo..yHi，排除 startR
    const yLo = Math.max(0, Math.min(H - 1, Math.floor(Math.min(y, ny))));
    const yHi = Math.max(0, Math.min(H - 1, Math.floor(Math.max(y, ny))));
    for (let r = yLo; r <= yHi; r++) {
      if (r === startR) continue;
      if (blocked[r * W + startC]) return true;
    }
  }
  if (dx !== 0) {
    const cy0 = Math.max(0, Math.min(H - 1, Math.floor(ny)));
    const xLo = Math.max(0, Math.min(W - 1, Math.floor(Math.min(x, nx))));
    const xHi = Math.max(0, Math.min(W - 1, Math.floor(Math.max(x, nx))));
    for (let c = xLo; c <= xHi; c++) {
      if (c === startC && cy0 === startR) continue;
      if (blocked[cy0 * W + c]) return true;
    }
  }
  return false;
}

const blocked = makeBlocked();
const DIRS = [[-1,0],[1,0],[0,-1],[0,1]]; // up down left right
const DIR_NAMES = ['up','down','left','right'];
const dist = CFG.stepLen; // 单 tick 位移

const cases = [];
let total = 0, passThrough = 0;

// 扫描泡周围的坐标（y: 3.0~10.0, x: 4.0~11.0, 步长 0.05）
for (let y = 3.0; y <= 10.0; y += 0.05) {
  for (let x = 4.0; x <= 11.0; x += 0.05) {
    // 跳过起点就在泡格内的（脚下有泡，允许出去，不算穿泡）
    if (Math.floor(y) === BOMB_R && Math.floor(x) === BOMB_C) continue;
    for (let d = 0; d < 4; d++) {
      const [dy, dx] = DIRS[d];
      const nyRaw = QQT.resolveAxis(y + dy * dist, dy * dist, x, y, x,
                                     blocked, rad, H, W, true);
      const nxRaw = QQT.resolveAxis(x + dx * dist, dx * dist, y, y, x,
                                     blocked, rad, H, W, false);
      // resolveAxis 沿单轴：dy!=0 只看 y 轴, dx!=0 只看 x 轴
      const ny = dy !== 0 ? nyRaw : y;
      const nx = dx !== 0 ? nxRaw : x;
      const moved = Math.abs(ny - y) > 0.001 || Math.abs(nx - x) > 0.001;
      total++;
      if (!moved) continue;
      // resolveAxis 放行了，检查中心路径是否 block
      const cpBlocks = centerPathBlocks(y, x, ny, nx, dy, dx, blocked);
      if (cpBlocks) {
        passThrough++;
        // 终点是否进入了泡格
        const endInBomb = Math.floor(ny) === BOMB_R && Math.floor(nx) === BOMB_C;
        cases.push({
          y: +y.toFixed(4), x: +x.toFixed(4),
          dir: DIR_NAMES[d], dy, dx,
          ny: +ny.toFixed(4), nx: +nx.toFixed(4),
          moved: true, centerPathBlocks: true,
          endInBombCell: endInBomb,
          bombR: BOMB_R, bombC: BOMB_C,
          radius: rad, stepLen: dist,
        });
      }
    }
  }
}

console.error(`扫描完成: ${total} 组, resolveAxis放行但中心路径block(=穿泡) ${passThrough} 组`);
console.error(`其中终点落入泡格 ${cases.filter(c=>c.endInBombCell).length} 组`);

// 输出前 50 条 + 统计
const sample = cases.slice(0, 50);
console.log(JSON.stringify({
  total_scanned: total,
  pass_through_count: passThrough,
  end_in_bomb_cell: cases.filter(c=>c.endInBombCell).length,
  radius: rad,
  stepLen: dist,
  bomb: [BOMB_R, BOMB_C],
  sample_cases: sample,
  all_cases_count: cases.length,
}, null, 2));
