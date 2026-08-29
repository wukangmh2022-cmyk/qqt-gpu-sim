'use strict';
// 用 JAX 参数(STEP=0.3, RADIUS=0.36, 13x15)在 JS 侧扫描所有穿泡数据点。
// 每条记录：起点(y,x)、方向、resolveAxis终点(=放行后位置)、中心路径block(=修复会拦截)。
// 输出 JSON 到 stdout 供 JAX 逐条复验。
// 用整数索引计算坐标（不浮点累加），输出用 toPrecision(15) 保留完整精度。
const path = require('path');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const { H, W } = QQT;

// ---- 用 JAX 参数覆盖 ----
const RADIUS = 0.36;
const STEP = 0.3;
const BOMB_R = 6, BOMB_C = 7;

function makeBlocked() {
  const b = new Uint8Array(H * W);
  b[BOMB_R * W + BOMB_C] = 1;
  return b;
}

// 中心路径检查（与 main.js frameMove / sim.js Sim.step 原始修复逻辑一致）
function centerPathBlocks(y, x, ny, nx, dy, dx, blocked) {
  const startR = Math.max(0, Math.min(H - 1, Math.floor(y)));
  const startC = Math.max(0, Math.min(W - 1, Math.floor(x)));
  if (dy !== 0) {
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
const DIRS = [[-1,0],[1,0],[0,-1],[0,1]];
const DIR_NAMES = ['up','down','left','right'];

const cases = [];
let total = 0, passThrough = 0;

// 扫描泡周围坐标（步长 0.02，整数索引不累加，精确覆盖边界情况）
for (let yi = 0; yi <= 450; yi++) {
  const y = 2.0 + yi * 0.02;
  for (let xi = 0; xi <= 450; xi++) {
    const x = 3.0 + xi * 0.02;
    // 跳过起点在泡格内的（脚下有泡，允许离开）
    if (Math.floor(y) === BOMB_R && Math.floor(x) === BOMB_C) continue;
    for (let d = 0; d < 4; d++) {
      const [dy, dx] = DIRS[d];
      total++;
      // resolveAxis 沿单轴
      let ny = y, nx = x;
      if (dy !== 0) {
        ny = QQT.resolveAxis(y + dy * STEP, dy * STEP, x, y, x,
                             blocked, RADIUS, H, W, true);
      }
      if (dx !== 0) {
        nx = QQT.resolveAxis(x + dx * STEP, dx * STEP, y, y, x,
                             blocked, RADIUS, H, W, false);
      }
      const moved = Math.abs(ny - y) > 0.0001 || Math.abs(nx - x) > 0.0001;
      if (!moved) continue;
      // resolveAxis 放行了，查中心路径
      const cpBlocks = centerPathBlocks(y, x, ny, nx, dy, dx, blocked);
      if (cpBlocks) {
        passThrough++;
        const endInBomb = Math.floor(ny) === BOMB_R && Math.floor(nx) === BOMB_C;
        cases.push({
          y: +y.toFixed(4), x: +x.toFixed(4),
          dir: d, dir_name: DIR_NAMES[d],
          dy, dx,
          // resolveAxis 返回的"放行后"终点
          ny: +ny.toFixed(6), nx: +nx.toFixed(6),
          moved: true,
          centerPathBlocks: true,
          endInBombCell: endInBomb,
          bombR: BOMB_R, bombC: BOMB_C,
          radius: RADIUS, step: STEP,
        });
      }
    }
  }
}

console.error(`扫描完成: ${total} 组, 穿泡(resolveAxis放行+中心路径block) ${passThrough} 组`);
console.error(`其中终点落入泡格 ${cases.filter(c=>c.endInBombCell).length} 组`);
const byDir = {};
for (const c of cases) byDir[c.dir_name] = (byDir[c.dir_name]||0)+1;
console.error('按方向分布:', JSON.stringify(byDir));

// 输出全部穿泡案例（用 toPrecision(15) 保留完整浮点精度，防 toFixed 截断）
const out = cases.map(c => ({
  y: parseFloat(c.y.toPrecision(15)), x: parseFloat(c.x.toPrecision(15)),
  dir: c.dir, dir_name: DIR_NAMES[c.dir],
  ny: parseFloat(c.ny.toPrecision(15)), nx: parseFloat(c.nx.toPrecision(15)),
  endInBomb: c.endInBombCell,
}));
console.log(JSON.stringify({
  total_scanned: total,
  pass_through_count: passThrough,
  end_in_bomb_cell: cases.filter(c=>c.endInBombCell).length,
  radius: RADIUS, step: STEP, H, W,
  bomb: [BOMB_R, BOMB_C],
  by_dir: byDir,
  cases: out,
}, null, 0));
