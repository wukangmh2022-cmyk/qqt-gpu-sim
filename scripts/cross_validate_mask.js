'use strict';
// 全面对比 JAX legal_mask vs JS legalMask
// 测试维度：不同位置 × 不同速度 × 不同方向 × 不同地图配置
// 输出每条数据的 JAX mask 和 JS mask 结果，供 Python 侧比对
const path = require('path');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const { H, W } = QQT;

// ====== JAX 参数 ======
const JAX_STEP = 0.3;
const JAX_RADIUS = 0.36;
const JAX_EPS = 1e-4;
const MAX_SWEEP = 3;
const DIRS = [[-1,0],[1,0],[0,-1],[0,1]];

// ====== JS resolveAxis + 中心路径（复刻 main.js frameMove 修复版）======
function jsResolveAxis(coord, delta, other, y, x, blocked, vertical) {
  return QQT.resolveAxis(coord, delta, other, y, x, blocked, JAX_RADIUS, H, W, vertical);
}

function jsMovePlayer(y, x, move, blocked, step, radius) {
  // 复刻 sim.js Sim.step 的移动逻辑（含中心路径约束）
  const [dy0, dx0] = DIRS[move];
  const dy = dy0 * step, dx = dx0 * step;
  let ny = jsResolveAxis(y + dy, dy, x, y, x, blocked, true);
  // 中心路径 y 段
  const startR = Math.max(0, Math.min(H-1, Math.floor(y)));
  const startC = Math.max(0, Math.min(W-1, Math.floor(x)));
  const yLo = Math.max(0, Math.min(H-1, Math.floor(Math.min(y, ny))));
  const yHi = Math.max(0, Math.min(H-1, Math.floor(Math.max(y, ny))));
  for (let r = yLo; r <= yHi; r++) {
    if (r === startR) continue;
    if (blocked[r * W + startC]) { ny = y; break; }
  }
  let nx = jsResolveAxis(x + dx, dx, y, ny, x, blocked, false);
  // 中心路径 x 段
  const xLo = Math.max(0, Math.min(W-1, Math.floor(Math.min(x, nx))));
  const xHi = Math.max(0, Math.min(W-1, Math.floor(Math.max(x, nx))));
  const cy0 = Math.max(0, Math.min(H-1, Math.floor(ny)));
  for (let c = xLo; c <= xHi; c++) {
    if (c === startC && cy0 === startR) continue;
    if (blocked[cy0 * W + c]) { nx = x; break; }
  }
  const outY = dy !== 0 ? ny : y;
  const outX = dx !== 0 ? nx : x;
  return [Math.max(radius, Math.min(H - radius, outY)),
          Math.max(radius, Math.min(W - radius, outX))];
}

// ====== JS legalMask（当前实现：resolveAxis only, base stepLen）======
function jsLegalMask_current(y, x, blocked, stepLen, radius) {
  const mask = [false, false, false, false, true]; // IDLE always legal
  const minMove = stepLen * 0.05;
  for (let mv = 0; mv < 4; mv++) {
    const [dy, dx] = DIRS[mv];
    let moved = false;
    if (dy !== 0) {
      const ny = jsResolveAxis(y + dy * stepLen, dy * stepLen, x, y, x, blocked, true);
      moved = Math.abs(ny - y) > minMove;
    } else {
      const nx = jsResolveAxis(x + dx * stepLen, dx * stepLen, y, y, x, blocked, false);
      moved = Math.abs(nx - x) > minMove;
    }
    mask[mv] = moved;
  }
  return mask;
}

// ====== JAX legalMask（用 _move_player，含中心路径 + 实际 spd_g）======
function jaxLegalMask(y, x, blocked, spdG) {
  const step = JAX_STEP * spdG;
  const mask = [false, false, false, false, true];
  for (let mv = 0; mv < 4; mv++) {
    const [ny, nx] = jsMovePlayer(y, x, mv, blocked, step, JAX_RADIUS);
    const moved = Math.abs(ny - y) > 2 * JAX_EPS || Math.abs(nx - x) > 2 * JAX_EPS;
    mask[mv] = moved;
  }
  return mask;
}

// ====== 测试场景生成 ======
const scenarios = [];

// 地图配置类型
const mapTypes = {
  'empty': { walls: [], bricks: [], bombs: [] },
  'bomb_center': { walls: [], bricks: [], bombs: [[6,7]] },
  'bomb_adjacent': { walls: [], bricks: [], bombs: [[5,7],[6,6],[6,8],[7,7]] },
  'wall_row': { walls: [[3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14]], bricks: [], bombs: [] },
  'brick_cross': { walls: [], bricks: [[6,6],[6,7],[6,8],[5,7],[7,7]], bombs: [] },
  'corner_wall': { walls: [[0,0],[0,1],[1,0],[1,1]], bricks: [], bombs: [] },
  'corridor': {
    walls: (() => { const w=[]; for(let c=0;c<W;c++){w.push([0,c]);w.push([1,c]);} return w; })(),
    bricks: (() => { const b=[]; for(let r=0;r<H;r++){b.push([r,5]);b.push([r,9]);} return b; })(),
    bombs: []
  },
  'mixed': {
    walls: [[0,0],[0,14],[12,0],[12,14],[2,3],[8,11]],
    bricks: [[4,5],[4,6],[7,8],[7,9],[9,3]],
    bombs: [[6,7],[5,10]]
  },
};

function buildBlocked(mapType) {
  const cfg = mapTypes[mapType];
  const blocked = new Uint8Array(H * W);
  for (const [r, c] of cfg.walls) blocked[r * W + c] = 1;
  for (const [r, c] of cfg.bricks) blocked[r * W + c] = 1;
  for (const [r, c] of cfg.bombs) blocked[r * W + c] = 1;
  return blocked;
}

// 速度档位
const speeds = [1.0, 1.3, 1.6, 2.1];

// 生成场景：不同位置 × 不同地图 × 不同速度
const positions = [];
for (let y = 2.0; y <= 11.0; y += 0.25) {
  for (let x = 2.0; x <= 13.0; x += 0.25) {
    positions.push([y, x]);
  }
}

let total = 0, agree = 0, disagree = 0;
const disagreements = [];

for (const mapType of Object.keys(mapTypes)) {
  const blocked = buildBlocked(mapType);
  for (const spdG of speeds) {
    for (const [y, x] of positions) {
      // 跳过起点在 blocked 格上的
      const r = Math.floor(y), c = Math.floor(x);
      if (blocked[r * W + c]) continue;

      const jaxMask = jaxLegalMask(y, x, blocked, spdG);
      // JS current mask uses base stepLen (0.3), NOT actual speed
      const jsMask = jsLegalMask_current(y, x, blocked, 0.3, JAX_RADIUS);

      for (let mv = 0; mv < 4; mv++) {
        total++;
        if (jaxMask[mv] === jsMask[mv]) {
          agree++;
        } else {
          disagree++;
          if (disagreements.length < 50) {
            disagreements.push({
              map: mapType, y: +y.toFixed(4), x: +x.toFixed(4),
              spdG, dir: mv,
              jax: jaxMask[mv], js: jsMask[mv],
            });
          }
        }
      }
    }
  }
}

console.log(JSON.stringify({
  total, agree, disagree,
  agree_rate: +(agree / total * 100).toFixed(2) + '%',
  maps: Object.keys(mapTypes),
  speeds,
  jax_params: { STEP: JAX_STEP, RADIUS: JAX_RADIUS },
  js_params: { stepLen: 0.3, radius: JAX_RADIUS },
  jax_mask_uses: '_move_player (center-path + actual spd_g)',
  js_mask_uses: 'resolveAxis only (no center-path, base stepLen=0.3)',
  disagreement_samples: disagreements,
}, null, 2));
