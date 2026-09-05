/**
 * NukemanAI - 基于 Nukeman 逆向算法（astar_rust/src/main.rs）的高级时空规则 AI
 *
 * 核心特性：
 * 1. Time-Aware A*：以毫秒到达时刻（arrival_time）为 g 值展开搜索；
 * 2. 连续时空危险窗（DangerMap）：每格维护 [(start_ms, end_ms)] 区间；
 * 3. 500ms 安全前置余量（SAFETY_MARGIN_MS）：到达时刻离起火不足 500ms 拒绝进入；
 * 4. 2 步回溯死胡同预判（would_be_trapped）：前瞻进入后是否有安全出口；
 * 5. 逃逸推演防自杀（canSafelyPlaceBomb）：放泡前假想模拟，无安全撤离路径坚决不放泡。
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.NukemanAI = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // U, D, L, R
  const MOVE_IDLE = 4;
  const SAFETY_MARGIN_MS = 500;  // 到达时刻离起火必须 >= 500ms
  const FLAME_LINGER_MS = 250;   // 火焰残留 250ms

  // 简易优先队列（二叉堆，Min-Heap by cost）
  class MinHeap {
    constructor() {
      this.data = [];
    }
    push(item) {
      this.data.push(item);
      this._up(this.data.length - 1);
    }
    pop() {
      if (this.data.length === 0) return null;
      const top = this.data[0];
      const bottom = this.data.pop();
      if (this.data.length > 0) {
        this.data[0] = bottom;
        this._down(0);
      }
      return top;
    }
    size() {
      return this.data.length;
    }
    _up(i) {
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (this.data[i].cost < this.data[p].cost) {
          const t = this.data[i];
          this.data[i] = this.data[p];
          this.data[p] = t;
          i = p;
        } else {
          break;
        }
      }
    }
    _down(i) {
      const len = this.data.length;
      while ((i << 1) + 1 < len) {
        let left = (i << 1) + 1;
        let right = left + 1;
        let best = left;
        if (right < len && this.data[right].cost < this.data[left].cost) {
          best = right;
        }
        if (this.data[best].cost < this.data[i].cost) {
          const t = this.data[i];
          this.data[i] = this.data[best];
          this.data[best] = t;
          i = best;
        } else {
          break;
        }
      }
    }
  }

  // 时空危险窗集合
  class DangerMap {
    constructor(W, H) {
      this.W = W;
      this.H = H;
      this.N = W * H;
      this.windows = Array.from({ length: this.N }, () => []);
    }

    addWindow(cell, start, end) {
      if (cell >= 0 && cell < this.N) {
        this.windows[cell].push([start, end]);
      }
    }

    // 在时刻 t 到达 cell（附加 safetyMargin 前置余量），是否落入任何危险窗
    hitTest(cell, t, safetyMargin = 0) {
      if (typeof cell !== 'number' || cell < 0 || cell >= this.N || !this.windows[cell]) return true;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        const s = wins[i][0], e = wins[i][1];
        if (t + safetyMargin >= s && t <= e) return true;
      }
      return false;
    }

    // 该格在 after 之后最早的危险窗开始时刻
    nextDangerStart(cell, after) {
      if (typeof cell !== 'number' || cell < 0 || cell >= this.N || !this.windows[cell]) return null;
      let minS = null;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        const s = wins[i][0];
        if (s >= after) {
          if (minS === null || s < minS) minS = s;
        }
      }
      return minS;
    }
  }

  class NukemanAI {
    constructor() {
      // 内部状态保持精简
    }

    // 构建时空危险窗
    buildDangerMap(sim, nowMs = 0, hypotheticalBomb = null) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const danger = new DangerMap(W, H);

      const addCross = (centerIdx, blastLen, boomAt) => {
        const r0 = (centerIdx / W) | 0, c0 = centerIdx % W;
        danger.addWindow(centerIdx, boomAt, boomAt + FLAME_LINGER_MS);
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRS[d];
          for (let k = 1; k <= blastLen; k++) {
            const nr = r0 + dr * k, nc = c0 + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const idx = nr * W + nc;
            if (sim.wall[idx]) break; // 墙挡住
            danger.addWindow(idx, boomAt, boomAt + FLAME_LINGER_MS);
            if (sim.brick[idx] || sim.pushable[idx]) break; // 砖挡住后续延伸
          }
        }
      };

      // 1. 在场真炸弹
      for (let i = 0; i < N; i++) {
        if (sim.fuse[i] > 0) {
          const boomAt = nowMs + sim.fuse[i] * 100; // 10Hz, 1 tick = 100ms
          const blast = sim.bombBlast[i] || 2;
          addCross(i, blast, boomAt);
        }
        // 2. 正在起火/余威中的格子
        if (sim.blastLinger[i] > 0) {
          danger.addWindow(i, nowMs, nowMs + sim.blastLinger[i] * 100);
        }
      }

      // 3. 假设放泡（用于防自杀模拟）
      if (hypotheticalBomb) {
        const boomAt = nowMs + (hypotheticalBomb.fuseTicks || 30) * 100;
        addCross(hypotheticalBomb.idx, hypotheticalBomb.blast || 2, boomAt);
      }

      return danger;
    }

    // 2 步回溯死胡同预判（astar_rust/src/main.rs 行 250-276 原汁原味还原）
    wouldBeTrapped(sim, danger, cell, arriveMs, stepMs, fromCell, extraBlocked = -1) {
      const W = sim.W || 15, H = sim.H || 13;
      const escapeDeadline = arriveMs + 2 * stepMs;
      const r = (cell / W) | 0, c = cell % W;

      let exits = 0;
      let safeExits = 0;

      for (let d = 0; d < 4; d++) {
        const nr = r + DIRS[d][0], nc = c + DIRS[d][1];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
        const np = nr * W + nc;
        if (np === fromCell) continue;
        // 阻挡检查（不可通行）
        if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0 || np === extraBlocked) continue;

        exits++;
        // 出口格在逃生截止时间前没有危险窗覆盖 → 可逃
        const nextStart = danger.nextDangerStart(np, arriveMs);
        const doomed = danger.hitTest(np, escapeDeadline, 0) ||
                       (nextStart !== null && nextStart <= escapeDeadline);
        if (!doomed) {
          safeExits++;
        }
      }
      return exits > 0 && safeExits === 0;
    }

    // 时间感知 A* 寻路（start -> goal）
    // 返回 { path: [start, ..., goal], arrivalTimes: [...] } 或 null
    search(sim, danger, start, goal, speedCellsPerSec, nowMs, options = {}) {
      if (start === goal) {
        return { path: [start], arrivalTimes: [nowMs] };
      }
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const stepMs = Math.round(1000 / speedCellsPerSec);
      const allowBreakBrick = !!options.allowBreakBrick;
      const extraBlocked = options.extraBlocked !== undefined ? options.extraBlocked : -1;

      const heap = new MinHeap();
      const bestArrival = new Float64Array(N).fill(Infinity);
      const parent = new Int32Array(N).fill(-1);

      const gr = (goal / W) | 0, gc = goal % W;
      const hMs = (cell) => {
        const cr = (cell / W) | 0, cc = cell % W;
        return (Math.abs(cr - gr) + Math.abs(cc - gc)) * stepMs;
      };

      bestArrival[start] = nowMs;
      heap.push({ cell: start, arrive: nowMs, cost: nowMs + hMs(start) });

      while (heap.size() > 0) {
        const cur = heap.pop();
        const curCell = cur.cell;
        const curT = cur.arrive;

        if (curCell === goal) {
          // 回溯路径
          const path = [];
          const arrivalTimes = [];
          let p = goal;
          while (p !== -1) {
            path.push(p);
            arrivalTimes.push(bestArrival[p]);
            p = parent[p];
          }
          path.reverse();
          arrivalTimes.reverse();
          return { path, arrivalTimes };
        }

        if (curT > bestArrival[curCell]) continue;

        const cr = (curCell / W) | 0, cc = curCell % W;
        for (let d = 0; d < 4; d++) {
          const nr = cr + DIRS[d][0], nc = cc + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const np = nr * W + nc;

          // 物理不可通
          if (sim.wall[np] || sim.fuse[np] > 0 || np === extraBlocked) continue;
          if (sim.brick[np] && !allowBreakBrick) continue;

          // 地形开销：普通 1x，砖块作为虚拟寻路 4x
          const costMul = sim.brick[np] ? 4 : 1;
          const arrive = curT + stepMs * costMul;

          // 时间剪枝①：撞爆炸（危险窗 + 500ms 安全余量）
          if (danger.hitTest(np, arrive, SAFETY_MARGIN_MS)) {
            continue;
          }

          // 时间剪枝②：死胡同检查（is_trap）
          if (this.wouldBeTrapped(sim, danger, np, arrive, stepMs, curCell, extraBlocked)) {
            continue;
          }

          if (arrive < bestArrival[np]) {
            bestArrival[np] = arrive;
            parent[np] = curCell;
            heap.push({ cell: np, arrive: arrive, cost: arrive + hMs(np) });
          }
        }
      }
      return null;
    }

    // 放泡安全前瞻（防自杀）：
    // 若在 ownIdx 放泡，能否在爆炸前（3000ms - 500ms）安全逃逸到十字范围之外的格子？
    canSafelyPlaceBomb(sim, ownIdx, blastLen, speedCellsPerSec, nowMs) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      // 模拟放置水泡
      const hypothetical = { idx: ownIdx, blast: blastLen, fuseTicks: 30 };
      const danger = this.buildDangerMap(sim, nowMs, hypothetical);
      const stepMs = Math.round(1000 / speedCellsPerSec);
      const boomAt = nowMs + 3000;
      const escapeDeadline = boomAt - SAFETY_MARGIN_MS; // 必须在 2500ms 前撤入安全格

      // 找出当前炸弹十字范围之外、且在该时间段安全的最近格子
      const r0 = (ownIdx / W) | 0, c0 = ownIdx % W;
      const isInsideCross = (idx) => {
        const r = (idx / W) | 0, c = idx % W;
        if (r === r0 && Math.abs(c - c0) <= blastLen) return true;
        if (c === c0 && Math.abs(r - r0) <= blastLen) return true;
        return false;
      };

      // 从脚下的 4 个可通行邻居寻找逃生出口
      for (let d = 0; d < 4; d++) {
        const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
        const np = nr * W + nc;
        if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0) continue;

        // 以 np 为起点，寻找逃离十字且安全的目标
        // 快速 BFS 探测逃生
        const q = [{ cell: np, t: nowMs + stepMs }];
        const visited = new Set([np, ownIdx]);
        while (q.length > 0) {
          const { cell, t } = q.shift();
          if (t > escapeDeadline) continue;

          // 成功逃出十字且该格安全
          if (!isInsideCross(cell) && !danger.hitTest(cell, t, SAFETY_MARGIN_MS)) {
            return true;
          }

          const cr = (cell / W) | 0, cc = cell % W;
          for (let nd = 0; nd < 4; nd++) {
            const nnr = cr + DIRS[nd][0], nnc = cc + DIRS[nd][1];
            if (nnr < 0 || nnr >= H || nnc < 0 || nnc >= W) continue;
            const nextCell = nnr * W + nnc;
            if (visited.has(nextCell)) continue;
            if (sim.wall[nextCell] || sim.brick[nextCell] || sim.fuse[nextCell] > 0) continue;
            if (danger.hitTest(nextCell, t + stepMs, SAFETY_MARGIN_MS)) continue;

            visited.add(nextCell);
            q.push({ cell: nextCell, t: t + stepMs });
          }
        }
      }
      return false; // 无路可逃，放泡必被炸死！
    }

    // 主决策入口：返回 [move, bomb]
    act(sim, pid) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = (sim.t || 0) * 100; // 10Hz, 1 tick = 100ms
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.buildDangerMap(sim, nowMs);

      // 敌方位置
      let oppIdx = -1;
      for (let o = 0; o < 2; o++) {
        if (o !== pid && sim.alive[o]) {
          const oc = sim.centerCell(o);
          oppIdx = oc[0] * W + oc[1];
          break;
        }
      }

      // ------------------------------------------------------------
      // 1. 紧急避险：脚下正处于危险窗中（或 500ms 内将爆炸）
      // ------------------------------------------------------------
      const underThreat = danger.hitTest(ownIdx, nowMs, SAFETY_MARGIN_MS);
      if (underThreat) {
        // 寻找全图最近的安全格
        let bestSafe = -1;
        let minSafeDist = Infinity;
        const r0 = own[0], c0 = own[1];
        for (let i = 0; i < N; i++) {
          if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0) continue;
          if (!danger.hitTest(i, nowMs + 300, 0)) {
            const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
            if (dist < minSafeDist) {
              minSafeDist = dist;
              bestSafe = i;
            }
          }
        }

        if (bestSafe !== -1) {
          const res = this.search(sim, danger, ownIdx, bestSafe, spd, nowMs);
          if (res && res.path.length > 1) {
            const nextCell = res.path[1];
            return [this._cellToMove(ownIdx, nextCell, W), 0];
          }
        }
      }

      // ------------------------------------------------------------
      // 2. 目标选择：吃道具 vs 追击敌人 / 破关键砖
      // ------------------------------------------------------------
      let goal = oppIdx;
      // 若有道具，优先寻路吃道具
      let nearestCrate = -1;
      let minCrateDist = Infinity;
      for (let i = 0; i < N; i++) {
        if (sim.crate[i]) {
          const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
          if (d < minCrateDist) {
            minCrateDist = d;
            nearestCrate = i;
          }
        }
      }

      // 属性未满时优先捡箱
      const blastCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const hungry = (sim.bombsCap && sim.bombsCap[pid] < 6) || blastCap < 5;
      if (hungry && nearestCrate !== -1) {
        const crateRes = this.search(sim, danger, ownIdx, nearestCrate, spd, nowMs);
        if (crateRes && crateRes.path.length > 1) {
          goal = nearestCrate;
        }
      }

      // 寻路到目标
      let pathRes = goal !== -1 ? this.search(sim, danger, ownIdx, goal, spd, nowMs) : null;

      // 若直接路径被砖墙完全封死，则允许虚拟穿砖寻路，定位关键阻断砖
      let targetBrick = -1;
      if (!pathRes && oppIdx !== -1) {
        const brickSearch = this.search(sim, danger, ownIdx, oppIdx, spd, nowMs, { allowBreakBrick: true });
        if (brickSearch && brickSearch.path.length > 1) {
          // 沿路径找第一个挡路的砖
          for (const cell of brickSearch.path) {
            if (sim.brick[cell]) {
              targetBrick = cell;
              break;
            }
          }
          if (targetBrick !== -1) {
            // 目标改为走到该砖相邻格
            pathRes = this.search(sim, danger, ownIdx, targetBrick, spd, nowMs, { allowBreakBrick: true });
          }
        }
      }

      let move = MOVE_IDLE;
      if (pathRes && pathRes.path.length > 1) {
        const nextCell = pathRes.path[1];
        if (!sim.brick[nextCell]) {
          move = this._cellToMove(ownIdx, nextCell, W);
        }
      }

      // ------------------------------------------------------------
      // 3. 放泡判定与防自杀拦截
      // ------------------------------------------------------------
      let placeBomb = 0;
      const { mm, bm } = sim.legalMask();
      const canDrop = bm[pid] === 1 && sim.fuse[ownIdx] === 0;

      if (canDrop) {
        // 条件 A: 与对手对齐并在自身爆炸威力内（且中间无永久墙隔断）
        let alignedOpp = false;
        let nearOpp = false;
        if (oppIdx !== -1) {
          const or = (oppIdx / W) | 0, oc = oppIdx % W;
          const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
          if ((dr === 0 && dc <= blastCap) || (dc === 0 && dr <= blastCap)) {
            let blockedByWall = false;
            if (dr === 0) {
              const minC = Math.min(own[1], oc), maxC = Math.max(own[1], oc);
              for (let c = minC + 1; c < maxC; c++) {
                if (sim.wall[own[0] * W + c]) { blockedByWall = true; break; }
              }
            } else {
              const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
              for (let r = minR + 1; r < maxR; r++) {
                if (sim.wall[r * W + own[1]]) { blockedByWall = true; break; }
              }
            }
            if (!blockedByWall) alignedOpp = true;
          }
          if (dr + dc <= blastCap + 1) nearOpp = true;
        }

        // 条件 B: 面前贴着阻碍通往敌人的关键砖
        let brickBlock = false;
        if (targetBrick !== -1) {
          const tbr = (targetBrick / W) | 0, tbc = targetBrick % W;
          const dbr = Math.abs(own[0] - tbr), dbc = Math.abs(own[1] - tbc);
          if (dbr + dbc === 1) brickBlock = true;
        }

        // 条件 C: 连锁老泡（blast 十字内有引信 <= 10 的老泡）
        let chain = false;
        for (let k = 1; k <= blastCap && !chain; k++) {
          for (let d = 0; d < 4; d++) {
            const nr = own[0] + DIRS[d][0] * k, nc = own[1] + DIRS[d][1] * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const idx = nr * W + nc;
            if (sim.wall[idx]) break;
            const f = sim.fuse[idx];
            if (f > 0 && f <= 10) { chain = true; break; }
          }
        }

        shouldDrop = alignedOpp || nearOpp || brickBlock || chain;

        // 核心关卡：放泡前必须进行假想前瞻防自杀！
        if (shouldDrop) {
          if (this.canSafelyPlaceBomb(sim, ownIdx, blastCap, spd, nowMs)) {
            placeBomb = 1;
          }
        }
      }

      // 验证移动动作物理合法性
      if (move !== MOVE_IDLE && mm[pid][move] !== 1) {
        move = MOVE_IDLE;
      }

      return [move, placeBomb];
    }

    _cellToMove(from, to, W) {
      const fr = (from / W) | 0, fc = from % W;
      const tr = (to / W) | 0, tc = to % W;
      if (tr < fr) return 0; // U
      if (tr > fr) return 1; // D
      if (tc < fc) return 2; // L
      if (tc > fc) return 3; // R
      return 4; // IDLE
    }
  }

  return NukemanAI;
});
