/**
 * TimeAStarAI - 高级时空规则 AI
 *
 * 核心架构：
 * 1. Time-Aware A*：以物理到达时刻（arrival_time，毫秒）作为真实物理时间展开时空搜索；
 * 2. 连续时空危险窗（DangerMap）：精准对齐动力学（(fuse - 1) * 100ms），包含多泡连环引爆链预测；
 * 3. 破砖开路（Break Brick to Advance）：穿砖代价启发式全局寻路，阻断时就地落子破障，开辟进攻走廊；
 * 4. 战术落子与穿梭连炮（Tactical Bombing & Infinite Chaining）：直瞄锁敌、近身压迫、连环老炮、游走铺雷；
 * 5. 闭环逃生路径承诺与即时物理安全过滤（Committed Escape & Immediate Danger Filter）：坚决不踏入任何起火或即将爆炸格；
 * 6. 优先吃箱发育（Crate Acquisition）：争抢物资增强威力、移速与泡容量，压制竞技对手。
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const AI = factory();
    module.exports = AI;
  } else {
    const AI = factory();
    root.TimeAStarAI = AI;
    root.NukemanAI = AI; // 兼容别名
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 0: 上, 1: 下, 2: 左, 3: 右
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;
  const SAFETY_MARGIN_MS = 350;  // 穿行安全前置余量（ms）
  const FLAME_LINGER_MS = 250;   // 爆炸余威残留（ms）

  // 快速最小二叉堆
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

  // 时空危险窗容器
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

    // 在时刻 t 处于该格（附加 safetyMargin 前置余量），是否与火焰时间窗冲突
    hitTest(cell, t, safetyMargin = 0) {
      if (cell < 0 || cell >= this.N || !this.windows[cell]) return true;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        if (t + safetyMargin >= wins[i][0] && t <= wins[i][1]) return true;
      }
      return false;
    }

    // 该格未来是否会有炸弹引爆/起火
    hasFutureDanger(cell, afterMs = 0) {
      if (cell < 0 || cell >= this.N || !this.windows[cell]) return false;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        if (wins[i][1] > afterMs) return true;
      }
      return false;
    }

    // 该格在 afterMs 之后最早的起火时刻
    nextDangerStart(cell, afterMs = 0) {
      if (cell < 0 || cell >= this.N || !this.windows[cell]) return null;
      let minS = null;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        if (wins[i][0] >= afterMs) {
          if (minS === null || wins[i][0] < minS) minS = wins[i][0];
        }
      }
      return minS;
    }
  }

  class TimeAStarAI {
    constructor() {
      this.targetCell = -1;
      this.targetPath = [];
      this.escapePath = [];
      this.escapeTarget = -1;
      this.lastDropTick = -999;
    }

    // 构建时空危险窗（严格对齐离散物理引爆时刻，含多泡连环引爆链预测）
    buildDangerMap(sim, nowMs = 0, extraBomb = null) {
      const W = sim.W || (sim.level && (sim.level.w || sim.level.width)) || 15;
      const H = sim.H || (sim.level && (sim.level.h || sim.level.height)) || 13;
      const N = W * H;
      const danger = new DangerMap(W, H);

      // 1. 收集在场真炸弹及已残留余威
      // 关键对齐：sim.fuse === 1 在当前 tick 的 step 中减为 0 立即引爆，所以距离爆炸剩余 (fuse - 1) * 100ms
      const bombs = [];
      for (let i = 0; i < N; i++) {
        if (sim.fuse[i] > 0) {
          bombs.push({
            idx: i,
            blast: sim.bombBlast[i] || 2,
            boomAt: nowMs + Math.max(0, sim.fuse[i] - 1) * 100
          });
        }
        if (sim.blastLinger[i] > 0) {
          danger.addWindow(i, nowMs, nowMs + sim.blastLinger[i] * 100);
        }
      }

      // 2. 模拟假设放泡（用于防自杀推演）
      if (extraBomb) {
        bombs.push({
          idx: extraBomb.idx,
          blast: extraBomb.blast || 2,
          boomAt: nowMs + (extraBomb.fuseTicks || 30) * 100
        });
      }

      // 3. 连锁引爆迭代更新（先起火的老泡引爆十字范围内的后续炸弹）
      let changed = true;
      let pass = 0;
      while (changed && pass < 12) {
        changed = false;
        pass++;
        for (let a = 0; a < bombs.length; a++) {
          const bA = bombs[a];
          const r0 = (bA.idx / W) | 0, c0 = bA.idx % W;
          for (let d = 0; d < 4; d++) {
            const [dr, dc] = DIRS[d];
            for (let k = 1; k <= bA.blast; k++) {
              const nr = r0 + dr * k, nc = c0 + dc * k;
              if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
              const idx = nr * W + nc;
              if (sim.wall[idx]) break;
              for (let b = 0; b < bombs.length; b++) {
                if (bombs[b].idx === idx && bombs[b].boomAt > bA.boomAt) {
                  bombs[b].boomAt = bA.boomAt;
                  changed = true;
                }
              }
              if (sim.brick[idx] || sim.pushable[idx]) break;
            }
          }
        }
      }

      // 4. 将收敛后的炸弹爆炸十字投影为连续危险窗
      for (let i = 0; i < bombs.length; i++) {
        const b = bombs[i];
        const r0 = (b.idx / W) | 0, c0 = b.idx % W;
        danger.addWindow(b.idx, b.boomAt, b.boomAt + FLAME_LINGER_MS);
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRS[d];
          for (let k = 1; k <= b.blast; k++) {
            const nr = r0 + dr * k, nc = c0 + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const idx = nr * W + nc;
            if (sim.wall[idx]) break;
            danger.addWindow(idx, b.boomAt, b.boomAt + FLAME_LINGER_MS);
            if (sim.brick[idx] || sim.pushable[idx]) break;
          }
        }
      }

      return danger;
    }

    // 纯物理时空 A* 寻路（arrive 严格保持真实物理时刻，严禁虚拟代价污染物理时间）
    search(sim, danger, start, goal, speedCellsPerSec, nowMs, options = {}) {
      if (start === goal) {
        return { path: [start], arrivalTimes: [nowMs] };
      }
      const W = sim.W || (sim.level && (sim.level.w || sim.level.width)) || 15;
      const H = sim.H || (sim.level && (sim.level.h || sim.level.height)) || 13;
      const N = W * H;
      const stepMs = Math.round(1000 / Math.max(0.5, speedCellsPerSec));
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

          // 物理不可通行检查
          if (sim.wall[np] || (sim.fuse[np] > 0 && np !== start) || np === extraBlocked) continue;
          if (sim.brick[np] && !allowBreakBrick) continue;

          // 真实物理到达时间计算：穿砖开路需放置炸弹并等待（等效 5 步延迟），走空地恒为 1 步
          const physicalStep = sim.brick[np] ? stepMs * 5 : stepMs;
          const arrive = curT + physicalStep;

          // 严格物理时间剪枝：到达该格时处于危险起火窗中（前置安全余量 SAFETY_MARGIN_MS）
          if (danger.hitTest(np, arrive, SAFETY_MARGIN_MS)) {
            continue;
          }

          // 虚拟启发代价：只参与小顶堆排序，绝不污染真实的 arrive
          let heuristicCost = arrive + hMs(np);
          if (danger.hasFutureDanger(np, nowMs)) {
            heuristicCost += 1500; // 偏好绕开有雷的通路，但不锁死物理通行
          }
          if (sim.brick[np]) {
            heuristicCost += 3000; // 偏好走现成通路，无路时才穿砖
          }

          if (arrive < bestArrival[np]) {
            bestArrival[np] = arrive;
            parent[np] = curCell;
            heap.push({ cell: np, arrive, cost: heuristicCost });
          }
        }
      }
      return null;
    }

    // 严格防自杀推演：确保在 ownIdx 放泡后，能有一条切实可行的安全路径在爆炸前撤至安全掩体
    canSafelyPlaceBomb(sim, ownIdx, blastLen, speedCellsPerSec, nowMs) {
      const W = sim.W || (sim.level && (sim.level.w || sim.level.width)) || 15;
      const H = sim.H || (sim.level && (sim.level.h || sim.level.height)) || 13;
      const N = W * H;

      const hypothetical = { idx: ownIdx, blast: blastLen, fuseTicks: 30 };
      const simDanger = this.buildDangerMap(sim, nowMs, hypothetical);
      const deadline = nowMs + 3000 - SAFETY_MARGIN_MS;

      const r0 = (ownIdx / W) | 0, c0 = ownIdx % W;
      const candidates = [];
      for (let i = 0; i < N; i++) {
        if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 || i === ownIdx) continue;
        if (!simDanger.hasFutureDanger(i, nowMs)) {
          const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
          candidates.push({ cell: i, dist });
        }
      }
      candidates.sort((a, b) => a.dist - b.dist);

      for (let c = 0; c < Math.min(candidates.length, 8); c++) {
        const target = candidates[c].cell;
        const res = this.search(sim, simDanger, ownIdx, target, speedCellsPerSec, nowMs, {
          extraBlocked: ownIdx,
          allowBreakBrick: false
        });
        if (res && res.path.length > 1) {
          if (res.arrivalTimes[res.arrivalTimes.length - 1] <= deadline) {
            this.lastEscapeTarget = target;
            this.lastEscapePath = res.path;
            return true;
          }
        }
      }
      this.lastEscapeTarget = -1;
      this.lastEscapePath = null;
      return false;
    }

    // 主决策入口：返回 [move, bomb]
    act(sim, pid) {
      const W = sim.W || (sim.level && (sim.level.w || sim.level.width)) || 15;
      const H = sim.H || (sim.level && (sim.level.h || sim.level.height)) || 13;
      const N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = (sim.t || 0) * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      // 脚下是否有即时危险（正在燃烧，或将在 1000ms 内起火）
      const nextStart = danger.nextDangerStart(ownIdx, nowMs);
      const inImminentDanger = danger.hitTest(ownIdx, nowMs, 0) || (nextStart !== null && nextStart - nowMs <= 1000);

      // ============================================================
      // 1. 承诺逃生路径（执行放泡后的单向撤离，绝不震荡）
      // ============================================================
      if (this.escapePath && this.escapePath.length > 1) {
        if (this.escapePath[0] === ownIdx) {
          const nextCell = this.escapePath[1];
          // 目标格安全验证：绝不能盲目迈入正在燃烧或即将爆炸的火线
          const nextStartCell = danger.nextDangerStart(nextCell, nowMs);
          const cellSafe = !sim.wall[nextCell] && !sim.brick[nextCell] && sim.fuse[nextCell] === 0 &&
                           !danger.hitTest(nextCell, nowMs, 0) &&
                           (nextStartCell === null || nextStartCell - nowMs > 500);
          if (cellSafe) {
            const mv = this._cellToMove(ownIdx, nextCell, W);
            if (mm[pid][mv] === 1) {
              this.escapePath.shift();
              return this._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
            }
          } else {
            // 撤退路线受阻，作废重算
            this.escapePath = [];
            this.escapeTarget = -1;
          }
        }
        if (ownIdx === this.escapeTarget || !danger.hasFutureDanger(ownIdx, nowMs)) {
          this.escapePath = [];
          this.escapeTarget = -1;
        }
      }

      // ============================================================
      // 2. 紧急避险：脚下即将爆炸，全力逃往无火线安全区
      // ============================================================
      if (inImminentDanger) {
        const safeCells = [];
        const r0 = own[0], c0 = own[1];
        for (let i = 0; i < N; i++) {
          if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0) continue;
          if (!danger.hasFutureDanger(i, nowMs)) {
            const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
            safeCells.push({ cell: i, dist });
          }
        }
        safeCells.sort((a, b) => a.dist - b.dist);

        for (let s = 0; s < Math.min(safeCells.length, 6); s++) {
          const res = this.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs, { allowBreakBrick: false });
          if (res && res.path.length > 1) {
            const mv = this._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              return this._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
            }
          }
        }

        // 贪心兜底：选起火时刻最晚的合法邻居
        let bestMv = MOVE_IDLE, maxWait = -1;
        for (let d = 0; d < 4; d++) {
          const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const np = nr * W + nc;
          if (mm[pid][d] !== 1) continue;
          const s = danger.nextDangerStart(np, nowMs) || 999999;
          if (s > maxWait) { maxWait = s; bestMv = d; }
        }
        return [bestMv, 0];
      }

      // ============================================================
      // 3. 宏观目标决策：吃宝箱发育 > 逼近对手压迫
      // ============================================================
      let oppIdx = -1;
      for (let o = 0; o < 2; o++) {
        if (o !== pid && sim.alive[o]) {
          const oc = sim.centerCell(o);
          oppIdx = oc[0] * W + oc[1];
          break;
        }
      }

      let targetGoal = oppIdx;
      // 优先吃宝箱升级（吃箱发育对对抗 Hunter / 模型至关重要）
      let nearestCrate = -1, minCrateDist = Infinity;
      for (let i = 0; i < N; i++) {
        if (sim.crate[i]) {
          const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
          if (d < minCrateDist) { minCrateDist = d; nearestCrate = i; }
        }
      }
      if (nearestCrate !== -1 && (minCrateDist <= 6 || (sim.spdG && sim.spdG[pid] < 1.8) || (sim.bombsCap && sim.bombsCap[pid] < 3))) {
        targetGoal = nearestCrate;
      }

      // 全局穿砖 A* 寻路（破砖开路）
      let pathRes = targetGoal !== -1 ? this.search(sim, danger, ownIdx, targetGoal, spd, nowMs, {
        allowBreakBrick: true
      }) : null;

      let move = MOVE_IDLE;
      let nextCellIsBrick = false;
      if (pathRes && pathRes.path.length > 1) {
        const nextCell = pathRes.path[1];
        if (sim.brick[nextCell]) {
          nextCellIsBrick = true; // 下一步是障碍砖，需要放泡破障！
        } else {
          move = this._cellToMove(ownIdx, nextCell, W);
        }
      }

      // ============================================================
      // 4. 战术落子：破砖开路 + 直瞄进攻 + 连环穿梭
      // ============================================================
      let placeBomb = 0;
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      const blastCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const cooldownTicks = (sim.t || 0) - this.lastDropTick;

      if (canDrop && !inImminentDanger) {
        // 战术 A: 面前紧贴阻碍前进的砖墙（破砖开路）
        const wantBreakBrick = nextCellIsBrick;

        // 战术 B: 与对手同轴直瞄攻击
        let directLineAttack = false;
        let nearOpp = false;
        if (oppIdx !== -1) {
          const or = (oppIdx / W) | 0, oc = oppIdx % W;
          const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
          if (dr === 0 && dc <= blastCap) {
            let blocked = false;
            const minC = Math.min(own[1], oc), maxC = Math.max(own[1], oc);
            for (let c = minC + 1; c < maxC; c++) {
              if (sim.wall[own[0] * W + c] || sim.brick[own[0] * W + c]) { blocked = true; break; }
            }
            if (!blocked) directLineAttack = true;
          } else if (dc === 0 && dr <= blastCap) {
            let blocked = false;
            const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
            for (let r = minR + 1; r < maxR; r++) {
              if (sim.wall[r * W + own[1]] || sim.brick[r * W + own[1]]) { blocked = true; break; }
            }
            if (!blocked) directLineAttack = true;
          }
          if (dr + dc <= blastCap + 1) {
            nearOpp = true;
          }
        }

        // 战术 C: 连锁老炮（时间差连环引爆）
        let chainOldBomb = false;
        for (let d = 0; d < 4 && !chainOldBomb; d++) {
          const [dr, dc] = DIRS[d];
          for (let k = 1; k <= blastCap; k++) {
            const nr = own[0] + dr * k, nc = own[1] + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const idx = nr * W + nc;
            if (sim.wall[idx] || sim.brick[idx]) break;
            const f = sim.fuse[idx];
            if (f >= 6 && f <= 24) { chainOldBomb = true; break; }
          }
        }

        // 战术 D: 穿梭游走落子（空场压迫）
        const roamBomb = cooldownTicks >= 8;

        const wantBomb = wantBreakBrick || directLineAttack || nearOpp || chainOldBomb || roamBomb;

        if (wantBomb && cooldownTicks >= 5) {
          if (this.canSafelyPlaceBomb(sim, ownIdx, blastCap, spd, nowMs)) {
            placeBomb = 1;
            this.lastDropTick = sim.t || 0;
            this.escapeTarget = this.lastEscapeTarget;
            this.escapePath = this.lastEscapePath;
          }
        }
      }

      return this._filterImmediateDanger(sim, danger, pid, move, placeBomb, nowMs, W, H);
    }

    // 最终即时物理安全过滤：绝对严禁迈入正在燃烧或 500ms 内起火的格子
    _filterImmediateDanger(sim, danger, pid, move, placeBomb, nowMs, W, H) {
      const { mm } = sim.legalMask();
      if (move === MOVE_IDLE) {
        return [MOVE_IDLE, placeBomb];
      }
      if (mm[pid][move] !== 1) {
        return [MOVE_IDLE, placeBomb];
      }

      const own = sim.centerCell(pid);
      const nr = own[0] + DIRS[move][0], nc = own[1] + DIRS[move][1];
      if (nr < 0 || nr >= H || nc < 0 || nc >= W) {
        return [MOVE_IDLE, placeBomb];
      }
      const targetCell = nr * W + nc;

      // 严禁踩入正在燃烧中或 500ms 内起火的格子
      if (danger.hitTest(targetCell, nowMs, 0)) {
        return [MOVE_IDLE, placeBomb];
      }
      const nextStart = danger.nextDangerStart(targetCell, nowMs);
      if (nextStart !== null && nextStart - nowMs <= 500) {
        return [MOVE_IDLE, placeBomb];
      }

      return [move, placeBomb];
    }

    _cellToMove(from, to, W) {
      const fr = (from / W) | 0, fc = from % W;
      const tr = (to / W) | 0, tc = to % W;
      if (tr < fr) return MOVE_UP;
      if (tr > fr) return MOVE_DOWN;
      if (tc < fc) return MOVE_LEFT;
      if (tc > fc) return MOVE_RIGHT;
      return MOVE_IDLE;
    }
  }

  TimeAStarAI.MinHeap = MinHeap;
  TimeAStarAI.DangerMap = DangerMap;
  return TimeAStarAI;
});
