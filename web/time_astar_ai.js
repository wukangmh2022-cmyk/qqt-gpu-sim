/**
 * TimeAStarAI - 高级时空规则 AI
 *
 * 核心设计：
 * 1. Time-Aware A*：以物理到达时刻（arrival_time，毫秒）作为 g 值展开启发式搜索；
 * 2. 连续时空危险窗（DangerMap）：每格维护 [(start_ms, end_ms)] 动态威胁区间；
 * 3. 连锁老炮引爆仿真（Chain Detonation）：前瞻递归预测多泡连环引爆的时间窗；
 * 4. 严禁安全区踏火（No Jitter / Anti-Oscillation）：处于安全区时严格禁止踏入任何未爆火线，彻底消除震荡抖动；
 * 5. 闭环逃生路径承诺（Committed Escape）：放泡时直接锁定推演出的绝对安全掩体路径，放完后丝滑直达掩体；
 * 6. 平滑无限走位与连环老炮（Smooth Roaming & Chaining）：游走走位、定期落子、连环老炮破障与压迫。
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
  const SAFETY_MARGIN_MS = 500;  // 穿行安全前置余量（ms）
  const FLAME_LINGER_MS = 250;   // 爆炸余威残留（ms）

  // 二叉小顶堆（Min-Heap by cost）
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
      if (typeof cell !== 'number' || cell < 0 || cell >= this.N || !this.windows[cell]) return true;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        const s = wins[i][0], e = wins[i][1];
        if (t + safetyMargin >= s && t <= e) return true;
      }
      return false;
    }

    // 该格未来是否会有炸弹引爆/起火（用于判定当前是否处于火线威胁区）
    hasFutureDanger(cell, afterMs = 0) {
      if (typeof cell !== 'number' || cell < 0 || cell >= this.N || !this.windows[cell]) return false;
      const wins = this.windows[cell];
      for (let i = 0; i < wins.length; i++) {
        if (wins[i][1] > afterMs) return true;
      }
      return false;
    }

    // 该格在 after 之后最早的起火时刻
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

  class TimeAStarAI {
    constructor() {
      this.targetCell = -1;
      this.targetPath = [];
      this.lastDropTick = -999;
      this.roamStep = 0;
    }

    // 构建时空危险窗（含多泡连环引爆链仿真）
    buildDangerMap(sim, nowMs = 0, extraBomb = null) {
      const W = sim.W || (sim.level && sim.level.width) || 15;
      const H = sim.H || (sim.level && sim.level.height) || 13;
      const N = W * H;
      const danger = new DangerMap(W, H);

      // 1. 收集在场真炸弹及已残留余威
      const bombs = [];
      for (let i = 0; i < N; i++) {
        if (sim.fuse[i] > 0) {
          bombs.push({
            idx: i,
            blast: sim.bombBlast[i] || 2,
            boomAt: nowMs + sim.fuse[i] * 100
          });
        }
        if (sim.blastLinger[i] > 0) {
          danger.addWindow(i, nowMs, nowMs + sim.blastLinger[i] * 100);
        }
      }

      // 2. 模拟假设放泡（用于防自杀模拟）
      if (extraBomb) {
        bombs.push({
          idx: extraBomb.idx,
          blast: extraBomb.blast || 2,
          boomAt: nowMs + (extraBomb.fuseTicks || 30) * 100
        });
      }

      // 3. 连锁老炮引爆计算（迭代收敛）
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

    // 2 步回溯死胡同预判：前瞻进入后是否有安全出口
    wouldBeTrapped(sim, danger, cell, arriveMs, stepMs, fromCell, extraBlocked = -1) {
      const W = sim.W || (sim.level && sim.level.width) || 15;
      const H = sim.H || (sim.level && sim.level.height) || 13;
      const escapeDeadline = arriveMs + 2 * stepMs;
      const r = (cell / W) | 0, c = cell % W;

      let exits = 0;
      let safeExits = 0;

      for (let d = 0; d < 4; d++) {
        const nr = r + DIRS[d][0], nc = c + DIRS[d][1];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
        const np = nr * W + nc;
        if (np === fromCell) continue;
        if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0 || np === extraBlocked) continue;

        exits++;
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
    search(sim, danger, start, goal, speedCellsPerSec, nowMs, options = {}) {
      if (start === goal) {
        return { path: [start], arrivalTimes: [nowMs] };
      }
      const W = sim.W || (sim.level && sim.level.width) || 15;
      const H = sim.H || (sim.level && sim.level.height) || 13;
      const N = W * H;
      const stepMs = Math.round(1000 / Math.max(0.5, speedCellsPerSec));
      const allowBreakBrick = !!options.allowBreakBrick;
      const forbidFutureDanger = !!options.forbidFutureDanger;
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

          // 物理障碍检查
          if (sim.wall[np] || (sim.fuse[np] > 0 && np !== start) || np === extraBlocked) continue;
          if (sim.brick[np] && !allowBreakBrick) continue;

          // 防震荡关键：在安全漫游模式下，绝对不主动踏入任何处于未来炸弹火线中的格子
          if (forbidFutureDanger && danger.hasFutureDanger(np, nowMs)) continue;

          const costMul = sim.brick[np] ? 4 : 1;
          const arrive = curT + stepMs * costMul;

          // 时间剪枝①：碰撞爆炸余威（500ms 安全前置裕量）
          if (danger.hitTest(np, arrive, SAFETY_MARGIN_MS)) {
            continue;
          }

          // 时间剪枝②：死胡同预判
          if (np !== goal && this.wouldBeTrapped(sim, danger, np, arrive, stepMs, curCell, extraBlocked)) {
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

    // 防自杀推演：模拟在 ownIdx 放泡后，能否在爆炸前安全撤离到绝对安全掩体
    canSafelyPlaceBomb(sim, ownIdx, blastLen, speedCellsPerSec, nowMs) {
      const W = sim.W || (sim.level && sim.level.width) || 15;
      const H = sim.H || (sim.level && sim.level.height) || 13;
      const N = W * H;

      const hypothetical = { idx: ownIdx, blast: blastLen, fuseTicks: 30 };
      const simDanger = this.buildDangerMap(sim, nowMs, hypothetical);
      const boomAt = nowMs + 3000;
      const deadline = boomAt - SAFETY_MARGIN_MS; // 必须在 2500ms 前完成撤离

      // 寻找全图真正安全的格子（没有任何炸弹威胁，包括假设的新泡）
      const candidates = [];
      const r0 = (ownIdx / W) | 0, c0 = ownIdx % W;
      for (let i = 0; i < N; i++) {
        if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 || i === ownIdx) continue;
        if (!simDanger.hasFutureDanger(i, nowMs)) {
          const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
          candidates.push({ cell: i, dist });
        }
      }
      candidates.sort((a, b) => a.dist - b.dist);

      for (let c = 0; c < Math.min(candidates.length, 6); c++) {
        const target = candidates[c].cell;
        const res = this.search(sim, simDanger, ownIdx, target, speedCellsPerSec, nowMs, { extraBlocked: ownIdx });
        if (res && res.path.length > 1) {
          const arriveTarget = res.arrivalTimes[res.arrivalTimes.length - 1];
          if (arriveTarget <= deadline) {
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
      const W = sim.W || (sim.level && sim.level.width) || 15;
      const H = sim.H || (sim.level && sim.level.height) || 13;
      const N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = (sim.t || 0) * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      const underThreat = danger.hasFutureDanger(ownIdx, nowMs);

      // ============================================================
      // 阶段 1: 绝对避险模式（脚下在火线上，严禁放泡，全速撤离）
      // ============================================================
      if (underThreat) {
        // 如果之前已有承诺的逃生目标且该目标依旧真正安全，保持目标不动摇
        let currentTarget = this.targetCell;
        if (currentTarget === -1 || currentTarget === ownIdx || danger.hasFutureDanger(currentTarget, nowMs)) {
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
            const res = this.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs);
            if (res && res.path.length > 1) {
              this.targetCell = safeCells[s].cell;
              this.targetPath = res.path;
              break;
            }
          }
        }

        // 沿逃生路径前进
        if (this.targetCell !== -1) {
          const res = this.search(sim, danger, ownIdx, this.targetCell, spd, nowMs);
          if (res && res.path.length > 1) {
            const mv = this._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              return [mv, 0];
            }
          }
        }

        // 终极避险兜底
        let bestMv = MOVE_IDLE, maxWait = -1;
        const r0 = own[0], c0 = own[1];
        for (let d = 0; d < 4; d++) {
          const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const np = nr * W + nc;
          if (mm[pid][d] !== 1) continue;
          const s = danger.nextDangerStart(np, nowMs) || 999999;
          if (s > maxWait) {
            maxWait = s;
            bestMv = d;
          }
        }
        return [bestMv, 0];
      }

      // ============================================================
      // 阶段 2: 处于安全区（平滑走位漫游，严禁踏入火线）
      // ============================================================
      if (this.targetCell === ownIdx) {
        this.targetCell = -1;
        this.targetPath = [];
      }

      // 若没有活跃目标，规划一个新的安全漫游目标
      if (this.targetCell === -1) {
        // 目标优先级 1: 吃无危险的道具箱
        let nearestCrate = -1, minCrateDist = Infinity;
        for (let i = 0; i < N; i++) {
          if (sim.crate[i] && !danger.hasFutureDanger(i, nowMs)) {
            const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
            if (d < minCrateDist) {
              minCrateDist = d;
              nearestCrate = i;
            }
          }
        }
        if (nearestCrate !== -1) {
          this.targetCell = nearestCrate;
        } else {
          // 目标优先级 2: 找一个开阔、安全的漫游格子（平滑走位）
          let oppIdx = -1;
          for (let o = 0; o < 2; o++) {
            if (o !== pid && sim.alive[o]) {
              const oc = sim.centerCell(o);
              oppIdx = oc[0] * W + oc[1];
              break;
            }
          }

          let bestScore = Infinity;
          let bestGoal = -1;
          const or = oppIdx !== -1 ? ((oppIdx / W) | 0) : 6;
          const oc = oppIdx !== -1 ? (oppIdx % W) : 7;

          for (let i = 0; i < N; i++) {
            if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 || danger.hasFutureDanger(i, nowMs)) continue;
            const ir = (i / W) | 0, ic = i % W;
            const dToOpp = Math.abs(ir - or) + Math.abs(ic - oc);
            const dToMe = Math.abs(ir - own[0]) + Math.abs(ic - own[1]);
            // 挑选距离自己 3~6 格、靠近对手方向的安全格
            const score = Math.abs(dToMe - 4) * 2 + dToOpp;
            if (score < bestScore) {
              bestScore = score;
              bestGoal = i;
            }
          }
          if (bestGoal !== -1) {
            this.targetCell = bestGoal;
          }
        }
      }

      let move = MOVE_IDLE;
      if (this.targetCell !== -1) {
        // 漫游寻路时严格禁止踏入任何炸弹的未爆火线（forbidFutureDanger: true）
        const pathRes = this.search(sim, danger, ownIdx, this.targetCell, spd, nowMs, { forbidFutureDanger: true });
        if (pathRes && pathRes.path.length > 1) {
          const nextCell = pathRes.path[1];
          move = this._cellToMove(ownIdx, nextCell, W);
        } else {
          // 目标不可达，重置
          this.targetCell = -1;
        }
      }

      // ============================================================
      // 阶段 3: 放新炮与连老炮（放泡后直接锁定撤离路线，消除抖动）
      // ============================================================
      let placeBomb = 0;
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0;
      const blastCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const cooldownPassed = (sim.t || 0) - this.lastDropTick >= 20; // 2.0秒节奏放泡

      if (canDrop && !underThreat && cooldownPassed) {
        // 连老炮判定：十字射程内是否有引信 8~22 ticks 的老炮
        let chainOldBomb = false;
        for (let d = 0; d < 4 && !chainOldBomb; d++) {
          const [dr, dc] = DIRS[d];
          for (let k = 1; k <= blastCap; k++) {
            const nr = own[0] + dr * k, nc = own[1] + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const idx = nr * W + nc;
            if (sim.wall[idx] || sim.brick[idx]) break;
            const f = sim.fuse[idx];
            if (f >= 8 && f <= 22) {
              chainOldBomb = true;
              break;
            }
          }
        }

        // 破砖判定：紧挨着阻断砖
        let brickAdjacent = false;
        for (let d = 0; d < 4; d++) {
          const nr = own[0] + DIRS[d][0], nc = own[1] + DIRS[d][1];
          if (nr >= 0 && nr < H && nc >= 0 && nc < W && sim.brick[nr * W + nc]) {
            brickAdjacent = true;
            break;
          }
        }

        // 走位开阔地放新泡
        const wantBomb = chainOldBomb || brickAdjacent || cooldownPassed;

        if (wantBomb) {
          if (this.canSafelyPlaceBomb(sim, ownIdx, blastCap, spd, nowMs)) {
            placeBomb = 1;
            this.lastDropTick = sim.t || 0;
            // 关键：放泡的瞬间直接锁定推演好的安全逃生目标和路径！
            this.targetCell = this.lastEscapeTarget;
            this.targetPath = this.lastEscapePath;
          }
        }
      }

      // 物理掩码最终校验
      if (move !== MOVE_IDLE && mm[pid][move] !== 1) {
        move = MOVE_IDLE;
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

  return TimeAStarAI;
});
