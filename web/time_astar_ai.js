/**
 * TimeAStarAI - 高级时空规则 AI
 *
 * 核心特性：
 * 1. Time-Aware A*：以物理到达时刻（arrival_time，毫秒）作为 g 值展开启发式搜索；
 * 2. 连续时空危险窗（DangerMap）：每格维护 [(start_ms, end_ms)] 动态威胁区间；
 * 3. 500ms 安全前置余量（SAFETY_MARGIN_MS）：穿行时间与爆炸起火时刻保持安全距离；
 * 4. 连锁老炮引爆仿真（Chain Reaction）：前瞻递归预测多泡连环引爆的时间窗；
 * 5. 全向逃逸推演防自杀（canSafelyPlaceBomb）：放泡前假想模拟，确保在爆炸前有明确的绝对安全掩体路径；
 * 6. 避险与自保绝对优先：脚下受火线威胁时严禁放泡，全速撤离至安全区。
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const AI = factory();
    module.exports = AI;
  } else {
    const AI = factory();
    root.TimeAStarAI = AI;
    root.NukemanAI = AI; // 保持向后兼容别名
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
      // 内部状态
      this.lastDropTick = -999;
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
            boomAt: nowMs + sim.fuse[i] * 100 // 10Hz, 1 tick = 100ms
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
              if (sim.wall[idx]) break; // 硬墙完全阻断火焰
              for (let b = 0; b < bombs.length; b++) {
                if (bombs[b].idx === idx && bombs[b].boomAt > bA.boomAt) {
                  bombs[b].boomAt = bA.boomAt;
                  changed = true;
                }
              }
              if (sim.brick[idx] || sim.pushable[idx]) break; // 砖墙阻断后续延伸
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

          // 物理通行障碍（除起点本身可离开外，其他带泡格不可通行）
          if (sim.wall[np] || (sim.fuse[np] > 0 && np !== start) || np === extraBlocked) continue;
          if (sim.brick[np] && !allowBreakBrick) continue;

          // 破砖代价加权
          const costMul = sim.brick[np] ? 4 : 1;
          const arrive = curT + stepMs * costMul;

          // 时间剪枝①：碰撞爆炸余威（500ms 安全前置裕量）
          if (danger.hitTest(np, arrive, SAFETY_MARGIN_MS)) {
            continue;
          }

          // 时间剪枝②：死胡同预判（仅当非目标格时剪枝）
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

      // 对最近的 6 个候选安全格进行逃生验证
      for (let c = 0; c < Math.min(candidates.length, 6); c++) {
        const target = candidates[c].cell;
        const res = this.search(sim, simDanger, ownIdx, target, speedCellsPerSec, nowMs, { extraBlocked: ownIdx });
        if (res && res.path.length > 1) {
          const arriveTarget = res.arrivalTimes[res.arrivalTimes.length - 1];
          if (arriveTarget <= deadline) {
            return true; // 存在安全无虞的撤离路径
          }
        }
      }
      return false; // 无路可逃，坚决不放泡！
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

      // 确定敌方位置
      let oppIdx = -1;
      for (let o = 0; o < 2; o++) {
        if (o !== pid && sim.alive[o]) {
          const oc = sim.centerCell(o);
          oppIdx = oc[0] * W + oc[1];
          break;
        }
      }

      const { mm, bm } = sim.legalMask();

      // ============================================================
      // 阶段 1: 绝对避险（自保最高优先级）
      // 脚下已在任何炸弹的火线覆盖范围中，或者正处于起火中 → 严禁放泡，全力撤离
      // ============================================================
      const underThreat = danger.hasFutureDanger(ownIdx, nowMs);
      if (underThreat) {
        // 寻找最近的绝对安全格（未来没有任何炸弹覆盖）
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
            const nextCell = res.path[1];
            const mv = this._cellToMove(ownIdx, nextCell, W);
            if (mm[pid][mv] === 1) {
              return [mv, 0]; // 逃跑过程中严格不放泡
            }
          }
        }

        // 终极避险兜底：若暂无直达绝对安全格的完整路径，选择起火时刻最晚的邻格（拖延时间等老泡炸完）
        let bestMv = MOVE_IDLE;
        let maxWait = -1;
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
      // 阶段 2: 目标规划与路径推进（逼近对手 / 收集道具）
      // ============================================================
      let goal = oppIdx;

      // 道具饥渴检测：若自身属性不足且场上有道具箱，优先搜寻
      const blastCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const hungry = (sim.bombsCap && sim.bombsCap[pid] < 5) || blastCap < 4;
      if (hungry) {
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
        if (nearestCrate !== -1 && minCrateDist <= 6) {
          const crateRes = this.search(sim, danger, ownIdx, nearestCrate, spd, nowMs);
          if (crateRes && crateRes.path.length > 1) {
            goal = nearestCrate;
          }
        }
      }

      // 寻路推进
      let pathRes = goal !== -1 ? this.search(sim, danger, ownIdx, goal, spd, nowMs) : null;

      // 若直接通路被砖阻断，启动虚拟破砖穿透寻路定位关键障碍砖
      let targetBrick = -1;
      if (!pathRes && oppIdx !== -1) {
        const brickSearch = this.search(sim, danger, ownIdx, oppIdx, spd, nowMs, { allowBreakBrick: true });
        if (brickSearch && brickSearch.path.length > 1) {
          for (const cell of brickSearch.path) {
            if (sim.brick[cell]) {
              targetBrick = cell;
              break;
            }
          }
          if (targetBrick !== -1) {
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

      // ============================================================
      // 阶段 3: 放泡与防自杀拦截
      // ============================================================
      let placeBomb = 0;
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0;
      const cooldownPassed = (sim.t || 0) - this.lastDropTick >= 10; // 至少间隔 1 秒（10 ticks），避免连续地毯式堆泡封死自身

      if (canDrop && !underThreat && cooldownPassed) {
        // 条件 A: 直瞄锁敌（同一直线，且无永久硬墙阻隔）
        let directLineAttack = false;
        if (oppIdx !== -1) {
          const or = (oppIdx / W) | 0, oc = oppIdx % W;
          const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
          if (dr === 0 && dc <= blastCap) {
            let blocked = false;
            const minC = Math.min(own[1], oc), maxC = Math.max(own[1], oc);
            for (let c = minC + 1; c < maxC; c++) {
              if (sim.wall[own[0] * W + c]) { blocked = true; break; }
            }
            if (!blocked) directLineAttack = true;
          } else if (dc === 0 && dr <= blastCap) {
            let blocked = false;
            const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
            for (let r = minR + 1; r < maxR; r++) {
              if (sim.wall[r * W + own[1]]) { blocked = true; break; }
            }
            if (!blocked) directLineAttack = true;
          }
        }

        // 条件 B: 破砖开路（面前贴着通往敌人的阻断砖）
        let brickAdjacent = false;
        if (targetBrick !== -1) {
          const tbr = (targetBrick / W) | 0, tbc = targetBrick % W;
          const dbr = Math.abs(own[0] - tbr), dbc = Math.abs(own[1] - tbc);
          if (dbr + dbc === 1) brickAdjacent = true;
        }

        // 条件 C: 连环连锁老泡（blast 十字内有引信 6~15 ticks 的老泡，可连锁压迫）
        let chainAttack = false;
        for (let d = 0; d < 4 && !chainAttack; d++) {
          const [dr, dc] = DIRS[d];
          for (let k = 1; k <= blastCap; k++) {
            const nr = own[0] + dr * k, nc = own[1] + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const idx = nr * W + nc;
            if (sim.wall[idx] || sim.brick[idx]) break;
            const f = sim.fuse[idx];
            // 严禁紧挨着 <5 ticks（即 0.5s 内要爆）的即爆泡放泡（否则瞬间引爆在脸上自爆）
            if (f >= 6 && f <= 15) {
              chainAttack = true;
              break;
            }
          }
        }

        const wantBomb = directLineAttack || brickAdjacent || chainAttack;

        // 核心：放泡前必须进行全向逃生推演验证！
        if (wantBomb) {
          if (this.canSafelyPlaceBomb(sim, ownIdx, blastCap, spd, nowMs)) {
            placeBomb = 1;
            this.lastDropTick = sim.t || 0;
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
