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
        if (afterMs >= wins[i][0] && afterMs <= wins[i][1]) {
          return afterMs;
        }
        if (wins[i][0] >= afterMs) {
          if (minS === null || wins[i][0] < minS) minS = wins[i][0];
        }
      }
      return minS;
    }
  }

  class TimeAStarAI {
    constructor(options = {}) {
      this.mode = options.mode || 'hunt'; // 'hunt' (竞技追猎版) 或 'roam' (经典漫游连炮版)
      this.reset();
    }

    reset() {
      this.targetCell = -1;
      this.targetPath = [];
      this.escapePath = [];
      this.escapeTarget = -1;
      this.lastEscapeTarget = -1;
      this.lastEscapePath = null;
      this.lastDropTick = -999;
      this.lastMove = -1;
      this.prevOwnIdx = -1;
      this.roamTarget = -1;
      this.roamTicks = 0;
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
      let extraIdx = -1;
      if (extraBomb) {
        extraIdx = bombs.length;
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

      if (extraIdx !== -1) {
        danger.extraBoomAt = bombs[extraIdx].boomAt;
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

          // 严格物理时间剪枝：在该格逗留期间（physicalStep）处于危险起火窗中（前置安全余量 SAFETY_MARGIN_MS）
          if (danger.hitTest(np, arrive, physicalStep + SAFETY_MARGIN_MS)) {
            continue;
          }

          // 虚拟启发代价：只参与小顶堆排序，绝不污染真实的 arrive
          let heuristicCost = arrive + hMs(np);
          if (danger.hasFutureDanger(np, nowMs)) {
            const dangerStart = danger.nextDangerStart(np, nowMs);
            if (dangerStart !== null && dangerStart - nowMs <= 1800) {
              heuristicCost += 8000; // 即将引爆的危险火线走廊，极高代价坚决绕行
            } else {
              heuristicCost += 3000;
            }
          }
          if (sim.brick[np]) {
            heuristicCost += 3000; // 偏好走现成通路，无路时才穿砖
          }
          if (options.prevCell !== undefined && np === options.prevCell && curCell === start) {
            heuristicCost += 2000; // 严禁第一步立即掉头走回刚刚离开的格子（消除 A* 等权格来回振荡）
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
      const actualBoomAt = simDanger.extraBoomAt !== undefined ? simDanger.extraBoomAt : (nowMs + 3000);
      const deadline = actualBoomAt - SAFETY_MARGIN_MS;

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
      const curTick = sim.t || 0;
      if (curTick <= 0 || curTick < this.lastDropTick) {
        this.reset();
      }
      const W = sim.W || (sim.level && (sim.level.w || sim.level.width)) || 15;
      const H = sim.H || (sim.level && (sim.level.h || sim.level.height)) || 13;
      const N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = (sim.t || 0) * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const blastCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const danger = this.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      // 脚下是否有即时危险（正在燃烧，或将在 700ms 内起火）
      const nextStart = danger.nextDangerStart(ownIdx, nowMs);
      const inImminentDanger = danger.hitTest(ownIdx, nowMs, 0) || (nextStart !== null && nextStart - nowMs <= 700);

      // ============================================================
      // 1. 承诺逃生路径（执行放泡后的单向撤离，绝不震荡）
      // ============================================================
      if (this.escapePath && this.escapePath.length > 1) {
        // 如果已经到达逃生终点，或者当前格完全脱离危险
        if (ownIdx === this.escapeTarget || (!danger.hasFutureDanger(ownIdx, nowMs) && !danger.hitTest(ownIdx, nowMs, 0))) {
          this.escapePath = [];
          this.escapeTarget = -1;
        } else {
          // 查找当前格在逃生路径中的推进位置
          let nextStepIdx = -1;
          const currIdx = this.escapePath.indexOf(ownIdx);
          if (currIdx !== -1) {
            nextStepIdx = currIdx + 1;
          } else {
            // 如果不在路径节点上，尝试找路径上与当前格相邻的最近后续节点（跳过第 0 个放雷点）
            for (let k = 1; k < this.escapePath.length; k++) {
              if (this._isAdjacent(ownIdx, this.escapePath[k], W)) {
                nextStepIdx = k;
                break;
              }
            }
          }

          if (nextStepIdx >= this.escapePath.length) {
            this.escapePath = [];
            this.escapeTarget = -1;
          } else if (nextStepIdx > 0) {
            const nextCell = this.escapePath[nextStepIdx];
            const nextStartCell = danger.nextDangerStart(nextCell, nowMs);
            const cellSafe = !sim.wall[nextCell] && !sim.brick[nextCell] && sim.fuse[nextCell] === 0 &&
                             !danger.hitTest(nextCell, nowMs, 0) &&
                             (nextStartCell === null || nextStartCell - nowMs > 400);
            if (cellSafe) {
              const mv = this._cellToMove(ownIdx, nextCell, W);
              if (mm[pid][mv] === 1) {
                const act = this._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
                this.lastMove = act[0];
                this.prevOwnIdx = ownIdx;
                return act;
              }
            } else {
              this.escapePath = [];
              this.escapeTarget = -1;
            }
          } else {
            // 彻底脱离逃生路径，作废并交由紧急避险重新计算
            this.escapePath = [];
            this.escapeTarget = -1;
          }
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

        for (let s = 0; s < Math.min(safeCells.length, 8); s++) {
          const res = this.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs, { allowBreakBrick: false });
          if (res && res.path.length > 1) {
            this.escapePath = res.path;
            this.escapeTarget = safeCells[s].cell;
            const mv = this._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              const act = this._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
              this.lastMove = act[0];
              this.prevOwnIdx = ownIdx;
              return act;
            }
          }
        }

        // 贪心兜底：选起火时刻最晚且未燃烧的合法邻居
        let bestMv = MOVE_IDLE, maxWait = -2;
        for (let d = 0; d < 4; d++) {
          const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const np = nr * W + nc;
          if (mm[pid][d] !== 1) continue;
          if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0) continue;
          let s = danger.nextDangerStart(np, nowMs);
          if (s === null) s = 999999;
          if (danger.hitTest(np, nowMs, 0)) s = -1;
          // 惯性偏好：避免在火海中 180 度来回折返跳
          if (this.lastMove !== undefined && this.lastMove >= 0 && this.lastMove < 4) {
            if (d === this.lastMove) s += 0.2;
            if (d === (this.lastMove ^ 1)) s -= 0.3;
          }
          if (s > maxWait) { maxWait = s; bestMv = d; }
        }
        const act = [bestMv, 0];
        this.lastMove = bestMv;
        this.prevOwnIdx = ownIdx;
        return act;
      }

      // ============================================================
      // 3. 宏观目标决策：双模式分流（竞技追猎 vs 经典漫游）
      // ============================================================
      let oppIdx = -1;
      for (let o = 0; o < 2; o++) {
        if (o !== pid && sim.alive[o]) {
          const oc = sim.centerCell(o);
          oppIdx = oc[0] * W + oc[1];
          break;
        }
      }

      let nearestCrate = -1, minCrateDist = Infinity;
      for (let i = 0; i < N; i++) {
        if (sim.crate[i]) {
          const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
          if (d < minCrateDist) { minCrateDist = d; nearestCrate = i; }
        }
      }

      let targetGoal = -1;

      if (this.mode === 'roam') {
        // === 经典漫游模式：不主动追踪对手，专注全图巡游、吃道具、破砖开荒 ===
        if (nearestCrate !== -1 && minCrateDist <= 8) {
          targetGoal = nearestCrate;
        } else {
          this.roamTicks = (this.roamTicks || 0) + 1;
          const needNewRoam = this.roamTarget === -1 ||
                              this.roamTarget === ownIdx ||
                              this.roamTicks > 25 ||
                              sim.wall[this.roamTarget] ||
                              danger.hitTest(this.roamTarget, nowMs, 0);
          if (needNewRoam) {
            this.roamTicks = 0;
            // 优先检查周围 1~3 格是否有障碍砖阻挡去路（开荒期主动破砖开路 hard block destroying）
            let bestBrick = -1, minBrickDist = Infinity;
            for (let i = 0; i < N; i++) {
              if (sim.brick[i]) {
                const r = (i / W) | 0, c = i % W;
                const dist = Math.abs(r - own[0]) + Math.abs(c - own[1]);
                if (dist < minBrickDist && dist <= 3) {
                  minBrickDist = dist;
                  bestBrick = i;
                }
              }
            }

            if (bestBrick !== -1 && (minBrickDist <= 2 || Math.random() < 0.5)) {
              this.roamTarget = bestBrick;
            } else {
              const candidates = [];
              for (let i = 0; i < N; i++) {
                if (sim.wall[i] || sim.fuse[i] > 0 || i === ownIdx) continue;
                if (danger.hitTest(i, nowMs, 0)) continue;
                const ds = danger.nextDangerStart(i, nowMs);
                if (ds !== null && ds - nowMs < 1200) continue;

                const r = (i / W) | 0, c = i % W;
                const dist = Math.abs(r - own[0]) + Math.abs(c - own[1]);
                if (dist >= 2 && dist <= 7) {
                  let openDegree = 0;
                  for (let d = 0; d < 4; d++) {
                    const nr = r + DIRS[d][0], nc = c + DIRS[d][1];
                    if (nr >= 0 && nr < H && nc >= 0 && nc < W && !sim.wall[nr * W + nc] && !sim.brick[nr * W + nc]) {
                      openDegree++;
                    }
                  }
                  // 连老炮加分：如果能与场上已有老炮 (fuse 10~25) 形成十字连线，给予倾向加分
                  let chainBonus = 0;
                  for (let d = 0; d < 4; d++) {
                    const [dr, dc] = DIRS[d];
                    for (let k = 1; k <= blastCap; k++) {
                      const nr = r + dr * k, nc = c + dc * k;
                      if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
                      const idx = nr * W + nc;
                      if (sim.wall[idx] || sim.brick[idx]) break;
                      if (sim.fuse[idx] >= 10 && sim.fuse[idx] <= 25) {
                        chainBonus += 20;
                        break;
                      }
                    }
                  }
                  const brickPenalty = sim.brick[i] ? 15 : 0;
                  candidates.push({ cell: i, score: openDegree * 10 - dist * 2 + chainBonus - brickPenalty });
                }
              }
              if (candidates.length > 0) {
                candidates.sort((a, b) => b.score - a.score);
                const pick = candidates[Math.floor(Math.random() * Math.min(3, candidates.length))];
                this.roamTarget = pick.cell;
              } else {
                this.roamTarget = -1;
              }
            }
          }
          targetGoal = this.roamTarget;
        }
      } else {
        // === 竞技追猎模式：直瞄锁敌、近身压迫、优先吃宝箱升级 ===
        targetGoal = oppIdx;
        if (nearestCrate !== -1 && (minCrateDist <= 12 || (sim.spdG && sim.spdG[pid] < 2.0) || (sim.bombsCap && sim.bombsCap[pid] < 4))) {
          targetGoal = nearestCrate;
        }
      }

      // 全局穿砖 A* 寻路（破砖开路）
      let pathRes = targetGoal !== -1 ? this.search(sim, danger, ownIdx, targetGoal, spd, nowMs, {
        allowBreakBrick: true,
        prevCell: this.prevOwnIdx
      }) : null;

      let move = MOVE_IDLE;
      let nextCellIsBrick = false;
      if (pathRes && pathRes.path.length > 1) {
        const nextCell = pathRes.path[1];
        if (sim.brick[nextCell]) {
          nextCellIsBrick = true; // 下一步是障碍砖，需要放泡破障！
        } else if (targetGoal === oppIdx && nextCell === oppIdx) {
          // 绝对防送头：严禁肉身直接跨入敌方所在的同一个格子，在 1 格外刹车就地架炮或拉扯
          move = MOVE_IDLE;
        } else {
          move = this._cellToMove(ownIdx, nextCell, W);
        }
      }

      // ============================================================
      // 4. 战术落子：破砖开路 + 连老炮 + 节奏铺雷 / 竞技压迫
      // ============================================================
      let placeBomb = 0;
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      const cooldownTicks = (sim.t || 0) - this.lastDropTick;

      if (canDrop && !inImminentDanger) {
        const wantBreakBrick = nextCellIsBrick;

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

        let wantBomb = false;

        if (this.mode === 'roam') {
          // 经典漫游模式：破砖开路 + 连环老炮 + 节奏巡航铺雷（每隔 8~10 tick 放一颗） + 自卫反击
          const roamRhythm = cooldownTicks >= 9;
          let selfDefense = false;
          if (oppIdx !== -1) {
            const or = (oppIdx / W) | 0, oc = oppIdx % W;
            const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
            if ((dr === 0 && dc <= 2) || (dc === 0 && dr <= 2)) {
              selfDefense = true;
            }
          }
          wantBomb = wantBreakBrick || chainOldBomb || roamRhythm || selfDefense;
        } else {
          // 竞技追猎模式：破砖 + 直瞄 + 贴身 + 连炮
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
            if (dr + dc <= 2) {
              nearOpp = true;
            }
          }
          wantBomb = wantBreakBrick || directLineAttack || nearOpp || chainOldBomb;
        }

        const minCooldown = this.mode === 'hunt'
          ? Math.max(6, Math.min(10, blastCap + 2))
          : 9;
        if (wantBomb && cooldownTicks >= minCooldown) {
          if (this.canSafelyPlaceBomb(sim, ownIdx, blastCap, spd, nowMs)) {
            const escMv = this.lastEscapePath && this.lastEscapePath.length > 1
              ? this._cellToMove(ownIdx, this.lastEscapePath[1], W)
              : MOVE_IDLE;
            if (escMv !== MOVE_IDLE && mm[pid][escMv] === 1) {
              placeBomb = 1;
              move = escMv;
              this.lastDropTick = sim.t || 0;
              this.escapeTarget = this.lastEscapeTarget;
              this.escapePath = this.lastEscapePath;
              if (this.mode === 'roam') {
                this.roamTarget = -1; // 放泡后重新规划下一个巡航路标
              }
            }
          }
        }
      }

      const finalAct = this._filterImmediateDanger(sim, danger, pid, move, placeBomb, nowMs, W, H);
      this.lastMove = finalAct[0];
      this.prevOwnIdx = ownIdx;
      return finalAct;
    }

    // 最终即时物理安全过滤：绝对严禁主动迈入正在燃烧或即将起火的危险格
    _filterImmediateDanger(sim, danger, pid, move, placeBomb, nowMs, W, H) {
      const { mm } = sim.legalMask();
      if (move === MOVE_IDLE) {
        return [MOVE_IDLE, placeBomb];
      }
      if (mm[pid][move] !== 1) {
        return [MOVE_IDLE, placeBomb];
      }

      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nr = own[0] + DIRS[move][0], nc = own[1] + DIRS[move][1];
      if (nr < 0 || nr >= H || nc < 0 || nc >= W) {
        return [MOVE_IDLE, placeBomb];
      }
      const targetCell = nr * W + nc;

      const ownHit = danger.hitTest(ownIdx, nowMs, 0);
      const ownStart = danger.nextDangerStart(ownIdx, nowMs);
      const ownImminent = ownHit || (ownStart !== null && ownStart - nowMs <= 500);

      const targetHit = danger.hitTest(targetCell, nowMs, 0);
      const targetStart = danger.nextDangerStart(targetCell, nowMs);
      const targetImminent = targetHit || (targetStart !== null && targetStart - nowMs <= 400);

      // 当前格安全时，严禁主动迈入危险格
      if (!ownImminent && targetImminent) {
        return [MOVE_IDLE, placeBomb];
      }

      // 当前格处于火海或即将爆炸时：
      if (ownImminent) {
        if (targetHit && !ownHit) {
          return [MOVE_IDLE, placeBomb]; // 不踏入已经着火的格子
        }
        if (targetStart !== null && ownStart !== null && targetStart < ownStart) {
          return [MOVE_IDLE, placeBomb]; // 不走向比脚下更早爆炸的格子
        }
      }

      return [move, placeBomb];
    }

    _isAdjacent(a, b, W) {
      if (a < 0 || b < 0) return false;
      const ra = (a / W) | 0, ca = a % W;
      const rb = (b / W) | 0, cb = b % W;
      return Math.abs(ra - rb) + Math.abs(ca - cb) === 1;
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
