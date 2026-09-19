/**
 * web/jev_ai.js - TypeSafe Jev (System One) 驱动的 QQT 对战 AI
 *
 * 核心架构：
 * 1. 结构化游戏态提取（extractJevState）：
 *    - 地图地形：墙体、可炸砖块、推箱、开阔通道
 *    - 道具感知：炸弹上限/威力/移速/超级宝箱的坐标、类型、曼哈顿距离及到达状态
 *    - 自身与对手：生命值、当前属性数值与关卡上限（bombs/blast/speed vs max）、饱和状态、双方相对距离
 *    - 危险场感知：炸弹引信倒计时、爆炸余威与连环引爆链
 *
 * 2. TypeSafe Jev 语义推理（System One）：
 *    - priority (Choice): [collect_crate, bomb_brick, attack_opponent, dodge_danger]
 *    - target (Choice): 在精选候选目标（最近道具、关键阻断砖、对手位置、安全掩体）中选择最佳目标
 *    - place_bomb (Noul): 是否就地落子破障或封路
 *
 * 3. 异步平滑执行与物理安全网（Execution & Safety Filter）：
 *    - 异步推理（~200ms）不阻塞 10Hz 仿真循环，采用航点巡航追踪
 *    - 具备即时危险规避（绝不主动踩火）与死胡同放泡自杀前瞻拦截
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const TimeAStarAI = require('./time_astar_ai.js');
    module.exports = factory(TimeAStarAI);
  } else {
    root.JevAI = factory(root.TimeAStarAI);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (TimeAStarAI) {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 0: 上, 1: 下, 2: 左, 3: 右
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;

  class JevAI {
    constructor(options = {}) {
      this.options = options;
      this.apiUrl = options.apiUrl || '/api/typesafe';
      this.model = options.model || 'jev-latest';
      this.inferIntervalTicks = options.inferIntervalTicks || 10; // 约每 1.0s 重新请求一次 Jev
      
      this.helperAi = new TimeAStarAI({ mode: 'roam' });
      this.reset();
    }

    reset() {
      this.helperAi.reset();
      this.lastInferTick = -999;
      this.isInferring = false;
      this.currentTarget = null;     // { type: 'crate'|'brick'|'enemy'|'safety', cell: number }
      this.currentPath = [];
      this.lastMove = MOVE_IDLE;
      this.lastTacticalDecision = null;
      this.decisionHistory = [];
      this.stats = {
        totalCalls: 0,
        bombsPlaced: 0,
        cratesCollected: 0,
        bricksDestroyed: 0,
        lastLatencyMs: 0
      };
      this.lastPlacedBombCell = -1;
    }

    computeConnectedComponent(sim, startCell) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const reachable = new Uint8Array(N);
      if (!Number.isFinite(startCell) || startCell < 0 || startCell >= N || sim.wall[startCell]) return reachable;
      reachable[startCell] = 1;
      const q = [startCell];
      let head = 0;
      while (head < q.length) {
        const cur = q[head++];
        const cr = (cur / W) | 0, cc = cur % W;
        for (let d = 0; d < 4; d++) {
          const nr = cr + DIRS[d][0], nc = cc + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const ni = nr * W + nc;
          if (sim.wall[ni] === 1) continue; // 实心墙阻断连通
          if (!reachable[ni]) {
            reachable[ni] = 1;
            q.push(ni);
          }
        }
      }
      return reachable;
    }

    computeWalkableComponent(sim, startCell) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const walkable = new Uint8Array(N);
      if (!Number.isFinite(startCell) || startCell < 0 || startCell >= N || sim.wall[startCell] || sim.brick[startCell]) return walkable;
      walkable[startCell] = 1;
      const q = [startCell];
      let head = 0;
      while (head < q.length) {
        const cur = q[head++];
        const cr = (cur / W) | 0, cc = cur % W;
        for (let d = 0; d < 4; d++) {
          const nr = cr + DIRS[d][0], nc = cc + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const ni = nr * W + nc;
          if (sim.wall[ni] === 1 || sim.brick[ni] === 1 || (sim.pushable && sim.pushable[ni] === 1)) continue;
          if (!walkable[ni]) {
            walkable[ni] = 1;
            q.push(ni);
          }
        }
      }
      return walkable;
    }

    // ------------------------------------------------------------ 提取结构化状态给 Jev
    extractJevState(sim, pid) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const opp = 1 - pid;
      const own = sim.centerCell(pid);
      const oppCell = sim.centerCell(opp);
      const ownIdx = own[0] * W + own[1];
      const oppIdx = oppCell[0] * W + oppCell[1];

      // 计算连通分量 (Reachable Component) 与即时直达分量 (Walkable Component)
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);
      const walkableMask = this.computeWalkableComponent(sim, ownIdx);

      // 1. 自身与对手属性
      const bCap = sim.bombsCap ? sim.bombsCap[pid] : 2;
      const bMax = sim.bombsMax || 10;
      const zCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const zMax = sim.blastMax || 8;
      const sCap = sim.spdG ? Number(sim.spdG[pid].toFixed(2)) : 1.3;
      const sMax = sim.speedMax !== undefined ? Number(sim.speedMax.toFixed(2)) : 2.4;

      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      // 2. 扫描地图道具（Crates，仅纳入物理连通可达的道具）
      const crates = [];
      for (let i = 0; i < N; i++) {
        if (sim.crate[i] === 1 && reachableMask[i] === 1) {
          const cr = (i / W) | 0, cc = i % W;
          const dist = Math.abs(cr - own[0]) + Math.abs(cc - own[1]);
          const ct = sim.crateType[i];
          const isSuper = sim.superCrate && sim.superCrate[i] === 1;
          let typeStr = 'bomb';
          if (ct === 1) typeStr = 'blast';
          else if (ct === 2) typeStr = 'speed';
          if (isSuper) typeStr = 'super_' + typeStr;
          
          crates.push({
            cell: i,
            pos: [cr, cc],
            type: typeStr,
            directly_walkable: walkableMask[i] === 1,
            dist: dist
          });
        }
      }
      crates.sort((a, b) => a.dist - b.dist);

      // 3. 扫描附近可炸砖块（Bricks，仅纳入物理连通可达的砖）
      const bricks = [];
      for (let i = 0; i < N; i++) {
        if (sim.brick[i] === 1 && (!sim.pushable || !sim.pushable[i]) && reachableMask[i] === 1) {
          const br = (i / W) | 0, bc = i % W;
          const dist = Math.abs(br - own[0]) + Math.abs(bc - own[1]);
          if (dist <= 6) { // 只挑附近的
            bricks.push({
              cell: i,
              pos: [br, bc],
              dist: dist
            });
          }
        }
      }
      bricks.sort((a, b) => a.dist - b.dist);

      // 4. 当前危险状态
      const nowMs = (sim.t || 0) * 100;
      const danger = this.helperAi.buildDangerMap(sim, nowMs);
      const isCurrentDanger = danger.hitTest(ownIdx, nowMs, 0) || danger.hasFutureDanger(ownIdx, nowMs);

      // 5. 直瞄火线与进攻态势计算（Direct Line-of-Sight & Attack Vectors）
      let directLineAttack = false;
      let lineDirection = null;
      if (oppIdx !== -1) {
        const or = oppCell[0], oc = oppCell[1];
        const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
        if (dr === 0 && dc <= zCap && dc > 0) {
          let blocked = false;
          const minC = Math.min(own[1], oc), maxC = Math.max(own[1], oc);
          for (let c = minC + 1; c < maxC; c++) {
            if (sim.wall[own[0] * W + c] || sim.brick[own[0] * W + c]) { blocked = true; break; }
          }
          if (!blocked) {
            directLineAttack = true;
            lineDirection = own[1] < oc ? 'right' : 'left';
          }
        } else if (dc === 0 && dr <= zCap && dr > 0) {
          let blocked = false;
          const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
          for (let r = minR + 1; r < maxR; r++) {
            if (sim.wall[r * W + own[1]] || sim.brick[r * W + own[1]]) { blocked = true; break; }
          }
          if (!blocked) {
            directLineAttack = true;
            lineDirection = own[0] < or ? 'down' : 'up';
          }
        }
      }
      const nearOpp = oppDist <= (zCap + 1);
      const isOneHitKill = (sim.hp && sim.hp[opp] === 1) || (sim.initialHp === 1);

      // 6. 四邻可移动方向
      const directions = {};
      const dirNames = ['up', 'down', 'left', 'right'];
      for (let d = 0; d < 4; d++) {
        const tr = own[0] + DIRS[d][0], tc = own[1] + DIRS[d][1];
        const name = dirNames[d];
        if (tr < 0 || tr >= H || tc < 0 || tc >= W) {
          directions[name] = 'wall_boundary';
        } else {
          const ti = tr * W + tc;
          if (sim.wall[ti]) directions[name] = 'indestructible_wall';
          else if (sim.brick[ti]) directions[name] = 'destructible_brick';
          else if (sim.fuse[ti] > 0) directions[name] = 'ticking_bomb';
          else if (danger.hitTest(ti, nowMs, 300)) directions[name] = 'danger_flame';
          else directions[name] = 'safe_open';
        }
      }

      return {
        step: sim.t || 0,
        player: {
          pos: own,
          hp: sim.hp ? sim.hp[pid] : 5,
          bombs: bCap,
          bombs_max: bMax,
          bombs_is_full: bCap >= bMax,
          blast: zCap,
          blast_max: zMax,
          blast_is_full: zCap >= zMax,
          speed: sCap,
          speed_max: sMax,
          speed_is_full: sCap >= sMax - 0.05,
          can_place_bomb: sim.liveBombs(pid) < bCap && sim.fuse[ownIdx] <= 0
        },
        opponent: {
          pos: oppCell,
          hp: sim.hp ? sim.hp[opp] : 5,
          distance: oppDist,
          alive: sim.alive ? Boolean(sim.alive[opp]) : true
        },
        tactical_context: {
          in_line_of_fire: directLineAttack,
          line_direction: lineDirection,
          near_opponent: nearOpp,
          is_one_hit_kill: isOneHitKill,
          opponent_distance: oppDist
        },
        surroundings: directions,
        under_fire: isCurrentDanger,
        nearby_crates: crates.slice(0, 5),
        nearby_bricks: bricks.slice(0, 5)
      };
    }

    // ------------------------------------------------------------ 向 Jev 发起异步推理
    async callJevAsync(sim, pid) {
      if (this.isInferring) return;
      this.isInferring = true;
      const t0 = performance.now();

      try {
        const state = this.extractJevState(sim, pid);

        // 构造候选目标列表给 Jev 选择
        const candidateCriteria = {};
        const candidateMap = new Map();

        // 候选 1: 避险
        candidateCriteria['dodge'] = 'Move towards the nearest safe shelter away from bomb flames';
        candidateMap.set('dodge', { type: 'safety' });

        // 候选 2: 进攻对手（优先）
        const oppDesc = state.tactical_context.is_one_hit_kill ? ' (1-HP LETHAL KILL)' : '';
        const fireDesc = state.tactical_context.in_line_of_fire ? ' [IN DIRECT LINE OF FIRE!]' : '';
        candidateCriteria['attack_enemy'] = `Pursue, corner and bomb opponent at (${state.opponent.pos[0]}, ${state.opponent.pos[1]}), distance ${state.opponent.distance}${oppDesc}${fireDesc}`;
        candidateMap.set('attack_enemy', { type: 'enemy', pos: state.opponent.pos });

        // 候选 3: 炸掉前 2 个关键阻断砖块
        state.nearby_bricks.slice(0, 2).forEach((b, idx) => {
          const key = `brick_${idx + 1}`;
          candidateCriteria[key] = `Approach and bomb blocking brick at (${b.pos[0]}, ${b.pos[1]}), distance ${b.dist} to open path`;
          candidateMap.set(key, { type: 'brick', cell: b.cell, pos: b.pos });
        });

        // 候选 4: 拾取前 2 个道具
        state.nearby_crates.slice(0, 2).forEach((c, idx) => {
          const key = `crate_${idx + 1}`;
          candidateCriteria[key] = `Collect ${c.type} crate at (${c.pos[0]}, ${c.pos[1]}), distance ${c.dist}`;
          candidateMap.set(key, { type: 'crate', cell: c.cell, pos: c.pos });
        });

        const questions = {
          tactical_priority: {
            type: 'choice',
            instructions: 'Based on tactical context (is_one_hit_kill, in_line_of_fire, opponent_distance, under_fire), what is the highest tactical priority?',
            criteria: {
              attack_opponent: 'Opponent has 1 HP, is in line of fire, nearby, or path is clear; aggressively pursue, corner, and bomb opponent to eliminate them.',
              dodge_danger: 'Current tile or path is under active bomb flame or imminent explosion; retreat to safe shelter.',
              bomb_brick: 'A destructible brick is blocking the direct corridor to opponent or item; place bomb to clear it.',
              collect_crate: 'A powerup crate is conveniently on the path; collect it to increase attributes.'
            }
          },
          target_selection: {
            type: 'choice',
            instructions: 'Select the best immediate objective to navigate towards.',
            criteria: candidateCriteria
          },
          place_bomb_now: {
            type: 'noul',
            instructions: 'Should player place a bomb at current location right now? (True if opponent is in line of fire, adjacent, trapped, or a brick directly blocks the path).'
          }
        };

        let url = this.apiUrl;
        const headers = { 'Content-Type': 'application/json' };

        // 在 Node.js 环境下，如果未配完整 URL 则优先连本地代理，或直连 TypeSafe API
        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
          if (url.startsWith('/')) {
            url = 'http://localhost:8080' + url;
          }
          const apiKey = process.env.TYPESAFE_API_KEY || '';
          if (url.includes('api.typesafe.ai')) {
            headers['Authorization'] = `Bearer ${apiKey}`;
          }
        }

        const resp = await fetch(url, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify({
            model: this.model,
            state: state,
            questions: questions
          })
        });

        if (!resp.ok) {
          throw new Error(`Jev API error: ${resp.status} ${resp.statusText}`);
        }

        const data = await resp.json();
        const answers = data.answers || {};
        const priority = answers.tactical_priority ? answers.tactical_priority.choice : 'collect_crate';
        const targetKey = answers.target_selection ? answers.target_selection.choice : 'dodge';
        const shouldBomb = answers.place_bomb_now ? (answers.place_bomb_now.noul > 0.6) : false;

        this.lastTacticalDecision = {
          priority,
          targetKey,
          targetObj: candidateMap.get(targetKey) || candidateMap.get('dodge'),
          shouldBomb,
          timestamp: Date.now()
        };

        const decisionEntry = {
          time: new Date().toLocaleTimeString(),
          tick: sim.t || 0,
          priority,
          targetKey,
          targetObj: this.lastTacticalDecision.targetObj,
          shouldBomb,
          inLineOfFire: state.tactical_context.in_line_of_fire,
          oppDist: state.opponent.distance,
          latencyMs: this.stats.lastLatencyMs
        };
        this.decisionHistory.push(decisionEntry);
        if (this.decisionHistory.length > 50) this.decisionHistory.shift();

        this.currentTarget = this.lastTacticalDecision.targetObj;
        this.stats.totalCalls++;
        this.stats.lastLatencyMs = Math.round(performance.now() - t0);

      } catch (err) {
        console.warn('[JevAI] Inference error, falling back to heuristic:', err);
      } finally {
        this.isInferring = false;
      }
    }

    // ------------------------------------------------------------ 主决策入口 act(sim, pid)
    act(sim, pid, rng) {
      const curTick = sim.t || 0;
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = curTick * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.helperAi.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      // 1. 定期触发 Jev 研判（或危险状态突变时立即重新研判）
      const nextStart = danger.nextDangerStart(ownIdx, nowMs);
      const inImminentDanger = danger.hitTest(ownIdx, nowMs, 0) || (nextStart !== null && nextStart - nowMs <= 800);

      if (!this.isInferring && (curTick - this.lastInferTick >= this.inferIntervalTicks || inImminentDanger)) {
        this.lastInferTick = curTick;
        // 异步发起，不阻塞本 tick
        this.callJevAsync(sim, pid);
      }

      // 2. 承诺逃生路径（放泡后单向撤出）
      if (this.helperAi.escapePath && this.helperAi.escapePath.length > 0) {
        if (ownIdx === this.helperAi.escapeTarget) {
          if (this.lastPlacedBombCell >= 0 && sim.fuse[this.lastPlacedBombCell] > 0) {
            this.lastMove = MOVE_IDLE;
            return [MOVE_IDLE, 0];
          } else {
            this.lastPlacedBombCell = -1;
            this.helperAi.escapePath = [];
            this.helperAi.escapeTarget = -1;
          }
        } else {
          const currIdxInPath = this.helperAi.escapePath.indexOf(ownIdx);
          if (currIdxInPath > 0) {
            this.helperAi.escapePath = this.helperAi.escapePath.slice(currIdxInPath);
          }
          if (this.helperAi.escapePath.length > 1 && this.helperAi.escapePath[0] === ownIdx) {
            const nextCell = this.helperAi.escapePath[1];
            const nextStartCell = danger.nextDangerStart(nextCell, nowMs);
            const cellSafe = !sim.wall[nextCell] && !sim.brick[nextCell] && sim.fuse[nextCell] === 0 &&
                             !danger.hitTest(nextCell, nowMs, 0) &&
                             (nextStartCell === null || nextStartCell - nowMs > 500);
            if (cellSafe) {
              const mv = this.helperAi._cellToMove(ownIdx, nextCell, W);
              if (mm[pid][mv] === 1) {
                const finalAct = this.helperAi._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
                this.lastMove = finalAct[0];
                return finalAct;
              }
            } else {
              this.helperAi.escapePath = [];
              this.helperAi.escapeTarget = -1;
            }
          }
        }
      }

      // 3. 物理安全网：脚下有火或即将爆炸，必须绝对避险
      if (inImminentDanger) {
        const safeCells = [];
        for (let i = 0; i < N; i++) {
          if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0) continue;
          if (!danger.hasFutureDanger(i, nowMs)) {
            const dist = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
            safeCells.push({ cell: i, dist });
          }
        }
        safeCells.sort((a, b) => a.dist - b.dist);
        for (let s = 0; s < Math.min(safeCells.length, 5); s++) {
          const res = this.helperAi.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs, { allowBreakBrick: false });
          if (res && res.path.length > 1) {
            const mv = this.helperAi._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              const safeAct = this.helperAi._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
              this.lastMove = safeAct[0];
              return safeAct;
            }
          }
        }
      }

      // 3. 执行 Jev 制定的目标与决策并做连通性校验与投影
      let targetCell = -1;
      let wantBomb = false;
      const dec = this.lastTacticalDecision;
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);

      if (dec) {
        wantBomb = dec.shouldBomb;
        if (dec.targetObj) {
          if (dec.targetObj.type === 'crate' && sim.crate[dec.targetObj.cell] === 1 && reachableMask[dec.targetObj.cell]) {
            targetCell = dec.targetObj.cell;
          } else if (dec.targetObj.type === 'brick' && sim.brick[dec.targetObj.cell] === 1 && reachableMask[dec.targetObj.cell]) {
            targetCell = dec.targetObj.cell;
          } else if (dec.targetObj.type === 'enemy') {
            const opp = 1 - pid;
            const oppC = sim.centerCell(opp);
            targetCell = oppC[0] * W + oppC[1];
          }
        }
      }

      // 若当前没有有效目标或目标已达成，默认进攻对手或寻找最近箱子
      if (targetCell < 0) {
        const opp = 1 - pid;
        const oppC = sim.centerCell(opp);
        targetCell = oppC[0] * W + oppC[1];
      }

      // 强连通性校验与就近投影：确保目标决不落在非联通格或实心墙上
      if (targetCell < 0 || targetCell >= N || !reachableMask[targetCell] || sim.wall[targetCell] === 1) {
        let bestDist = Infinity;
        let bestC = ownIdx;
        for (let i = 0; i < N; i++) {
          if (reachableMask[i] && sim.wall[i] === 0) {
            const ir = (i / W) | 0, ic = i % W;
            const tr = (targetCell / W) | 0, tc = targetCell % W;
            const d = Math.abs(ir - tr) + Math.abs(ic - tc);
            if (d < bestDist) {
              bestDist = d;
              bestC = i;
            }
          }
        }
        targetCell = bestC;
      }

      // 4. 寻路前往目标
      let searchRes = this.helperAi.search(sim, danger, ownIdx, targetCell, spd, nowMs, {
        allowBreakBrick: true,
        lastMove: this.lastMove
      });

      // 寻路兜底：仅在寻路真正受阻且未到达目标时触发
      if (!searchRes && ownIdx !== targetCell) {
        const candidates = [];
        for (let i = 0; i < N; i++) {
          if (reachableMask[i] && (sim.crate[i] === 1 || sim.brick[i] === 1)) {
            const dist = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
            candidates.push({ cell: i, dist });
          }
        }
        candidates.sort((a, b) => a.dist - b.dist);
        for (let c = 0; c < Math.min(candidates.length, 5); c++) {
          const altRes = this.helperAi.search(sim, danger, ownIdx, candidates[c].cell, spd, nowMs, {
            allowBreakBrick: true,
            lastMove: this.lastMove
          });
          if (altRes && altRes.path.length > 1) {
            searchRes = altRes;
            targetCell = candidates[c].cell;
            break;
          }
        }
      }

      // 确保路径起点严格与当前 ownIdx 对齐（若错位则自动截断）
      if (searchRes && searchRes.path.length > 0) {
        const ownPosInPath = searchRes.path.indexOf(ownIdx);
        if (ownPosInPath > 0) {
          searchRes.path = searchRes.path.slice(ownPosInPath);
        }
      }

      // 暴露给外部与前端录屏高亮的目标属性
      this.targetCell = targetCell;
      this.targetPos = targetCell >= 0 ? [(targetCell / W) | 0, targetCell % W] : null;
      this.targetType = dec?.targetObj?.type || 'enemy';
      this.targetLabel = dec?.targetKey || 'attack_enemy';
      this.targetPriority = dec?.priority || 'attack_opponent';
      this.targetShouldBomb = wantBomb;
      this.currentSearchPath = searchRes ? searchRes.path : [];

      let chosenMove = MOVE_IDLE;
      let finalBomb = 0;
      let nextStepIsBrick = false;

      if (searchRes && searchRes.path.length > 1 && searchRes.path[0] === ownIdx) {
        const nextCell = searchRes.path[1];
        if (sim.brick[nextCell]) {
          nextStepIsBrick = true; // 下一步是障碍砖，需要放泡破障！
        } else {
          chosenMove = this.helperAi._cellToMove(ownIdx, nextCell, W);
        }
      }

      // 5. 实时进攻态势（直瞄锁敌 / 贴身锁死）
      const opp = 1 - pid;
      const oppCell = sim.centerCell(opp);
      const oppIdx = oppCell[0] * W + oppCell[1];
      const zCap = sim.blastCap ? sim.blastCap[pid] : 2;
      let directLineAttack = false;
      if (oppIdx !== -1 && sim.alive && sim.alive[opp]) {
        const or = oppCell[0], oc = oppCell[1];
        const dr = Math.abs(own[0] - or), dc = Math.abs(own[1] - oc);
        if (dr === 0 && dc <= zCap && dc > 0) {
          let blocked = false;
          const minC = Math.min(own[1], oc), maxC = Math.max(own[1], oc);
          for (let c = minC + 1; c < maxC; c++) {
            if (sim.wall[own[0] * W + c] || sim.brick[own[0] * W + c]) { blocked = true; break; }
          }
          if (!blocked) directLineAttack = true;
        } else if (dc === 0 && dr <= zCap && dr > 0) {
          let blocked = false;
          const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
          for (let r = minR + 1; r < maxR; r++) {
            if (sim.wall[r * W + own[1]] || sim.brick[r * W + own[1]]) { blocked = true; break; }
          }
          if (!blocked) directLineAttack = true;
        }
      }
      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      // 6. 放泡执行：
      // 条件 A: 路径前方为障碍砖需炸开
      // 条件 B: 对手进入直瞄十字火线 (directLineAttack)，立刻放泡必杀
      // 条件 C: 贴身压迫对手 (oppDist <= 1) 且战术处于进攻模式
      // 条件 D: Jev 决策 shouldBomb
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      if (canDrop && !inImminentDanger) {
        const shouldDropForBrick = nextStepIsBrick;
        const shouldDropForDirectHit = directLineAttack;
        const shouldDropForNearOpp = (oppDist <= 1) && (wantBomb || (dec && dec.priority === 'attack_opponent'));
        const shouldDropForJev = wantBomb && (nextStepIsBrick || oppDist <= 1 || directLineAttack);
        if (shouldDropForBrick || shouldDropForDirectHit || shouldDropForNearOpp || shouldDropForJev) {
          const safeToDrop = this.helperAi.canSafelyPlaceBomb(sim, ownIdx, sim.blastCap[pid], spd, nowMs);
          if (safeToDrop) {
            finalBomb = 1;
            this.stats.bombsPlaced++;
            this.lastDropTick = curTick;
            this.lastPlacedBombCell = ownIdx;
            // 立即消耗决策，防止持续多 tick 重复盲目落子
            if (dec) dec.shouldBomb = false;
            this.targetShouldBomb = false;
            // 锁定逃生路线：放完泡后立即沿着已验证的安全掩体路径撤离
            if (this.helperAi.lastEscapePath && this.helperAi.lastEscapePath.length > 1) {
              this.helperAi.escapePath = this.helperAi.lastEscapePath.slice();
              this.helperAi.escapeTarget = this.helperAi.lastEscapeTarget;
              chosenMove = this.helperAi._cellToMove(ownIdx, this.helperAi.escapePath[1], W);
            }
          }
        }
      }

      // 7. 最终即时危险过滤（绝不踏进已燃烧格）
      const finalAct = this.helperAi._filterImmediateDanger(sim, danger, pid, chosenMove, finalBomb, nowMs, W, H);
      this.lastMove = finalAct[0];
      return finalAct;
    }
  }

  return JevAI;
});
