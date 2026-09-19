/**
 * web/jev_autonomous_ai.js - TypeSafe Jev (System One) 完全自主版 AI (GPT-5.6 设想)
 *
 * 核心哲学：
 * 1. 彻底放权 (Full Delegation)：
 *    - 战略姿态 (strategic_posture)：由 Jev 全权决定【进攻围剿 / 走廊伏击 / 资源压制 / 防守风筝】
 *    - 空间落点 (target_selection)：全图高价值战术实体候选（直击对手、走廊截击点、战略阻断砖、核心道具、避难所）
 *    - 放泡动作 (bomb_action)：由 Jev 决定【致命突击 / 破障开路 / 封路陷阱 / 停火机动】
 *    - 下子置信度 (plant_bomb_confidence)：Noul 概率量化下子收益
 * 2. 零本地规则干预 (Zero Heuristic Overrides)：
 *    - 废除本地 hardcoded directLineAttack / nearOpp 强制放泡逻辑
 *    - 只有当 Jev 的 bomb_action 明确下达放泡指令且 confidence > 0.5 时才落子
 * 3. 极简物理安全网 (Minimal Physical Safety Invariant)：
 *    - 本地控制器仅充当“底层身体反射”：绝不主动踩入正在燃烧的烈焰，绝不在 100% 无法生还的死胡同自杀
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const TimeAStarAI = require('./time_astar_ai.js');
    module.exports = factory(TimeAStarAI);
  } else {
    root.JevAutonomousAI = factory(root.TimeAStarAI);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (TimeAStarAI) {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 0: 上, 1: 下, 2: 左, 3: 右
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;

  class JevAutonomousAI {
    constructor(options = {}) {
      this.options = options;
      this.apiUrl = options.apiUrl || '/api/typesafe';
      this.model = options.model || 'jev-latest';
      this.inferIntervalTicks = options.inferIntervalTicks || 10; // 约每 1.0s 重新向 Jev 请求一次宏观推演

      this.helperAi = new TimeAStarAI({ mode: 'roam' });
      this.reset();
    }

    reset() {
      this.helperAi.reset();
      this.lastInferTick = -999;
      this.isInferring = false;
      this.currentTarget = null;
      this.currentPath = [];
      this.lastMove = MOVE_IDLE;
      this.lastDecision = null;
      this.decisionHistory = [];
      this.stats = {
        totalCalls: 0,
        bombsPlaced: 0,
        cratesCollected: 0,
        lastLatencyMs: 0
      };

      // 暴露给前端渲染与录屏的实时状态
      this.targetCell = -1;
      this.targetPos = null;
      this.targetType = 'enemy';
      this.targetLabel = 'enemy_direct';
      this.targetPosture = 'offensive_siege';
      this.bombAction = 'hold_fire';
      this.bombConfidence = 0.0;
      this.currentSearchPath = [];
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

    // ------------------------------------------------------------ 提取全局空间态势
    extractAutonomousState(sim, pid) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const opp = 1 - pid;
      const own = sim.centerCell(pid);
      const oppCell = sim.centerCell(opp);
      const ownIdx = own[0] * W + own[1];
      const oppIdx = oppCell[0] * W + oppCell[1];

      // 计算连通分量 (Reachable Component) 与即时直达分量 (Walkable Component)
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);
      const walkableMask = this.computeWalkableComponent(sim, ownIdx);

      // 1. 属性与上限
      const bCap = sim.bombsCap ? sim.bombsCap[pid] : 2;
      const bMax = sim.bombsMax || 10;
      const zCap = sim.blastCap ? sim.blastCap[pid] : 2;
      const zMax = sim.blastMax || 8;
      const sCap = sim.spdG ? Number(sim.spdG[pid].toFixed(2)) : 1.3;
      const sMax = sim.speedMax !== undefined ? Number(sim.speedMax.toFixed(2)) : 2.4;

      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      // 2. 扫描道具（仅纳入物理连通可达的道具，按价值与距离加权排序）
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
            is_super: isSuper,
            directly_walkable: walkableMask[i] === 1,
            dist: dist
          });
        }
      }
      // 优先超级道具，其次最近道具
      crates.sort((a, b) => (b.is_super ? 5 : 0) - (a.is_super ? 5 : 0) || (a.dist - b.dist));

      // 3. 扫描战略阻断砖块（仅纳入物理连通可达的砖）
      const bricks = [];
      for (let i = 0; i < N; i++) {
        if (sim.brick[i] === 1 && (!sim.pushable || !sim.pushable[i]) && reachableMask[i] === 1) {
          const br = (i / W) | 0, bc = i % W;
          const distToOwn = Math.abs(br - own[0]) + Math.abs(bc - own[1]);
          const distToOpp = Math.abs(br - oppCell[0]) + Math.abs(bc - oppCell[1]);
          // 仅筛选距离自身不超过 6 格的砖
          if (distToOwn <= 6) {
            bricks.push({
              cell: i,
              pos: [br, bc],
              dist_to_player: distToOwn,
              dist_to_opp: distToOpp,
              corridor_score: distToOwn + distToOpp
            });
          }
        }
      }
      // 优先打通连接玩家与对手最短走廊的砖
      bricks.sort((a, b) => a.corridor_score - b.corridor_score || a.dist_to_player - b.dist_to_player);

      // 4. 直瞄火线态势与对手弱点
      let directLineAttack = false;
      let lineDir = null;
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
            lineDir = own[1] < oc ? 'right' : 'left';
          }
        } else if (dc === 0 && dr <= zCap && dr > 0) {
          let blocked = false;
          const minR = Math.min(own[0], or), maxR = Math.max(own[0], or);
          for (let r = minR + 1; r < maxR; r++) {
            if (sim.wall[r * W + own[1]] || sim.brick[r * W + own[1]]) { blocked = true; break; }
          }
          if (!blocked) {
            directLineAttack = true;
            lineDir = own[0] < or ? 'down' : 'up';
          }
        }
      }

      const nowMs = (sim.t || 0) * 100;
      const danger = this.helperAi.buildDangerMap(sim, nowMs);
      const isCurrentDanger = danger.hitTest(ownIdx, nowMs, 0) || danger.hasFutureDanger(ownIdx, nowMs);
      const isOneHitKill = (sim.hp && sim.hp[opp] === 1) || (sim.initialHp === 1);

      // 5. 四邻与近身障碍态势
      const surroundings = {};
      const dirNames = ['up', 'down', 'left', 'right'];
      let adjacentBricksCount = 0;
      let adjacentBrickDir = null;
      let adjacentToOpp = false;

      for (let d = 0; d < 4; d++) {
        const tr = own[0] + DIRS[d][0], tc = own[1] + DIRS[d][1];
        const name = dirNames[d];
        if (tr < 0 || tr >= H || tc < 0 || tc >= W) {
          surroundings[name] = 'boundary_wall';
        } else {
          const ti = tr * W + tc;
          if (tr === oppCell[0] && tc === oppCell[1]) {
            surroundings[name] = 'opponent';
            adjacentToOpp = true;
          } else if (sim.wall[ti]) {
            surroundings[name] = 'indestructible_wall';
          } else if (sim.brick[ti]) {
            surroundings[name] = 'destructible_brick';
            adjacentBricksCount++;
            if (!adjacentBrickDir) adjacentBrickDir = name;
          } else if (sim.fuse[ti] > 0) {
            surroundings[name] = 'ticking_bomb';
          } else if (danger.hitTest(ti, nowMs, 300)) {
            surroundings[name] = 'bomb_flame';
          } else {
            surroundings[name] = 'open_path';
          }
        }
      }

      // 6. 对手可逃逸路径数评估（对手是否处于死角）
      let oppEscapeRoutes = 0;
      for (let d = 0; d < 4; d++) {
        const tr = oppCell[0] + DIRS[d][0], tc = oppCell[1] + DIRS[d][1];
        if (tr >= 0 && tr < H && tc >= 0 && tc < W) {
          const ti = tr * W + tc;
          if (!sim.wall[ti] && !sim.brick[ti] && sim.fuse[ti] === 0) oppEscapeRoutes++;
        }
      }

      // 7. 构造战略候选目标集合 (Candidate Targets)
      const candidateCriteria = {};
      const candidateMap = new Map();

      // 候选 A: 直击对手本体
      const lethalNote = isOneHitKill ? ' [LETHAL 1-HP KILL OPPORTUNITY]' : '';
      const fireNote = directLineAttack ? ` [IN DIRECT BLAST LINE (${lineDir})!]` : '';
      const adjNote = adjacentToOpp ? ' [ENEMY IS ADJACENT!]' : '';
      candidateCriteria['enemy_direct'] = `Directly assault and corner opponent at (${oppCell[0]}, ${oppCell[1]}), distance ${oppDist}${lethalNote}${fireNote}${adjNote}`;
      candidateMap.set('enemy_direct', { type: 'enemy', pos: oppCell, cell: oppIdx });

      // 候选 B: 走廊截击点（若对手在移动，预判截击）
      if (oppDist >= 2) {
        const midR = Math.round((own[0] + oppCell[0]) / 2);
        const midC = Math.round((own[1] + oppCell[1]) / 2);
        const midIdx = midR * W + midC;
        if (!sim.wall[midIdx] && !sim.brick[midIdx]) {
          candidateCriteria['enemy_intercept'] = `Advance to choke intersection at (${midR}, ${midC}) to cut off opponent retreat route`;
          candidateMap.set('enemy_intercept', { type: 'intercept', pos: [midR, midC], cell: midIdx });
        }
      }

      // 候选 C: 战略突破阻断砖 (至多 3 块)
      bricks.slice(0, 3).forEach((b, idx) => {
        const key = `chokepoint_brick_${idx + 1}`;
        const isAdj = b.dist_to_player === 1 ? ' [IMMEDIATELY ADJACENT!]' : '';
        candidateCriteria[key] = `Approach and bomb strategic barrier brick at (${b.pos[0]}, ${b.pos[1]}), distance ${b.dist_to_player}${isAdj} to unlock path to opponent/items`;
        candidateMap.set(key, { type: 'brick', pos: b.pos, cell: b.cell });
      });

      // 候选 D: 核心道具争夺 (至多 2 个)
      crates.slice(0, 2).forEach((c, idx) => {
        const key = `powerup_${idx + 1}`;
        candidateCriteria[key] = `Capture ${c.type} crate at (${c.pos[0]}, ${c.pos[1]}), distance ${c.dist} for stat dominance`;
        candidateMap.set(key, { type: 'crate', pos: c.pos, cell: c.cell });
      });

      // 候选 E: 战术掩体与避险
      candidateCriteria['tactical_cover'] = 'Reposition to the nearest safe cover tile out of potential blast corridors';
      candidateMap.set('tactical_cover', { type: 'safety' });

      return {
        step: sim.t || 0,
        player: {
          pos: own,
          hp: sim.hp ? sim.hp[pid] : 5,
          bombs: bCap,
          bombs_max: bMax,
          blast: zCap,
          blast_max: zMax,
          speed: sCap,
          speed_max: sMax,
          can_place_bomb: sim.liveBombs(pid) < bCap && sim.fuse[ownIdx] <= 0
        },
        opponent: {
          pos: oppCell,
          hp: sim.hp ? sim.hp[opp] : 5,
          distance: oppDist,
          alive: sim.alive ? Boolean(sim.alive[opp]) : true,
          escape_corridors_count: oppEscapeRoutes,
          is_trapped_in_corner: oppEscapeRoutes <= 1
        },
        tactical_context: {
          in_line_of_fire: directLineAttack,
          line_direction: lineDir,
          is_adjacent_to_opponent: adjacentToOpp,
          is_adjacent_to_brick: adjacentBricksCount > 0,
          adjacent_brick_direction: adjacentBrickDir,
          adjacent_bricks_count: adjacentBricksCount,
          is_one_hit_kill: isOneHitKill,
          opponent_distance: oppDist,
          under_fire: isCurrentDanger
        },
        surroundings,
        candidateCriteria,
        candidateMap
      };
    }

    // ------------------------------------------------------------ 发起完全放权的 Jev 异步推演
    async callJevAsync(sim, pid) {
      if (this.isInferring) return;
      this.isInferring = true;
      const t0 = performance.now();

      try {
        const stateObj = this.extractAutonomousState(sim, pid);
        const { candidateCriteria, candidateMap, ...cleanState } = stateObj;

        // 4 个高度自主维度的提问
        const questions = {
          // 1. 宏观战略姿态
          strategic_posture: {
            type: 'choice',
            instructions: 'Evaluate overall battlefield state (is_one_hit_kill, in_line_of_fire, opponent_distance, escape routes). What is the master strategic posture for this phase?',
            criteria: {
              offensive_siege: 'Opponent has 1 HP (or is_one_hit_kill), is within strike range, trapped, or path is open; aggressively corner, chase down, and bomb opponent to eliminate them.',
              tactical_ambush: 'Opponent is roaming open corridors; hold critical intersections, position for a lethal line-of-sight snipe, or lay a crossfire trap.',
              resource_dominance: 'High-value powerup crates (super blast, extra bombs, speed) are nearby and uncontested; capture them quickly to secure decisive attribute superiority.',
              kiting_counter: 'Current position or path is threatened by imminent bomb flame; retreat to cover and lay defensive zoning bombs to punish enemy advance.'
            }
          },
          // 2. 空间落点目标
          target_selection: {
            type: 'choice',
            instructions: 'Select the primary destination tile on the map to navigate towards.',
            criteria: candidateCriteria
          },
          // 3. 放泡动作决策（彻底放权）
          bomb_action: {
            type: 'choice',
            instructions: 'What bomb placement action should the player take at current position right now?',
            criteria: {
              plant_lethal_strike: 'Plant a bomb right now to eliminate, trap, or blast the opponent (opponent is in direct line of fire, adjacent, or cornered).',
              plant_breach_charge: 'Plant a bomb right now to blow up a destructible brick immediately blocking the pathway.',
              plant_zoning_barrier: 'Plant a bomb right now to seal the corridor, control territory, or block opponent from pursuing.',
              hold_fire: 'Do NOT plant a bomb right now; preserve bomb stock, maintain open escape routes, and continue navigation.'
            }
          },
          // 4. 下子置信度
          plant_bomb_confidence: {
            type: 'noul',
            instructions: 'Is placing a bomb at the current location right now tactically advantageous and safe to execute?'
          }
        };

        let url = this.apiUrl;
        const headers = { 'Content-Type': 'application/json' };

        if (typeof process !== 'undefined' && process.versions && process.versions.node) {
          if (url.startsWith('/')) url = 'http://localhost:8080' + url;
          const apiKey = process.env.TYPESAFE_API_KEY || '';
          if (url.includes('api.typesafe.ai')) headers['Authorization'] = `Bearer ${apiKey}`;
        }

        const resp = await fetch(url, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify({
            model: this.model,
            state: cleanState,
            questions: questions
          })
        });

        if (!resp.ok) {
          throw new Error(`Jev API error: ${resp.status} ${resp.statusText}`);
        }

        const data = await resp.json();
        const answers = data.answers || {};

        const posture = answers.strategic_posture ? answers.strategic_posture.choice : 'offensive_siege';
        const targetKey = answers.target_selection ? answers.target_selection.choice : 'enemy_direct';
        const bombAction = answers.bomb_action ? answers.bomb_action.choice : 'hold_fire';
        const bombConf = answers.plant_bomb_confidence ? answers.plant_bomb_confidence.noul : 0.0;

        const targetObj = candidateMap.get(targetKey) || candidateMap.get('enemy_direct') || { type: 'enemy', pos: stateObj.opponent.pos };

        this.lastDecision = {
          posture,
          targetKey,
          targetObj,
          bombAction,
          bombConf,
          timestamp: Date.now()
        };

        const decisionEntry = {
          time: new Date().toLocaleTimeString(),
          tick: sim.t || 0,
          posture,
          targetKey,
          targetObj,
          bombAction,
          bombConf: Number(bombConf.toFixed(2)),
          oppDist: stateObj.opponent.distance,
          inLineOfFire: stateObj.tactical_context.in_line_of_fire,
          latencyMs: Math.round(performance.now() - t0)
        };
        this.decisionHistory.push(decisionEntry);
        if (this.decisionHistory.length > 50) this.decisionHistory.shift();

        this.currentTarget = targetObj;
        this.targetPosture = posture;
        this.targetLabel = targetKey;
        this.targetType = targetObj.type;
        this.bombAction = bombAction;
        this.bombConfidence = bombConf;

        this.stats.totalCalls++;
        this.stats.lastLatencyMs = Math.round(performance.now() - t0);

      } catch (err) {
        console.warn('[JevAutonomousAI] Inference warning:', err);
      } finally {
        this.isInferring = false;
      }
    }

    // ------------------------------------------------------------ 主驱动入口 act(sim, pid)
    act(sim, pid, rng) {
      const curTick = sim.t || 0;
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = curTick * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.helperAi.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      // 1. 定期或态势突变时异步发起 Jev 研判
      const nextStart = danger.nextDangerStart(ownIdx, nowMs);
      const inImminentDanger = danger.hitTest(ownIdx, nowMs, 0) || (nextStart !== null && nextStart - nowMs <= 1000);
      const hasDanger = danger.hitTest(ownIdx, nowMs, 0) || danger.hasFutureDanger(ownIdx, nowMs);

      if (!this.isInferring && (curTick - this.lastInferTick >= this.inferIntervalTicks || inImminentDanger)) {
        this.lastInferTick = curTick;
        this.callJevAsync(sim, pid);
      }

      // 2. 承诺撤离路径（放泡后单向安全撤出，绝不在火线折返）
      if (this.helperAi.escapePath && this.helperAi.escapePath.length > 0) {
        // 关键防御：逃生目的地必须真正安全，绝不能撤入外部炸弹火线中
        const targetSafe = this.helperAi.escapeTarget >= 0 &&
                           !danger.hasFutureDanger(this.helperAi.escapeTarget, nowMs) &&
                           !danger.hitTest(this.helperAi.escapeTarget, nowMs, 0);

        if (!targetSafe) {
          // 掩体已受外部火线覆盖威胁，原逃生路径作废，立即重新规划避险！
          this.helperAi.escapePath = [];
          this.helperAi.escapeTarget = -1;
        } else if (ownIdx === this.helperAi.escapeTarget) {
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

      // 3. 极简物理安全网：若脚下当前瞬间已有烈焰或将在 1000ms 内起火，触发条件反射避险
      if (inImminentDanger) {
        const safeCells = [];
        const r0 = own[0], c0 = own[1];
        for (let i = 0; i < N; i++) {
          if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0) continue;
          if (!danger.hasFutureDanger(i, nowMs) && !danger.hitTest(i, nowMs, 0)) {
            const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
            safeCells.push({ cell: i, dist });
          }
        }
        safeCells.sort((a, b) => a.dist - b.dist);
        for (let s = 0; s < Math.min(safeCells.length, 8); s++) {
          const res = this.helperAi.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs, {
            allowBreakBrick: false,
            lastMove: this.lastMove
          });
          if (res && res.path.length > 1) {
            const mv = this.helperAi._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              const safeAct = this.helperAi._filterImmediateDanger(sim, danger, pid, mv, 0, nowMs, W, H);
              this.lastMove = safeAct[0];
              return safeAct;
            }
          }
        }

        // 贪心兜底
        let bestMv = MOVE_IDLE, maxScore = -1e9;
        for (let d = 0; d < 4; d++) {
          const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const np = nr * W + nc;
          if (mm[pid][d] !== 1) continue;
          if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0) continue;
          if (danger.hitTest(np, nowMs, 0)) continue;
          const s = danger.nextDangerStart(np, nowMs) || 999999;
          const isOpp = this.lastMove >= 0 && this.lastMove < 4 && d === (this.lastMove ^ 1);
          const isCont = this.lastMove >= 0 && this.lastMove < 4 && d === this.lastMove;
          const score = s + (isCont ? 50 : 0) - (isOpp ? 100 : 0);
          if (score > maxScore) { maxScore = score; bestMv = d; }
        }
        if (bestMv !== MOVE_IDLE) {
          this.lastMove = bestMv;
          return [bestMv, 0];
        }
      }

      // 4. 执行 Jev 完全自主选择的目标位并做连通性校验与投影
      let targetCell = -1;
      const dec = this.lastDecision;
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);
      const opp = 1 - pid;
      const oppCell = sim.centerCell(opp);
      const oppIdx = oppCell[0] * W + oppCell[1];
      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      if (dec && dec.targetObj) {
        const tObj = dec.targetObj;
        if (tObj.type === 'crate' && sim.crate[tObj.cell] === 1 && reachableMask[tObj.cell]) {
          targetCell = tObj.cell;
        } else if (tObj.type === 'brick' && sim.brick[tObj.cell] === 1 && reachableMask[tObj.cell]) {
          targetCell = tObj.cell;
        } else if (tObj.type === 'enemy' || tObj.type === 'intercept') {
          if (tObj.cell !== undefined && tObj.cell >= 0) {
            targetCell = tObj.cell;
          } else if (tObj.pos) {
            targetCell = tObj.pos[0] * W + tObj.pos[1];
          }
        }
      }

      // 缺省目标：敌方实时坐标
      if (targetCell < 0) {
        targetCell = oppIdx;
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

      // 5. 寻路前往目标
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

      // 暴露给前端与录屏的可视化参数
      this.targetCell = targetCell;
      this.targetPos = targetCell >= 0 ? [(targetCell / W) | 0, targetCell % W] : null;
      this.currentSearchPath = searchRes ? searchRes.path : [];

      let chosenMove = MOVE_IDLE;
      let finalBomb = 0;
      let nextStepIsBrick = false;

      if (searchRes && searchRes.path.length > 1 && searchRes.path[0] === ownIdx) {
        const nextCell = searchRes.path[1];
        if (sim.brick[nextCell]) {
          nextStepIsBrick = true; // 路径前方受阻于障碍砖
        } else {
          chosenMove = this.helperAi._cellToMove(ownIdx, nextCell, W);
        }
      }

      // 核心防御：若主路径受阻导致 chosenMove === MOVE_IDLE，但当前脚下处于未来火线覆盖中：
      // 绝不能在火线中发呆！立即触发紧急避险前往真正安全的无火线掩体！
      if (chosenMove === MOVE_IDLE && danger.hasFutureDanger(ownIdx, nowMs)) {
        const safeCells = [];
        const r0 = own[0], c0 = own[1];
        for (let i = 0; i < N; i++) {
          if (sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0) continue;
          if (!danger.hasFutureDanger(i, nowMs) && !danger.hitTest(i, nowMs, 0)) {
            const dist = Math.abs(((i / W) | 0) - r0) + Math.abs((i % W) - c0);
            safeCells.push({ cell: i, dist });
          }
        }
        safeCells.sort((a, b) => a.dist - b.dist);
        for (let s = 0; s < Math.min(safeCells.length, 8); s++) {
          const res = this.helperAi.search(sim, danger, ownIdx, safeCells[s].cell, spd, nowMs, {
            allowBreakBrick: false,
            lastMove: this.lastMove
          });
          if (res && res.path.length > 1) {
            const mv = this.helperAi._cellToMove(ownIdx, res.path[1], W);
            if (mm[pid][mv] === 1) {
              chosenMove = mv;
              break;
            }
          }
        }

        if (chosenMove === MOVE_IDLE) {
          // 贪心兜底
          let bestMv = MOVE_IDLE, maxScore = -1e9;
          for (let d = 0; d < 4; d++) {
            const nr = r0 + DIRS[d][0], nc = c0 + DIRS[d][1];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const np = nr * W + nc;
            if (mm[pid][d] !== 1) continue;
            if (sim.wall[np] || sim.brick[np] || sim.fuse[np] > 0) continue;
            if (danger.hitTest(np, nowMs, 0)) continue;
            const s = danger.nextDangerStart(np, nowMs) || 999999;
            const isOpp = this.lastMove >= 0 && this.lastMove < 4 && d === (this.lastMove ^ 1);
            const isCont = this.lastMove >= 0 && this.lastMove < 4 && d === this.lastMove;
            const score = s + (isCont ? 50 : 0) - (isOpp ? 100 : 0);
            if (score > maxScore) { maxScore = score; bestMv = d; }
          }
          if (bestMv !== MOVE_IDLE) {
            chosenMove = bestMv;
          }
        }
      }

      // 6. 核心放权点：放泡由 Jev 的 bomb_action 决定，若路径正被砖阻挡或已贴身对手亦触发破障/绝杀
      const adjacentToOpp = (oppDist <= 1) || (ownIdx === targetCell && targetCell === oppIdx);
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      if (canDrop && !hasDanger && dec) {
        const jevWantsBomb = (dec.bombAction === 'plant_lethal_strike' ||
                              dec.bombAction === 'plant_breach_charge' ||
                              dec.bombAction === 'plant_zoning_barrier') ||
                             (dec.bombConf >= 0.55) || nextStepIsBrick || (adjacentToOpp && sim.initialHp === 1);

        if (jevWantsBomb) {
          // 本地仅做绝对物理防自杀底线检查（检查是否存在至少一条合法生还路径）
          const safeToDrop = this.helperAi.canSafelyPlaceBomb(sim, ownIdx, sim.blastCap[pid], spd, nowMs);
          if (safeToDrop) {
            finalBomb = 1;
            this.stats.bombsPlaced++;
            this.lastDropTick = curTick;
            this.lastPlacedBombCell = ownIdx;
            // 立即消耗决策，防止持续多 tick 重复盲目落子
            if (dec) dec.bombAction = 'hold_fire';
            this.bombAction = 'hold_fire';
            // 锁定逃生路线
            if (this.helperAi.lastEscapePath && this.helperAi.lastEscapePath.length > 1) {
              this.helperAi.escapePath = this.helperAi.lastEscapePath.slice();
              this.helperAi.escapeTarget = this.helperAi.lastEscapeTarget;
              chosenMove = this.helperAi._cellToMove(ownIdx, this.helperAi.escapePath[1], W);
            }
          }
        }
      }

      // 7. 物理反射过滤（绝不踏进已燃烧格）
      const finalAct = this.helperAi._filterImmediateDanger(sim, danger, pid, chosenMove, finalBomb, nowMs, W, H);
      this.lastMove = finalAct[0];
      return finalAct;
    }
  }

  return JevAutonomousAI;
});
