/**
 * web/jev_grid_ai.js - TypeSafe Jev 高维坐标全图版 (Grid Coordinate Output Edition)
 *
 * 核心设计：
 * 1. 全图 15×13 二维空间矩阵输入 (Full 2D Map Grid Input)：
 *    - 像 ViTModel 一样将整张地图的所有信息注入模型输入，但采用适合大语言模型的 15×13 二维字符矩阵
 *    - 明确标识：# (不可炸墙), B (可炸砖), X (推箱), C (道具及类型), P (我方), E (敌方), ! (炸弹/烈焰), . (通路)
 *    - 附带各关键实体确切坐标表，辅助精确对齐
 *
 * 2. 时序历史动作上下文 (Temporal Action & Intent History)：
 *    - 上下文中注入过去若干步（约 200ms 间隔）的历史决策序列：
 *      { tick, elapsed_ms, chosen_target: [r, c], chosen_intent, moves_taken, bomb_placed }
 *    - 让 Jev 具备时间连续性，能够感知“上一次我想去吃 (1, 10) 处的道具，目前已行进了 2 步，是否继续坚持或调整目标”
 *
 * 3. 高维坐标直接输出 (Direct Coordinate Output)：
 *    - Jev 直接输出空间落点：
 *      - target_row: Choice [r0 .. r12] (13 个行选项)
 *      - target_col: Choice [c0 .. c14] (15 个列选项)
 *      - strategic_intent: Choice [hunt_opponent, gather_powerup, breach_obstacle, evade_danger]
 *      - bomb_decision: Choice [plant_bomb_now, hold_bomb]
 *
 * 4. 底层时空 A* 驱动与防自杀底线 (Low-level A* Motor & Physical Safety):
 *    - 底层 A* 接收 Jev 输出的 (target_row, target_col)，计算 100ms 微观路径
 *    - 若目标格为实心墙，自动向周围最近的合法连通格松弛
 *    - 底层保障不踩火、不自杀
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    const TimeAStarAI = require('./time_astar_ai.js');
    module.exports = factory(TimeAStarAI);
  } else {
    root.JevGridAI = factory(root.TimeAStarAI);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (TimeAStarAI) {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 0: 上, 1: 下, 2: 左, 3: 右
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;
  const MOVE_NAMES = ['up', 'down', 'left', 'right', 'idle'];

  class JevGridAI {
    constructor(options = {}) {
      this.options = options;
      this.apiUrl = options.apiUrl || '/api/typesafe';
      this.model = options.model || 'jev-latest';
      this.inferIntervalTicks = options.inferIntervalTicks || 10; // 约每 1.0s 重新推演，或态势剧变时触发

      this.helperAi = new TimeAStarAI({ mode: 'roam' });
      this.reset();
    }

    reset() {
      this.helperAi.reset();
      this.lastInferTick = -999;
      this.lastInferTimeMs = Date.now();
      this.isInferring = false;
      this.currentPath = [];
      this.lastMove = MOVE_IDLE;
      this.recentMoves = []; // 最近几个 tick 的实际移动动作

      // 决策历史缓冲区（带 200ms~1000ms 时间戳）
      this.temporalHistory = [];
      this.lastDecision = null;

      // 暴露给前端画布与横幅的参数
      this.targetRow = -1;
      this.targetCol = -1;
      this.targetCell = -1;
      this.targetPos = null;
      this.targetIntent = 'hunt_opponent';
      this.bombDecision = 'hold_bomb';
      this.currentSearchPath = [];

      this.stats = {
        totalCalls: 0,
        bombsPlaced: 0,
        lastLatencyMs: 0
      };
    }

    // ------------------------------------------------------------ 编码 15×13 二维空间矩阵与完整全局上下文
    extractGridState(sim, pid) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const opp = 1 - pid;
      const own = sim.centerCell(pid);
      const oppCell = sim.centerCell(opp);
      const ownIdx = own[0] * W + own[1];
      const oppIdx = oppCell[0] * W + oppCell[1];
      const nowMs = (sim.t || 0) * 100;
      const danger = this.helperAi.buildDangerMap(sim, nowMs);

      // 1. 构建 15×13 二维字符矩阵
      const gridRows = [];
      const cratesDetail = [];
      const bricksList = [];
      const dangerTiles = [];

      for (let r = 0; r < H; r++) {
        let rowStr = '';
        for (let c = 0; c < W; c++) {
          const idx = r * W + c;
          if (r === own[0] && c === own[1]) {
            rowStr += 'P'; // 我方玩家
          } else if (r === oppCell[0] && c === oppCell[1]) {
            rowStr += 'E'; // 敌方对手
          } else if (sim.fuse[idx] > 0 || danger.hitTest(idx, nowMs, 0)) {
            rowStr += '!'; // 炸弹或烈焰
            dangerTiles.push([r, c]);
          } else if (sim.wall[idx] === 1) {
            rowStr += '#'; // 不可炸实心墙
          } else if (sim.brick[idx] === 1) {
            rowStr += 'B'; // 可炸砖块
            bricksList.push([r, c]);
          } else if (sim.pushable && sim.pushable[idx] === 1) {
            rowStr += 'X'; // 可推箱
          } else if (sim.crate[idx] === 1) {
            rowStr += 'C'; // 道具宝箱
            const ct = sim.crateType[idx];
            const isSuper = sim.superCrate && sim.superCrate[idx] === 1;
            let typeStr = 'bomb';
            if (ct === 1) typeStr = 'blast';
            else if (ct === 2) typeStr = 'speed';
            if (isSuper) typeStr = 'super_' + typeStr;
            cratesDetail.push({ pos: [r, c], type: typeStr, dist: Math.abs(r - own[0]) + Math.abs(c - own[1]) });
          } else {
            rowStr += '.'; // 畅通道路
          }
        }
        gridRows.push(rowStr);
      }

      cratesDetail.sort((a, b) => a.dist - b.dist);

      // 2. 四邻状态与路径障碍
      const surroundings = {};
      const dirNames = ['up', 'down', 'left', 'right'];
      let adjacentBrickDir = null;
      for (let d = 0; d < 4; d++) {
        const tr = own[0] + DIRS[d][0], tc = own[1] + DIRS[d][1];
        const name = dirNames[d];
        if (tr < 0 || tr >= H || tc < 0 || tc >= W) {
          surroundings[name] = 'boundary_wall';
        } else {
          const ti = tr * W + tc;
          if (sim.wall[ti]) surroundings[name] = 'indestructible_wall';
          else if (sim.brick[ti]) {
            surroundings[name] = 'destructible_brick';
            if (!adjacentBrickDir) adjacentBrickDir = name;
          }
          else if (sim.fuse[ti] > 0) surroundings[name] = 'ticking_bomb';
          else if (danger.hitTest(ti, nowMs, 300)) surroundings[name] = 'danger_flame';
          else surroundings[name] = 'open_path';
        }
      }

      // 检查当前路径下一步是否为砖块
      let nextStepIsBrick = false;
      if (this.currentSearchPath && this.currentSearchPath.length > 1) {
        const nextCell = this.currentSearchPath[1];
        if (sim.brick[nextCell]) nextStepIsBrick = true;
      }

      // 3. 构造时序历史动作上下文（包含时间间隔）
      const historyContext = this.temporalHistory.slice(-5).map(h => ({
        tick: h.tick,
        dt_ms: h.dtMs,
        target_coordinate: h.target,
        intent: h.intent,
        moves_executed: h.moves,
        bomb_placed: h.bombPlaced
      }));

      return {
        step: sim.t || 0,
        map_dimensions: { width: 15, height: 13 },
        map_legend: {
          "P": `player_position (${own[0]}, ${own[1]})`,
          "E": `enemy_position (${oppCell[0]}, ${oppCell[1]})`,
          "C": "powerup_crate (collectible)",
          "B": "destructible_brick",
          "X": "pushable_box",
          "#": "indestructible_wall",
          "!": "bomb_or_lethal_flame",
          ".": "open_walkable_path"
        },
        map_grid_15x13: gridRows,
        immediate_surroundings: surroundings,
        path_status: {
          next_step_blocked_by_brick: nextStepIsBrick,
          adjacent_brick_direction: adjacentBrickDir
        },
        key_coordinates: {
          player: [own[0], own[1]],
          enemy: [oppCell[0], oppCell[1]],
          crates: cratesDetail.slice(0, 5),
          nearby_bricks: bricksList.filter(b => Math.abs(b[0] - own[0]) + Math.abs(b[1] - own[1]) <= 6).slice(0, 5),
          danger_zones: dangerTiles
        },
        player_stats: {
          hp: sim.hp ? sim.hp[pid] : 5,
          bombs_cap: sim.bombsCap ? sim.bombsCap[pid] : 2,
          blast_cap: sim.blastCap ? sim.blastCap[pid] : 2,
          speed: sim.spdG ? Number(sim.spdG[pid].toFixed(2)) : 1.3,
          can_place_bomb: sim.liveBombs(pid) < sim.bombsCap[pid] && sim.fuse[ownIdx] <= 0
        },
        enemy_stats: {
          hp: sim.hp ? sim.hp[opp] : 5,
          distance: Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]),
          alive: sim.alive ? Boolean(sim.alive[opp]) : true
        },
        recent_temporal_history: historyContext
      };
    }

    // ------------------------------------------------------------ 异步向 Jev 发起高维坐标推演
    async callJevAsync(sim, pid) {
      if (this.isInferring) return;
      this.isInferring = true;
      const t0 = performance.now();
      const curTick = sim.t || 0;
      const nowWallMs = Date.now();
      const dtMs = nowWallMs - this.lastInferTimeMs;
      this.lastInferTimeMs = nowWallMs;

      try {
        const state = this.extractGridState(sim, pid);

        // 构造 13 个行选项 (r0 .. r12)
        const rowCriteria = {};
        for (let r = 0; r < 13; r++) {
          rowCriteria[`r${r}`] = `Row ${r} (${r === 0 ? 'top boundary' : (r === 12 ? 'bottom boundary' : `interior row ${r}`)})`;
        }

        // 构造 15 个列选项 (c0 .. c14)
        const colCriteria = {};
        for (let c = 0; c < 15; c++) {
          colCriteria[`c${c}`] = `Col ${c} (${c === 0 ? 'left boundary' : (c === 14 ? 'right boundary' : `interior col ${c}`)})`;
        }

        const questions = {
          strategic_intent: {
            type: 'choice',
            instructions: 'Based on map_grid_15x13, player/enemy stats, and recent_temporal_history, what is the primary strategic objective?',
            criteria: {
              hunt_opponent: 'Aggressively navigate towards opponent E to corner, trap, or blast them.',
              gather_powerup: 'Navigate towards a high-value crate C on the map to collect it for attribute upgrades.',
              breach_obstacle: 'Navigate towards a blocking brick B that cuts off corridors or blocks access to enemy/crates.',
              evade_danger: 'Navigate away from bombs/flames ! to a secure shelter tile.'
            }
          },
          target_row: {
            type: 'choice',
            instructions: 'Select the exact target destination row index (0 to 12) for the player to navigate towards on the 15x13 map grid.',
            criteria: rowCriteria
          },
          target_col: {
            type: 'choice',
            instructions: 'Select the exact target destination column index (0 to 14) for the player to navigate towards on the 15x13 map grid.',
            criteria: colCriteria
          },
          bomb_decision: {
            type: 'choice',
            instructions: 'Should player place a bomb at current location right now?',
            criteria: {
              plant_bomb_now: 'Place bomb right now (path_status.next_step_blocked_by_brick is true, a brick is directly adjacent to player, or opponent is within blast line).',
              hold_bomb: 'Do not place bomb; path is open, no blocking obstacle adjacent, and player is actively cruising towards target.'
            }
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
            state: state,
            questions: questions
          })
        });

        if (!resp.ok) {
          throw new Error(`Jev API error: ${resp.status} ${resp.statusText}`);
        }

        const data = await resp.json();
        const answers = data.answers || {};

        const intent = answers.strategic_intent ? answers.strategic_intent.choice : 'hunt_opponent';
        const rowStr = answers.target_row ? answers.target_row.choice : 'r1';
        const colStr = answers.target_col ? answers.target_col.choice : 'c1';
        const bombChoice = answers.bomb_decision ? answers.bomb_decision.choice : 'hold_bomb';

        const row = parseInt(rowStr.replace('r', ''), 10);
        const col = parseInt(colStr.replace('c', ''), 10);

        this.lastDecision = {
          intent,
          row: isNaN(row) ? 1 : Math.max(0, Math.min(12, row)),
          col: isNaN(col) ? 1 : Math.max(0, Math.min(14, col)),
          bombChoice,
          timestamp: Date.now()
        };

        // 记入时序历史
        this.temporalHistory.push({
          tick: curTick,
          dtMs: dtMs,
          target: [this.lastDecision.row, this.lastDecision.col],
          intent: intent,
          moves: this.recentMoves.slice(-4),
          bombPlaced: bombChoice === 'plant_bomb_now'
        });
        if (this.temporalHistory.length > 20) this.temporalHistory.shift();
        this.recentMoves = [];

        this.targetRow = this.lastDecision.row;
        this.targetCol = this.lastDecision.col;
        this.targetIntent = intent;
        this.bombDecision = bombChoice;

        this.stats.totalCalls++;
        this.stats.lastLatencyMs = Math.round(performance.now() - t0);

      } catch (err) {
        console.warn('[JevGridAI] Inference error:', err);
      } finally {
        this.isInferring = false;
      }
    }

    // ------------------------------------------------------------ 驱动执行 act(sim, pid)
    act(sim, pid, rng) {
      const curTick = sim.t || 0;
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const nowMs = curTick * 100;
      const spd = 3.0 * (sim.spdG ? sim.spdG[pid] : 1.0);
      const danger = this.helperAi.buildDangerMap(sim, nowMs);
      const { mm, bm } = sim.legalMask();

      // 1. 定期或态势突变时异步发起研判
      const nextStart = danger.nextDangerStart(ownIdx, nowMs);
      const inImminentDanger = danger.hitTest(ownIdx, nowMs, 0) || (nextStart !== null && nextStart - nowMs <= 800);
      const blockedByBrick = (this.currentSearchPath && this.currentSearchPath.length > 1 && sim.brick[this.currentSearchPath[1]]);

      if (!this.isInferring && (curTick - this.lastInferTick >= this.inferIntervalTicks || inImminentDanger || (blockedByBrick && curTick - this.lastInferTick >= 3))) {
        this.lastInferTick = curTick;
        this.callJevAsync(sim, pid);
      }

      // 2. 承诺逃生路径（放泡后单向撤出）
      if (this.helperAi.escapePath && this.helperAi.escapePath.length > 0) {
        if (ownIdx === this.helperAi.escapeTarget || !danger.hasFutureDanger(ownIdx, nowMs)) {
          this.helperAi.escapePath = [];
          this.helperAi.escapeTarget = -1;
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
                this.recentMoves.push(MOVE_NAMES[finalAct[0]]);
                return finalAct;
              }
            } else {
              this.helperAi.escapePath = [];
              this.helperAi.escapeTarget = -1;
            }
          }
        }
      }

      // 3. 极简物理反射：脚下有火立即撤离
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
              this.recentMoves.push(MOVE_NAMES[safeAct[0]]);
              return safeAct;
            }
          }
        }
      }

      // 4. 解析 Jev 直出的目标坐标 (Row, Col)
      let targetRow = this.targetRow >= 0 ? this.targetRow : own[0];
      let targetCol = this.targetCol >= 0 ? this.targetCol : own[1];
      let targetCell = targetRow * W + targetCol;

      // 坐标松弛：如果 Jev 选中的是一个不可炸的实心外墙或柱子，自动松弛到周围最近的连通地格
      if (sim.wall[targetCell] === 1) {
        let bestDist = 999;
        let bestC = targetCell;
        for (let d = 0; d < 4; d++) {
          const nr = targetRow + DIRS[d][0], nc = targetCol + DIRS[d][1];
          if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
            const ni = nr * W + nc;
            if (sim.wall[ni] === 0) {
              const dToOwn = Math.abs(nr - own[0]) + Math.abs(nc - own[1]);
              if (dToOwn < bestDist) {
                bestDist = dToOwn;
                bestC = ni;
              }
            }
          }
        }
        targetCell = bestC;
        targetRow = (targetCell / W) | 0;
        targetCol = targetCell % W;
      }

      // 5. A* 寻路前往目标
      const searchRes = this.helperAi.search(sim, danger, ownIdx, targetCell, spd, nowMs, {
        allowBreakBrick: true,
        lastMove: this.lastMove
      });

      this.targetPos = [targetRow, targetCol];
      this.targetCell = targetCell;
      this.currentSearchPath = searchRes ? searchRes.path : [];

      let chosenMove = MOVE_IDLE;
      let finalBomb = 0;

      if (searchRes && searchRes.path.length > 1) {
        const nextCell = searchRes.path[1];
        if (!sim.brick[nextCell]) {
          chosenMove = this.helperAi._cellToMove(ownIdx, nextCell, W);
        }
      }

      // 6. 放泡执行：完全由 Jev 的 bomb_decision 决定
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      if (canDrop && !inImminentDanger && this.lastDecision) {
        if (this.lastDecision.bombChoice === 'plant_bomb_now') {
          // 物理防自杀底线检查
          const safeToDrop = this.helperAi.canSafelyPlaceBomb(sim, ownIdx, sim.blastCap[pid], spd, nowMs);
          if (safeToDrop) {
            finalBomb = 1;
            this.stats.bombsPlaced++;
            if (this.helperAi.lastEscapePath && this.helperAi.lastEscapePath.length > 1) {
              this.helperAi.escapePath = this.helperAi.lastEscapePath.slice();
              this.helperAi.escapeTarget = this.helperAi.lastEscapeTarget;
            }
          }
        }
      }

      // 7. 物理反射过滤
      const finalAct = this.helperAi._filterImmediateDanger(sim, danger, pid, chosenMove, finalBomb, nowMs, W, H);
      this.lastMove = finalAct[0];
      this.recentMoves.push(MOVE_NAMES[finalAct[0]]);
      return finalAct;
    }
  }

  return JevGridAI;
});
