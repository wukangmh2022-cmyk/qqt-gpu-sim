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
      this.lastPlacedBombCell = -1;

      this.stats = {
        totalCalls: 0,
        bombsPlaced: 0,
        lastLatencyMs: 0
      };
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

      // 计算角色在物理引擎中的精确连续浮点坐标与格子内 [0, 1) 的 diff 偏移量
      const ownRawR = sim.pos ? sim.pos[pid * 2] : own[0] + 0.5;
      const ownRawC = sim.pos ? sim.pos[pid * 2 + 1] : own[1] + 0.5;
      const ownDiffR = Number((ownRawR - own[0]).toFixed(3));
      const ownDiffC = Number((ownRawC - own[1]).toFixed(3));

      const oppRawR = sim.pos ? sim.pos[opp * 2] : oppCell[0] + 0.5;
      const oppRawC = sim.pos ? sim.pos[opp * 2 + 1] : oppCell[1] + 0.5;
      const oppDiffR = Number((oppRawR - oppCell[0]).toFixed(3));
      const oppDiffC = Number((oppRawC - oppCell[1]).toFixed(3));

      // 计算玩家当前所在连通分量 (Reachable Connected Component)
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);
      const walkableMask = this.computeWalkableComponent(sim, ownIdx);
      const reachableRows = new Set();
      const reachableCols = new Set();
      for (let i = 0; i < N; i++) {
        if (reachableMask[i] && sim.wall[i] === 0) {
          reachableRows.add((i / W) | 0);
          reachableCols.add(i % W);
        }
      }

      // 提取在场所有炸弹的结构化信息（引线倒计时与威力半径）
      const activeBombs = [];
      for (let i = 0; i < N; i++) {
        if (sim.fuse[i] > 0) {
          const br = (i / W) | 0, bc = i % W;
          const fuseTicks = sim.fuse[i];
          const fuseMs = Math.max(0, fuseTicks - 1) * 100;
          const digit = Math.min(9, Math.max(1, Math.ceil(fuseMs / 300)));
          activeBombs.push({
            pos: [br, bc],
            countdown: digit,
            fuse_ticks: fuseTicks,
            fuse_ms: fuseMs,
            blast_radius: sim.bombBlast ? (sim.bombBlast[i] || 2) : 2
          });
        }
      }

      // 1. 构建 15×13 二维字符矩阵（0~9 量化危险倒计时热力图）
      // 0: 当前正在燃烧 (Lethal)
      // 1~3: 100~900ms 临界引爆（当前 1000ms 决策周期内即将爆炸）
      // 4~6: 1000~1800ms 中期倒计时
      // 7~9: 1900~3000ms 安全充裕倒计时（可快速借道通过）
      // .: 绝对安全道路（无任何爆炸预定）
      const gridRows = [];
      const cratesDetail = [];
      const bricksList = [];
      const dangerTiles = [];
      const minRowCountdown = new Array(H).fill(null);
      const minColCountdown = new Array(W).fill(null);
      const rowDangerCounts = new Array(H).fill(0);
      const colDangerCounts = new Array(W).fill(0);

      for (let r = 0; r < H; r++) {
        let rowStr = '';
        for (let c = 0; c < W; c++) {
          const idx = r * W + c;
          const isReachable = reachableMask[idx] === 1;

          // 计算当前格的危险倒计时
          let cellCountdown = null;
          let cellThreatMs = null;
          const isBurningNow = (sim.blastLinger && sim.blastLinger[idx] > 0) || danger.hitTest(idx, nowMs, 0);

          if (isBurningNow) {
            cellCountdown = 0;
            cellThreatMs = 0;
          } else {
            const nextBlast = danger.nextDangerStart(idx, nowMs);
            if (sim.fuse[idx] > 0) {
              const bMs = Math.max(0, sim.fuse[idx] - 1) * 100;
              cellThreatMs = (nextBlast !== null && nextBlast - nowMs < bMs) ? (nextBlast - nowMs) : bMs;
              cellCountdown = Math.min(9, Math.max(1, Math.ceil(cellThreatMs / 300)));
            } else if (nextBlast !== null && nextBlast - nowMs <= 3000) {
              cellThreatMs = nextBlast - nowMs;
              cellCountdown = Math.min(9, Math.max(1, Math.ceil(cellThreatMs / 300)));
            }
          }

          if (isReachable && cellCountdown !== null) {
            if (minRowCountdown[r] === null || cellCountdown < minRowCountdown[r]) minRowCountdown[r] = cellCountdown;
            if (minColCountdown[c] === null || cellCountdown < minColCountdown[c]) minColCountdown[c] = cellCountdown;
            if (cellCountdown <= 3) {
              rowDangerCounts[r]++;
              colDangerCounts[c]++;
            }
          }

          if (r === own[0] && c === own[1]) {
            rowStr += 'P'; // 我方玩家
          } else if (r === oppCell[0] && c === oppCell[1]) {
            rowStr += isReachable ? 'E' : 'e'; // E=连通可达对手, e=非连通隔断对手
          } else if (sim.wall[idx] === 1) {
            rowStr += '#'; // 不可炸实心墙
          } else if (sim.brick[idx] === 1) {
            rowStr += isReachable ? 'B' : '?'; // 仅连通砖标为B，非连通孤立格标为?
            if (isReachable) bricksList.push([r, c]);
          } else if (sim.pushable && sim.pushable[idx] === 1) {
            rowStr += isReachable ? 'X' : '?';
          } else if (sim.crate[idx] === 1) {
            const ct = sim.crateType ? sim.crateType[idx] : -1;
            const isSuper = sim.superCrate && sim.superCrate[idx] === 1;
            let crateChar = 'c';
            let typeStr = 'mystery';
            if (ct === 0) { crateChar = 'b'; typeStr = 'bomb_cap'; }
            else if (ct === 1) { crateChar = 'f'; typeStr = 'blast_power'; }
            else if (ct === 2) { crateChar = 's'; typeStr = 'speed_potion'; }
            if (isSuper) typeStr = 'super_' + typeStr;

            rowStr += isReachable ? crateChar : '?';
            if (isReachable) {
              cratesDetail.push({
                pos: [r, c],
                symbol: crateChar,
                type: typeStr,
                directly_walkable: walkableMask[idx] === 1,
                dist: Math.abs(r - own[0]) + Math.abs(c - own[1]),
                threat_countdown: cellCountdown,
                threat_ms: cellThreatMs
              });
            }
          } else if (cellCountdown !== null) {
            rowStr += String(cellCountdown); // 0~9 量化倒计时热力图
            if (isReachable) {
              dangerTiles.push({
                pos: [r, c],
                countdown: cellCountdown,
                threat_ms: cellThreatMs,
                is_bomb: (sim.fuse[idx] > 0)
              });
            }
          } else {
            rowStr += isReachable ? '.' : '?'; // 畅通道路 vs 孤岛盲区
          }
        }
        gridRows.push(rowStr);
      }

      cratesDetail.sort((a, b) => a.dist - b.dist);

      // 2. 四邻状态与路径障碍（精确到起火毫秒与倒计时）
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
          else if (sim.blastLinger && sim.blastLinger[ti] > 0) surroundings[name] = 'active_flame_countdown_0_(lingering_for_0.3s)';
          else if (sim.fuse[ti] > 0) {
            const bDt = Math.max(0, sim.fuse[ti] - 1) * 100;
            const bDigit = Math.min(9, Math.max(1, Math.ceil(bDt / 300)));
            surroundings[name] = `bomb_center_countdown_${bDigit}_(${bDt}ms)`;
          }
          else {
            const nextFire = danger.nextDangerStart(ti, nowMs);
            if (nextFire !== null && nextFire - nowMs <= 3000) {
              const dt = nextFire - nowMs;
              const digit = Math.min(9, Math.max(1, Math.ceil(dt / 300)));
              surroundings[name] = `blast_corridor_countdown_${digit}_(${dt}ms)`;
            } else {
              surroundings[name] = 'safe_open_path';
            }
          }
        }
      }

      // 检查当前路径下一步是否为砖块
      let nextStepIsBrick = false;
      if (this.currentSearchPath && this.currentSearchPath.length > 1) {
        const nextCell = this.currentSearchPath[1];
        if (sim.brick[nextCell]) nextStepIsBrick = true;
      }

      // 检查对手是否在直接水柱火线上（无障碍隔挡）
      let opponentInBlastLine = false;
      if (oppIdx !== -1 && sim.alive && sim.alive[opp]) {
        const dr = Math.abs(own[0] - oppCell[0]), dc = Math.abs(own[1] - oppCell[1]);
        const zCap = sim.blastCap ? sim.blastCap[pid] : 2;
        if (dr === 0 && dc <= zCap && dc > 0) {
          let blocked = false;
          const minC = Math.min(own[1], oppCell[1]), maxC = Math.max(own[1], oppCell[1]);
          for (let c = minC + 1; c < maxC; c++) {
            if (sim.wall[own[0] * W + c] || sim.brick[own[0] * W + c]) { blocked = true; break; }
          }
          if (!blocked) opponentInBlastLine = true;
        } else if (dc === 0 && dr <= zCap && dr > 0) {
          let blocked = false;
          const minR = Math.min(own[0], oppCell[0]), maxR = Math.max(own[0], oppCell[0]);
          for (let r = minR + 1; r < maxR; r++) {
            if (sim.wall[r * W + own[1]] || sim.brick[r * W + own[1]]) { blocked = true; break; }
          }
          if (!blocked) opponentInBlastLine = true;
        }
      }

      const ownCap = sim.bombsCap ? sim.bombsCap[pid] : 2;
      const ownLive = sim.liveBombs ? sim.liveBombs(pid) : 0;
      const ownAvail = Math.max(0, ownCap - ownLive);

      const oppCap = sim.bombsCap ? sim.bombsCap[opp] : 2;
      const oppLive = sim.liveBombs ? sim.liveBombs(opp) : 0;
      const oppAvail = Math.max(0, oppCap - oppLive);

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
        combat_tactics: {
          opponent_in_direct_blast_line: opponentInBlastLine,
          opponent_adjacent: Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]) <= 1,
          available_bombs: ownAvail,
          can_place_bomb_now: ownAvail > 0 && sim.fuse[ownIdx] <= 0
        },
        map_legend: {
          "P": `player_position (${own[0]}, ${own[1]})`,
          "E": `enemy_position (${oppCell[0]}, ${oppCell[1]})`,
          "s": "powerup_crate_speed (+0.2 speed potion, HIGHEST PRIORITY when slower than enemy to avoid being outrun/cornered)",
          "f": "powerup_crate_flame (+1 fire blast range)",
          "b": "powerup_crate_bomb (+1 max bomb capacity)",
          "c": "powerup_crate_mystery (random upgrade: speed, flame, or bomb)",
          "B": "destructible_brick",
          "X": "pushable_box",
          "#": "indestructible_wall",
          "0": "active_lethal_flame (active explosion & residual flame lingering for 0.3s~0.5s. NEVER touch, deals 1 HP damage!)",
          "1-9": "danger_countdown (1=100-300ms imminent blast, 2=400-600ms, 3=700-900ms <=1s window, 4-6=1-1.8s medium, 7-9=1.9-3.0s safe delay). Represents bomb center or blast line. See key_coordinates.active_bombs for bomb centers.",
          ".": "open_safe_path (no explosion scheduled, completely safe)",
          "?": "isolated_unreachable_tile"
        },
        map_grid_15x13: gridRows,
        immediate_surroundings: surroundings,
        path_status: {
          next_step_blocked_by_brick: nextStepIsBrick,
          adjacent_brick_direction: adjacentBrickDir
        },
        reachable_rows: Array.from(reachableRows).sort((a, b) => a - b),
        reachable_cols: Array.from(reachableCols).sort((a, b) => a - b),
        row_threats: minRowCountdown,
        col_threats: minColCountdown,
        row_danger_counts: rowDangerCounts,
        col_danger_counts: colDangerCounts,
        match_rules: {
          core_gameplay_loop: "CORE GAMEPLAY LOOP: 1) Plant bombs near destructible bricks (B) to blast them open and reveal upgrade crates (s, f, b, c). 2) Collect crates to increase attributes (speed, blast_cap, bombs_cap). 3) Corner enemy (E) using bombs as solid blocks or corridor traps. 4) Hit enemy with bomb flames or chain reaction explosions to reduce their HP to 0 for VICTORY.",
          powerup_upgrade_rules: "POWERUP UPGRADE RULES (DEVELOPMENT ADVANTAGE): 1) 's' Speed potion: boosts movement speed (+0.2). If enemy speed is higher than yours, collecting 's' is TOP TACTICAL PRIORITY to prevent being chased down or cornered! 2) 'f' Flame blast: increases bomb cross fire reach by 1 tile. 3) 'b' Bomb capacity: allows holding +1 more bomb on the field. 4) 'c' Mystery crate: awards a random upgrade.",
          chain_reaction_rule: "CHAIN REACTION MECHANIC (CRITICAL TACTIC): When any bomb explodes, its cross-line flame INSTANTLY detonates ALL other bombs within its blast reach immediately without waiting for their timer! TACTICAL OFFENSE: Placing bombs in a line or grid triggers a simultaneous multi-bomb chain blast covering long corridors and giving enemies 0 reaction time. DEFENSIVE WARNING: If you plant a bomb in the blast line of an older ticking bomb (countdown 1-3), your new bomb will explode early together with it! Never place a bomb near a ticking bomb unless you have an immediate escape route!",
          damage_rule: "Touching any flame (0) or exploding bomb blast (1-2) deducts 1 HP.",
          flame_linger_rule: "LATENT RUNTIME RULE: When countdown reaches 0, the explosion flame persists and LINGERS for 0.3s~0.5s (250~300ms / 2~3 ticks). A cell marked '0' is in active combustion; touching it during this 0.3s window still causes 1 HP damage! Never step onto '0' until it turns back to safe path '.'.",
          bomb_inventory_rule: `BOMB INVENTORY RULE: Player bomb capacity is ${ownCap}. Currently active on map: ${ownLive}, available in hand: ${ownAvail}. Placing a bomb deploys 1 bomb at current tile and consumes 1 slot; once it detonates and flame clears, the slot returns to inventory.`,
          victory_condition: "Reducing enemy HP to 0 achieves immediate VICTORY.",
          defeat_condition: "When player HP reaches 0, player is ELIMINATED (instant DEFEAT / GAME OVER).",
          current_player_hp: sim.hp ? sim.hp[pid] : 5,
          current_enemy_hp: sim.hp ? sim.hp[opp] : 5,
          survival_alert: (sim.hp && sim.hp[pid] <= 1)
            ? "🚨 CRITICAL SURVIVAL ALERT: Your HP is 1! Any damage immediately terminates the match with DEFEAT. You MUST avoid all flames (0) and blast lines (1-3)!"
            : `Healthy: Current HP is ${sim.hp ? sim.hp[pid] : 5}.`,
          kill_opportunity: (sim.hp && sim.hp[opp] <= 1)
            ? "🎯 LETHAL OPPORTUNITY: Enemy HP is 1! Hitting enemy with a single bomb blast secures immediate VICTORY!"
            : `Enemy HP is ${sim.hp ? sim.hp[opp] : 5}.`
        },
        continuous_positions: {
          player_desc: `Player P exact continuous position is (${ownRawR.toFixed(2)}, ${ownRawC.toFixed(2)}), in grid [${own[0]}, ${own[1]}] with in-tile float offset (+${ownDiffR.toFixed(2)} row, +${ownDiffC.toFixed(2)} col).`,
          enemy_desc: `Enemy E exact continuous position is (${oppRawR.toFixed(2)}, ${oppRawC.toFixed(2)}), in grid [${oppCell[0]}, ${oppCell[1]}] with in-tile float offset (+${oppDiffR.toFixed(2)} row, +${oppDiffC.toFixed(2)} col).`,
          player_continuous: [Number(ownRawR.toFixed(2)), Number(ownRawC.toFixed(2))],
          enemy_continuous: [Number(oppRawR.toFixed(2)), Number(oppRawC.toFixed(2))],
          player_in_tile_diff: [ownDiffR, ownDiffC],
          enemy_in_tile_diff: [oppDiffR, oppDiffC],
          continuous_euclidean_distance: Number(Math.hypot(ownRawR - oppRawR, ownRawC - oppRawC).toFixed(2))
        },
        key_coordinates: {
          player: [own[0], own[1]],
          player_exact: [Number(ownRawR.toFixed(2)), Number(ownRawC.toFixed(2))],
          enemy: [oppCell[0], oppCell[1]],
          enemy_exact: [Number(oppRawR.toFixed(2)), Number(oppRawC.toFixed(2))],
          crates: cratesDetail.slice(0, 5),
          nearby_bricks: bricksList.filter(b => Math.abs(b[0] - own[0]) + Math.abs(b[1] - own[1]) <= 6).slice(0, 5),
          active_bombs: activeBombs,
          danger_zones: dangerTiles.slice(0, 15)
        },
        player_stats: {
          hp: sim.hp ? sim.hp[pid] : 5,
          bombs_cap: ownCap,
          active_bombs: ownLive,
          available_bombs: ownAvail,
          blast_cap: sim.blastCap ? sim.blastCap[pid] : 2,
          speed: sim.spdG ? Number(sim.spdG[pid].toFixed(2)) : 1.3,
          can_place_bomb: ownAvail > 0 && sim.fuse[ownIdx] <= 0,
          continuous_pos: [Number(ownRawR.toFixed(2)), Number(ownRawC.toFixed(2))],
          in_tile_diff: [ownDiffR, ownDiffC]
        },
        enemy_stats: {
          hp: sim.hp ? sim.hp[opp] : 5,
          bombs_cap: oppCap,
          active_bombs: oppLive,
          available_bombs: oppAvail,
          blast_cap: sim.blastCap ? sim.blastCap[opp] : 2,
          distance: Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]),
          continuous_distance: Number(Math.hypot(ownRawR - oppRawR, ownRawC - oppRawC).toFixed(2)),
          alive: sim.alive ? Boolean(sim.alive[opp]) : true,
          continuous_pos: [Number(oppRawR.toFixed(2)), Number(oppRawC.toFixed(2))],
          in_tile_diff: [oppDiffR, oppDiffC]
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
        const ownR = state.key_coordinates.player[0];
        const ownC = state.key_coordinates.player[1];
        const enemyR = state.key_coordinates.enemy[0];
        const enemyC = state.key_coordinates.enemy[1];

        // 构造 13 个行选项 (r0 .. r12)，明确标注连通性、危险倒计时与目标引导
        const rowCriteria = {};
        for (let r = 0; r < 13; r++) {
          const isReachable = state.reachable_rows.includes(r);
          if (!isReachable) {
            rowCriteria[`r${r}`] = `Row ${r} [UNREACHABLE / CUT OFF BY WALLS - DO NOT SELECT]`;
          } else {
            const tags = [];
            if (r === enemyR) tags.push('ENEMY E DESTINATION (primary target for hunt_opponent)');
            const cratesInRow = state.key_coordinates.crates.filter(c => c.pos[0] === r).map(c => `${c.type} at col ${c.pos[1]}`);
            if (cratesInRow.length) tags.push('crates: ' + cratesInRow.join(', '));
            const bricksInRow = state.key_coordinates.nearby_bricks.filter(b => b[0] === r).map(b => `brick at col ${b[1]}`);
            if (bricksInRow.length) tags.push('bricks: ' + bricksInRow.slice(0, 2).join(', '));
            if (r === ownR) tags.push('CURRENT PLAYER ROW (select ONLY if intentionally holding position)');

            const rThreat = state.row_threats ? state.row_threats[r] : null;
            const rDangerCount = state.row_danger_counts ? state.row_danger_counts[r] : 0;
            if (rThreat !== null && rThreat !== undefined) {
              if (rThreat === 0) tags.push('⚠️ FLAME BURNING (countdown 0)');
              else if (rThreat <= 3) {
                if (rDangerCount >= 3) tags.push(`🚨 CRITICAL DANGER: Heavy blast line along row (countdown ${rThreat})`);
                else tags.push(`⚠️ Crossed by blast at ${rDangerCount} cell(s) (countdown ${rThreat})`);
              }
              else if (rThreat <= 6) tags.push(`medium danger (countdown ${rThreat})`);
              else tags.push(`safe delay (countdown ${rThreat})`);
            } else {
              tags.push('SAFE corridor');
            }

            rowCriteria[`r${r}`] = `Row ${r} [REACHABLE: ${tags.join(' | ')}]`;
          }
        }

        // 构造 15 个列选项 (c0 .. c14)，明确标注连通性、危险倒计时与目标引导
        const colCriteria = {};
        for (let c = 0; c < 15; c++) {
          const isReachable = state.reachable_cols.includes(c);
          if (!isReachable) {
            colCriteria[`c${c}`] = `Col ${c} [UNREACHABLE / CUT OFF BY WALLS - DO NOT SELECT]`;
          } else {
            const tags = [];
            if (c === enemyC) tags.push('ENEMY E DESTINATION (primary target for hunt_opponent)');
            const cratesInCol = state.key_coordinates.crates.filter(cObj => cObj.pos[1] === c).map(cObj => `${cObj.type} at row ${cObj.pos[0]}`);
            if (cratesInCol.length) tags.push('crates: ' + cratesInCol.join(', '));
            if (c === ownC) tags.push('CURRENT PLAYER COL (select ONLY if intentionally holding position)');

            const cThreat = state.col_threats ? state.col_threats[c] : null;
            const cDangerCount = state.col_danger_counts ? state.col_danger_counts[c] : 0;
            if (cThreat !== null && cThreat !== undefined) {
              if (cThreat === 0) tags.push('⚠️ FLAME BURNING (countdown 0)');
              else if (cThreat <= 3) {
                if (cDangerCount >= 3) tags.push(`🚨 CRITICAL DANGER: Heavy blast line along col (countdown ${cThreat})`);
                else tags.push(`⚠️ Crossed by blast at ${cDangerCount} cell(s) (countdown ${cThreat})`);
              }
              else if (cThreat <= 6) tags.push(`medium danger (countdown ${cThreat})`);
              else tags.push(`safe delay (countdown ${cThreat})`);
            } else {
              tags.push('SAFE corridor');
            }

            colCriteria[`c${c}`] = `Col ${c} [REACHABLE: ${tags.join(' | ')}]`;
          }
        }

        const curHp = state.player_stats.hp;
        const oppHp = state.enemy_stats.hp;
        const hpRulesSummary = `WIN/LOSS RULES: Touching any flame/blast loses 1 HP; HP=0 is instant GAME OVER (Defeat). Reducing enemy HP to 0 is immediate VICTORY. Your HP: ${curHp}, Enemy HP: ${oppHp}.${curHp <= 1 ? ' [WARNING: ONE-HIT DEATH MODE - ANY DAMAGE IS FATAL!]' : ''}`;

        const cratesList = state.key_coordinates.crates || [];
        const questions = {
          strategic_intent: {
            type: 'choice',
            instructions: `${hpRulesSummary} Based on map_grid_15x13 (where s=speed, f=flame, b=bomb, c=mystery crates, and 0-9=danger countdowns), match_rules, continuous_positions, and history, what is the primary strategic objective?`,
            criteria: {
              hunt_opponent: `Aggressively advance towards opponent E (at row ${enemyR}, col ${enemyC}) along SAFE corridors (. or countdown >= 5). AVOID corridors with imminent countdown 0-3!`,
              gather_powerup: cratesList.length > 0
                ? `PRIORITIZE POWERUPS FIRST (发育优先 / 先吃道具): Navigate towards nearest safe powerup (${cratesList.slice(0, 3).map(cr => `${cr.type}('${cr.symbol}') at [${cr.pos[0]},${cr.pos[1]}] dist ${cr.dist}`).join(', ')}). TOP PRIORITY when speed (${state.player_stats.speed}) or bombs (${state.player_stats.bombs_cap}) are behind enemy (${state.enemy_stats.speed} speed, ${state.enemy_stats.bombs_cap} bombs), or to establish early-game attribute dominance before fighting!`
                : 'Navigate towards a safe powerup crate on the map to collect it for attribute upgrades.',
              breach_obstacle: 'Navigate towards a blocking brick B to place a bomb and open corridors.',
              evade_danger: curHp <= 1
                ? '🚨 URGENT EVASION: Navigate away from danger corridors (countdown 0-3) to a secure shelter tile (.). HP IS 1 (ONE-HIT DEATH), SURVIVAL IS ABSOLUTE TOP PRIORITY!'
                : 'Navigate away from danger corridors (countdown 0-3) to a secure shelter tile (.).'
            }
          },
          target_row: {
            type: 'choice',
            instructions: `Select target destination row index (0 to 12). CRITICAL: Your HP is ${curHp}. Prefer rows tagged [SAFE] or paths with '.' or countdown >= 5. DO NOT route into rows with imminent danger (countdown 0-3) unless intentionally evading.`,
            criteria: rowCriteria
          },
          target_col: {
            type: 'choice',
            instructions: `Select target destination column index (0 to 14). CRITICAL: Your HP is ${curHp}. Prefer columns tagged [SAFE] or paths with '.' or countdown >= 5. DO NOT route into columns with imminent danger (countdown 0-3) unless intentionally evading.`,
            criteria: colCriteria
          },
          bomb_decision: {
            type: 'choice',
            instructions: `Should player place a bomb at current location right now? (Available in hand: ${state.player_stats.available_bombs}/${state.player_stats.bombs_cap}, active on map: ${state.player_stats.active_bombs}, blast reach: ${state.player_stats.blast_cap} tiles). (CHAIN REACTION: Bombs touching any blast line detonate immediately! SAFETY RULE: Bombs are solid obstacles. NEVER place if you are in an enclosed corner with no exit corridor!).`,
            criteria: {
              plant_bomb_now: state.player_stats.available_bombs > 0
                ? 'Place a bomb at current location right now (only if you have an open escape route to step back into, AND an opponent is nearby, blocking brick is directly ahead, or creating a chain reaction).'
                : 'No bombs available to place right now (all bombs active on field).',
              hold_bomb: 'Do not place bomb; keep corridor open, or player is actively moving towards target without dropping a bomb.'
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

      // 1. 定期、到达目标格、或路径被砖块阻断时发起研判
      const blockedByBrick = (this.currentSearchPath && this.currentSearchPath.length > 1 && sim.brick[this.currentSearchPath[1]]);
      const arrivedAtTarget = (this.targetCell >= 0 && this.targetCell === ownIdx);

      if (!this.isInferring && (curTick - this.lastInferTick >= this.inferIntervalTicks || (blockedByBrick && curTick - this.lastInferTick >= 3) || (arrivedAtTarget && curTick - this.lastInferTick >= 4))) {
        this.lastInferTick = curTick;
        this.callJevAsync(sim, pid);
      }

      // 2. 解析 Jev 直出的目标坐标 (Row, Col) 并执行连通性校验与就近投影
      const reachableMask = this.computeConnectedComponent(sim, ownIdx);
      const opp = 1 - pid;
      const oppCell = sim.centerCell(opp);
      const oppIdx = oppCell[0] * W + oppCell[1];
      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      let targetRow = this.targetRow >= 0 ? this.targetRow : own[0];
      let targetCol = this.targetCol >= 0 ? this.targetCol : own[1];
      let targetCell = targetRow * W + targetCol;

      // 如果选中的目标不在连通分量中（孤岛/非联通格）或者自身为不可穿透实心墙，就近投影到最近的连通格
      if (targetCell < 0 || targetCell >= N || !reachableMask[targetCell] || sim.wall[targetCell] === 1) {
        let bestDist = Infinity;
        let bestC = ownIdx;
        for (let i = 0; i < N; i++) {
          if (reachableMask[i] && sim.wall[i] === 0) {
            const ir = (i / W) | 0, ic = i % W;
            const d = Math.abs(ir - targetRow) + Math.abs(ic - targetCol);
            if (d < bestDist) {
              bestDist = d;
              bestC = i;
            }
          }
        }
        targetCell = bestC;
        targetRow = (targetCell / W) | 0;
        targetCol = targetCell % W;
      }

      // 关键防卡死：若模型输出的目标就是当前所在格，或角色已经到达目标格：
      // 严禁原地原地发呆！根据 strategic_intent 自主向下一个目标推进
      if (targetCell === ownIdx) {
        if (this.targetIntent === 'hunt_opponent' && oppIdx !== -1 && reachableMask[oppIdx]) {
          targetCell = oppIdx;
          targetRow = oppCell[0];
          targetCol = oppCell[1];
        } else if (this.targetIntent === 'gather_powerup') {
          let nearestCrate = -1, nearestDist = Infinity;
          for (let i = 0; i < N; i++) {
            if (sim.crate[i] === 1 && reachableMask[i] && i !== ownIdx) {
              const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
              if (d < nearestDist) {
                nearestDist = d;
                nearestCrate = i;
              }
            }
          }
          if (nearestCrate !== -1) {
            targetCell = nearestCrate;
            targetRow = (targetCell / W) | 0;
            targetCol = targetCell % W;
          } else if (oppIdx !== -1 && reachableMask[oppIdx]) {
            targetCell = oppIdx;
            targetRow = oppCell[0];
            targetCol = oppCell[1];
          }
        } else if (this.targetIntent === 'breach_obstacle') {
          let nearestBrick = -1, nearestDist = Infinity;
          for (let i = 0; i < N; i++) {
            if (sim.brick[i] === 1 && reachableMask[i]) {
              const d = Math.abs(((i / W) | 0) - own[0]) + Math.abs((i % W) - own[1]);
              if (d < nearestDist) {
                nearestDist = d;
                nearestBrick = i;
              }
            }
          }
          if (nearestBrick !== -1) {
            targetCell = nearestBrick;
            targetRow = (targetCell / W) | 0;
            targetCol = targetCell % W;
          } else if (oppIdx !== -1 && reachableMask[oppIdx]) {
            targetCell = oppIdx;
            targetRow = oppCell[0];
            targetCol = oppCell[1];
          }
        } else {
          if (oppIdx !== -1 && reachableMask[oppIdx]) {
            targetCell = oppIdx;
            targetRow = oppCell[0];
            targetCol = oppCell[1];
          }
        }

        // 核心持久化：必须将推进的新目标同步持久保存至 this.targetRow 和 this.targetCol！
        // 绝对禁止留在旧格，否则角色迈出一步后下一 tick 就会把旧格误当成目的地往回走，导致远处/当前格高频切换、人体抽搐抖动！
        this.targetRow = targetRow;
        this.targetCol = targetCol;
      }

      // 3. A* 寻路前往目标（ignoreDanger: true，火是可以踩的，完全由大模型基于危险观测做决策！）
      let searchRes = this.helperAi.search(sim, danger, ownIdx, targetCell, spd, nowMs, {
        allowBreakBrick: true,
        lastMove: this.lastMove,
        ignoreDanger: true
      });

      // 寻路兜底：仅在寻路真正受阻且未到达目标时才尝试备用目标
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
            lastMove: this.lastMove,
            ignoreDanger: true
          });
          if (altRes && altRes.path.length > 1) {
            searchRes = altRes;
            targetCell = candidates[c].cell;
            targetRow = (targetCell / W) | 0;
            targetCol = targetCell % W;
            this.targetRow = targetRow;
            this.targetCol = targetCol;
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

      this.targetRow = targetRow;
      this.targetCol = targetCol;
      this.targetPos = [targetRow, targetCol];
      this.targetCell = targetCell;
      this.currentSearchPath = searchRes ? searchRes.path : [];

      let chosenMove = MOVE_IDLE;
      let finalBomb = 0;
      let nextStepIsBrick = false;

      if (searchRes && searchRes.path.length > 1 && searchRes.path[0] === ownIdx) {
        const nextCell = searchRes.path[1];
        if (sim.brick[nextCell]) {
          nextStepIsBrick = true; // 路径前方受阻于砖块，需放泡破障
        } else {
          chosenMove = this.helperAi._cellToMove(ownIdx, nextCell, W);
        }
      }

      // 若前方受阻于砖块或者当前格有正在倒计时的炸弹，但此时 chosenMove 停滞在 MOVE_IDLE：
      // 必须立刻从当前格向周边合法开放格（非墙非砖无雷）机动撤退，避免原地等死！
      if ((nextStepIsBrick || sim.fuse[ownIdx] > 0) && chosenMove === MOVE_IDLE) {
        for (let d = 0; d < 4; d++) {
          if (mm[pid][d] === 1) {
            const nr = own[0] + DIRS[d][0], nc = own[1] + DIRS[d][1];
            if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
              const ni = nr * W + nc;
              if (!sim.wall[ni] && !sim.brick[ni] && sim.fuse[ni] === 0) {
                chosenMove = d;
                break;
              }
            }
          }
        }
      }

      // 4. 放泡执行：完全由大模型决策 plant_bomb_now，或前方被砖块阻断时破障
      // 不设底层安全拦截，允许火中放泡与近身死斗
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
      const adjacentToOpp = (oppDist <= 1) || (ownIdx === targetCell && targetCell === oppIdx);

      // 4. 放泡执行：完全交由大模型决策 bomb_decision，底层严禁越权放泡
      const canDrop = bm[pid][1] === 1 && sim.fuse[ownIdx] === 0 && sim.liveBombs(pid) < sim.bombsCap[pid];
      if (canDrop) {
        const shouldDropForJev = this.lastDecision && this.lastDecision.bombChoice === 'plant_bomb_now';
        if (shouldDropForJev) {
          finalBomb = 1;
          this.stats.bombsPlaced++;
          this.lastPlacedBombCell = ownIdx;
          if (this.lastDecision) this.lastDecision.bombChoice = 'hold_bomb';
          this.bombDecision = 'hold_bomb';
        }
      }

      // 5. 100ms 临界底层防自杀底线 (100ms Imminent Anti-Suicide Gate):
      // 允许踩初级/远期火焰与穿雷走位（绝不因初级火焰被 stop），但当炸弹处于 <= 100ms (1 tick) 临界起火爆炸或当前格正在燃烧时：
      // 绝对别走进去（别过去），若身处险境则紧急机动脱险！
      const [safeMove, safeBomb] = this.filter100msSuicide(sim, pid, chosenMove, finalBomb, targetCell);
      chosenMove = safeMove;
      finalBomb = safeBomb;

      // 物理引擎有效性校验（撞墙/越界保护）
      if (chosenMove !== MOVE_IDLE && mm[pid][chosenMove] !== 1) {
        chosenMove = MOVE_IDLE;
      }

      this.lastMove = chosenMove;
      this.recentMoves.push(MOVE_NAMES[chosenMove]);
      return [chosenMove, finalBomb];
    }

    getImminentLethalMask(sim) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const lethal = new Uint8Array(N);

      // 1. 正在燃烧的烈焰残威
      if (sim.blastLinger) {
        for (let i = 0; i < N; i++) {
          if (sim.blastLinger[i] > 0) lethal[i] = 1;
        }
      }

      // 2. 查找所有还有 <= 1 tick (<= 100ms) 爆炸的炸弹
      const detonatingBombs = [];
      const bombList = [];
      if (sim.fuse) {
        for (let i = 0; i < N; i++) {
          if (sim.fuse[i] > 0) {
            const bObj = {
              idx: i,
              r: (i / W) | 0,
              c: i % W,
              blast: sim.bombBlast ? (sim.bombBlast[i] || 2) : 2,
              fuse: sim.fuse[i],
              willExplode: sim.fuse[i] <= 1
            };
            bombList.push(bObj);
            if (bObj.willExplode) detonatingBombs.push(bObj);
          }
        }
      }

      // 3. 连锁引爆传播：fuse <= 1 的炸弹引发的连锁引爆
      let changed = true;
      let pass = 0;
      while (changed && pass < 10) {
        changed = false;
        pass++;
        for (let d = 0; d < detonatingBombs.length; d++) {
          const bA = detonatingBombs[d];
          for (let dir = 0; dir < 4; dir++) {
            const [dr, dc] = DIRS[dir];
            for (let k = 1; k <= bA.blast; k++) {
              const nr = bA.r + dr * k, nc = bA.c + dc * k;
              if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
              const ni = nr * W + nc;
              if (sim.wall && sim.wall[ni]) break;
              for (let b = 0; b < bombList.length; b++) {
                const bB = bombList[b];
                if (bB.idx === ni && !bB.willExplode) {
                  bB.willExplode = true;
                  detonatingBombs.push(bB);
                  changed = true;
                }
              }
              if ((sim.brick && sim.brick[ni]) || (sim.pushable && sim.pushable[ni])) break;
            }
          }
        }
      }

      // 4. 涂布所有将在 <= 100ms 内致命起火爆炸的十字范围
      for (let d = 0; d < detonatingBombs.length; d++) {
        const b = detonatingBombs[d];
        lethal[b.idx] = 1;
        for (let dir = 0; dir < 4; dir++) {
          const [dr, dc] = DIRS[dir];
          for (let k = 1; k <= b.blast; k++) {
            const nr = b.r + dr * k, nc = b.c + dc * k;
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) break;
            const ni = nr * W + nc;
            if (sim.wall && sim.wall[ni]) break;
            lethal[ni] = 1;
            if ((sim.brick && sim.brick[ni]) || (sim.pushable && sim.pushable[ni])) break;
          }
        }
      }

      return lethal;
    }

    filter100msSuicide(sim, pid, chosenMove, finalBomb, targetCell) {
      const W = sim.W || 15, H = sim.H || 13;
      const lethal = this.getImminentLethalMask(sim);
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const { mm } = sim.legalMask();

      // Case A: 自身当前格处于 <= 100ms 即刻爆炸火线或燃烧中！必须紧急机动脱险
      if (lethal[ownIdx] === 1) {
        let moveIsSafe = false;
        if (chosenMove !== MOVE_IDLE && mm[pid][chosenMove] === 1) {
          const nr = own[0] + DIRS[chosenMove][0], nc = own[1] + DIRS[chosenMove][1];
          if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
            const ni = nr * W + nc;
            if (lethal[ni] === 0 && !sim.wall[ni] && !sim.brick[ni]) {
              moveIsSafe = true;
            }
          }
        }

        if (!moveIsSafe) {
          let bestD = MOVE_IDLE;
          let bestDist = Infinity;
          const targetR = targetCell >= 0 ? (targetCell / W) | 0 : own[0];
          const targetC = targetCell >= 0 ? targetCell % W : own[1];

          for (let d = 0; d < 4; d++) {
            if (mm[pid][d] === 1) {
              const nr = own[0] + DIRS[d][0], nc = own[1] + DIRS[d][1];
              if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
                const ni = nr * W + nc;
                if (lethal[ni] === 0 && !sim.wall[ni] && !sim.brick[ni]) {
                  const dist = Math.abs(nr - targetR) + Math.abs(nc - targetC);
                  if (dist < bestDist) {
                    bestDist = dist;
                    bestD = d;
                  }
                }
              }
            }
          }
          chosenMove = bestD;
        }
        finalBomb = 0; // 濒死脱险瞬间取消放泡，全力逃生
      }
      // Case B: 自身当前格安全，但拟迈入的格子处于 <= 100ms 临界爆炸区！“别过去”拦截！
      else if (chosenMove !== MOVE_IDLE) {
        const nr = own[0] + DIRS[chosenMove][0], nc = own[1] + DIRS[chosenMove][1];
        if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
          const ni = nr * W + nc;
          if (lethal[ni] === 1) {
            // 目标格即将爆炸：绝对别走进去！在当前安全格驻留等待起火结束
            chosenMove = MOVE_IDLE;
          }
        }
      }

      return [chosenMove, finalBomb];
    }
  }

  return JevGridAI;
});
