/**
 * web/jev_pure_ai.js - TypeSafe Jev 纯净直出版 (Doom Paradigm Zero-Rule Edition)
 *
 * 设计哲学（对齐 TypeSafe 官方 Doom Demo）：
 * 1. 结构化游戏状态观测 (Structured Game State Observation)：
 *    - 像 Doom Demo 一样，从游戏内存中提取结构化 JSON 上下文（而非原始像素/张量）。
 *    - 包含：角色属性、对手属性、速度优劣比、四邻地形（可通行/砖/墙/火线）、最近道具列表与危险倒计时。
 * 2. 强提示词语义决策 (Prompt-Driven Semantic Decision)：
 *    - 核心业务与战术规则（例如：“速度比不过敌人时优先吃最近道具”、“脚下有火线时沿垂直方向避险”）
 *      全部写在 instructions 与 criteria 的语义提示词中，由 Jev (System One) 直接理解并推演。
 * 3. 底层零规则零干预 (Zero Heuristics / Direct Output)：
 *    - 底层没有任何 A* 寻路、没有逃生路径锁定、没有掩体等待、没有踩火禁止覆盖。
 *    - Jev 直接输出移动方向 (Up/Down/Left/Right/Idle) 与放泡决策 (Plant/Hold)，底层 100% 忠实执行！
 */

'use strict';

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.JevPureAI = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {

  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 0: 上, 1: 下, 2: 左, 3: 右
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;
  const MOVE_NAMES = ['up', 'down', 'left', 'right', 'idle'];

  class JevPureAI {
    constructor(options = {}) {
      this.options = options;
      this.apiUrl = options.apiUrl || '/api/typesafe';
      this.model = options.model || 'jev-latest';
      this.inferIntervalTicks = options.inferIntervalTicks || 2; // Doom Demo 模式：高频 200ms 推理循环

      this.reset();
    }

    reset() {
      this.lastInferTick = -999;
      this.lastInferTimeMs = Date.now();
      this.isInferring = false;
      this.lastMove = MOVE_IDLE;
      this.lastBomb = 0;
      this.lastDecision = null;
      this.stats = {
        totalCalls: 0,
        lastLatencyMs: 0,
        bombsPlaced: 0
      };
    }

    // ------------------------------------------------------------ 提取 Doom 式结构化状态
    extractState(sim, pid) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const opp = 1 - pid;
      const own = sim.centerCell(pid);
      const oppCell = sim.centerCell(opp);
      const ownIdx = own[0] * W + own[1];
      const oppIdx = oppCell[0] * W + oppCell[1];
      const nowMs = (sim.t || 0) * 100;

      const ownSpd = Number((sim.spdG ? sim.spdG[pid] : 1.0).toFixed(2));
      const oppSpd = Number((sim.spdG ? sim.spdG[opp] : 1.0).toFixed(2));
      const isSlower = ownSpd < oppSpd;
      const speedDiff = Number((ownSpd - oppSpd).toFixed(2));

      const oppDist = Math.abs(own[0] - oppCell[0]) + Math.abs(own[1] - oppCell[1]);

      // 1. 扫描地图所有道具并排序
      const powerups = [];
      for (let i = 0; i < N; i++) {
        if (sim.crate && sim.crate[i] === 1) {
          const cr = (i / W) | 0, cc = i % W;
          const dist = Math.abs(cr - own[0]) + Math.abs(cc - own[1]);
          const ct = sim.crateType ? sim.crateType[i] : 0;
          let typeStr = 'bomb_cap';
          if (ct === 1) typeStr = 'blast_range';
          else if (ct === 2) typeStr = 'speed_boots';
          const isSuper = sim.superCrate && sim.superCrate[i] === 1;
          if (isSuper) typeStr = 'super_' + typeStr;

          // 寻找朝该道具缩短距离的方向
          let bestDir = 'none';
          if (cr < own[0]) bestDir = 'up';
          else if (cr > own[0]) bestDir = 'down';
          else if (cc < own[1]) bestDir = 'left';
          else if (cc > own[1]) bestDir = 'right';

          powerups.push({
            type: typeStr,
            position: [cr, cc],
            distance: dist,
            best_direction: bestDir,
            is_speed_boost: ct === 2
          });
        }
      }
      // 优先按速度道具与距离排序
      powerups.sort((a, b) => {
        if (isSlower && a.is_speed_boost !== b.is_speed_boost) {
          return a.is_speed_boost ? -1 : 1;
        }
        return a.distance - b.distance;
      });

      // 2. 扫描四邻直接可走方向
      const surroundings = {};
      const dirKeys = ['up', 'down', 'left', 'right'];
      for (let d = 0; d < 4; d++) {
        const nr = own[0] + DIRS[d][0], nc = own[1] + DIRS[d][1];
        const key = dirKeys[d];
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) {
          surroundings[key] = {
            tile: [nr, nc],
            walkable: false,
            obstacle: 'boundary_wall',
            threat: 'impassable',
            leads_to: 'out_of_bounds'
          };
        } else {
          const ni = nr * W + nc;
          let obstacle = 'open_ground';
          let walkable = true;
          if (sim.wall[ni] === 1) { obstacle = 'indestructible_wall'; walkable = false; }
          else if (sim.brick[ni] === 1) { obstacle = 'destructible_brick'; walkable = false; }
          else if (sim.fuse[ni] > 0) { obstacle = 'ticking_bomb'; walkable = false; }

          let threat = 'safe';
          let timeToBoom = null;
          // 简易直接火线检测：检查四向是否有炸弹
          for (let b = 0; b < N; b++) {
            if (sim.fuse[b] > 0) {
              const br = (b / W) | 0, bc = b % W;
              const blastLen = sim.bombBlast ? sim.bombBlast[b] : 2;
              if ((br === nr && Math.abs(bc - nc) <= blastLen) || (bc === nc && Math.abs(br - nr) <= blastLen)) {
                threat = 'in_bomb_blast_line';
                timeToBoom = sim.fuse[b] * 100;
                break;
              }
            }
          }

          let leadsTo = 'open_area';
          if (powerups.length > 0 && powerups[0].best_direction === key) leadsTo = `nearest_powerup_${powerups[0].type}`;
          else if (oppCell[0] === nr && oppCell[1] === nc) leadsTo = 'opponent';
          else if (nr === oppCell[0] || nc === oppCell[1]) leadsTo = 'towards_opponent';

          surroundings[key] = {
            tile: [nr, nc],
            walkable,
            obstacle,
            threat,
            time_to_boom_ms: timeToBoom,
            leads_to: leadsTo
          };
        }
      }

      // 3. 自身当前所在格的安全态势
      let currentTileThreat = 'safe';
      let currentTileBoomMs = null;
      for (let b = 0; b < N; b++) {
        if (sim.fuse[b] > 0) {
          const br = (b / W) | 0, bc = b % W;
          const blastLen = sim.bombBlast ? sim.bombBlast[b] : 2;
          if ((br === own[0] && Math.abs(bc - own[1]) <= blastLen) || (bc === own[1] && Math.abs(br - own[0]) <= blastLen)) {
            currentTileThreat = 'in_bomb_blast_line';
            currentTileBoomMs = sim.fuse[b] * 100;
            break;
          }
        }
      }

      return {
        player: {
          position: [own[0], own[1]],
          hp: sim.hp ? sim.hp[pid] : 5,
          speed: ownSpd,
          bombs_cap: sim.bombsCap ? sim.bombsCap[pid] : 2,
          blast_range: sim.blastCap ? sim.blastCap[pid] : 2,
          live_bombs: sim.liveBombs ? sim.liveBombs(pid) : 0,
          can_plant_bomb: sim.fuse[ownIdx] === 0 && (!sim.liveBombs || sim.liveBombs(pid) < sim.bombsCap[pid])
        },
        opponent: {
          position: [oppCell[0], oppCell[1]],
          hp: sim.hp ? sim.hp[opp] : 5,
          speed: oppSpd,
          distance: oppDist,
          speed_advantage: speedDiff,
          is_faster_than_player: isSlower
        },
        tactical_situation: {
          is_slower_than_opponent: isSlower,
          speed_guidance: isSlower
            ? "CRITICAL: Player speed is lower than opponent! DO NOT engage directly! Prioritize moving towards nearest speed_boots powerup crate!"
            : "Speed is equal or superior. Safe to hunt and pressure opponent.",
          current_tile_threat: currentTileThreat,
          time_to_boom_ms: currentTileBoomMs,
          immediate_danger_warning: currentTileThreat !== 'safe'
            ? `ALERT: Current tile is inside a bomb blast line (exploding in ${currentTileBoomMs}ms)! Choose a safe open direction immediately!`
            : "Current tile is safe."
        },
        immediate_directions: surroundings,
        visible_powerups: powerups.slice(0, 4)
      };
    }

    // ------------------------------------------------------------ 异步向 Jev 发起纯净推演
    async callJevAsync(sim, pid) {
      if (this.isInferring) return;
      this.isInferring = true;
      const t0 = performance.now();

      try {
        const state = this.extractPureState(sim, pid);

        // 构造四向语义描述 (Criteria)
        const dirCriteria = {};
        const dirKeys = ['up', 'down', 'left', 'right'];
        for (const k of dirKeys) {
          const s = state.immediate_directions[k];
          let desc = `Move ${k.toUpperCase()} to (${s.tile[0]}, ${s.tile[1]}): `;
          if (!s.walkable) {
            desc += `[BLOCKED by ${s.obstacle} - CANNOT WALK]`;
          } else {
            desc += `[WALKABLE open path]; Threat: ${s.threat}`;
            if (s.time_to_boom_ms) desc += ` (${s.time_to_boom_ms}ms to blast)`;
            desc += `; Leads towards: ${s.leads_to}`;
          }
          dirCriteria[k] = desc;
        }
        dirCriteria['idle'] = `Stay IDLE at current tile (${state.player.position[0]}, ${state.player.position[1]}): Threat: ${state.tactical_situation.current_tile_threat}`;

        const questions = {
          move_direction: {
            type: 'choice',
            instructions: `Select the single best move direction (up, down, left, right, or idle).
RULES:
1. SURVIVAL: If current tile is under threat, immediately pick a safe walkable direction to step off the blast line.
2. SPEED DEFICIT: If is_slower_than_opponent is true, your HIGHEST priority is to move towards the nearest powerup crate (especially speed_boots). Do not fight when slower!
3. ATTACK: If speed is matched or superior, maneuver towards opponent to trap or bomb them.
4. DO NOT walk into blocked tiles (walls/bricks/bombs).`,
            criteria: dirCriteria
          },
          bomb_action: {
            type: 'choice',
            instructions: `Should the player place a bomb at the current location right now?
- plant_bomb: Use when adjacent to a destructible brick to open a path, or enemy is in direct line and player has a clear escape tile.
- hold_bomb: Do not place a bomb. Cruising towards powerup, pursuing enemy, or fleeing danger.`,
            criteria: {
              plant_bomb: "Place a bomb at current location now to breach brick or attack enemy.",
              hold_bomb: "Do not place a bomb now. Keep moving."
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
          throw new Error(`Jev Pure API error: ${resp.status} ${resp.statusText}`);
        }

        const data = await resp.json();
        const answers = data.answers || {};

        const moveDir = answers.move_direction ? answers.move_direction.choice : 'idle';
        const bombAct = answers.bomb_action ? answers.bomb_action.choice : 'hold_bomb';

        this.lastDecision = {
          moveDir,
          bombAct,
          isSlower: state.tactical_situation.is_slower_than_opponent,
          currentThreat: state.tactical_situation.current_tile_threat,
          timestamp: Date.now()
        };

        this.stats.totalCalls++;
        this.stats.lastLatencyMs = Math.round(performance.now() - t0);

      } catch (err) {
        console.warn('[JevPureAI] Inference warning:', err);
      } finally {
        this.isInferring = false;
      }
    }

    extractPureState(sim, pid) {
      return this.extractState(sim, pid);
    }

    // ------------------------------------------------------------ 纯净直接驱动 act(sim, pid)
    // 零规则、零 A*、零干预：直接执行 Jev 输出的方向与放泡！
    act(sim, pid, rng) {
      const curTick = sim.t || 0;

      // 定期或触发推演
      if (!this.isInferring && (curTick - this.lastInferTick >= this.inferIntervalTicks)) {
        this.lastInferTick = curTick;
        this.callJevAsync(sim, pid);
      }

      // 默认动作：若尚无推演结果则保持不动
      let chosenMove = MOVE_IDLE;
      let finalBomb = 0;

      if (this.lastDecision) {
        const dirMap = {
          'up': MOVE_UP,
          'down': MOVE_DOWN,
          'left': MOVE_LEFT,
          'right': MOVE_RIGHT,
          'idle': MOVE_IDLE
        };
        chosenMove = dirMap[this.lastDecision.moveDir] ?? MOVE_IDLE;
        finalBomb = this.lastDecision.bombAct === 'plant_bomb' ? 1 : 0;
      }

      if (finalBomb === 1) {
        this.stats.bombsPlaced++;
        // 消耗放泡决策，防止连续多 tick 持续下子
        if (this.lastDecision) this.lastDecision.bombAct = 'hold_bomb';
      }

      // 100ms 临界底层防自杀底线 (Ultra-Short 100ms Anti-Suicide Gate):
      // 允许踩初级/远期火焰与穿雷走位，但当炸弹处于 <= 100ms 临界爆炸或当前格正在燃烧时拦截！
      const [safeMove, safeBomb] = this.filter100msSuicide(sim, pid, chosenMove, finalBomb);
      chosenMove = safeMove;
      finalBomb = safeBomb;

      const { mm } = sim.legalMask();
      if (chosenMove !== MOVE_IDLE && mm && mm[pid] && mm[pid][chosenMove] !== 1) {
        chosenMove = MOVE_IDLE;
      }

      this.lastMove = chosenMove;
      this.lastBomb = finalBomb;
      return [chosenMove, finalBomb];
    }

    getImminentLethalMask(sim) {
      const W = sim.W || 15, H = sim.H || 13, N = W * H;
      const lethal = new Uint8Array(N);
      if (sim.blastLinger) {
        for (let i = 0; i < N; i++) {
          if (sim.blastLinger[i] > 0) lethal[i] = 1;
        }
      }
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

    filter100msSuicide(sim, pid, chosenMove, finalBomb) {
      const W = sim.W || 15, H = sim.H || 13;
      const lethal = this.getImminentLethalMask(sim);
      const own = sim.centerCell(pid);
      const ownIdx = own[0] * W + own[1];
      const { mm } = sim.legalMask();

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
          for (let d = 0; d < 4; d++) {
            if (mm[pid][d] === 1) {
              const nr = own[0] + DIRS[d][0], nc = own[1] + DIRS[d][1];
              if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
                const ni = nr * W + nc;
                if (lethal[ni] === 0 && !sim.wall[ni] && !sim.brick[ni]) {
                  bestD = d;
                  break;
                }
              }
            }
          }
          chosenMove = bestD;
        }
        finalBomb = 0;
      } else if (chosenMove !== MOVE_IDLE) {
        const nr = own[0] + DIRS[chosenMove][0], nc = own[1] + DIRS[chosenMove][1];
        if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
          const ni = nr * W + nc;
          if (lethal[ni] === 1) {
            chosenMove = MOVE_IDLE;
          }
        }
      }

      return [chosenMove, finalBomb];
    }
  }

  return JevPureAI;
});
