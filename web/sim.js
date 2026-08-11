// sim.js —— 泡泡堂 1v1 模拟器的浏览器版（单局、2 人），纯 JS 无依赖。
//
// 这是 sim/torch_sim.py + sim/blast.py + sim/move.py + sim/obs.py 的**标量移植**：
// 与训练侧语义对齐（13×13、10Hz、连续坐标 AABB 碰撞、连锁爆炸、危险图、
// 14 通道观测、5×2 因子化动作）。浏览器 + Node（冒烟测试）双端可用。
//
// 布局约定：
//   - 网格 (H=13, W=13)，一维索引 i = r*W + c；
//   - pos 是 [p0_y, p0_x, p1_y, p1_x] 的 Float64Array（角色中心，格坐标）；
//   - 动作：[move, bomb]，move∈{0上,1下,2左,3右,4松手}，bomb∈{0不放,1放}；
//   - 模型权重由 deploy/export_ckpt.py 导出（已折 pid=0 视角），JS 侧直接对
//     共享观测做普通 MLP 前向；AI 走玩家 1 时观测先做通道互换（自己→通道 0）。
//
// 规则常量与 play/duel.py 的 open / corridor 两档配置对齐（见 CFG）。

(function (root) {
  'use strict';

  // ---------------------------------------------------------------- 常量
  const H = 13, W = 13, N = H * W;
  const N_PLAYERS = 2;
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;
  const N_MOVES = 5, N_BOMB = 2;
  // (dy, dx)，索引与方向编码对齐
  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const EPS = 1e-4;

  // duel.py 的配置（map_mode=corridor + open_fraction=1.0 即 open 关；
  // corridor 关用 growth_*_start 起步 + 顶墙/侧砖）。
  const CFG = {
    tickHz: 10, speed: 3.0, radius: 0.3, maxSteps: 1800,
    fuse: 30, blast: 2, maxBombs: 10, maxChain: 16,
    maxHp: 5, invulnTicks: 30,
    stepLen: 3.0 / 10,                 // 0.3 格/tick
    // 成长（corridor 起步）
    growthBombsStart: 2, growthBlastStart: 2, growthSpeedStart: 1.6,
    growthBombsMax: 10, growthBlastMax: 7,
    growthSpeedMax: 2.1, growthSpeedStep: 0.15,
    // open 关起步 = 上限 80%（与训练一致）
    openGrowthBombs: Math.ceil(10 * 0.8),        // 8
    openGrowthBlast: Math.ceil(7 * 0.8),         // 6
    openGrowthSpeed: Math.round(2.1 * 0.8 * 100) / 100,  // 1.68
    // 地图
    corridorWidth: 5, topWallRows: 4,
    // 宝箱
    growthCrateProb: 0.5, recycleCrateProb: 1.0, hitAttrPenalty: 2,
    openCrateCross: true,
    // 危险图
    dangerExp: 2.0,
  };

  // ---------------------------------------------------------------- 随机
  // mulberry32：可播种的简易 PRNG（web 端不要求与 torch RNG 对齐，行为等价即可）。
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------- base64 → Float32Array
  function decodeB64(b64) {
    // 浏览器用 atob，Node 用 Buffer；统一返回承载 float32 的 ArrayBuffer。
    if (typeof atob === 'function') {
      const bin = atob(b64);
      const buf = new ArrayBuffer(bin.length);
      const u8 = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return buf;
    }
    const buf = Buffer.from(b64, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  // ---------------------------------------------------------------- 碰撞
  // 单格不可通行：越界、或被固体挡住且碰撞盒当前没压在这一格上
  // （“脚下刚放的泡必须能走出去”：压着就放行）。
  function impassable(blocked, row, col, y, x, rad, h, w) {
    if (row < 0 || row >= h || col < 0 || col >= w) return true;
    if (!blocked[row * w + col]) return false;
    const r0 = Math.floor(y - rad), r1 = Math.floor(y + rad);
    const c0 = Math.floor(x - rad), c1 = Math.floor(x + rad);
    const inside = row >= r0 && row <= r1 && col >= c0 && col <= c1;
    return !inside;
  }

  // 沿单轴消解碰撞（sim/move.py::_resolve_axis 的标量版）：撞上就贴着停。
  function resolveAxis(coord, delta, other, y, x, blocked, rad, h, w, vertical) {
    const sgn = Math.sign(delta);
    if (sgn === 0) return coord;
    const oldLead = Math.floor(coord - delta + sgn * rad);
    const newLead = Math.floor(coord + sgn * rad);
    const lo = Math.min(oldLead, newLead);
    const hi = Math.max(oldLead, newLead);
    const span0 = Math.floor(other - rad);
    const span1 = Math.floor(other + rad);
    const hitAt = (lead) => {
      if (vertical) {
        return impassable(blocked, lead, span0, y, x, rad, h, w) ||
          impassable(blocked, lead, span1, y, x, rad, h, w);
      }
      return impassable(blocked, span0, lead, y, x, rad, h, w) ||
        impassable(blocked, span1, lead, y, x, rad, h, w);
    };
    const firstLead = sgn > 0 ? hi : lo;
    const secondLead = sgn > 0 ? lo : hi;
    const firstHit = hitAt(firstLead);
    const secondHit = hitAt(secondLead);
    const has = firstHit || secondHit;
    const first = firstHit ? firstLead : (secondHit ? secondLead : 0);
    const stopPos = sgn > 0 ? first - rad - EPS : first + 1 + rad + EPS;
    return has ? stopPos : coord;
  }

  // ---------------------------------------------------------------- 模拟器
  class Sim {
    constructor(seed) {
      this.rng = mulberry32(seed == null ? 1 : seed);
      this.reset('open');
    }

    // mode: 'open'（纯空场 + 中心十字宝箱 + 80% 起步）| 'corridor'（顶墙侧砖 + 2/2/1 起步）
    reset(mode) {
      this.mode = mode === 'corridor' ? 'corridor' : 'open';
      this.wall = new Uint8Array(N);
      this.brick = new Uint8Array(N);
      this.crate = new Uint8Array(N);
      this.recycle = new Uint8Array(N);
      this.fuse = new Int16Array(N);
      this.owner = new Int8Array(N);
      this.owner.fill(-1);
      this.bombBlast = new Int16Array(N);
      this.pos = new Float64Array(4);
      this.alive = [true, true];
      this.hp = [CFG.maxHp, CFG.maxHp];
      this.invuln = [0, 0];
      this.sinceBomb = [0, 0];
      this.bombsCap = [0, 0];
      this.blastCap = [0, 0];
      this.spdG = [1.0, 1.0];
      this.loBombs = [0, 0];
      this.loBlast = [0, 0];
      this.loSpeed = [1.0, 1.0];
      this.t = 0;
      this.done = false;
      this.winner = null;        // 0 / 1 / null（平局或未结束）
      this.lastCovered = null;   // 最近一次爆炸覆盖掩码（渲染用）
      this.lastDied = [false, false];

      if (this.mode === 'corridor') {
        // 顶部 topWallRows 行全部永久墙
        for (let r = 0; r < CFG.topWallRows; r++) {
          for (let c = 0; c < W; c++) this.wall[r * W + c] = 1;
        }
        // 左右两侧 brick（可通行区 = 中间 corridorWidth 列，[c0, c1)）
        const c0 = (W - CFG.corridorWidth) >> 1;
        const c1 = c0 + CFG.corridorWidth;
        for (let r = CFG.topWallRows; r < H; r++) {
          for (let c = 0; c < W; c++) {
            if (c < c0 || c >= c1) this.brick[r * W + c] = 1;
          }
        }
        // 出生点（cfg.spawn_pos）：空旷区中心
        this.pos[0] = 8.5; this.pos[1] = 5.5;   // (row=8.5, col=5.5)
        this.pos[2] = 8.5; this.pos[3] = 8.5;
        const startB = CFG.growthBombsStart, startZ = CFG.growthBlastStart,
              startS = CFG.growthSpeedStart;
        for (let p = 0; p < 2; p++) {
          this.bombsCap[p] = startB; this.blastCap[p] = startZ; this.spdG[p] = startS;
          this.loBombs[p] = startB; this.loBlast[p] = startZ; this.loSpeed[p] = startS;
        }
      } else {
        // open：无墙无砖
        this.pos[0] = 6.5; this.pos[1] = 4.5;   // (6.5, 4.5)
        this.pos[2] = 6.5; this.pos[3] = 8.5;   // (6.5, 8.5)
        const b = CFG.openGrowthBombs, z = CFG.openGrowthBlast, s = CFG.openGrowthSpeed;
        for (let p = 0; p < 2; p++) {
          this.bombsCap[p] = b; this.blastCap[p] = z; this.spdG[p] = s;
          this.loBombs[p] = b; this.loBlast[p] = z; this.loSpeed[p] = s;
        }
        if (CFG.openCrateCross) this._placeOpenCrossCrates();
      }

      // 位置对称化：约一半对局交换 P0/P1 出生点（消除恒打一侧偏置）
      if (this.rng() < 0.5) {
        const y0 = this.pos[0], x0 = this.pos[1];
        this.pos[0] = this.pos[2]; this.pos[1] = this.pos[3];
        this.pos[2] = y0; this.pos[3] = x0;
      }
      this.sinceBomb[0] = 0; this.sinceBomb[1] = 0;
    }

    _placeOpenCrossCrates() {
      // 中心十字带：行 {cy-1, cy} 全宽 ∪ 列 {cx-1, cx} 全高，扣除出生点及四邻
      const cy = (H - 1) >> 1, cx = (W - 1) >> 1;   // 6, 6
      const excl = new Set();
      const spawns = [[6.5, 4.5], [6.5, 8.5]];
      for (const [rr, cc] of spawns) {
        const r = Math.floor(rr), c = Math.floor(cc);
        excl.add(r * W + c);
        for (const [dr, dc] of DIRS) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < H && nc >= 0 && nc < W) excl.add(nr * W + nc);
        }
      }
      for (let c = 0; c < W; c++) {
        for (const rr of [cy - 1, cy]) {
          if (!excl.has(rr * W + c)) this.crate[rr * W + c] = 1;
        }
      }
      for (let r = 0; r < H; r++) {
        for (const cc of [cx - 1, cx]) {
          if (!excl.has(r * W + cc)) this.crate[r * W + cc] = 1;
        }
      }
    }

    centerCell(p) {
      return [Math.floor(this.pos[p * 2]), Math.floor(this.pos[p * 2 + 1])];
    }

    liveBombs(p) {
      let n = 0;
      for (let i = 0; i < N; i++) if (this.owner[i] === p && this.fuse[i] > 0) n++;
      return n;
    }

    // ------------------------------------------------------- 一个 tick
    // actions: [[move0, bomb0], [move1, bomb1]]
    step(actions) {
      const alive0 = [this.alive[0], this.alive[1]];
      const hpBefore = [this.hp[0], this.hp[1]];
      this.lastDied = [false, false];
      this.lastCovered = new Uint8Array(N);

      // 1. 引信递减
      for (let i = 0; i < N; i++) if (this.fuse[i] > 0) this.fuse[i]--;

      // 2. 放泡（在移动前，落在起始中心格；威力按当前档位快照）
      const placed = [false, false];
      for (let p = 0; p < 2; p++) {
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        const ok = alive0[p] && actions[p][1] === 1 && this.fuse[i] <= 0 &&
          !this.brick[i] && this.liveBombs(p) < this.bombsCap[p];
        if (ok) {
          this.fuse[i] = CFG.fuse;
          this.owner[i] = p;
          this.bombBlast[i] = this.blastCap[p];
          placed[p] = true;
        }
      }
      // 被动计时
      this.sinceBomb[0]++; this.sinceBomb[1]++;
      if (placed[0]) this.sinceBomb[0] = 0;
      if (placed[1]) this.sinceBomb[1] = 0;

      // 3. 连续移动 + AABB 滑动碰撞（速度 = 基础速 × 成长倍率）
      const blocked = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        blocked[i] = this.wall[i] || this.brick[i] || this.fuse[i] > 0 ? 1 : 0;
      }
      for (let p = 0; p < 2; p++) {
        const mv = actions[p][0];
        if (!alive0[p] || mv === MOVE_IDLE) continue;
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        const dist = CFG.stepLen * this.spdG[p];
        const [dy, dx] = DIRS[mv];
        if (dy !== 0) {
          this.pos[p * 2] = resolveAxis(y + dy * dist, dy * dist, x, y, x,
                                        blocked, CFG.radius, H, W, true);
        }
        if (dx !== 0) {
          this.pos[p * 2 + 1] = resolveAxis(x + dx * dist, dx * dist, y, y, x,
                                            blocked, CFG.radius, H, W, false);
        }
        // 边界夹紧（防穿出地图）
        this.pos[p * 2] = Math.min(Math.max(this.pos[p * 2], CFG.radius), H - CFG.radius);
        this.pos[p * 2 + 1] = Math.min(Math.max(this.pos[p * 2 + 1], CFG.radius), W - CFG.radius);
      }

      // 4. 爆炸与连锁（sim/blast.py::resolve_explosions 的标量版）
      const { covered, triggered } = this._resolveExplosions(blocked);
      this.lastCovered = covered;
      // 炸掉的砖 → 宝箱；摧毁砖
      for (let i = 0; i < N; i++) {
        if (covered[i] && this.brick[i]) {
          this.crate[i] = 1;
          this.brick[i] = 0;
        }
      }

      // 5. 伤害判定：移动后的中心格着火 → 扣血（无敌期内不掉血）
      for (let p = 0; p < 2; p++) {
        if (!alive0[p]) continue;
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        const hit = covered[i] && this.invuln[p] <= 0;
        if (hit) {
          this.hp[p] = Math.max(0, this.hp[p] - 1);
          if (this.hp[p] === 0) {
            this.alive[p] = false;
            this.lastDied[p] = true;
          }
        }
      }
      // 无敌期递减（≥0）；实际掉血者重新进入无敌期
      this.invuln[0] = Math.max(0, this.invuln[0] - 1);
      this.invuln[1] = Math.max(0, this.invuln[1] - 1);
      for (let p = 0; p < 2; p++) {
        if (hpBefore[p] > this.hp[p]) this.invuln[p] = CFG.invulnTicks;
      }

      // 6. 清场（引爆的泡归还额度）
      for (let i = 0; i < N; i++) {
        if (triggered[i]) {
          this.fuse[i] = 0; this.owner[i] = -1; this.bombBlast[i] = 0;
        }
      }

      // 6.5 掉血属性惩罚 + 宝箱回收（hit_attr_penalty）
      if (CFG.hitAttrPenalty > 0) {
        for (let p = 0; p < 2; p++) {
          const dmg = hpBefore[p] - this.hp[p];
          if (dmg <= 0 || !alive0[p]) continue;
          const nb = Math.max(this.bombsCap[p] - CFG.hitAttrPenalty, this.loBombs[p]);
          const nz = Math.max(this.blastCap[p] - CFG.hitAttrPenalty, this.loBlast[p]);
          const ns = Math.max(this.spdG[p] - CFG.hitAttrPenalty * CFG.growthSpeedStep,
                              this.loSpeed[p]);
          const lost = (this.bombsCap[p] - nb) + (this.blastCap[p] - nz) +
            Math.round((this.spdG[p] - ns) / CFG.growthSpeedStep);
          this.bombsCap[p] = nb; this.blastCap[p] = nz; this.spdG[p] = ns;
          this._scatterRecycle(p, lost);
        }
      }

      // 7. 计步 + 宝箱拾取成长 + 终局
      this.t++;
      for (let p = 0; p < 2; p++) {
        if (!this.alive[p]) continue;
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        if (!this.crate[i]) continue;
        this.crate[i] = 0;
        const isRecycle = this.recycle[i] === 1;
        this.recycle[i] = 0;
        const prob = isRecycle ? CFG.recycleCrateProb
          : (this.mode === 'open' ? 1.0 : CFG.growthCrateProb);
        if (this.rng() < prob) this._grow(p);
      }

      const nAlive = (this.alive[0] ? 1 : 0) + (this.alive[1] ? 1 : 0);
      if (nAlive <= 1) {
        this.done = true;
        this.winner = nAlive === 1 ? (this.alive[0] ? 0 : 1) : null;
      } else if (this.t >= CFG.maxSteps) {
        this.done = true;
        this.winner = this.hp[0] === this.hp[1] ? null : (this.hp[0] > this.hp[1] ? 0 : 1);
      }
      return { placed, covered, triggered, died: this.lastDied.slice() };
    }

    _grow(p) {
      const attr = Math.floor(this.rng() * 3);   // 0 泡 / 1 威 / 2 速，均匀
      if (attr === 0) this.bombsCap[p] = Math.min(this.bombsCap[p] + 1, CFG.growthBombsMax);
      else if (attr === 1) this.blastCap[p] = Math.min(this.blastCap[p] + 1, CFG.growthBlastMax);
      else this.spdG[p] = Math.min(this.spdG[p] + CFG.growthSpeedStep, CFG.growthSpeedMax);
    }

    _scatterRecycle(p, lost) {
      if (lost <= 0) return;
      // 可通行格（无墙无砖），排除出生点四邻 + open 十字带
      const excl = new Set();
      for (const [rr, cc] of [[6.5, 4.5], [6.5, 8.5], [8.5, 5.5], [8.5, 8.5]]) {
        const r = Math.floor(rr), c = Math.floor(cc);
        excl.add(r * W + c);
        for (const [dr, dc] of DIRS) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < H && nc >= 0 && nc < W) excl.add(nr * W + nc);
        }
      }
      const cy = (H - 1) >> 1, cx = (W - 1) >> 1;
      for (let c = 0; c < W; c++) for (const rr of [cy - 1, cy]) excl.add(rr * W + c);
      for (let r = 0; r < H; r++) for (const cc of [cx - 1, cx]) excl.add(r * W + cc);

      const pool = [];
      for (let i = 0; i < N; i++) {
        if (!this.wall[i] && !this.brick[i] && !excl.has(i)) pool.push(i);
      }
      // 不放回抽样
      for (let k = 0; k < Math.min(lost, pool.length); k++) {
        const j = k + Math.floor(this.rng() * (pool.length - k));
        const tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
        this.crate[pool[k]] = 1;
        this.recycle[pool[k]] = 1;
      }
    }

    // ------------------------------------------------------- 爆炸与连锁
    _rays(sources /*Uint8Array*/, blastMap /*Int16Array*/, brick /*Uint8Array*/) {
      const covered = new Uint8Array(N);
      const bombed = new Uint8Array(N);
      for (let i = 0; i < N; i++) if (this.fuse[i] > 0) bombed[i] = 1;
      for (let s = 0; s < N; s++) {
        if (!sources[s]) continue;
        const sr = (s / W) | 0, sc = s % W;
        covered[s] = 1;
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRS[d];
          let r = sr, c = sc;
          for (let k = 0; k < blastMap[s]; k++) {
            r += dr; c += dc;
            if (r < 0 || r >= H || c < 0 || c >= W) break;
            const i = r * W + c;
            if (this.wall[i]) break;          // 永久墙：不覆盖、不穿透
            covered[i] = 1;
            if (bombed[i] || brick[i]) break; // 泡/砖：覆盖但挡火
          }
        }
      }
      return covered;
    }

    _resolveExplosions() {
      const triggered = new Uint8Array(N);
      let any = false;
      for (let i = 0; i < N; i++) {
        if (this.fuse[i] === 0 && this.owner[i] >= 0) { triggered[i] = 1; any = true; }
      }
      if (!any) {
        return { covered: new Uint8Array(N), triggered };
      }
      // 每颗泡自己的威力（0 回退 cfg.blast）
      const blastMap = new Int16Array(N);
      for (let i = 0; i < N; i++) blastMap[i] = this.bombBlast[i] > 0 ? this.bombBlast[i] : CFG.blast;

      let covered = this._rays(triggered, blastMap, this.brick);
      for (let round = 0; round < CFG.maxChain - 1; round++) {
        const newly = new Uint8Array(N);
        let anyNew = false;
        for (let i = 0; i < N; i++) {
          if (this.fuse[i] > 0 && covered[i] && !triggered[i]) {
            newly[i] = 1; triggered[i] = 1; anyNew = true;
          }
        }
        if (!anyNew) break;
        const more = this._rays(newly, blastMap, this.brick);
        for (let i = 0; i < N; i++) if (more[i]) covered[i] = 1;
      }
      return { covered, triggered };
    }

    // ------------------------------------------------------- 危险图
    // sim/blast.py::danger_map 的标量版：阶段 A 炮格间连锁（组内取最危险），
    // 阶段 B 从每颗炮按自身威力扩散（墙挡、泡/砖挡穿透但覆盖）。
    dangerMap() {
      const out = new Float32Array(N);
      let anyBomb = false;
      for (let i = 0; i < N; i++) if (this.fuse[i] > 0) { anyBomb = true; break; }
      if (!anyBomb) return out;

      const weight = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        if (this.fuse[i] > 0) {
          const wr = Math.max(0, 1 - (this.fuse[i] - 1) / CFG.fuse);
          weight[i] = Math.pow(wr, CFG.dangerExp);
        }
      }
      const blastOf = (i) => this.bombBlast[i] > 0 ? this.bombBlast[i] : CFG.blast;

      // 阶段 A：连锁关系（i 的火焰能覆盖 j）→ 组内取最大权重。
      // 与 rays 同规则：墙挡；泡/砖覆盖后挡穿透。
      const bombs = [];
      for (let i = 0; i < N; i++) if (this.fuse[i] > 0) bombs.push(i);
      const adj = new Map();
      for (const s of bombs) {
        const reach = [];
        const sr = (s / W) | 0, sc = s % W;
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRS[d];
          let r = sr, c = sc;
          for (let k = 0; k < blastOf(s); k++) {
            r += dr; c += dc;
            if (r < 0 || r >= H || c < 0 || c >= W) break;
            const i = r * W + c;
            if (this.wall[i]) break;
            if (this.fuse[i] > 0) reach.push(i);
            if (this.fuse[i] > 0 || this.brick[i]) break;   // 覆盖后不穿透
          }
        }
        adj.set(s, reach);
      }
      // 不动点：eff[j] = max(eff[j], eff[i]) 对所有 i→j（含传递链）
      const eff = Float64Array.from(weight);
      let changed = true;
      while (changed) {
        changed = false;
        for (const [s, reach] of adj) {
          for (const j of reach) {
            if (eff[s] > eff[j]) { eff[j] = eff[s]; changed = true; }
          }
        }
      }

      // 阶段 B：从每颗炮（修正权重）扩散危险范围
      for (const s of bombs) {
        const sr = (s / W) | 0, sc = s % W;
        if (this.wall[s]) continue;                 // 炮在墙格（不可能状态）：torch 的
                                                    // seed 乘了 passable → 全 0 不传播
        out[s] = Math.max(out[s], eff[s]);
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRS[d];
          let r = sr, c = sc;
          for (let k = 0; k < blastOf(s); k++) {
            r += dr; c += dc;
            if (r < 0 || r >= H || c < 0 || c >= W) break;
            const i = r * W + c;
            if (this.wall[i]) break;                    // 墙：无危险
            out[i] = Math.max(out[i], eff[s]);
            if (this.fuse[i] > 0 || this.brick[i]) break;  // 泡/砖：覆盖后不穿透
          }
        }
      }
      return out;
    }

    // ------------------------------------------------------- 观测
    // 14 通道共享观测（与 sim/obs.py::encode_obs 一致）：
    //   0,1 玩家位置(splat) | 2,3 各玩家泡引信 | 4 墙|砖 | 5 危险图 | 6 进度
    //   7 宝箱 | 8,9 无敌标记 | 10,11 可用泡/上限 | 12,13 泡上限/上限档
    encodeObs() {
      const C = 14;
      const o = new Float32Array(C * N);
      const splat = (ch, p) => {
        const yx = this.pos[p * 2];
        if (!this.alive[p]) return;
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        const fy = Math.min(Math.max(y - 0.5, 0), H - 1);
        const fx = Math.min(Math.max(x - 0.5, 0), W - 1);
        const y0 = Math.min(Math.floor(fy), H - 1);
        const x0 = Math.min(Math.floor(fx), W - 1);
        const y1 = Math.min(y0 + 1, H - 1);
        const x1 = Math.min(x0 + 1, W - 1);
        const wy = Math.min(Math.max(fy - y0, 0), 1);
        const wx = Math.min(Math.max(fx - x0, 0), 1);
        o[ch * N + y0 * W + x0] += (1 - wy) * (1 - wx);
        o[ch * N + y0 * W + x1] += (1 - wy) * wx;
        o[ch * N + y1 * W + x0] += wy * (1 - wx);
        o[ch * N + y1 * W + x1] += wy * wx;
      };
      splat(0, 0); splat(1, 1);
      for (let i = 0; i < N; i++) {
        if (this.owner[i] === 0 && this.fuse[i] > 0) o[2 * N + i] = this.fuse[i] / CFG.fuse;
        if (this.owner[i] === 1 && this.fuse[i] > 0) o[3 * N + i] = this.fuse[i] / CFG.fuse;
        if (this.wall[i] || this.brick[i]) o[4 * N + i] = 1;
        if (this.crate[i]) o[7 * N + i] = 1;
      }
      const danger = this.dangerMap();
      for (let i = 0; i < N; i++) o[5 * N + i] = danger[i];
      const tv = this.t / CFG.maxSteps;
      for (let i = 0; i < N; i++) o[6 * N + i] = tv;
      // 无敌标记 + 可用泡/上限（玩家自己的格）
      for (let p = 0; p < 2; p++) {
        if (!this.alive[p]) continue;
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        if (this.invuln[p] > 0) o[(8 + p) * N + i] = 1;
        const cap = Math.max(1, this.bombsCap[p]);
        const avail = Math.max(0, this.bombsCap[p] - this.liveBombs(p));
        o[(10 + p) * N + i] = avail / cap;
        o[(12 + p) * N + i] = this.bombsCap[p] / CFG.growthBombsMax;
      }
      return o;
    }

    // ------------------------------------------------------- 合法动作掩码
    // 方向掩码：探针 4 方向（按 step_len 试探碰撞消解，动了才算合法）；IDLE 恒合法。
    // 放泡掩码：中心格可放泡（can_place）；bomb=0 恒合法。
    legalMask() {
      const mm = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]];
      const bm = [[1, 1], [1, 1]];
      const blocked = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        blocked[i] = this.wall[i] || this.brick[i] || this.fuse[i] > 0 ? 1 : 0;
      }
      for (let p = 0; p < 2; p++) {
        if (!this.alive[p]) continue;
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        const dist = CFG.stepLen;      // 与 torch legal_mask 一致：用基础步长探测
        for (let mv = 0; mv < 4; mv++) {
          const [dy, dx] = DIRS[mv];
          let moved = false;
          if (dy !== 0) {
            const ny = resolveAxis(y + dy * dist, dy * dist, x, y, x,
                                   blocked, CFG.radius, H, W, true);
            moved = Math.abs(ny - y) > EPS * 2;
          } else {
            const nx = resolveAxis(x + dx * dist, dx * dist, y, y, x,
                                   blocked, CFG.radius, H, W, false);
            moved = Math.abs(nx - x) > EPS * 2;
          }
          mm[p][mv] = moved ? 1 : 0;
        }
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        const can = this.fuse[i] <= 0 && !this.brick[i] &&
          this.liveBombs(p) < this.bombsCap[p];
        bm[p][1] = can ? 1 : 0;
      }
      return { mm, bm };
    }
  }

  // ---------------------------------------------------------------- 模型
  // 权重 = deploy/export_ckpt.py 导出的 base64(float32 全部张量) + 偏移表。
  class MLPModel {
    constructor(doc) {
      this.meta = doc.meta;
      this.tensors = doc.tensors;
      this.buf = new Float32Array(decodeB64(doc.flat));
      this.obsShape = doc.meta.obs_shape;
    }

    T(name) {
      const [off, cnt] = this.tensors[name];
      return this.buf.subarray(off, off + cnt);
    }

    // obs: 共享观测 Float32Array(C*H*W)，pid=0 视角已折进权重 → 普通 MLP 前向
    forward(obs) {
      const [C, h, w] = this.obsShape;
      const inDim = C * h * w;
      const W1 = this.T('shared0_w'), b1 = this.T('shared0_b');
      const ln1w = this.T('ln1_w'), ln1b = this.T('ln1_b');
      const W2 = this.T('shared3_w'), b2 = this.T('shared3_b');
      const ln2w = this.T('ln2_w'), ln2b = this.T('ln2_b');

      // x = W1 @ obs + b1 ; LN ; ReLU ; W2 ; LN ; ReLU
      let x = new Float64Array(128);
      for (let i = 0; i < 128; i++) {
        let s = b1[i];
        for (let j = 0; j < inDim; j++) s += W1[i * inDim + j] * obs[j];
        x[i] = s;
      }
      this._lnRelu(x, ln1w, ln1b);
      const h2 = new Float64Array(128);
      for (let i = 0; i < 128; i++) {
        let s = b2[i];
        for (let j = 0; j < 128; j++) s += W2[i * 128 + j] * x[j];
        h2[i] = s;
      }
      this._lnRelu(h2, ln2w, ln2b);

      // 头：Linear(128→64)→ReLU→Linear(64→out)
      const head = (w0, b0, w2, b2, outDim) => {
        const h = new Float64Array(64);
        for (let i = 0; i < 64; i++) {
          let s = b0[i];
          for (let j = 0; j < 128; j++) s += w0[i * 128 + j] * h2[j];
          h[i] = Math.max(0, s);
        }
        const out = new Float64Array(outDim);
        for (let i = 0; i < outDim; i++) {
          let s = b2[i];
          for (let j = 0; j < 64; j++) s += w2[i * 64 + j] * h[j];
          out[i] = s;
        }
        return out;
      };
      return {
        move: head(this.T('move0_w'), this.T('move0_b'),
                   this.T('move2_w'), this.T('move2_b'), 5),
        bomb: head(this.T('bomb0_w'), this.T('bomb0_b'),
                   this.T('bomb2_w'), this.T('bomb2_b'), 2),
      };
    }

    _lnRelu(x, ww, bb) {
      let mean = 0;
      for (let i = 0; i < x.length; i++) mean += x[i];
      mean /= x.length;
      let varSum = 0;
      for (let i = 0; i < x.length; i++) varSum += (x[i] - mean) * (x[i] - mean);
      const std = Math.sqrt(varSum / x.length + 1e-5);
      for (let i = 0; i < x.length; i++) {
        x[i] = Math.max(0, (x[i] - mean) / std * ww[i] + bb[i]);
      }
    }

    // AI 决策：pid 是物理玩家位。模型统一用 pid=0 视角 —— 玩家 1 时观测
    // 先做通道互换把自己搬到通道 0（play/duel.py::_swap_player_channels）。
    act(sim, pid, rng) {
      const obs = sim.encodeObs();
      if (pid === 1) this._swapChannels(obs);
      const { mm, bm } = sim.legalMask();
      const logits = this.forward(obs);
      const aM = this._sampleMasked(logits.move, mm[pid], rng);
      const aB = this._sampleMasked(logits.bomb, bm[pid], rng);
      return [aM, aB];
    }

    _swapChannels(obs) {
      const [C, h, w] = this.obsShape;
      const n = h * w;
      const swap = (a, b) => {
        for (let i = 0; i < n; i++) {
          const t = obs[a * n + i];
          obs[a * n + i] = obs[b * n + i];
          obs[b * n + i] = t;
        }
      };
      swap(0, 1); swap(2, 3);
      if (C > 7) { swap(8, 9); swap(10, 11); swap(12, 13); }
    }

    _sampleMasked(logits, mask, rng) {
      const negInf = -1e30;
      let max = -Infinity;
      const n = logits.length;
      for (let i = 0; i < n; i++) {
        const v = mask[i] ? logits[i] : negInf;
        if (v > max) max = v;
      }
      let sum = 0;
      const probs = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        probs[i] = mask[i] ? Math.exp(logits[i] - max) : 0;
        sum += probs[i];
      }
      if (sum <= 0) return 0;   // 全掩码兜底（IDLE/不放恒合法，正常不会到这）
      let r = rng() * sum;
      for (let i = 0; i < n; i++) {
        r -= probs[i];
        if (r <= 0) return i;
      }
      return n - 1;
    }
  }

  // ---------------------------------------------------------------- 规则 AI
  // sim/bots.py::astar_attack(eat_crates=True) 的标量移植 —— 启动器的「纯进攻
  // 寻路 hunter」：恒 aggressive（逼近+放泡），成长属性 ≥2 项不满 70% 且有
  // 宝箱时优先寻路吃箱补属性（适合 corridor）。13×13 每 tick 决策一次，
  // 169 格 Dijkstra 毫秒级，浏览器无压力。
  class HunterAI {
    constructor() {
      // 独立 rng：噪声用（Python 侧是 torch.rand 独立生成器），不污染 sim 的
      // 步进随机序列（crate 概率等）。
      this.rng = mulberry32(0xC0FFEE);
    }

    // 多源 Dijkstra 价值场（Bellman-Ford）：dist[j] = min over i∈4邻 (dist[i]+cost[j])。
    // source 为 1 的格是源（dist=0）；cost = 1 + 2×danger，blocked 格 cost=∞。
    _dijkstra(source, danger, blocked) {
      const inf = Infinity;
      const dist = new Float64Array(N);
      const cost = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        dist[i] = source[i] ? 0 : inf;
        cost[i] = blocked[i] ? inf : 1 + 2 * danger[i];
      }
      let changed = true;
      let passes = 0;
      while (changed && passes < N) {
        changed = false;
        passes++;
        for (let i = 0; i < N; i++) {
          if (dist[i] === inf) continue;
          const r = (i / W) | 0, c = i % W;
          for (let d = 0; d < 4; d++) {
            const nr = r + DIRS[d][0], nc = c + DIRS[d][1];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const j = nr * W + nc;
            if (cost[j] === inf) continue;
            const nd = dist[i] + cost[j];
            if (nd < dist[j]) { dist[j] = nd; changed = true; }
          }
        }
      }
      return dist;
    }

    // hunter 决策：返回 [move, bomb]（与 MLPModel.act 同签名）。
    // pid 是物理玩家位；策略直接读 sim 状态（规则 AI 不需要视角置换）。
    act(sim, pid) {
      const danger = sim.dangerMap();
      const own = sim.centerCell(pid);            // [r, c]
      const ownIdx = own[0] * W + own[1];
      const ownDng = danger[ownIdx];
      const r = sim.pos[pid * 2], c = sim.pos[pid * 2 + 1];
      const rIdx = Math.min(Math.max(Math.floor(r), 0), H - 1);
      const cIdx = Math.min(Math.max(Math.floor(c), 0), W - 1);

      // 威胁区：自己名下在场泡的爆炸范围（膨胀 growthBlastMax 次，绕行惩罚）
      const threat = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        if (sim.owner[i] === pid && sim.fuse[i] > 0) threat[i] = 1;
      }
      for (let k = 0; k < CFG.growthBlastMax; k++) {
        const nb = new Uint8Array(N);
        for (let i = 0; i < N; i++) {
          if (!threat[i]) continue;
          const rr = (i / W) | 0, cc = i % W;
          for (let d = 0; d < 4; d++) {
            const nr = rr + DIRS[d][0], nc = cc + DIRS[d][1];
            if (nr >= 0 && nr < H && nc >= 0 && nc < W) nb[nr * W + nc] = 1;
          }
        }
        threat.set(nb);
      }

      // 障碍 = 墙 | 砖 | 在场泡（与 legal_mask 的 blocked 一致）
      const blocked = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        blocked[i] = sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 ? 1 : 0;
      }

      // 两个基础场：逃生场（安全格为源）+ 逼近场（对手为源）
      const safeSrc = new Uint8Array(N);
      for (let i = 0; i < N; i++) safeSrc[i] = danger[i] < 0.35 ? 1 : 0;
      const V_safe = this._dijkstra(safeSrc, danger, blocked);
      const oppSrc = new Uint8Array(N);
      for (let o = 0; o < 2; o++) {
        if (o === pid || !sim.alive[o]) continue;
        const [ro, co] = sim.centerCell(o);
        oppSrc[ro * W + co] = 1;
      }
      const V_opp = this._dijkstra(oppSrc, danger, blocked);

      // 5 个候选动作的目标格（4 方向 + IDLE = 当前格）
      const cells = new Int32Array(5);
      const dngC = new Float64Array(5), thrC = new Float64Array(5);
      for (let mv = 0; mv < 4; mv++) {
        let nr = rIdx + DIRS[mv][0], nc = cIdx + DIRS[mv][1];
        nr = Math.min(Math.max(nr, 0), H - 1);
        nc = Math.min(Math.max(nc, 0), W - 1);
        cells[mv] = nr * W + nc;
      }
      cells[MOVE_IDLE] = ownIdx;
      for (let mv = 0; mv < 5; mv++) {
        dngC[mv] = danger[cells[mv]];
        thrC[mv] = threat[cells[mv]];
      }
      const noise = [];
      for (let mv = 0; mv < 5; mv++) noise.push(0.05 * this.rng());

      // 合法掩码（已含泡挡路），hunter 额外排除砖墙格（与 Python blk_c 一致）
      const { mm, bm } = sim.legalMask();
      const legal = [];
      for (let mv = 0; mv < 5; mv++) {
        legal.push(mm[pid][mv] === 1 &&
          !(sim.wall[cells[mv]] || sim.brick[cells[mv]]));
      }

      // --- 吃道具层（hunter 专属，高优先级）---
      // 成长属性 ≥2 项低于上限 70% 且场上有宝箱 → 朝最近宝箱寻路（走路踩箱
      // 升级），优先级高于逼近、低于逃生（命要紧）。
      const bMax = CFG.growthBombsMax, zMax = CFG.growthBlastMax,
            sMax = CFG.growthSpeedMax;
      const fracs = [sim.bombsCap[pid] / bMax, sim.blastCap[pid] / zMax,
                     sim.spdG[pid] / sMax];
      const hungry = fracs.filter((f) => f < 0.7).length >= 2;
      let hasCrate = false;
      for (let i = 0; i < N; i++) if (sim.crate[i]) { hasCrate = true; break; }
      let eatOn = hungry && hasCrate && sim.alive[pid];

      const app = [], esc = [], eat = [];
      const inf = Infinity;
      const VoppC = [], VsafeC = [];
      for (let mv = 0; mv < 5; mv++) {
        VoppC.push(V_opp[cells[mv]]);
        VsafeC.push(V_safe[cells[mv]]);
      }
      let V_crate = null, VcrateC = null;
      if (eatOn) {
        const crateSrc = new Uint8Array(N);
        for (let i = 0; i < N; i++) if (sim.crate[i]) crateSrc[i] = 1;
        V_crate = this._dijkstra(crateSrc, danger, blocked);
        VcrateC = [];
        for (let mv = 0; mv < 5; mv++) VcrateC.push(V_crate[cells[mv]]);
      }

      for (let mv = 0; mv < 5; mv++) {
        // 逼近：沿 V_opp 最速下降；dng≥0.5（快爆的泡）禁行
        let a = VoppC[mv] + 2 * dngC[mv] + 2 * thrC[mv] + noise[mv];
        if (dngC[mv] >= 0.5 || !legal[mv]) a = inf;
        app.push(a);
        // 逃生：只按 Vsafe（安全距离），顺带绕威胁；脚下危险时禁止停（IDLE）
        let e = VsafeC[mv] * 100 + 2 * thrC[mv] + noise[mv];
        if (!legal[mv]) e = inf;
        esc.push(e);
        // 吃箱：朝最近宝箱，避险（dng≥0.5 禁行）
        let t = eatOn
          ? (VcrateC[mv] * 100 + 2 * dngC[mv] + noise[mv]) : inf;
        if (eatOn && (dngC[mv] >= 0.5 || !legal[mv])) t = inf;
        eat.push(t);
      }
      // 吃箱禁停：四周有更优非停方向就不许停（防原地发呆被炸）
      if (eatOn) {
        let eatNoIdle = inf;
        for (let mv = 0; mv < 4; mv++) eatNoIdle = Math.min(eatNoIdle, eat[mv]);
        if (eatNoIdle < eat[MOVE_IDLE]) eat[MOVE_IDLE] = inf;
        // 吃箱路径必须可达（全 inf 时回退逼近）
        let eatMin = inf;
        for (let mv = 0; mv < 5; mv++) eatMin = Math.min(eatMin, eat[mv]);
        if (!isFinite(eatMin)) { eatOn = false; eat.fill(inf); }
      }

      // 逼近无路 / 脚下危险 → 转逃生
      let appOk = false;
      for (let mv = 0; mv < 5; mv++) if (isFinite(app[mv])) appOk = true;
      const useEsc = ownDng >= 0.35 || !appOk;
      if (useEsc) {
        // 逃生禁停：停 = 留在必受伤的格（被围死时 fallback 兜底允许）
        esc[MOVE_IDLE] = inf;
      }

      // 合并打分：逃生 > 吃箱 > 逼近（逃生最优先）
      const score = [];
      for (let mv = 0; mv < 5; mv++) {
        score.push(useEsc ? esc[mv] : (eatOn ? eat[mv] : app[mv]));
      }
      let move = MOVE_IDLE;
      let best = inf;
      for (let mv = 0; mv < 5; mv++) {
        if (score[mv] < best) { best = score[mv]; move = mv; }
      }
      // 终极兜底：打分全 inf（被泡/火完全围死）→ 选最小 danger 的合法格；
      // 优先非停方向（走比等死强），全方向更危险才允许停
      let anyLegal = false;
      for (let mv = 0; mv < 5; mv++) if (legal[mv]) { anyLegal = true; break; }
      let noPath = true;
      for (let mv = 0; mv < 5; mv++) if (isFinite(score[mv])) { noPath = false; break; }
      if (noPath && anyLegal) {
        let bm = MOVE_IDLE, bs = inf, bm2 = MOVE_IDLE, bs2 = inf;
        for (let mv = 0; mv < 5; mv++) {
          if (!legal[mv]) continue;
          if (dngC[mv] < bs2) { bs2 = dngC[mv]; bm2 = mv; }      // 含停兜底
          if (mv !== MOVE_IDLE && dngC[mv] < bs) { bs = dngC[mv]; bm = mv; }
        }
        move = isFinite(bs) ? bm : bm2;
      }

      // --- 放泡：十字内打得到对手 / 连锁老泡 / 近身压制，且不自爆、能撤 ---
      const cap = sim.blastCap[pid];
      const orow = [], ocol = [];
      for (let o = 0; o < 2; o++) {
        if (o === pid || !sim.alive[o]) continue;
        const [ro, co] = sim.centerCell(o);
        orow.push(ro); ocol.push(co);
      }
      let alignedOpp = false, manh = 1e9;
      for (let o = 0; o < orow.length; o++) {
        const drO = Math.abs(rIdx - orow[o]), dcO = Math.abs(cIdx - ocol[o]);
        if ((drO === 0 && dcO <= cap) || (dcO === 0 && drO <= cap)) alignedOpp = true;
        manh = Math.min(manh, drO + dcO);
      }
      const nearOpp = manh <= cap + 1;
      // 连锁：附近（blast 十字内）有引信 ≤10 的老泡可"时间差连锁"
      let chain = false;
      for (let k = 1; k <= cap && !chain; k++) {
        for (let d = 0; d < 4; d++) {
          const nr = rIdx + DIRS[d][0] * k, nc = cIdx + DIRS[d][1] * k;
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const f = sim.fuse[nr * W + nc];
          if (f > 0 && f <= 10) { chain = true; break; }
        }
      }
      // 能撤：有合法且低危险且非威胁的目标格
      let canEscape = false;
      for (let mv = 0; mv < 5; mv++) {
        if (legal[mv] && dngC[mv] < 0.35 && !thrC[mv]) { canEscape = true; break; }
      }
      let bomb = 0;
      if (bm[pid][1] === 1 && (alignedOpp || nearOpp || chain) &&
          ownDng < 0.2 && canEscape && sim.alive[pid]) {
        bomb = 1;
      }
      if (!sim.alive[pid]) { move = MOVE_IDLE; bomb = 0; }
      return [move, bomb];
    }
  }

  const QQT = {
    H, W, N, N_PLAYERS, N_MOVES, N_BOMB,
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_IDLE,
    DIRS, EPS, CFG,
    Sim, MLPModel, HunterAI, mulberry32, resolveAxis, decodeB64,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = QQT;
  else root.QQT = QQT;
})(typeof globalThis !== 'undefined' ? globalThis : this);
