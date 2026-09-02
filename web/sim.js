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
  // 全部地图(含空场景)统一 15×13（QQ堂竞技标准尺寸，241 张原版图）
  const H = 13, W = 15, N = H * W;
  const PUSH_TIME = 0.3;   // 推箱子: 持续推 ≥0.3s 才动一格
  const BUSH_EID = 6003;   // 野外绿色躲猫猫草丛（可通行、可被爆炸清除）
  const ORT_RUN_TIMEOUT_MS = 1500;  // WebGPU 卡住时及时回退纯 JS，不能阻塞游戏 tick
  const N_PLAYERS = 2;
  const MOVE_UP = 0, MOVE_DOWN = 1, MOVE_LEFT = 2, MOVE_RIGHT = 3, MOVE_IDLE = 4;
  const N_MOVES = 5, N_BOMB = 2;
  // (dy, dx)，索引与方向编码对齐
  const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const EPS = 1e-4;

  // duel.py 的配置（map_mode=corridor + open_fraction=1.0 即 open 关；
  // corridor 关用 growth_*_start 起步 + 顶墙/侧砖）。
  const CFG = {
    tickHz: 10, speed: 3.0, radius: 0.36, maxSteps: 1800,
    fuse: 30, blast: 2, maxBombs: 10, maxChain: 16,
    blastLingerTicks: 3,                 // 爆炸后余威 0.3s（10Hz）
    maxHp: 5, invulnTicks: 30,
    stepLen: 3.0 / 10,                 // 0.3 格/tick
    // 成长（corridor 起步）
    growthBombsStart: 2, growthBlastStart: 2, growthSpeedStart: 1.3,
    growthBombsMax: 10, growthBlastMax: 7,
    growthSpeedMax: 2.1, growthSpeedStep: 0.8 / 7,   // 7档: (2.1-1.3)/7 ≈ 0.1143
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
  // （“脚下刚放的泡必须能走出去”：压着就放行 —— 这也是 QQ堂“穿炮穿墙”
  // 的源头：盒子(跨度)压着障碍格就能穿过去，Feature 保留）。
  function impassable(blocked, row, col, y, x, rad, h, w) {
    if (row < 0 || row >= h || col < 0 || col >= w) return true;
    if (!blocked[row * w + col]) return false;
    // 豁免 = 盒子**身体真正进入**该格(上边界用 ceil, 严格大于):
    // 底/右边线刚好贴住格边界(如 y+R=3.0 贴着行3)不算压着 → 炸弹/墙照常阻挡,
    // 否则"角刚好擦到炸弹"会被误判可走, 自动转向永远不触发。
    const r0 = Math.floor(y - rad), r1 = Math.ceil(y + rad);
    const c0 = Math.floor(x - rad), c1 = Math.ceil(x + rad);
    const inside = row >= r0 && row < r1 && col >= c0 && col < c1;
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
    const stopPos = sgn > 0 ? first - CFG.radius - EPS : first + 1 + CFG.radius + EPS;
    return has ? stopPos : coord;
  }

  // ---------------------------------------------------------------- 模拟器
  class Sim {
    constructor(seed) {
      this.rng = mulberry32(seed == null ? 1 : seed);
      this.reset('open');
    }

    // mode: 'open'（纯空场 + 中心十字宝箱 + 80% 起步）| 'corridor'（顶墙侧砖 + 2/2/1 起步）
    //       | 关卡对象（level: {wall, brick, spawns, initial_stats, crate_rate, ...}）
    // opts.oldMode: 旧 13x13 模型兼容 —— 地图加载后把第 13/14 列填为不可通行墙,
    //               encodeObs 按 13 宽输出(与旧版训练环境一致)。
    reset(mode, opts) {
      this.oldMode = !!(opts && opts.oldMode);
      // 可推箱运行时状态(所有模式都初始化; 关卡模式在 _loadLevel 填充)
      this.pushable = new Uint8Array(N);
      this.pushT = new Float32Array(N);
      this.pushBoxAt = new Int32Array(N).fill(-1);
      this.pushSprite = new Int32Array(N).fill(-1);
      this.pushBoxes = [];
      const isLevel = mode !== null && typeof mode === 'object';
      this.mode = isLevel ? 'level' : (mode === 'corridor' ? 'corridor' : 'open');
      this.level = isLevel ? mode : null;
      this._gen = (this._gen || 0) + 1;   // 代际计数：模型每 tick 缓存失效用
      this.wall = new Uint8Array(N);
      this.brick = new Uint8Array(N);
      this.cover = new Uint8Array(N);      // 房子: 可通行 + 不可炸 + 藏人
      this.bush = new Uint8Array(N);       // 灌木: 可通行 + 可炸 + 藏人
      this.crate = new Uint8Array(N);
      this.superCrate = new Uint8Array(N);   // 1=超级宝箱(拾取+4档)
      this.crateType = new Int8Array(N);    // 宝箱种类: -1=随机(问号), 0/1/2=泡/威/速(炸开时定)
      this.recycle = new Uint8Array(N);
      this.fuse = new Int16Array(N);
      this.owner = new Int8Array(N);
      this.owner.fill(-1);
      this.bombBlast = new Int16Array(N);
      this.blastLinger = new Int8Array(N);
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
      this.lastReplayCovered = null;
      this.lastReplayTriggered = null;
      this.lastReplayPlaced = [false, false];
      this.spawnCells = null;    // 本局出生点（回收排除用）
      // 属性上限默认 = CFG 上限（open/corridor；关卡模式在 _loadLevel 覆盖）
      this.bombsMax = CFG.growthBombsMax;
      this.blastMax = CFG.growthBlastMax;
      this.speedMax = CFG.growthSpeedMax;

      if (isLevel) {
        this._loadLevel(mode);
      } else if (this.mode === 'corridor') {
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

    // Serializable logical state used by replay export. Typed arrays are copied to
    // plain arrays so the JSON document is independent from the live simulation.
    snapshotReplay(info) {
      const arr = (v) => v == null ? null : Array.from(v);
      return {
        t: this.t,
        pos: arr(this.pos),
        alive: this.alive.slice(),
        hp: this.hp.slice(),
        invuln: this.invuln.slice(),
        sinceBomb: this.sinceBomb.slice(),
        bombsCap: this.bombsCap.slice(),
        blastCap: this.blastCap.slice(),
        spdG: this.spdG.slice(),
        loBombs: this.loBombs.slice(),
        loBlast: this.loBlast.slice(),
        loSpeed: this.loSpeed.slice(),
        wall: arr(this.wall),
        brick: arr(this.brick),
        cover: arr(this.cover),
        bush: arr(this.bush),
        crate: arr(this.crate),
        superCrate: arr(this.superCrate),
        crateType: arr(this.crateType),
        recycle: arr(this.recycle),
        fuse: arr(this.fuse),
        owner: arr(this.owner),
        bombBlast: arr(this.bombBlast),
        blastLinger: arr(this.blastLinger),
        pushable: arr(this.pushable),
        pushT: arr(this.pushT),
        pushBoxAt: arr(this.pushBoxAt),
        pushSprite: arr(this.pushSprite),
        pushBoxes: this.pushBoxes.map((b) => ({
          o: b.o, cells: b.cells.slice(), eid: b.eid, dead: !!b.dead,
        })),
        done: this.done,
        winner: this.winner,
        lastDied: this.lastDied.slice(),
        covered: info && info.covered ? arr(info.covered) : arr(this.lastReplayCovered),
        triggered: info && info.triggered ? arr(info.triggered) : arr(this.lastReplayTriggered),
        died: info && info.died ? info.died.slice() : this.lastDied.slice(),
        placed: info && info.placed ? info.placed.slice() : this.lastReplayPlaced.slice(),
      };
    }

    // Restore a frame for deterministic replay rendering; it intentionally does
    // not advance the simulation or consume its random stream.
    restoreReplay(frame) {
      const copy = (name, Type) => {
        if (frame[name] == null) return;
        this[name] = Type ? new Type(frame[name]) : frame[name].slice();
      };
      copy('pos', Float64Array);
      this.alive = frame.alive.slice();
      this.hp = frame.hp.slice();
      this.invuln = frame.invuln.slice();
      this.sinceBomb = frame.sinceBomb.slice();
      this.bombsCap = frame.bombsCap.slice();
      this.blastCap = frame.blastCap.slice();
      this.spdG = frame.spdG.slice();
      this.loBombs = frame.loBombs.slice();
      this.loBlast = frame.loBlast.slice();
      this.loSpeed = frame.loSpeed.slice();
      for (const name of ['wall', 'brick', 'cover', 'bush', 'crate', 'superCrate',
        'recycle', 'fuse', 'owner', 'bombBlast', 'blastLinger', 'pushable', 'pushT', 'pushBoxAt',
        'pushSprite']) {
        const old = this[name];
        if (frame[name] == null) continue;
        this[name] = new old.constructor(frame[name]);
      }
      if (frame.blastLinger == null) this.blastLinger.fill(0);
      if (frame.crateType != null) this.crateType = new Int8Array(frame.crateType);
      this.pushBoxes = (frame.pushBoxes || []).map((b) => ({
        o: b.o, cells: b.cells.slice(), eid: b.eid, dead: !!b.dead,
      }));
      this.t = frame.t;
      this.done = !!frame.done;
      this.winner = frame.winner == null ? null : frame.winner;
      this.lastDied = (frame.lastDied || frame.died || [false, false]).slice();
      this.lastReplayCovered = frame.covered ? new Uint8Array(frame.covered) : null;
      this.lastReplayTriggered = frame.triggered ? new Uint8Array(frame.triggered) : null;
      this.lastReplayPlaced = (frame.placed || [false, false]).slice();
      this.lastCovered = this.lastReplayCovered;
      this._gen = (this._gen || 0) + 1;
      return this;
    }

    // 加载一张新地图关卡（web/assets/maps/levels.json 导出的 241 张之一）：
    //   wall/brick 通行性、出生点随机二选、初始属性按地图配置、炸砖爆率按地图
    //   crate_rate（= 57/W，保证全图砖清完 ≈300% 单人满属性）、空场景中心十字宝箱。
    _loadLevel(level) {
      for (let i = 0; i < N; i++) {
        this.wall[i] = level.wall[i];
        this.brick[i] = level.brick[i];
        if (level.cover) this.cover[i] = level.cover[i];
        if (level.bush) this.bush[i] = level.bush[i];
        // 旧版 Web 地图导出可能只有 layers，没有同步 bush 布尔层。
        // 野外 6003 是可通行且可炸的躲猫猫草丛，运行时必须补回状态层。
        if (!this.bush[i] && level.layers && Math.abs(level.layers[0][i] || 0) === BUSH_EID) {
          this.bush[i] = 1;
        }
      }
      // 旧模型兼容: 多出的两列(13,14)填为不可通行墙 → 环境与旧版 13 宽一致
      if (this.oldMode) {
        for (let r = 0; r < H; r++) {
          this.wall[r * W + W - 1] = 1;
          this.wall[r * W + W - 2] = 1;
        }
        // 兼容墙可能覆盖关卡自带的初始道具；墙体生成后统一清掉重叠项。
        this._clearBlockedCrates();
      }
      // 可推箱运行时状态: push_boxes [[r,c,w,h]...] → 足迹格/推动计时/精灵
      const pb = level.push_boxes || [];
      for (const [r, c, bw, bh] of pb) {
        const o = r * W + c;
        const cells = [];
        for (let dr = 0; dr < bh; dr++) {
          for (let dc = 0; dc < bw; dc++) cells.push((r + dr) * W + (c + dc));
        }
        const bi = this.pushBoxes.length;
        for (const cell of cells) {
          this.pushable[cell] = 1;
          this.pushBoxAt[cell] = bi;
        }
        this.pushSprite[o] = level.layers && level.layers[1] ? Math.abs(level.layers[1][o]) : 0;
        this.pushBoxes.push({ o, cells, eid: this.pushSprite[o] || 0, dead: false });
        this.pushT[o] = 0;
      }
      // 初始属性（比武/不足300% → 3/3/1.2；普通 → 2/2/1.2；空场景 → 8/6/1.68）
      const st = level.initial_stats || { bombs: 2, blast: 2, speed: 1.0 };
      for (let p = 0; p < 2; p++) {
        this.bombsCap[p] = st.bombs;
        this.blastCap[p] = st.blast;
        this.spdG[p] = st.speed;
        this.loBombs[p] = st.bombs;
        this.loBlast[p] = st.blast;
        this.loSpeed[p] = st.speed;
      }
      // 出生点：从地图出生点列表里随机挑两个（打乱后取前二，保证不同开局）
      const sp = level.spawns.map((s) => [s[0], s[1]]);
      for (let i = sp.length - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        const t = sp[i]; sp[i] = sp[j]; sp[j] = t;
      }
      const s0 = sp[0], s1 = sp.length > 1 ? sp[1] : sp[0];
      this.pos[0] = s0[0] + 0.5; this.pos[1] = s0[1] + 0.5;
      this.pos[2] = s1[0] + 0.5; this.pos[3] = s1[1] + 0.5;
      this.spawnCells = sp;
      // 炸砖 → 宝箱的爆率（地图配置；空场景无砖，crate_rate=0 不走炸砖路径）
      this.crateRate = (level.crate_rate != null && level.crate_rate > 0)
        ? level.crate_rate : 1.0;
      // 宝箱中超级占比（超级威力/泡泡/速度 整体 = 普通爆率的 10% -> 1/11）
      this.superFraction = level.crate_super_fraction || 0;
      // 属性上限标准化在地图文件里（比武图泡/威上限=7, 空场景速上限=2.2）
      this.bombsMax = level.bombs_max || CFG.growthBombsMax;
      this.blastMax = level.blast_max || CFG.growthBlastMax;
      this.speedMax = level.speed_max || CFG.growthSpeedMax;
      // 空场景：开局中心十字宝箱直接放地上（踩到必升）
      if (level.initial_crates && level.initial_crates.length) {
        for (const [r, c] of level.initial_crates) {
          const i = r * W + c;
          if (!this._crateBlocked(i)) {
            this.crate[i] = 1;
            this.crateType[i] = -1;   // 空场景随机宝箱(问号)
          }
        }
      }
      // 初始宝箱写入发生在兼容墙生成之后；再次清理可防止地图数据越界覆盖。
      this._clearBlockedCrates();
    }

    _crateBlocked(i) {
      return i < 0 || i >= N || this.wall[i] || this.brick[i] || this.pushable[i];
    }

    _clearBlockedCrates() {
      for (let i = 0; i < N; i++) {
        if (!this._crateBlocked(i)) continue;
        this.crate[i] = 0;
        this.superCrate[i] = 0;
        this.recycle[i] = 0;
        this.crateType[i] = -1;
      }
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
          const i = rr * W + c;
          if (!excl.has(i) && !this._crateBlocked(i)) this.crate[i] = 1;
        }
      }
      for (let r = 0; r < H; r++) {
        for (const cc of [cx - 1, cx]) {
          const i = r * W + cc;
          if (!excl.has(i) && !this._crateBlocked(i)) this.crate[i] = 1;
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
      this.lastReplayCovered = new Uint8Array(N);
      this.lastReplayTriggered = new Uint8Array(N);
      this.lastReplayPlaced = [false, false];

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
      // Keep event masks with the logical frame for deterministic replay.
      this.lastReplayPlaced = placed.slice();
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
                // 推箱子: 前缘顶着可推箱 → 累计推动时间(每tick 0.1s), ≥PUSH_TIME 后箱子移一格
        if (dy !== 0 || dx !== 0) {
          const pr = dy !== 0 ? (dy > 0 ? Math.floor(y + CFG.radius + EPS * 8) : Math.floor(y - CFG.radius - EPS * 8)) : Math.floor(y);
          const pc = dx !== 0 ? (dx > 0 ? Math.floor(x + CFG.radius + EPS * 8) : Math.floor(x - CFG.radius - EPS * 8)) : Math.floor(x);
          const pi = pr * W + pc;
          const bi = pi >= 0 && pi < N ? this.pushBoxAt[pi] : -1;
          if (bi >= 0) {
            const box = this.pushBoxes[bi];
            let ok = true;
            const targetCells = [];
            for (const cell of box.cells) {
              const rr = (cell / W) | 0, cc = cell % W;
              const tr = rr + dy, tc = cc + dx;
              if (tr < 0 || tr >= H || tc < 0 || tc >= W) { ok = false; break; }
              const ti = tr * W + tc;
              // 目标格必须全空: 无墙/砖/泡/道具(宝箱)/其他箱子
              if (this.wall[ti] || this.brick[ti] || this.fuse[ti] > 0 || this.crate[ti] || this.pushable[ti]) { ok = false; break; }
              targetCells.push(ti);
            }
            if (ok) {
              this.pushT[box.o] += 0.1;
              if (this.pushT[box.o] >= PUSH_TIME) {
                for (let k = 0; k < box.cells.length; k++) {
                  const ci = box.cells[k], ti = targetCells[k];
                  this.brick[ci] = 0; this.brick[ti] = 1;
                  this.pushable[ci] = 0; this.pushable[ti] = 1;
                  this.pushBoxAt[ci] = -1; this.pushBoxAt[ti] = bi;
                  this.pushSprite[ti] = this.pushSprite[ci]; this.pushSprite[ci] = -1;
                }
                box.cells = targetCells;
                box.o = targetCells[0];
                this.pushT[box.o] = 0;
              }
            } else {
              this.pushT[box.o] = 0;   // 推不动(目标被挡) → 重置计时
            }
          }
        }
        // 中心路径硬约束 + 贪婪转向：模型输出=目标相邻格，直走被挡自动试垂直方向
        const [ny, nx] = this._steer(y, x, mv, blocked, dist);
        this.pos[p * 2] = ny;
        this.pos[p * 2 + 1] = nx;
        // 边界夹紧（防穿出地图）
        this.pos[p * 2] = Math.min(Math.max(this.pos[p * 2], CFG.radius), H - CFG.radius);
        this.pos[p * 2 + 1] = Math.min(Math.max(this.pos[p * 2 + 1], CFG.radius), W - CFG.radius);
      }

      // 4. 爆炸与连锁（sim/blast.py::resolve_explosions 的标量版）
      const { covered, triggered } = this._resolveExplosions(blocked);
      this.lastCovered = covered;
      this.lastReplayCovered = covered;
      this.lastReplayTriggered = triggered;
      // 炸掉的砖 → 宝箱（按地图爆率 crate_rate 判定；摧毁砖）
      // 灌木(bush)：可通行 + **可炸毁** —— 被火焰覆盖即摧毁（不给宝箱）
      for (let i = 0; i < N; i++) {
        if (covered[i] && this.brick[i]) {
          this.brick[i] = 0;
          // 被炸的可推箱: 整箱移除(足迹清空)
          const biX = this.pushBoxAt[i];
          if (biX >= 0 && !this.pushBoxes[biX].dead) {
            const bx = this.pushBoxes[biX];
            bx.dead = true;
            for (const cell of bx.cells) {
              this.pushable[cell] = 0;
              this.pushBoxAt[cell] = -1;
              this.pushSprite[cell] = -1;
            }
          }
          if (this.rng() < this.crateRate && !this._crateBlocked(i)) {
            this.crate[i] = 1;
            this.superCrate[i] = this.rng() < this.superFraction ? 1 : 0;
            this.crateType[i] = Math.floor(this.rng() * 3);   // 吃到啥在炸开时定
          }
        }
        if (covered[i] && this.bush[i]) this.bush[i] = 0;
      }

      // 5. 伤害判定：当前爆炸 + 之前 0.3s 余威覆盖的中心格均可扣血。
      //    余威只在后续 3 tick 生效；无敌期防止重复掉血。
      for (let p = 0; p < 2; p++) {
        if (!alive0[p]) continue;
        const [r, c] = this.centerCell(p);
        const i = r * W + c;
        const hit = (covered[i] || this.blastLinger[i] > 0) && this.invuln[p] <= 0;
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

      // 当前爆炸已经在本 tick 结算过；把覆盖范围续留到后续 tick。
      // 逐格计时可正确处理不同时间发生、范围重叠的多次爆炸。
      for (let i = 0; i < N; i++) {
        if (this.blastLinger[i] > 0) this.blastLinger[i]--;
        if (covered[i]) this.blastLinger[i] = CFG.blastLingerTicks;
      }

      // 6. 清场（引爆的泡归还额度）
      for (let i = 0; i < N; i++) {
        if (triggered[i]) {
          this.fuse[i] = 0; this.owner[i] = -1; this.bombBlast[i] = 0;
        }
      }

      // 6.5 掉血属性惩罚 + 宝箱回收（每项扣 clamp(round(25%×当前值), 1, 2) 档）
      if (CFG.hitAttrPenalty > 0) {
        for (let p = 0; p < 2; p++) {
          const dmg = hpBefore[p] - this.hp[p];
          if (dmg <= 0 || !alive0[p]) continue;
          // 扣减档数: min 1 档, max 2 档, 中间 = 现在属性的 25%
          //   泡 = 当前泡上限的 25%; 威 = 当前威力的 25%;
          //   速 = 当前速度相对起点档数的 25%
          const lossOf = (cur, start, step) =>
            Math.min(2, Math.max(1, Math.round(0.25 * (cur - start) / step)));
          const lb = lossOf(this.bombsCap[p], this.loBombs[p], 1);
          const lz = lossOf(this.blastCap[p], this.loBlast[p], 1);
          const ls = lossOf(this.spdG[p], this.loSpeed[p], CFG.growthSpeedStep);
          const nb = Math.max(this.bombsCap[p] - lb, this.loBombs[p]);
          const nz = Math.max(this.blastCap[p] - lz, this.loBlast[p]);
          const ns = Math.max(this.spdG[p] - ls * CFG.growthSpeedStep, this.loSpeed[p]);
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
        const isSuper = this.superCrate[i] === 1;   // 超级宝箱 +4 档
        const fAttr = this.crateType[i];            // 炸开时定好的种类(-1=随机)
        this.recycle[i] = 0;
        this.superCrate[i] = 0;
        this.crateType[i] = -1;
        // 爆率已在"炸砖→生箱"时判定过，踩到必升（种类提前定，见 _grow）
        if (this.rng() < 1.0) this._grow(p, isSuper, fAttr >= 0 ? fAttr : null);
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

    _grow(p, isSuper, attrFixed) {
      const add = isSuper ? 4 : 1;                 // 超级宝箱 +4 档
      // 种类：炸开时已定(0/1/2)则用定好的；随机宝箱(问号)踩到才掷
      const attr = attrFixed != null ? attrFixed : Math.floor(this.rng() * 3);
      if (attr === 0) this.bombsCap[p] = Math.min(this.bombsCap[p] + add, this.bombsMax);
      else if (attr === 1) this.blastCap[p] = Math.min(this.blastCap[p] + add, this.blastMax);
      else this.spdG[p] = Math.min(this.spdG[p] + add * CFG.growthSpeedStep, this.speedMax);
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
        // 一次遍历: 可通行 && 非出生区 && **当前无道具**(不落在已有宝箱格上)
        if (!this.wall[i] && !this.brick[i] && !excl.has(i) && !this.crate[i]) pool.push(i);
      }
      // 不放回抽样
      for (let k = 0; k < Math.min(lost, pool.length); k++) {
        const j = k + Math.floor(this.rng() * (pool.length - k));
        const tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
        this.crate[pool[k]] = 1;
        this.recycle[pool[k]] = 1;
        this.crateType[pool[k]] = -1;   // 回收宝箱=随机(问号)
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
      // 余威不是在场炸弹，不能参与连锁传播；但仍是当前可伤害区域。
      for (let i = 0; i < N; i++) if (this.blastLinger[i] > 0) out[i] = 1;
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
      const ow = this.oldMode ? 13 : W;    // 旧 13x13 模型: 观测按 13 宽输出
      const n = H * ow;
      const o = new Float32Array(C * n);
      const splat = (ch, p) => {
        if (!this.alive[p]) return;
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        const fy = Math.min(Math.max(y - 0.5, 0), H - 1);
        const fx = Math.min(Math.max(x - 0.5, 0), ow - 1);
        const y0 = Math.min(Math.floor(fy), H - 1);
        const x0 = Math.min(Math.floor(fx), ow - 1);
        const y1 = Math.min(y0 + 1, H - 1);
        const x1 = Math.min(x0 + 1, ow - 1);
        const wy = Math.min(Math.max(fy - y0, 0), 1);
        const wx = Math.min(Math.max(fx - x0, 0), 1);
        o[ch * n + y0 * ow + x0] += (1 - wy) * (1 - wx);
        o[ch * n + y0 * ow + x1] += (1 - wy) * wx;
        o[ch * n + y1 * ow + x0] += wy * (1 - wx);
        o[ch * n + y1 * ow + x1] += wy * wx;
      };
      splat(0, 0); splat(1, 1);
      // 全网格(15宽) → 旧网格(13宽) 逐格搬运; 旧列 c<13 与全网格列一致
      for (let r = 0; r < H; r++) {
        for (let c = 0; c < ow; c++) {
          const fi = r * W + c, oi = r * ow + c;
          if (this.owner[fi] === 0 && this.fuse[fi] > 0) o[2 * n + oi] = this.fuse[fi] / CFG.fuse;
          if (this.owner[fi] === 1 && this.fuse[fi] > 0) o[3 * n + oi] = this.fuse[fi] / CFG.fuse;
          if (this.wall[fi] || this.brick[fi]) o[4 * n + oi] = 1;
          if (this.crate[fi]) o[7 * n + oi] = 1;
        }
      }
      const danger = this.dangerMap();
      for (let r = 0; r < H; r++) {
        for (let c = 0; c < ow; c++) {
          const fi = r * W + c, oi = r * ow + c;
          o[5 * n + oi] = danger[fi];
          o[6 * n + oi] = this.t / CFG.maxSteps;
        }
      }
      // 无敌标记 + 可用泡/上限（玩家自己的格）
      for (let p = 0; p < 2; p++) {
        if (!this.alive[p]) continue;
        const [r, c] = this.centerCell(p);
        const i = r * ow + c;
        if (this.invuln[p] > 0) o[(8 + p) * n + i] = 1;
        const cap = Math.max(1, this.bombsCap[p]);
        const avail = Math.max(0, this.bombsCap[p] - this.liveBombs(p));
        o[(10 + p) * n + i] = avail / cap;
        o[(12 + p) * n + i] = this.bombsCap[p] / CFG.growthBombsMax;
      }
      return o;
    }

    // ------------------------------------------------------- JAX 视角观测
    // jax_bomb/jax_env.py::make_obs 的浏览器移植：玩家 pid 自己的 13 通道视角。
    // transformer 专用（每玩家一份观测直接喂网络）；MLP/CNN 仍是上面的 14 通道
    // 共享观测 + 折视角。通道顺序与 make_obs 完全一致：
    //   ch0 我位置(splat) ch1 我的泡引信  ch2 敌位置(splat) ch3 敌泡引信
    //   ch4 墙|砖  ch5 危险图  ch6 进度  ch7 宝箱存在  ch8 灌木
    //   ch9 泡道具(1/4) ch10 威力道具(2/5) ch11 速度道具(3/6) ch12 超级(4/5/6)
    encodeObsJAX(pid, C = 14) {
      // C=14（新标准，ch13=可推箱）；旧 ViT ckpt obs_shape[0]=13 → 不含箱子
      // 通道，由调用方传模型的 obs_shape[0] 保持逐位对齐。
      const o = new Float32Array(C * N);
      const opp = 1 - pid;
      const splat = (ch, p) => {
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        if (!this.alive[p]) return;
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
      splat(0, pid); splat(2, opp);
      for (let i = 0; i < N; i++) {
        if (this.owner[i] === pid && this.fuse[i] > 0) o[1 * N + i] = this.fuse[i] / CFG.fuse;
        if (this.owner[i] === opp && this.fuse[i] > 0) o[3 * N + i] = this.fuse[i] / CFG.fuse;
        if (this.wall[i] || this.brick[i]) o[4 * N + i] = 1;
        if (this.crate[i] > 0) o[7 * N + i] = 1;
        if (this.bush[i]) o[8 * N + i] = 1;
        if (this.crate[i] > 0) {
          const ct = this.crateType[i], sup = this.superCrate[i] === 1;
          if (ct === 0) o[9 * N + i] = 1;          // 泡道具（普通 1 / 超级 4）
          else if (ct === 1) o[10 * N + i] = 1;    // 威力道具（2 / 5）
          else if (ct === 2) o[11 * N + i] = 1;    // 速度道具（3 / 6）
          if (sup) o[12 * N + i] = 1;              // 超级档（4/5/6，+4）
        }
      }
      const danger = this.dangerMap();
      for (let i = 0; i < N; i++) o[5 * N + i] = danger[i];
      const tv = this.t / CFG.maxSteps;
      for (let i = 0; i < N; i++) o[6 * N + i] = tv;
      if (C >= 14) {
        for (let i = 0; i < N; i++) o[13 * N + i] = this.pushable[i];
      }
      return o;
    }

    // jax_env.global_vec 的浏览器移植：玩家 pid 视角 24 维全局向量
    // （state token 输入）。buffs/debuffs/items/gametype 训练侧当前全 0
    // （预留位），Web 端同样给 0，与训练分布一致。
    encodeStateJAX(pid) {
      const opp = 1 - pid;
      const g = new Float64Array(24);
      g[0] = this.t / CFG.maxSteps;
      g[1] = this.hp[pid] / CFG.maxHp;
      g[2] = this.hp[opp] / CFG.maxHp;
      g[3] = this.bombsCap[pid] / CFG.growthBombsMax;
      g[4] = this.blastCap[pid] / CFG.growthBlastMax;
      g[5] = this.spdG[pid] / CFG.growthSpeedMax;
      g[6] = this.bombsCap[opp] / CFG.growthBombsMax;
      g[7] = this.blastCap[opp] / CFG.growthBlastMax;
      g[8] = this.spdG[opp] / CFG.growthSpeedMax;
      g[9] = this.alive[pid] ? 1 : 0;
      g[10] = this.alive[opp] ? 1 : 0;
      return g;   // g[11..23] = 0（预留）
    }

    // 单方向移动尝试（原 Sim.step 移动块提取）：resolveAxis AABB 滑动碰撞 +
    // 中心路径硬约束（起点格脚下豁免）。对齐 JAX _move_player，逐位一致。
    _tryMove(y, x, mv, blocked, dist) {
      const [dy, dx] = DIRS[mv];
      const startR = Math.max(0, Math.min(H - 1, Math.floor(y)));
      const startC = Math.max(0, Math.min(W - 1, Math.floor(x)));
      let ny = y, nx = x;
      if (dy !== 0) {
        ny = resolveAxis(y + dy * dist, dy * dist, x, y, x,
                         blocked, CFG.radius, H, W, true);
        // 中心沿起点列扫过的行段必须全可通行（起点行脚下豁免）
        const yLo = Math.max(0, Math.min(H - 1, Math.floor(Math.min(y, ny))));
        const yHi = Math.max(0, Math.min(H - 1, Math.floor(Math.max(y, ny))));
        for (let r = yLo; r <= yHi; r++) {
          if (r === startR) continue;
          if (blocked[r * W + startC]) { ny = y; break; }
        }
      }
      if (dx !== 0) {
        nx = resolveAxis(x + dx * dist, dx * dist, y, ny, x,
                         blocked, CFG.radius, H, W, false);
        // 中心沿 ny 行扫过的列段必须全可通行（起点格脚下豁免）
        const xLo = Math.max(0, Math.min(W - 1, Math.floor(Math.min(x, nx))));
        const xHi = Math.max(0, Math.min(W - 1, Math.floor(Math.max(x, nx))));
        const cy0 = Math.max(0, Math.min(H - 1, Math.floor(ny)));
        for (let c = xLo; c <= xHi; c++) {
          if (c === startC && cy0 === startR) continue;
          if (blocked[cy0 * W + c]) { nx = x; break; }
        }
      }
      return [ny, nx];
    }

    // 贪婪转向适配器（对齐 JAX _steer）：模型输出=目标相邻格，选第一个能动的
    // 方向。优先级：直走(mv) > 垂直偏转1 > 垂直偏转2。每 tick 无状态决策——
    // 不跨 tick 承诺方向，不会振荡。
    // 垂直回退方向按目标行/列的**斜对角开闭**排序：楔死时朝开口侧滑——朝墙
    // 侧滑永远进不了目标行/列，会背向目标绕路（点上方开口却左滑绕墙的 bug）。
    // 两边同开/同堵 → 朝目标格中心线归中。固定偏左/偏上会让角色在目标
    // 方向受阻时走向错误一侧，尤其是角色中心尚未对齐当前格中心的情况。
    _steer(y, x, mv, blocked, dist) {
      if (mv >= 4) return [y, x];
      let [ny, nx] = this._tryMove(y, x, mv, blocked, dist);
      const moved = Math.abs(ny - y) + Math.abs(nx - x);
      // 完整直走直接结束；只走到半步（常见于碰撞盒擦住侧砖）仍需尝试
      // 垂直修正，否则会卡在砖边缘。推箱目标例外：保留直走动作累计 pushT。
      const fullStep = moved >= dist * 0.95;
      const r0 = Math.max(0, Math.min(H - 1, Math.floor(y)));
      const c0 = Math.max(0, Math.min(W - 1, Math.floor(x)));
      const [dr, dc] = DIRS[mv];
      const tr0 = r0 + dr, tc0 = c0 + dc;
      // 可推箱必须持续收到同一方向动作累计 PUSH_TIME。箱子仍属于 blocked，
      // 但这里不能把“顶箱未移动”误判成失败后侧滑，否则计时会中断且角色
      // 产生上下/左右偏移。箱子移走后的下一 tick 会正常直行进入原箱格。
      if (this.pushable && tr0 >= 0 && tr0 < H && tc0 >= 0 && tc0 < W &&
          this.pushable[tr0 * W + tc0]) return [ny, nx];
      if (fullStep) return [ny, nx];
      const open = (r, c) => r >= 0 && r < H && c >= 0 && c < W && !blocked[r * W + c];
      let perp;
      if (mv < 2) {
        // 上/下：目标行 tr；左斜(tr,c0-1)堵且右斜(tr,c0+1)开 → 先右滑
        const tr = r0 + (mv === 0 ? -1 : 1);
        const leftOpen = open(tr, c0 - 1), rightOpen = open(tr, c0 + 1);
        if (leftOpen !== rightOpen) perp = rightOpen ? [3, 2] : [2, 3];
        else perp = x < c0 + 0.5 ? [3, 2] : [2, 3];
      } else {
        // 左/右：目标列 tc；上斜(r0-1,tc)堵且下斜(r0+1,tc)开 → 先下滑
        const tc = c0 + (mv === 2 ? -1 : 1);
        const upOpen = open(r0 - 1, tc), downOpen = open(r0 + 1, tc);
        if (upOpen !== downOpen) perp = downOpen ? [1, 0] : [0, 1];
        else perp = y < r0 + 0.5 ? [1, 0] : [0, 1];
      }
      const p1 = this._tryMove(y, x, perp[0], blocked, dist);
      const p2 = this._tryMove(y, x, perp[1], blocked, dist);
      // 目标前方是一格宽通道（两侧都堵）时，侧移只负责归中，不能一次
      // 越过中心线，否则下一 tick 会反向修正并左右/上下振荡。
      if (mv < 2 && !open(r0 + (mv === 0 ? -1 : 1), c0 - 1) &&
          !open(r0 + (mv === 0 ? -1 : 1), c0 + 1)) {
        const center = c0 + 0.5;
        if (perp[0] === MOVE_LEFT) p1[1] = Math.max(p1[1], center);
        else p1[1] = Math.min(p1[1], center);
        if (perp[1] === MOVE_LEFT) p2[1] = Math.max(p2[1], center);
        else p2[1] = Math.min(p2[1], center);
      } else if (mv >= 2 && !open(r0 - 1, c0 + (mv === 2 ? -1 : 1)) &&
                 !open(r0 + 1, c0 + (mv === 2 ? -1 : 1))) {
        const center = r0 + 0.5;
        if (perp[0] === MOVE_UP) p1[0] = Math.max(p1[0], center);
        else p1[0] = Math.min(p1[0], center);
        if (perp[1] === MOVE_UP) p2[0] = Math.max(p2[0], center);
        else p2[0] = Math.min(p2[0], center);
      }
      const moved1 = Math.abs(p1[0] - y) + Math.abs(p1[1] - x) > 2 * EPS;
      const moved2 = Math.abs(p2[0] - y) + Math.abs(p2[1] - x) > 2 * EPS;
      if (moved1) return p1;
      if (moved2) return p2;
      return moved > 2 * EPS ? [ny, nx] : [y, x];
    }

    // 方向掩码：目标格 blocked 查表（O(1)，不探测碰撞物理——对齐 JAX legal_mask）；
    // IDLE 恒合法。放泡掩码：中心格可放泡（can_place）；bomb=0 恒合法。
    legalMask() {
      const mm = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]];
      const bm = [[1, 1], [1, 1]];
      const blocked = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        // 推箱格豁免：mask 把朝可推箱方向标记为合法（模型才会选这个方向去推），
        // 但 Sim.step 的移动仍用含 brick 的 blocked 挡住（推箱期间贴箱不动，
        // 3 tick 后箱子移走玩家跟进）。
        blocked[i] = this.wall[i] || (this.brick[i] && !this.pushable[i]) || this.fuse[i] > 0 ? 1 : 0;
      }
      for (let p = 0; p < 2; p++) {
        if (!this.alive[p]) continue;
        const y = this.pos[p * 2], x = this.pos[p * 2 + 1];
        // 目标格查询：floor(位置) + DIRS 的 4 个相邻格，blocked 查表
        const r0 = Math.max(0, Math.min(H - 1, Math.floor(y)));
        const c0 = Math.max(0, Math.min(W - 1, Math.floor(x)));
        for (let mv = 0; mv < 4; mv++) {
          const [dr, dc] = DIRS[mv];
          const tr = r0 + dr, tc = c0 + dc;
          if (tr < 0 || tr >= H || tc < 0 || tc >= W) { mm[p][mv] = 0; continue; }
          mm[p][mv] = blocked[tr * W + tc] ? 0 : 1;   // 目标格非障碍 = 合法
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

  // ---------------------------------------------------------------- CNN 模型
  // train/model.py arch="cnn" 的浏览器移植：conv0(3×3) → [LN(16)+ReLU] →
  // conv(16→32) → [LN+ReLU] → conv(32→64) → [LN+ReLU] → conv1×1(64→8) →
  // [LN+ReLU] → flatten(8·h·w) → shared MLP(128→128) → 双头。
  // 与 MLP 的差异只在特征提取：conv 里的 LayerNorm 归一化范围是**整个
  // (C,H,W) 三维**（weight/bias 逐元素），shared 层与 MLPModel 完全同构
  // （weight key 同名，复用 _lnRelu/_sampleMasked）。pid=0 视角已折进
  // conv0 输入通道（deploy/export_ckpt.py::fold_conv_perm_pid0）。
  class CNNModel extends MLPModel {
    forward(obs) {
      const [C, h, w] = this.obsShape;
      const N2 = h * w;
      const T = (n) => this.T(n);

      // 3×3 卷积（padding=1 保持分辨率）：先把输入零填充到 (H+2)×(W+2)，
      // 内层无边界分支，kernel 3×3 显式展开成 9 次读（比 44ms 的带分支
      // 版本快 ~4×，AI 对打双模型也要保证 10Hz 逻辑预算内）。
      const conv3 = (inp, inC, outC, Wt, Bt) => {
        const hh = h + 2, ww = w + 2, PW = hh * ww;
        const pad = new Float64Array(inC * PW);
        for (let i = 0; i < inC; i++) {
          const src = i * N2, dst = i * PW + hh + 1;
          for (let r = 0; r < h; r++)
            for (let c2 = 0; c2 < w; c2++)
              pad[dst + r * ww + c2] = inp[src + r * w + c2];
        }
        const out = new Float64Array(outC * N2);
        for (let o = 0; o < outC; o++) {
          const bo = Bt[o], wb = o * inC * 9;
          for (let r = 0; r < h; r++) {
            const pr = r * ww;
            for (let cc = 0; cc < w; cc++) {
              let s = bo;
              for (let i = 0; i < inC; i++) {
                const base = i * PW + pr + cc, w9 = wb + i * 9;
                s += Wt[w9] * pad[base] + Wt[w9 + 1] * pad[base + 1]
                  + Wt[w9 + 2] * pad[base + 2] + Wt[w9 + 3] * pad[base + ww]
                  + Wt[w9 + 4] * pad[base + ww + 1] + Wt[w9 + 5] * pad[base + ww + 2]
                  + Wt[w9 + 6] * pad[base + 2 * ww]
                  + Wt[w9 + 7] * pad[base + 2 * ww + 1]
                  + Wt[w9 + 8] * pad[base + 2 * ww + 2];
              }
              out[o * N2 + r * w + cc] = s;
            }
          }
        }
        return out;
      };
      // 1×1 卷积 = 每像素跨通道加权和
      const conv1x1 = (inp, inC, outC, Wt, Bt) => {
        const out = new Float64Array(outC * N2);
        for (let o = 0; o < outC; o++) {
          const bo = Bt[o], wob = o * inC;
          for (let i = 0; i < N2; i++) {
            let s = bo;
            for (let cc = 0; cc < inC; cc++) s += Wt[wob + cc] * inp[cc * N2 + i];
            out[o * N2 + i] = s;
          }
        }
        return out;
      };
      // 3D LayerNorm：mean/var 统计整个 C*H*W（每样本一个标量对），w/b 逐元素
      const ln3dRelu = (x, ww, bb) => {
        let mean = 0;
        for (let i = 0; i < x.length; i++) mean += x[i];
        mean /= x.length;
        let v = 0;
        for (let i = 0; i < x.length; i++) v += (x[i] - mean) * (x[i] - mean);
        const std = Math.sqrt(v / x.length + 1e-5);
        for (let i = 0; i < x.length; i++) {
          x[i] = Math.max(0, (x[i] - mean) / std * ww[i] + bb[i]);
        }
      };

      let x = conv3(obs, C, 16, T('conv0_w'), T('conv0_b'));
      ln3dRelu(x, T('cn1_w'), T('cn1_b'));
      x = conv3(x, 16, 32, T('conv1_w'), T('conv1_b'));
      ln3dRelu(x, T('cn2_w'), T('cn2_b'));
      x = conv3(x, 32, 64, T('conv2_w'), T('conv2_b'));
      ln3dRelu(x, T('cn3_w'), T('cn3_b'));
      x = conv1x1(x, 64, 8, T('conv3_w'), T('conv3_b'));
      ln3dRelu(x, T('cn4_w'), T('cn4_b'));

      // flatten(8·h·w) → shared MLP（与 MLPModel 同构，weight key 同名）
      const inDim = 8 * N2;
      const W1 = T('shared0_w'), b1 = T('shared0_b');
      const ln1w = T('ln1_w'), ln1b = T('ln1_b');
      const W2 = T('shared3_w'), b2 = T('shared3_b');
      const ln2w = T('ln2_w'), ln2b = T('ln2_b');
      const f = new Float64Array(128);
      for (let i = 0; i < 128; i++) {
        let s = b1[i];
        for (let j = 0; j < inDim; j++) s += W1[i * inDim + j] * x[j];
        f[i] = s;
      }
      this._lnRelu(f, ln1w, ln1b);
      const h2 = new Float64Array(128);
      for (let i = 0; i < 128; i++) {
        let s = b2[i];
        for (let j = 0; j < 128; j++) s += W2[i * 128 + j] * f[j];
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
        move: head(T('move0_w'), T('move0_b'), T('move2_w'), T('move2_b'), 5),
        bomb: head(T('bomb0_w'), T('bomb0_b'), T('bomb2_w'), T('bomb2_b'), 2),
      };
    }
  }

  // ---------------------------------------------------------------- Transformer 模型
  // jax_bomb/jax_net.py::init_transformer / transformer_forward 的浏览器移植
  // （ViT 风格：patch 切块 + state token + pre-norm MHA/FFN + patch-token
  // 均值池化 + 三头）。与 MLP/CNN 不同：输入是**每玩家视角** obs（13 通道
  // encodeObsJAX） + 24 维 global_vec（encodeStateJAX），不走共享观测/折视角。
  // 数值对齐 jax_bomb 训练：LN eps=1e-6，softmax 前 scores 先 round 到 fp32，
  // q/k/v 按 (T,heads,d) 切分后转置成 (heads,T,d)（不能直接 reshape(heads,T,d)）。
  class TransformerModel extends MLPModel {
    constructor(doc) {
      super(doc);
      this.embed = doc.meta.embed;
      this.patch = doc.meta.patch;
      this.depth = doc.meta.depth;
      const E = this.embed, P = this.patch;
      const gp = Math.ceil(this.obsShape[1] / P), nTok = gp * gp, T17 = nTok + 1;
      this._nTok = nTok; this._T17 = T17;
      const F = this.T('b0_ff1_w').length / E;      // ff_factor*E
      // 一次性分配全部中间缓冲（forward 内零分配；GC 是纯 JS 前向的最大
      // 隐性开销之一）。容量按 2 玩家（观战批处理）分配，单玩家只用前半。
      this._x = new Float64Array(2 * nTok * E);
      this._cur = new Float64Array(2 * T17 * E);
      this._ln1 = new Float64Array(2 * T17 * E);
      this._ln2 = new Float64Array(2 * T17 * E);
      this._q = new Float64Array(2 * T17 * E);
      this._k = new Float64Array(2 * T17 * E);
      this._v = new Float64Array(2 * T17 * E);
      this._scores = new Float64Array(2 * 4 * T17 * T17);
      this._w = new Float64Array(2 * 4 * T17 * T17);
      this._att = new Float64Array(2 * T17 * E);
      this._ff = new Float64Array(2 * T17 * F);
      this._g = new Float64Array(2 * E);
      this._cSim = null; this._cGen = -1; this._cT = [-1, -1]; this._cA = [null, null];
      this._inferMs = 0;               // 推理耗时累加器（[prof] 每秒读取后清零）
      // 矩阵乘权重**预转置**成 (N,K) 布局（Float32）：forward 的 4 路展开
      // 内层按 j 连续读（纯 JS 标量循环约 2.5× 提速，见 matmul 微基准）。
      this._tokT = this._transpose(this.T('tok_w'), E);
      this._qT = []; this._kT = []; this._vT = []; this._prT = [];
      this._ff1T = []; this._ff2T = [];
      for (let i = 0; i < this.depth; i++) {
        const p = String(i);
        this._qT.push(this._transpose(this.T('b' + p + '_q_w'), E));
        this._kT.push(this._transpose(this.T('b' + p + '_k_w'), E));
        this._vT.push(this._transpose(this.T('b' + p + '_v_w'), E));
        this._prT.push(this._transpose(this.T('b' + p + '_proj_w'), E));
        this._ff1T.push(this._transpose(this.T('b' + p + '_ff1_w'), E));   // (E,F)→(F,E)
        this._ff2T.push(this._transpose(this.T('b' + p + '_ff2_w'), F));   // (F,E)→(E,F)
      }
    }

    // w 按 (K,N) 行主序 → 返回 (N,K) Float32 转置
    _transpose(w, K) {
      const n = w.length / K, t = new Float32Array(w.length);
      for (let j = 0; j < K; j++) {
        const src = j * n, dst = j;
        for (let i = 0; i < n; i++) t[dst + i * K] = w[src + i];
      }
      return t;
    }

    // 4 路展开矩阵乘（无分支专用版，V8 易内联）：out[t*N+e] = sum_j a[t*K+j]*wT[e*K+j] + bias[e]。
    // 要求 N % 4 == 0（本模型 E=392、F=1568 均满足）；4 个独立局部累加链是 V8
    // 标量循环最快的形态（微基准 ~2.5× 于朴素三重循环）。
    _mmB(out, a, wT, bias, M, K, N) {
      for (let t = 0; t < M; t++) {
        const arow = t * K, orow = t * N;
        for (let e = 0; e < N; e += 4) {
          const w0 = e * K, w1 = w0 + K, w2 = w1 + K, w3 = w2 + K;
          let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
          for (let j = 0; j < K; j++) {
            const av = a[arow + j];
            s0 += av * wT[w0 + j];
            s1 += av * wT[w1 + j];
            s2 += av * wT[w2 + j];
            s3 += av * wT[w3 + j];
          }
          out[orow + e] = s0 + bias[e];
          out[orow + e + 1] = s1 + bias[e + 1];
          out[orow + e + 2] = s2 + bias[e + 2];
          out[orow + e + 3] = s3 + bias[e + 3];
        }
      }
    }

    // 同上，但累加到 out（out += a@wT + bias），用于残差连接/FFN 第二段
    _mmA(out, a, wT, bias, M, K, N) {
      for (let t = 0; t < M; t++) {
        const arow = t * K, orow = t * N;
        for (let e = 0; e < N; e += 4) {
          const w0 = e * K, w1 = w0 + K, w2 = w1 + K, w3 = w2 + K;
          let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
          for (let j = 0; j < K; j++) {
            const av = a[arow + j];
            s0 += av * wT[w0 + j];
            s1 += av * wT[w1 + j];
            s2 += av * wT[w2 + j];
            s3 += av * wT[w3 + j];
          }
          out[orow + e] += s0 + bias[e];
          out[orow + e + 1] += s1 + bias[e + 1];
          out[orow + e + 2] += s2 + bias[e + 2];
          out[orow + e + 3] += s3 + bias[e + 3];
        }
      }
    }

    // obs: Float32Array(C*H*W) 每玩家视角；state: Float64Array(24)
    // 返回 { move: [5], bomb: [2], value: number }（move/bomb 与 MLP 同签名）
    forward(obs, state) {
      const t0 = performance.now();
      const out = this._run(1, [obs], [state])[0];
      this._inferMs += performance.now() - t0;   // [prof] 推理耗时累计
      return out;
    }

    // 双玩家批处理：一次前向同时出两名玩家的 logits（观战 AI vs AI 用，
    // 权重只读一遍，比两次单玩家前向省 ~35%）。返回 [{move,bomb,value}×2]。
    forward2(o0, s0, o1, s1) {
      return this._run(2, [o0, o1], [s0, s1]);
    }

    // 数值对齐 jax_bomb 训练：LN eps=1e-6，softmax 前 scores 先 round 到
    // fp32，q/k/v 按 (T,heads,d) 切分（t*E + h*d + dd 索引，等价
    // (T,E)→(T,heads,d)，不额外物化转置）。
    _run(n, obsList, stateList) {
      const [C, h, w] = this.obsShape;
      const E = this.embed, P = this.patch, heads = 4;
      const gp = Math.ceil(h / P), nTok = this._nTok, T17 = this._T17;
      const d = E / heads;
      const tokW = this.T('tok_w'), tokB = this.T('tok_b');
      const pos = this.T('pos');              // (T17, E)
      const N2 = h * w;
      const TS = T17 * E, AS = heads * T17 * T17;   // 玩家行距
      const x = this._x, cur = this._cur, ln1 = this._ln1, ln2 = this._ln2;
      const q = this._q, k = this._k, v = this._v;
      const scores = this._scores, wm = this._w, att = this._att,
            ff = this._ff, g = this._g;
      const rows = n * T17;                    // LN/线性层按行展开（跨玩家）

      // 1) patch embedding：obs (C,h,w) 补零到 (C, gp*P, gp*P)，按
      //    (patch行, patch列, 通道, pr, pc) 展平 → tok 线性 + 位置编码。
      //    x 是复用缓冲且用 += 累加，必须先清零。x 玩家行距 = nTok*E。
      x.fill(0);
      for (let p = 0; p < n; p++) {
        const obs = obsList[p];
        for (let t = 0; t < nTok; t++) {
          const tr = (t / gp) | 0, tc = t % gp;
          const base = p * nTok * E + t * E;
          for (let ch = 0; ch < C; ch++) {
            const chOff = ch * P * P * E;
            const oCh = ch * N2;              // 每通道基址；格索引 = r*W+cc（r/cc 已含 patch 偏移）
            for (let pr = 0; pr < P; pr++) {
              const r = tr * P + pr;
              if (r >= h) break;              // 补零区
              for (let pc = 0; pc < P; pc++) {
                const cc = tc * P + pc;
                if (cc >= w) break;           // 补零区
                const ov = obs[oCh + r * W + cc];
                if (ov === 0) continue;
                const wOff = chOff + (pr * P + pc) * E;
                for (let e = 0; e < E; e++) x[base + e] += ov * tokW[wOff + e];
              }
            }
          }
          for (let e = 0; e < E; e++) x[base + e] += tokB[e] + pos[t * E + e];
        }
      }

      // 2) state token：global_vec(24) @ state_w + state_b + 自己的位置编码
      const stateW = this.T('state_w'), stateB = this.T('state_b');
      for (let p = 0; p < n; p++) {
        const cb = p * TS, xb = p * nTok * E;
        for (let t = 0; t < nTok; t++)
          for (let e = 0; e < E; e++) cur[cb + t * E + e] = x[xb + t * E + e];
        const state = stateList[p];
        for (let e = 0; e < E; e++) {
          let s = stateB[e] + pos[nTok * E + e];
          for (let j = 0; j < state.length; j++) s += state[j] * stateW[j * E + e];
          cur[cb + nTok * E + e] = s;
        }
      }

      // 3) pre-norm MHA / FFN blocks（LN eps=1e-6，与 jax_net 一致）
      for (let i = 0; i < this.depth; i++) {
        const p = String(i);
        const ln1g = this.T('b' + p + '_ln1_g'), ln1b = this.T('b' + p + '_ln1_b');
        const qb = this.T('b' + p + '_q_b');
        const kb = this.T('b' + p + '_k_b');
        const vb = this.T('b' + p + '_v_b');
        const prb = this.T('b' + p + '_proj_b');
        const ln2g = this.T('b' + p + '_ln2_g'), ln2b = this.T('b' + p + '_ln2_b');
        const f1b = this.T('b' + p + '_ff1_b');
        const f2b = this.T('b' + p + '_ff2_b');

        // LN1（逐行，跨玩家）
        for (let t = 0; t < rows; t++) {
          let m = 0;
          const row = t * E;
          for (let e = 0; e < E; e++) m += cur[row + e];
          m /= E;
          let vv = 0;
          for (let e = 0; e < E; e++) { const dd = cur[row + e] - m; vv += dd * dd; }
          const inv = 1 / Math.sqrt(vv / E + 1e-6);
          for (let e = 0; e < E; e++)
            ln1[row + e] = (cur[row + e] - m) * inv * ln1g[e] + ln1b[e];
        }
        // q/k/v：(rows,E) @ (E,E)，转置权重 + 4 路展开（bias 尾加，对齐 JAX）
        this._mmB(q, ln1, this._qT[i], qb, rows, E, E);
        this._mmB(k, ln1, this._kT[i], kb, rows, E, E);
        this._mmB(v, ln1, this._vT[i], vb, rows, E, E);
        // scores = einsum("htd,hTd->htT") / sqrt(d) —— attention 只在玩家
        // 自己的 17 个 token 内（跨玩家不混合）
        for (let p = 0; p < n; p++) {
          const qo = p * TS, so = p * AS;
          for (let h = 0; h < heads; h++) {
            for (let t = 0; t < T17; t++) {
              const sBase = so + h * T17 * T17 + t * T17;
              const qt = qo + t * E + h * d;
              for (let T2 = 0; T2 < T17; T2++) {
                let s = 0;
                const kt = qo + T2 * E + h * d;
                for (let dd = 0; dd < d; dd++) s += q[qt + dd] * k[kt + dd];
                scores[sBase + T2] = s / Math.sqrt(d);
              }
            }
          }
        }
        // softmax：scores 先 round 到 fp32（训练 softmax 在 fp32 算，bf16 下更稳）
        for (let p = 0; p < n; p++) {
          const so = p * AS;
          for (let h = 0; h < heads; h++) {
            for (let t = 0; t < T17; t++) {
              const base = so + h * T17 * T17 + t * T17;
              let mx = -Infinity;
              for (let T2 = 0; T2 < T17; T2++) {
                const f = Math.fround(scores[base + T2]);
                if (f > mx) mx = f;
              }
              let sum = 0;
              for (let T2 = 0; T2 < T17; T2++) {
                const pp = Math.exp(Math.fround(scores[base + T2]) - mx);
                wm[base + T2] = pp; sum += pp;
              }
              for (let T2 = 0; T2 < T17; T2++) wm[base + T2] /= sum;
            }
          }
        }
        // att = einsum("htT,hTd->htd")，输出直接写 (T,heads,d) 展平
        for (let p = 0; p < n; p++) {
          const qo = p * TS, so = p * AS;
          for (let t = 0; t < T17; t++) {
            const at = qo + t * E;
            for (let h = 0; h < heads; h++) {
              const wBase = so + h * T17 * T17 + t * T17;
              for (let dd = 0; dd < d; dd++) {
                let s = 0;
                const ai = at + h * d + dd;
                for (let T2 = 0; T2 < T17; T2++)
                  s += wm[wBase + T2] * v[qo + T2 * E + h * d + dd];
                att[ai] = s;
              }
            }
          }
        }
        // x = x + att @ proj + proj_b（累加到 cur）
        this._mmA(cur, att, this._prT[i], prb, rows, E, E);
        // LN2 + ReLU FFN：relu(ln2 @ ff1 + f1b) @ ff2 + f2b
        for (let t = 0; t < rows; t++) {
          let m = 0;
          const row = t * E;
          for (let e = 0; e < E; e++) m += cur[row + e];
          m /= E;
          let vv = 0;
          for (let e = 0; e < E; e++) { const dd = cur[row + e] - m; vv += dd * dd; }
          const inv = 1 / Math.sqrt(vv / E + 1e-6);
          for (let e = 0; e < E; e++)
            ln2[row + e] = (cur[row + e] - m) * inv * ln2g[e] + ln2b[e];
        }
        const F = this._ff1T[i].length / E;     // ff1: (E, F)，F = ff_factor*E
        this._mmB(ff, ln2, this._ff1T[i], f1b, rows, E, F);
        for (let j = 0; j < ff.length; j++) if (ff[j] < 0) ff[j] = 0;   // ReLU
        this._mmA(cur, ff, this._ff2T[i], f2b, rows, F, E);
      }

      // 4) patch-token 均值池化 → 三头（每玩家）
      const hmw = this.T('head_wm_w'), hmb = this.T('head_wm_b');   // (E, 5)
      const hbw = this.T('head_wb_w'), hbb = this.T('head_wb_b');   // (E, 2)
      const hvw = this.T('head_wv_w'), hvb = this.T('head_wv_b');   // (E, 1) or (E, 128)
      const numBins = hvb.length;
      const outs = [];
      for (let p = 0; p < n; p++) {
        const base = p * TS;
        for (let e = 0; e < E; e++) {
          let s = 0;
          for (let t = 0; t < nTok; t++) s += cur[base + t * E + e];
          g[e] = s / nTok;
        }
        const move = new Float64Array(5), bomb = new Float64Array(2);
        for (let j = 0; j < E; j++) {
          const gj = g[j], j5 = j * 5, j2 = j * 2;
          for (let o = 0; o < 5; o++) move[o] += gj * hmw[j5 + o];
          for (let o = 0; o < 2; o++) bomb[o] += gj * hbw[j2 + o];
        }
        for (let o = 0; o < 5; o++) move[o] += hmb[o];
        for (let o = 0; o < 2; o++) bomb[o] += hbb[o];
        let value = 0;
        if (numBins === 128) {
          const vLogits = new Float64Array(128);
          for (let b = 0; b < 128; b++) vLogits[b] = hvb[b];
          for (let j = 0; j < E; j++) {
            const gj = g[j], jB = j * 128;
            for (let b = 0; b < 128; b++) vLogits[b] += gj * hvw[jB + b];
          }
          let maxLogit = -Infinity;
          for (let b = 0; b < 128; b++) if (vLogits[b] > maxLogit) maxLogit = vLogits[b];
          let expSum = 0;
          const exps = new Float64Array(128);
          for (let b = 0; b < 128; b++) {
            exps[b] = Math.exp(vLogits[b] - maxLogit);
            expSum += exps[b];
          }
          for (let b = 0; b < 128; b++) {
            const prob = exps[b] / expSum;
            const center = -1.0 + (2.0 * b) / 127.0;
            value += prob * center;
          }
        } else {
          value = hvb[0];
          for (let j = 0; j < E; j++) value += g[j] * hvw[j];
        }
        outs.push({ move, bomb, value });
      }
      return outs;
    }

    // AI 决策：每玩家自己的视角，不需要通道互换。按 (sim, pid, tick) 缓存：
    // 同一 tick 重复询问同一玩家不再重复推理；不同玩家各算各的（人机对局
    // 只算 pid=1，不白跑 pid=0）。inferEvery>1 时每隔 N tick 才推理一次，
    // 中间 tick 复用上一次的动作（省 CPU 的玩法选项；默认 1 = 每 tick）。
    act(sim, pid, rng) {
      const every = this.inferEvery || 1;
      const changed = sim !== this._cSim || sim._gen !== this._cGen;
      if (changed) {
        this._cT[0] = -1; this._cT[1] = -1;
        this._cA[0] = null; this._cA[1] = null;
      }
      const need = changed || this._cT[pid] < 0 ||
        (sim.t - this._cT[pid] >= every);
      if (need) {
        this._cSim = sim; this._cGen = sim._gen;
        const { mm, bm } = sim.legalMask();
        this._cA[pid] = this._decide(sim, pid, mm, bm, rng);
        this._cT[pid] = sim.t;
      }
      return this._cA[pid];
    }

    _decide(sim, pid, mm, bm, rng) {
      const logits = this.forward(sim.encodeObsJAX(pid, this.obsShape[0]),
                                  sim.encodeStateJAX(pid));
      if (!this._lastVal) this._lastVal = [0, 0];
      this._lastVal[pid] = logits.value;
      const aM = this._sampleMasked(logits.move, mm[pid], rng);
      const aB = this._sampleMasked(logits.bomb, bm[pid], rng);
      return [aM, aB];
    }

    // 观战双模型：一次批处理前向出双玩家动作（权重只读一遍，比两次单玩家
    // 前向省 ~35%）。返回 [a0, a1]，缓存语义与 act 一致（含降频）。
    bothAct(sim, rng) {
      const every = this.inferEvery || 1;
      const need = sim !== this._cSim || sim._gen !== this._cGen ||
        this._cT[0] < 0 || (sim.t - this._cT[0] >= every);
      if (need) {
        this._cSim = sim; this._cGen = sim._gen;
        const { mm, bm } = sim.legalMask();
        const [f0, f1] = this.forward2(
          sim.encodeObsJAX(0, this.obsShape[0]), sim.encodeStateJAX(0),
          sim.encodeObsJAX(1, this.obsShape[0]), sim.encodeStateJAX(1));
        this._lastVal = [f0.value, f1.value];
        this._cA[0] = [this._sampleMasked(f0.move, mm[0], rng),
                      this._sampleMasked(f0.bomb, bm[0], rng)];
        this._cA[1] = [this._sampleMasked(f1.move, mm[1], rng),
                      this._sampleMasked(f1.bomb, bm[1], rng)];
        this._cT[0] = sim.t; this._cT[1] = sim.t;
      }
      return this._cA;
    }
  }

  // ---------------------------------------------------------------- ORT Transformer 模型
  // onnxruntime-web 推理（WebGPU EP 优先，WASM 回退；session 由 main.js 创建，
  // 创建失败时直接用上面的纯 JS TransformerModel 兜底）。与 JS 版同一套
  // act/bothAct 缓存语义（每 tick、每玩家、含降频），只是 forward 走 session。
  // session.run 是异步的 → act/bothAct 返回 Promise，游戏 tick 需 await。
  class ORTTransformerModel extends TransformerModel {
    constructor(doc, session) {
      super(doc);
      this.session = session;
    }

    async _runWithTimeout(inputs, shape, label) {
      if (this._ortDisabled) return null;
      let timer;
      try {
        const run = this.session.run(inputs);
        const timeout = new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`ORT ${label} 超时（>${ORT_RUN_TIMEOUT_MS}ms），回退纯 JS`)),
                              ORT_RUN_TIMEOUT_MS);
        });
        return await Promise.race([run, timeout]);
      } catch (e) {
        this._ortDisabled = true;
        this._lastInferError = String(e && e.message ? e.message : e);
        console.warn('[ort] 推理失败，回退纯 JS：', this._lastInferError);
        return null;
      } finally {
        if (timer) clearTimeout(timer);
      }
    }

    _fallbackForward(obs, state) {
      return TransformerModel.prototype.forward.call(this, obs, state);
    }

    _fallbackForward2(o0, s0, o1, s1) {
      return TransformerModel.prototype.forward2.call(this, o0, s0, o1, s1);
    }

    async forward(obs, state) {
      const t0 = performance.now();
      const out = await this._runWithTimeout({
        obs: new ort.Tensor('float32', obs, [1, this.obsShape[0], this.obsShape[1], this.obsShape[2]]),
        state: new ort.Tensor('float32', Float32Array.from(state), [1, state.length]),
      }, [1, this.obsShape[0], this.obsShape[1], this.obsShape[2]], '单玩家');
      if (!out) return this._fallbackForward(obs, state);
      this._inferMs += performance.now() - t0;   // [prof] 推理耗时累计
      return { move: out.move.data, bomb: out.bomb.data, value: out.value.data[0] };
    }

    // 双玩家批处理（观战）：一次 session.run 出两人（batch=2）
    async forward2(o0, s0, o1, s1) {
      const t0 = performance.now();
      const C = this.obsShape[0], h = this.obsShape[1], w = this.obsShape[2];
      const obs = new Float32Array(2 * C * h * w);
      obs.set(o0, 0); obs.set(o1, C * h * w);
      const st = new Float32Array(48);
      st.set(Float32Array.from(s0), 0); st.set(Float32Array.from(s1), 24);
      const out = await this._runWithTimeout({
        obs: new ort.Tensor('float32', obs, [2, C, h, w]),
        state: new ort.Tensor('float32', st, [2, 24]),
      }, [2, C, h, w], '双玩家');
      if (!out) return this._fallbackForward2(o0, s0, o1, s1);
      this._inferMs += performance.now() - t0;   // [prof] 推理耗时累计
      const n5 = 5, n2 = 2;
      const f0 = {
        move: out.move.data.subarray(0, n5),
        bomb: out.bomb.data.subarray(0, n2),
        value: out.value.data[0],
      };
      const f1 = {
        move: out.move.data.subarray(n5, n5 + n5),
        bomb: out.bomb.data.subarray(n2, n2 + n2),
        value: out.value.data[1],
      };
      return [f0, f1];
    }

    async act(sim, pid, rng) {
      const every = this.inferEvery || 1;
      const changed = sim !== this._cSim || sim._gen !== this._cGen;
      if (changed) {
        this._cT[0] = -1; this._cT[1] = -1;
        this._cA[0] = null; this._cA[1] = null;
      }
      const need = changed || this._cT[pid] < 0 ||
        (sim.t - this._cT[pid] >= every);
      if (need) {
        this._cSim = sim; this._cGen = sim._gen;
        const { mm, bm } = sim.legalMask();
        this._cA[pid] = await this._decide(sim, pid, mm, bm, rng);
        this._cT[pid] = sim.t;
      }
      return this._cA[pid];
    }

    async bothAct(sim, rng) {
      const every = this.inferEvery || 1;
      const need = sim !== this._cSim || sim._gen !== this._cGen ||
        this._cT[0] < 0 || (sim.t - this._cT[0] >= every);
      if (need) {
        this._cSim = sim; this._cGen = sim._gen;
        const { mm, bm } = sim.legalMask();
        const [f0, f1] = await this.forward2(
          sim.encodeObsJAX(0, this.obsShape[0]), sim.encodeStateJAX(0),
          sim.encodeObsJAX(1, this.obsShape[0]), sim.encodeStateJAX(1));
        this._lastVal = [f0.value, f1.value];
        this._cA[0] = [this._sampleMasked(f0.move, mm[0], rng),
                      this._sampleMasked(f0.bomb, bm[0], rng)];
        this._cA[1] = [this._sampleMasked(f1.move, mm[1], rng),
                      this._sampleMasked(f1.bomb, bm[1], rng)];
        this._cT[0] = sim.t; this._cT[1] = sim.t;
      }
      return this._cA;
    }

    async _decide(sim, pid, mm, bm, rng) {
      const logits = await this.forward(
        sim.encodeObsJAX(pid, this.obsShape[0]), sim.encodeStateJAX(pid));
      if (!this._lastVal) this._lastVal = [0, 0];
      this._lastVal[pid] = logits.value;
      const aM = this._sampleMasked(logits.move, mm[pid], rng);
      const aB = this._sampleMasked(logits.bomb, bm[pid], rng);
      return [aM, aB];
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

      // 威胁区：自己名下在场泡的爆炸范围（绕行惩罚）。
      // **膨胀轮数 = 当前 blastCap**（不是 growthBlastMax 上限）：Python 端
      // 用 7 是训练 batch 里各 env 威力不同的工程妥协（避免每 tick 同步读
      // max），单局下 7 轮把威胁区膨胀到 60% 场地 → hunter 自己的泡堵死
      // 逃生/放泡空间 → 贴脸后 canEscape 恒 false、逼近打分全被惩罚 →
      // 发呆（Python 版 open 实测 IDLE 97%）。用当前威力膨胀威胁区才准确。
      // **每轮保留源格**（Python: nb = threat.clone() 后 4 邻扩张）。
      const threat = new Uint8Array(N);
      for (let i = 0; i < N; i++) {
        if (sim.owner[i] === pid && sim.fuse[i] > 0) threat[i] = 1;
      }
      const thrRounds = Math.max(1, sim.blastCap[pid]);
      for (let k = 0; k < thrRounds; k++) {
        const nb = new Uint8Array(threat);        // 保留源格（= clone）
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
      // 逼近场用**只挡永久墙**的版本：可炸砖视为可通（炸开即可过）—— 否则对手
      // 被砖封在舱室里时 V_opp=∞，Hunter 会放弃追击转逃生乱走。真正的移动
      // 合法性仍排除砖（不能走进砖），破砖交给下方"破砖开路"放泡逻辑。
      const blockedWalls = new Uint8Array(N);
      for (let i = 0; i < N; i++) blockedWalls[i] = sim.wall[i] ? 1 : 0;

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
      const V_opp = this._dijkstra(oppSrc, danger, blockedWalls);   // 砖可通，逼近场恒有限

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
      // 破砖开路：面前（4 邻）有可炸砖、且在"忽略砖"逼近场上比脚下更接近
      // 对手（砖在通往对手的路径上）→ 放泡炸开，而不是绕圈/发呆。
      let brickBlock = false;
      if (sim.bombsCap[pid] > 0 && sim.alive[pid]) {
        const vOwn = V_opp[ownIdx];
        for (let d = 0; d < 4; d++) {
          const nr = rIdx + DIRS[d][0], nc = cIdx + DIRS[d][1];
          if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
          const j = nr * W + nc;
          if (sim.brick[j] && !sim.wall[j] && V_opp[j] < vOwn) { brickBlock = true; break; }
        }
      }
      let bomb = 0;
      if (bm[pid][1] === 1 && (alignedOpp || nearOpp || chain || brickBlock) &&
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
    Sim, MLPModel, CNNModel, TransformerModel, ORTTransformerModel, HunterAI,
    mulberry32, resolveAxis, decodeB64,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = QQT;
  else root.QQT = QQT;
})(typeof globalThis !== 'undefined' ? globalThis : this);
