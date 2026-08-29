#!/usr/bin/env node
/* 无头对战测试：不启动浏览器，直接 Node 调用 web/sim.js 跑模型对战，采集局内指标。

指标（对齐训练语义）：
  - 结果：胜/负/平、终局原因（击杀/超时）、时长、终局血差
  - 自杀率：死亡 tick 被自己即将引爆的泡（fuse==1、曼哈顿距离≤blast）覆盖
  - 探索率：本局 visited 格数 / 总格数（13×15=195），每格首次到达才计
  - 攻击性：放炮数/局、命中（造成对方掉血）次数/局、炸墙数/局、吃箱数/局
  - 移动率：走动 tick 占比

对战模式（--opp）：
  self   = 模型 vs 自己（内耗/自杀/踱步）
  hunter = 模型 vs 规则 HunterAI（对抗压力参考；Hunter 纸面很强）
  cross  = 模型列表相邻配对对打（进化曲线；也可 cross:ai,bj 指定）

用法：
  node scripts/headless_test.js [--models a,b,c] [--opp self,hunter,cross] \
      [--maps N(每主题抽样)] [--ep N(每地图局数)] [--max-tick N] [--seed S] [--out out.json]

默认：所有 transformer ViTModel_* ckpt + 1 个 MLP 参考；self + cross；每主题 2 张 × 每张 3 局。
*/
'use strict';
const path = require('path');
const fs = require('fs');
const ROOT = path.join(__dirname, '..');
// ORTTransformerModel 内部用 ort.Tensor —— Node 侧注入 onnxruntime-node
const ortNode = require('onnxruntime-node');
global.ort = ortNode;
const QQT = require(path.join(ROOT, 'web', 'sim.js'));
const { Sim, MLPModel, CNNModel, ORTTransformerModel, HunterAI, H, W, CFG, mulberry32 } = QQT;

const MODELS = path.join(ROOT, 'web', 'models');
const MAPS_JSON = path.join(ROOT, 'web', 'assets', 'maps', 'levels.json');
const TOTAL_CELLS = H * W;

// ---------------- CLI ----------------
const argv = process.argv.slice(2);
function arg(name, dflt) {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
}
const MODELS_ARG = arg('--models', '').split(',').filter(Boolean);
const OPP_ARG = arg('--opp', 'self,cross').split(',').filter(Boolean);
// --vs a,b,c：每个模型 vs 指定陪练（模型名，出胜率），如 --vs ViTModel_500,duel_nobc_5.95B
const VS_ARG = arg('--vs', '').split(',').filter(Boolean);
// --pairs "0,5;1,2"：显式索引配对（--models 列表内），分号分隔
const PAIRS_ARG = arg('--pairs', '');
const MAPS_PER_THEME = parseInt(arg('--maps', '2'), 10);
const MAP_SOURCE = arg('--map-source', '').split(',').filter(Boolean); // 指定 source 列表（逗号分隔）
const EP_PER_MAP = parseInt(arg('--ep', '3'), 10);
const MAX_TICK = parseInt(arg('--max-tick', '900'), 10);
const SEED = parseInt(arg('--seed', '20260821'), 10);
const OUT = arg('--out', '');
// --spawns "r0,c0;r1,c1"：reset 后固定覆盖双方出生点（绕过地图随机打乱）。
//   实验用：验证"对称出生点 → 镜像同步 → P0 先手放大"；不对称出生点应打破同步。
const SPAWNS_ARG = arg('--spawns', '');
// --swap：每局交替双方位置（奇数局对调）。同模型 self 时可直接对比
//   "模型在 P0" vs "模型在 P1" 的胜率 → 检测位置偏置（P0/P1 观测不对称）。
const SWAP = argv.includes('--swap');

// ---------------- 模型加载 ----------------
async function loadModel(name) {
  const doc = JSON.parse(fs.readFileSync(path.join(MODELS, `${name}.json`), 'utf8'));
  const arch = doc.meta.arch;
  let m;
  if (arch === 'transformer') {
    const sess = await ortNode.InferenceSession.create(
      path.join(MODELS, `${name}.onnx`), { executionProviders: ['cpu'] });
    m = new ORTTransformerModel(doc, sess);
  } else if (arch === 'cnn') {
    m = new CNNModel(doc);
  } else {
    m = new MLPModel(doc);
  }
  m.inferEvery = 1;
  return { name, arch, m, gs: doc.meta.global_step || 0 };
}

function discoverModels() {
  const files = fs.readdirSync(MODELS).filter((f) => f.endsWith('.json') && f !== 'index.json');
  const list = [];
  for (const f of files) {
    const name = f.replace(/\.json$/, '');
    const doc = JSON.parse(fs.readFileSync(path.join(MODELS, f), 'utf8'));
    list.push({ name, arch: doc.meta.arch, gs: doc.meta.global_step || 0 });
  }
  list.sort((a, b) => a.gs - b.gs);
  return list;
}

// ---------------- 地图抽样（分层：每主题抽 N，附隔离度标注） ----------------
function sampleMaps(nPerTheme) {
  const maps = JSON.parse(fs.readFileSync(MAPS_JSON, 'utf8'));
  const list = Array.isArray(maps) ? maps : (maps.levels || maps.maps);
  if (MAP_SOURCE.length) {
    const hit = list.filter((m) => MAP_SOURCE.includes(m.source || ''));
    if (hit.length) {
      // 只测指定图(隔离度标注照旧)
      return hit.map((m) => { m._isolated = 0; return m; });
    }
  }
  const byTheme = {};
  for (const m of list) {
    const t = m.theme || m.category || '其他';   // levels.json 主题在 theme 字段
    (byTheme[t] = byTheme[t] || []).push(m);
  }
  const out = [];
  const themes = Object.keys(byTheme).sort();
  for (const t of themes) {
    const arr = byTheme[t].sort((a, b) => (a.id || 0) - (b.id || 0));
    const take = Math.min(nPerTheme, arr.length);
    for (let i = 0; i < take; i++) {
      const idx = take === 1 ? 0 : Math.round(i * (arr.length - 1) / (take - 1));
      out.push(arr[idx]);
    }
  }
  // 隔离度：出生点之间被 wall/brick 完全隔开 = 开局隔离（踱步高风险）
  for (const m of out) {
    const [s0, s1] = (m.spawns || []).slice(0, 2).map((s) => [s[0], s[1]]);
    let isolated = null;
    if (s0 && s1) {
      const wall = m.wall, brick = m.brick, w = m.w;
      const blocked = new Uint8Array(w * (m.h || H));
      for (let i = 0; i < blocked.length; i++) blocked[i] = wall[i] || brick[i] ? 1 : 0;
      const key = (r, c) => r * w + c;
      const q = [key(s0[0], s0[1])], seen = new Set(q);
      const D = [[0, 1], [0, -1], [1, 0], [-1, 0]];
      while (q.length) {
        const k = q.pop();
        const r = Math.floor(k / w), c = k % w;
        for (const [dr, dc] of D) {
          const nr = r + dr, nc = c + dc;
          if (nr < 0 || nc < 0 || nr >= (m.h || H) || nc >= w) continue;
          const nk = key(nr, nc);
          if (blocked[nk] || seen.has(nk)) continue;
          seen.add(nk); q.push(nk);
        }
      }
      isolated = !seen.has(key(s1[0], s1[1]));
    }
    m._isolated = isolated === true ? 1 : 0;
  }
  return out;
}

// ---------------- 单局对战 ----------------
function snapshotBombs(sim) {
  const bs = [];
  for (let i = 0; i < W * H; i++) {
    if (sim.fuse[i] > 0) bs.push({ i, owner: sim.owner[i], blast: sim.bombBlast[i], fuse: sim.fuse[i] });
  }
  return bs;
}

function isSuicide(sim, p, preBombs, diedCell) {
  const [r, c] = diedCell;
  for (const b of preBombs) {
    if (b.owner !== p || b.fuse !== 1) continue;   // fuse==1 → 本 tick 递减后引爆
    const br = Math.floor(b.i / W), bc = b.i % W;
    if (Math.abs(br - r) + Math.abs(bc - c) <= b.blast) return true;
  }
  return false;
}

async function playOne(level, seed, actors, maxTick, oldMode) {
  const sim = new Sim(seed);
  sim.reset(level, oldMode ? { oldMode: true } : undefined);
  if (SPAWNS_ARG) {
    // 实验：固定覆盖双方出生点（绕过地图随机打乱；reset 后直接写 pos）
    const [a, b] = SPAWNS_ARG.split(';').map((s) => s.split(',').map(Number));
    if (a && b) {
      sim.pos[0] = a[0] + 0.5; sim.pos[1] = a[1] + 0.5;
      sim.pos[2] = b[0] + 0.5; sim.pos[3] = b[1] + 0.5;
    }
  }
  const rng = mulberry32(seed ^ 0x9e3779b9);
  const st = {
    winner: null, cause: 'timeout', ticks: 0, hpEnd: [sim.hp[0], sim.hp[1]],
    suicide: [false, false], explored: [new Set(), new Set()],
    bombs: [0, 0], hits: [0, 0], walls: [0, 0], crates: [0, 0], moves: [0, 0], ticksAlive: [0, 0],
  };
  const [r0, c0] = sim.centerCell(0); st.explored[0].add(r0 * W + c0);
  const [r1, c1] = sim.centerCell(1); st.explored[1].add(r1 * W + c1);
  while (!sim.done && sim.t < maxTick) {
    const preBombs = snapshotBombs(sim);
    const preBrick = Array.from(sim.brick);
    const preCrate = Array.from(sim.crate);
    const hpBefore = [sim.hp[0], sim.hp[1]];
    const aliveBefore = [sim.alive[0], sim.alive[1]];
    const a0 = await actors[0](sim, 0, rng);
    const a1 = await actors[1](sim, 1, rng);
    sim.step([a0, a1]);
    st.ticks = sim.t;
    for (let p = 0; p < 2; p++) {
      const ap = p === 0 ? a0 : a1;
      if (aliveBefore[p]) {
        st.ticksAlive[p]++;
        if (ap[0] !== 4) st.moves[p]++;
        if (ap[1] === 1) st.bombs[p]++;
        const [r, c] = sim.centerCell(p);
        st.explored[p].add(r * W + c);
        if (hpBefore[p] > sim.hp[p] && sim.alive[p]) st.hits[1 - p]++;   // 我掉血 = 对手命中
        const ci = r * W + c;
        if (preCrate[ci] === 1 && sim.crate[ci] === 0) st.crates[p]++;
      }
      if (aliveBefore[p] && !sim.alive[p]) {
        const dc = sim.centerCell(p);
        st.suicide[p] = isSuicide(sim, p, preBombs, dc);
      }
    }
    for (let i = 0; i < W * H; i++) {
      if (preBrick[i] === 1 && sim.brick[i] === 0) {
        st.walls[sim.owner[i] >= 0 ? sim.owner[i] : 0]++;
      }
    }
  }
  st.winner = sim.winner;
  st.cause = sim.done && sim.t >= maxTick ? 'max_tick' : (sim.done ? 'end' : 'cutoff');
  st.hpEnd = [sim.hp[0], sim.hp[1]];
  return st;
}

// ---------------- 聚合 ----------------
function aggStat(rows, pick) {
  const vs = rows.map(pick).filter((x) => Number.isFinite(x));
  if (!vs.length) return 0;
  return vs.reduce((a, b) => a + b, 0) / vs.length;
}

function summarize(games, pid) {
  const n = games.length;
  const wins = games.filter((g) => g.st.winner === pid).length;
  const draws = games.filter((g) => g.st.winner === null).length;
  const sui = games.filter((g) => g.st.suicide[pid]).length;
  return {
    n, wins, draws, losses: n - wins - draws,
    winRate: n ? wins / n : 0,
    suicideRate: n ? sui / n : 0,
    ticks: aggStat(games, (g) => g.st.ticks),
    explore: aggStat(games, (g) => g.st.explored[pid].size / TOTAL_CELLS),
    bombs: aggStat(games, (g) => g.st.bombs[pid]),
    hits: aggStat(games, (g) => g.st.hits[pid]),
    walls: aggStat(games, (g) => g.st.walls[pid]),
    crates: aggStat(games, (g) => g.st.crates[pid]),
    moveRate: aggStat(games, (g) => (g.st.ticksAlive[pid] ? g.st.moves[pid] / g.st.ticksAlive[pid] : 0)),
  };
}

// With --swap, model A alternates between P0 and P1. Keep the model role
// attached to each game so cross/vs win rates do not mix the two actors.
function summarizeRole(games, role) {
  const pidOf = (g) => role === 'a' ? (g.swapped ? 1 : 0) : (g.swapped ? 0 : 1);
  const n = games.length;
  const wins = games.filter((g) => g.st.winner === pidOf(g)).length;
  const draws = games.filter((g) => g.st.winner === null).length;
  const sui = games.filter((g) => g.st.suicide[pidOf(g)]).length;
  return {
    n, wins, draws, losses: n - wins - draws,
    winRate: n ? wins / n : 0,
    suicideRate: n ? sui / n : 0,
    ticks: aggStat(games, (g) => g.st.ticks),
    explore: aggStat(games, (g) => g.st.explored[pidOf(g)].size / TOTAL_CELLS),
    bombs: aggStat(games, (g) => g.st.bombs[pidOf(g)]),
    hits: aggStat(games, (g) => g.st.hits[pidOf(g)]),
    walls: aggStat(games, (g) => g.st.walls[pidOf(g)]),
    crates: aggStat(games, (g) => g.st.crates[pidOf(g)]),
    moveRate: aggStat(games, (g) => {
      const pid = pidOf(g);
      return g.st.ticksAlive[pid] ? g.st.moves[pid] / g.st.ticksAlive[pid] : 0;
    }),
  };
}

// ---------------- 主流程 ----------------
(async () => {
  const t0 = Date.now();
  const maps = sampleMaps(MAPS_PER_THEME);
  const isolatedMaps = maps.filter((m) => m._isolated);
  console.log(`地图抽样: ${maps.length} 张（${maps.filter((m) => !m._isolated).length} 通 / ${isolatedMaps.length} 隔离）`);

  let wanted;
  if (MODELS_ARG.length) {
    wanted = MODELS_ARG;
  } else {
    const all = discoverModels();
    const trans = all.filter((x) => x.arch === 'transformer');
    const mlp = all.find((x) => x.arch === 'mlp' && x.gs >= 5e9) || all.find((x) => x.arch === 'mlp');
    wanted = [...trans.map((x) => x.name), ...(mlp ? [mlp.name] : [])];
  }
  const models = [];
  for (const n of wanted) {
    try { models.push(await loadModel(n)); }
    catch (e) { console.log(`跳过 ${n}: ${e.message}`); }
  }

  // 对战任务清单
  const jobs = [];
  for (const mod of models) {
    if (OPP_ARG.includes('self')) jobs.push({ kind: 'self', a: mod, b: null });
    if (OPP_ARG.includes('hunter')) jobs.push({ kind: 'hunter', a: mod, b: null });
  }
  if (OPP_ARG.includes('cross')) {
    // 默认相邻配对（进化曲线）；cross:i,j 或 cross:i 支持显式指定
    let pairs = [];
    const ci = OPP_ARG.find((o) => o.startsWith('cross:'));
    if (ci) {
      const idx = ci.split(':')[1].split(',').map((x) => parseInt(x, 10));
      pairs.push([models[idx[0]], idx.length > 1 ? models[idx[1]] : models[idx[0] + 1]]);
    } else {
      for (let i = 0; i + 1 < models.length; i++) pairs.push([models[i], models[i + 1]]);
    }
    for (const [a, b] of pairs) jobs.push({ kind: 'cross', a, b });
  }
  if (VS_ARG.length) {
    // 每个模型 vs 指定陪练（--vs a,b,c），出胜率
    const bench = [];
    for (const n of VS_ARG) {
      const m = models.find((x) => x.name === n);
      if (m) bench.push(m);
    }
    for (const mod of models) {
      for (const b of bench) {
        if (b.name !== mod.name) jobs.push({ kind: 'vs', a: mod, b });
      }
    }
  }
  if (PAIRS_ARG) {
    // --pairs "0,5;1,2"：显式索引配对（cross）
    for (const pr of PAIRS_ARG.split(';')) {
      const [ia, ib] = pr.split(',').map((x) => parseInt(x, 10));
      if (models[ia] && models[ib]) jobs.push({ kind: 'cross', a: models[ia], b: models[ib] });
    }
  }

  // 跑任务
  const res = {};   // key = `${a.name}|${b.name||'hunter'}` → { all:[games], isolated:[games] }
  for (const job of jobs) {
    const key = job.kind === 'cross' || job.kind === 'vs'
      ? `${job.kind}|${job.a.name}|${job.b.name}` : `${job.a.name}|${job.kind}`;
    const rec = { all: [], isolated: [] };
    let actorA, actorB;
    if (job.kind === 'hunter') {
      const h = new HunterAI();
      actorA = async (sim, pid, rng) => await job.a.m.act(sim, pid, rng);
      actorB = (sim, pid) => h.act(sim, pid);
    } else {
      actorA = async (sim, pid, rng) => await job.a.m.act(sim, pid, rng);
      actorB = job.b ? async (sim, pid, rng) => await job.b.m.act(sim, pid, rng)
                     : async (sim, pid, rng) => await job.a.m.act(sim, pid, rng);
    }
    const actors = [actorA, actorB];
    // 旧模型兼容: 任一 actor 是 13 宽旧模型 且 地图是空场景 → oldMode
    const isOld = (mm) => mm && mm.meta && mm.meta.obs_shape && mm.meta.obs_shape[2] === 13;
    let gi = 0;
    for (const lv of maps) {
      const oldMode = (lv.source || '') === 'empty_scene' &&
        (isOld(job.a.m) || (job.b && isOld(job.b.m)));
      for (let e = 0; e < EP_PER_MAP; e++) {
        const seed = SEED + gi * 7919 + (job.kind === 'self' ? 0 : 31337);
        const swapped = SWAP && (gi % 2 === 1);       // 奇数局对调双方位置
        const use = swapped ? [actors[1], actors[0]] : actors;
        const st = await playOne(lv, seed, use, MAX_TICK, oldMode);
        const game = { level: lv, st, swapped };
        rec.all.push(game);
        if (lv._isolated) rec.isolated.push(game);
        gi++;
      }
    }
    res[key] = rec;
    console.log(`[进度] ${key} 完成 (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  }

  // ---------------- 输出 ----------------
  const pad = (s, n) => String(s).padEnd(n);
  console.log('\n================ 指标口径：探索率=visited/195，自杀=被自己泡炸死，命中=造成对方掉血 ================');
  for (const job of jobs) {
    const key = job.kind === 'cross' || job.kind === 'vs'
      ? `${job.kind}|${job.a.name}|${job.b.name}` : `${job.a.name}|${job.kind}`;
    const rec = res[key];
    if (job.kind === 'self') {
      const s = summarizeRole(rec.all, 'a');
      console.log(`\n[同模型内耗] ${job.a.name}`);
      console.log(`  自杀 ${(s.suicideRate * 100).toFixed(1)}%  探索 ${(s.explore * 100).toFixed(0)}%  ` +
                  `放炮 ${s.bombs.toFixed(0)}/局  命中 ${s.hits.toFixed(1)}/局  炸墙 ${s.walls.toFixed(1)}  吃箱 ${s.crates.toFixed(1)}  ` +
                  `移动率 ${(s.moveRate * 100).toFixed(0)}%  平均局长 ${s.ticks.toFixed(0)} tick  n=${s.n}`);
      const iso = summarizeRole(rec.isolated, 'a');
      if (rec.isolated.length) {
        console.log(`  [隔离地图] 自杀 ${(iso.suicideRate * 100).toFixed(1)}%  探索 ${(iso.explore * 100).toFixed(0)}%  ` +
                    `放炮 ${iso.bombs.toFixed(0)}/局  局长 ${iso.ticks.toFixed(0)}  n=${iso.n}`);
      }
      if (SWAP) {
        const p0 = summarize(rec.all.filter((g) => !g.swapped), 0);
        const p1 = summarize(rec.all.filter((g) => g.swapped), 1);
        console.log(`  [位置偏置] 模型在 P0(我方): 胜率 ${(p0.winRate * 100).toFixed(1)}% (n=${p0.n})  |  ` +
                    `模型在 P1(敌方): 胜率 ${(p1.winRate * 100).toFixed(1)}% (n=${p1.n})`);
      }
    } else if (job.kind === 'hunter') {
      const s = summarizeRole(rec.all, 'a');
      console.log(`\n[vs Hunter] ${job.a.name}  胜率 ${(s.winRate * 100).toFixed(1)}%（${s.wins}胜/${s.draws}平/${s.losses}负）`);
      console.log(`  自杀 ${(s.suicideRate * 100).toFixed(1)}%  探索 ${(s.explore * 100).toFixed(0)}%  ` +
                  `放炮 ${s.bombs.toFixed(0)}/局  命中 ${s.hits.toFixed(1)}/局  吃箱 ${s.crates.toFixed(1)}  局长 ${s.ticks.toFixed(0)}  n=${s.n}`);
    } else {
      const a = summarizeRole(rec.all, 'a'), b = summarizeRole(rec.all, 'b');
      console.log(`\n[${job.kind === 'vs' ? 'vs基准' : '交叉'}] ${job.a.name} vs ${job.b.name}`);
      console.log(`  A 胜率 ${(a.winRate * 100).toFixed(1)}%（${a.wins}胜/${a.draws}平/${a.losses}负）  B 胜率 ${(b.winRate * 100).toFixed(1)}%`);
      console.log(`  A: 自杀 ${(a.suicideRate * 100).toFixed(1)}% 探索 ${(a.explore * 100).toFixed(0)}% 放炮 ${a.bombs.toFixed(0)} 命中 ${a.hits.toFixed(1)} 吃箱 ${a.crates.toFixed(1)} 局长 ${a.ticks.toFixed(0)}`);
      console.log(`  B: 自杀 ${(b.suicideRate * 100).toFixed(1)}% 探索 ${(b.explore * 100).toFixed(0)}% 放炮 ${b.bombs.toFixed(0)} 命中 ${b.hits.toFixed(1)} 吃箱 ${b.crates.toFixed(1)} 局长 ${b.ticks.toFixed(0)}`);
      if (SWAP) {
        const aP0 = summarize(rec.all.filter((g) => !g.swapped), 0);
        const aP1 = summarize(rec.all.filter((g) => g.swapped), 1);
        console.log(`  [位置偏置] ${job.a.name} 在 P0(我方): 胜率 ${(aP0.winRate * 100).toFixed(1)}% (n=${aP0.n})  |  ` +
                    `在 P1(敌方): 胜率 ${(aP1.winRate * 100).toFixed(1)}% (n=${aP1.n})`);
      }
    }
  }

  if (OUT) {
    const out = {};
    for (const job of jobs) {
      const key = job.kind === 'cross' || job.kind === 'vs'
      ? `${job.kind}|${job.a.name}|${job.b.name}` : `${job.a.name}|${job.kind}`;
      const rec = res[key];
      out[key] = {
        all: {
          a: summarizeRole(rec.all, 'a'), b: summarizeRole(rec.all, 'b'),
          p0: summarize(rec.all, 0), p1: summarize(rec.all, 1),
        },
        isolated: {
          a: summarizeRole(rec.isolated, 'a'), b: summarizeRole(rec.isolated, 'b'),
          p0: summarize(rec.isolated, 0), p1: summarize(rec.isolated, 1),
        },
      };
    }
    out._meta = { maps: maps.length, isolated: isolatedMaps.length, epPerMap: EP_PER_MAP,
                  maxTick: MAX_TICK, seed: SEED, elapsedSec: ((Date.now() - t0) / 1000).toFixed(1) };
    fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
    console.log(`\nJSON 已写: ${OUT}`);
  }
  console.log(`\n耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
})().catch((e) => { console.error('FATAL:', e); process.exit(1); });
