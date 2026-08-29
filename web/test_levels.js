#!/usr/bin/env node
// test_levels.js —— 新地图系统 Node 冒烟测试：
// 加载 web/assets/maps/levels.json，在代表性关卡上跑 Sim + 规则 Hunter vs 随机，
// 校验 15×13 不变量（坐标界内、血量/引信合法、出生点可通行、爆率路径不崩）。
'use strict';

const fs = require('fs');
const path = require('path');
const QQT = require('./sim.js');

const { Sim, CFG, MOVE_IDLE } = QQT;
const H = QQT.H, W = QQT.W, N = QQT.N;

const levels = JSON.parse(fs.readFileSync(path.join(__dirname, 'assets/maps/levels.json'), 'utf8'));
const byId = new Map(levels.map((l) => [l.id, l]));

function assert(c, msg) { if (!c) throw new Error('ASSERT FAIL: ' + msg); }
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randomAct(r, sim, pid) {
  const { mm, bm } = sim.legalMask();
  const mv = mm[pid].filter(Boolean).length;
  let k = Math.floor(r() * mv), a = -1;
  for (let i = 0; i < 5; i++) if (mm[pid][i]) { if (k === 0) { a = i; break; } k--; }
  return [a, r() < 0.12 && bm[pid] ? 1 : 0];
}
const hunter = new QQT.HunterAI();

function run(level, seed, ticks) {
  const sim = new Sim(seed);
  sim.reset(level);
  // 出生点校验：脚下无墙无砖
  for (let p = 0; p < 2; p++) {
    const i = Math.floor(sim.pos[p * 2]) * W + Math.floor(sim.pos[p * 2 + 1]);
    assert(!sim.wall[i] && !sim.brick[i], `${level.source} p${p} 出生点被挡`);
  }
  // 初始属性校验
  const st = level.initial_stats;
  assert(sim.bombsCap[0] === st.bombs && sim.blastCap[0] === st.blast &&
         Math.abs(sim.spdG[0] - st.speed) < 1e-9, `${level.source} 初始属性不符`);
  // 空场景十字宝箱
  if (level.initial_crates && level.initial_crates.length) {
    let n = 0;
    for (let i = 0; i < N; i++) if (sim.crate[i]) n++;
    assert(n === level.initial_crates.length, `${level.source} 十字宝箱数 ${n} != ${level.initial_crates.length}`);
  }
  const r = rng(seed ^ 0xABCDEF);
  for (let t = 0; t < ticks; t++) {
    for (let p = 0; p < 2; p++) {
      const y = sim.pos[p * 2], x = sim.pos[p * 2 + 1];
      assert(isFinite(y) && isFinite(x), `t${t} p${p} 非有限`);
      assert(y >= CFG.radius - 1e-9 && y <= H - CFG.radius + 1e-9, `t${t} p${p} y越界 ${y}`);
      assert(x >= CFG.radius - 1e-9 && x <= W - CFG.radius + 1e-9, `t${t} p${p} x越界 ${x}`);
    }
    const a0 = hunter.act(sim, 0), a1 = randomAct(r, sim, 1);
    const res = sim.step([a0, a1]);
    if (sim.done) return { done: true, winner: sim.winner, t: sim.t };
  }
  return { done: false, winner: null, t: ticks };
}

// 代表关卡：普通竞技/比武/夺宝/推箱子/空场景 各挑 1-2 张 + 多格元件图(比武01)
const picks = [
  'desert01_4.map', 'town10_8.map', 'contest01_8.map', 'match01_2.map',
  'treasure01_4.map', 'box01_8.map', 'empty_scene',
];
const sel = [];
for (const src of picks) {
  const lv = levels.find((l) => l.source === src);
  if (lv) sel.push(lv); else console.log('(无此图,跳过)', src);
}

// 旧 13 列模型兼容：空场景右侧两列是永久墙，任何道具都不得重叠。
const empty = levels.find((l) => l.source === 'empty_scene');
if (empty) {
  try {
    const sim = new Sim(123);
    sim.reset(empty, { oldMode: true });
    let overlap = 0;
    for (let r = 0; r < H; r++) {
      for (const c of [W - 2, W - 1]) {
        const i = r * W + c;
        assert(sim.wall[i] === 1, `兼容空场景右侧墙缺失 (${r},${c})`);
        if (sim.crate[i] || sim.superCrate[i] || sim.recycle[i]) overlap++;
      }
    }
    assert(overlap === 0, `兼容空场景右侧墙有 ${overlap} 个重叠道具`);
    console.log('✅ 兼容空场景右侧两列墙体无重叠道具');
  } catch (e) {
    console.log(`❌ 兼容空场景墙体道具: ${e.message}`);
    process.exitCode = 1;
  }
}
let fails = 0;
// 野外躲猫猫草丛：旧导出可能 bush 数组为空，但 layers[0] 的 6003 仍必须可炸。
const field = levels.find((l) => l.source === 'field01_4.map');
if (field) {
  try {
    const bushCells = [];
    for (let i = 0; i < N; i++) {
      if (Math.abs(field.layers[0][i] || 0) === 6003) bushCells.push(i);
    }
    assert(bushCells.length > 0, '野外01 没有找到 6003 草丛');
    const sim = new Sim(456);
    sim.reset(field);
    for (const i of bushCells) assert(sim.bush[i] === 1, `草丛状态未加载 (${i})`);
    const target = bushCells.find((i) => {
      const r = (i / W) | 0, c = i % W;
      return r > 0 && r < H - 1 && c > 0 && c < W - 1 &&
        !sim.wall[i] && !sim.brick[i];
    });
    assert(target != null, '没有可测试的草丛格');
    const tr = (target / W) | 0, tc = target % W;
    // 从相邻的纯地面格连续走入草丛中心，覆盖 AABB/中心路径两条约束。
    const entry = [[tr - 1, tc], [tr + 1, tc], [tr, tc - 1], [tr, tc + 1]]
      .find(([r, c]) => r >= 0 && r < H && c >= 0 && c < W &&
        !sim.wall[r * W + c] && !sim.brick[r * W + c]);
    assert(entry, `草丛没有可进入的相邻地面 (${tr},${tc})`);
    sim.pos[0] = entry[0] + 0.5; sim.pos[1] = entry[1] + 0.5;
    sim.pos[2] = 1.5; sim.pos[3] = 1.5;
    const move = entry[0] < tr ? 1 : entry[0] > tr ? 0 : entry[1] < tc ? 3 : 2;
    for (let t = 0; t < 8; t++) sim.step([[move, 0], [MOVE_IDLE, 0]]);
    assert(Math.floor(sim.pos[0]) === tr && Math.floor(sim.pos[1]) === tc,
      `无法进入草丛 (${entry} -> ${tr},${tc}), 实际=${sim.pos[0]},${sim.pos[1]}`);
    // 草丛中心放泡，连续推进到引爆；草丛应消失且不生成砖/箱阻挡。
    sim.pos[0] = tr + 0.5; sim.pos[1] = tc + 0.5;
    sim.step([[MOVE_IDLE, 1], [MOVE_IDLE, 0]]);
    for (let t = 0; t < 35 && sim.bush[target]; t++) sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
    assert(sim.bush[target] === 0, `草丛被爆炸覆盖后仍存在 (${tr},${tc})`);
    assert(sim.brick[target] === 0 && sim.wall[target] === 0,
      `草丛清除后错误变成阻挡 (${tr},${tc})`);
    console.log('✅ 野外躲猫猫草丛可被爆炸清除');
  } catch (e) {
    console.log(`❌ 野外草丛爆炸: ${e.message}`);
    process.exitCode = 1;
  }
}
for (const lv of sel) {
  try {
    const out = run(lv, 42, 400);
    console.log(`✅ ${lv.source.padEnd(18)} ${lv.name || ''} (${lv.category}) ` +
      `${lv.initial_stats.bombs}泡/${lv.initial_stats.blast}威/${lv.initial_stats.speed}速 ` +
      `rate=${lv.crate_rate} → ${out.done ? `终局 胜者=${out.winner} @${out.t}tick` : '400 tick 未终局'}`);
  } catch (e) {
    fails++;
    console.log(`❌ ${lv.source}: ${e.message}`);
  }
}
// 全量关卡快速跑 60 tick（覆盖所有图不崩）
let allFail = 0;
for (const lv of levels) {
  try { run(lv, 7, 60); } catch (e) { allFail++; console.log(`❌ 全量 ${lv.source}: ${e.message}`); }
}
console.log(`\n代表关卡 ${sel.length - fails}/${sel.length} 通过；全量 241 关 60 tick 失败 ${allFail}`);

// ---- 宝箱语义（与 jax_bomb 对照：sim.js:336 炸砖时掷 crate_rate、拾取必升） ----
// 1) 炸砖爆率统计：p0 首 tick 放泡后全停，40 tick 内引爆。
//    注意不能每 tick 按放泡：放泡判定在爆炸结算前（fuse<=0 即重放），原地连按会
//    无限刷新引信——与 jax step 同款顺序，行为一致（两边都要只放一次）。
function brickCrateRate(level, seed0, trials, ticks) {
  let destroyed = 0, created = 0;
  for (let t = 0; t < trials; t++) {
    const sim = new Sim(seed0 + t);
    sim.reset(level);
    let bStart = 0, cStart = 0;
    for (let i = 0; i < N; i++) { bStart += sim.brick[i]; cStart += sim.crate[i]; }
    for (let k = 0; k < ticks; k++) sim.step([[MOVE_IDLE, k === 0 ? 1 : 0], [MOVE_IDLE, 0]]);
    let bEnd = 0, cEnd = 0;
    for (let i = 0; i < N; i++) { bEnd += sim.brick[i]; if (sim.crate[i] && !sim.recycle[i]) cEnd++; }
    destroyed += bStart - bEnd;
    created += cEnd - cStart;       // 排除掉血回收箱（recycle）
  }
  return { destroyed, created, rate: destroyed ? created / destroyed : NaN };
}
// 2) 拾取必升：p0 脚下直接放箱（普通 + 回收两种 rec 标记），1 tick IDLE 后必长一层
function pickupAlwaysGrows(level, seed, withRecycle) {
  const sim = new Sim(seed);
  sim.reset(level);
  const i = Math.floor(sim.pos[0]) * W + Math.floor(sim.pos[1]);
  sim.crate[i] = 1;
  if (withRecycle) sim.recycle[i] = 1;
  const b0 = sim.bombsCap[0], z0 = sim.blastCap[0], s0 = sim.spdG[0];
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  const grew = (sim.bombsCap[0] - b0) + (sim.blastCap[0] - z0)
    + (sim.spdG[0] - s0) / CFG.growthSpeedStep;
  return { grew, grewOk: Math.abs(grew - 1) < 1e-6 };
}
let crateFail = 0;
const lv4 = levels.find((l) => l.id === 4);          // 沙漠03: 期望 = 关卡 crate_rate (60/W)
const lv5 = levels.find((l) => l.id === 5);          // desert04 rate=1.0
try {
  const r4 = brickCrateRate(lv4, 2000, 1200, 40);
  const ok4 = Math.abs(r4.rate - lv4.crate_rate) < 0.03;
  console.log(`${ok4 ? '✅' : '❌'} JS 炸砖爆率 id=4 实测 ${r4.rate.toFixed(4)}（期望 ${lv4.crate_rate.toFixed(4)}, 样本 ${r4.destroyed}）`);
  if (!ok4 || r4.destroyed < 1000) crateFail++;
  const r5 = brickCrateRate(lv5, 5000, 1200, 40);
  const ok5 = Math.abs(r5.rate - lv5.crate_rate) < 0.03;
  console.log(`${ok5 ? '✅' : '❌'} JS 炸砖爆率 id=5 实测 ${r5.rate.toFixed(4)}（期望 ${lv5.crate_rate.toFixed(4)}, 样本 ${r5.destroyed}）`);
  if (!ok5) crateFail++;
  for (const rec of [false, true]) {
    let bad = 0;
    for (let s = 0; s < 2000; s++) if (!pickupAlwaysGrows(lv4, 9000 + s, rec).grewOk) bad++;
    console.log(`${bad === 0 ? '✅' : '❌'} JS 拾取必升(${rec ? '回收箱rec' : '普通宝箱'}) 2000 次失败 ${bad}`);
    if (bad) crateFail++;
  }
} catch (e) {
  crateFail++;
  console.log(`❌ 宝箱语义测试: ${e.message}`);
}
console.log(`宝箱语义对照：${crateFail === 0 ? '全部通过' : `FAIL ${crateFail} 项`}`);
process.exit(fails || allFail || crateFail ? 1 : 0);
