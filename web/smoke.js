#!/usr/bin/env node
// 冒烟测试：Node 端跑通「sim 引擎 + 模型推理」整条链路，不依赖浏览器。
//
// 用法： node web/smoke.js [models/course_1023m.json]
// 输出：模型 vs 随机 一局的胜负/时长/关键事件 + 不变式检查结果。

'use strict';

const fs = require('fs');
const path = require('path');
const QQT = require('./sim.js');

const { Sim, MLPModel, CFG, MOVE_IDLE } = QQT;

function assert(cond, msg) {
  if (!cond) throw new Error('ASSERT FAIL: ' + msg);
}

// 每 tick 的不变式：坐标在界内、血量/引信合法、无 NaN
function checkInvariants(sim, tick) {
  for (let p = 0; p < 2; p++) {
    const y = sim.pos[p * 2], x = sim.pos[p * 2 + 1];
    assert(isFinite(y) && isFinite(x), `tick ${tick} p${p} pos 非有限`);
    assert(y >= CFG.radius - 1e-9 && y <= 13 - CFG.radius + 1e-9, `tick ${tick} p${p} y 越界: ${y}`);
    assert(x >= CFG.radius - 1e-9 && x <= 13 - CFG.radius + 1e-9, `tick ${tick} p${p} x 越界: ${x}`);
    assert(sim.hp[p] >= 0 && sim.hp[p] <= CFG.maxHp, `tick ${tick} p${p} hp 非法`);
    assert(sim.invuln[p] >= 0, `tick ${tick} p${p} invuln 非法`);
    assert(sim.bombsCap[p] >= 0 && sim.blastCap[p] >= 0, `tick ${tick} p${p} 成长非法`);
  }
  for (let i = 0; i < QQT.N; i++) {
    assert(sim.fuse[i] >= 0 && sim.fuse[i] <= CFG.fuse, `tick ${tick} fuse 非法 @${i}`);
    assert(sim.owner[i] >= -1 && sim.owner[i] < 2, `tick ${tick} owner 非法 @${i}`);
  }
}

function randomAct(rng, sim, pid) {
  const { mm, bm } = sim.legalMask();
  const mv = mm[pid].filter(Boolean).length;
  let k = Math.floor(rng() * mv), a = -1;
  for (let i = 0; i < 5; i++) if (mm[pid][i]) { if (k === 0) { a = i; break; } k--; }
  const b = (bm[pid][1] && rng() < 0.15) ? 1 : 0;
  return [a, b];
}

function runMatch(model, mode, seed, aiPid, tag) {
  const sim = new Sim(seed);
  sim.reset(mode);
  const rng = QQT.mulberry32(seed ^ 0xABCDEF);
  let placed = 0, picked = 0, killed = false;
  while (!sim.done) {
    checkInvariants(sim, sim.t);
    const a0 = aiPid === 0 ? model.act(sim, 0, rng) : randomAct(rng, sim, 0);
    const a1 = aiPid === 1 ? model.act(sim, 1, rng) : randomAct(rng, sim, 1);
    const before = sim.liveBombs(0) + sim.liveBombs(1);
    const info = sim.step([a0, a1]);
    if (info.placed[0] || info.placed[1]) placed++;
    if (info.died[0] || info.died[1]) killed = true;
  }
  const winner = sim.winner;
  const nAlive = (sim.alive[0] ? 1 : 0) + (sim.alive[1] ? 1 : 0);
  return {
    tag, ticks: sim.t, winner, nAlive,
    hp: [sim.hp[0], sim.hp[1]], placed, killed,
    grow: [sim.bombsCap, sim.blastCap, sim.spdG],
  };
}

function main() {
  const modelPath = process.argv[2] ||
    path.join(__dirname, 'models', 'course_1023m.json');
  const doc = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  const model = new MLPModel(doc);
  console.log(`模型: ${doc.meta.name}  step=${doc.meta.global_step}  elo=${doc.meta.elo}`);
  console.log(`观测: ${doc.meta.obs_shape.join('x')}  参数: ${Object.values(doc.tensors)
    .reduce((s, [_, n]) => s + n, 0)}`);

  // 1) 纯随机对局 ×2（open + corridor），验证引擎不炸
  for (const mode of ['open', 'corridor']) {
    const r = runMatch(null, mode, 7, -1, `random-vs-random/${mode}`);
    console.log(`  [sim] ${r.tag}: ${r.ticks} ticks, winner=${r.winner}, ` +
      `hp=${r.hp}, 放泡=${r.placed}, 击杀=${r.killed}`);
    assert(r.ticks > 0 && r.ticks <= CFG.maxSteps, '对局时长非法');
  }

  // 2) 模型 vs 随机（AI 走 P1，和人打 AI 同视角），双地图
  for (const mode of ['open', 'corridor']) {
    const r = runMatch(model, mode, 42, 1, `model-vs-random/${mode}`);
    const win = r.winner === 1;
    console.log(`  [ai]  ${r.tag}: ${r.ticks} ticks, winner=${r.winner}` +
      `(模型${win ? '赢' : r.winner === 0 ? '输' : '平'}), hp=${r.hp}, ` +
      `成长=[泡${r.grow[0]} 威${r.grow[1]} 速${r.grow[2].map(v => v.toFixed(2))}], ` +
      `放泡=${r.placed}, 击杀=${r.killed}`);
    assert(r.ticks <= CFG.maxSteps, 'AI 对局时长非法');
  }

  // 3) 模型自对弈 3 局（观战模式），确认无死锁（双方都可能放泡、能打完）
  let totalTicks = 0;
  for (let s = 100; s < 103; s++) {
    const sim = new Sim(s);
    sim.reset('open');
    const rng = QQT.mulberry32(s * 7 + 1);
    while (!sim.done) {
      checkInvariants(sim, sim.t);
      const a0 = model.act(sim, 0, rng);
      const a1 = model.act(sim, 1, rng);
      sim.step([a0, a1]);
    }
    totalTicks += sim.t;
    console.log(`  [ai]  自对弈 seed=${s}: ${sim.t} ticks, winner=${sim.winner}, hp=${sim.hp}`);
  }
  assert(totalTicks > 0, '自对弈没跑起来');

  // 4) 观测与危险图的数值合法性（抽样 tick）
  const sim = new Sim(5);
  sim.reset('corridor');
  const rng = QQT.mulberry32(99);
  for (let k = 0; k < 300 && !sim.done; k++) {
    const a0 = randomAct(rng, sim, 0);
    const a1 = randomAct(rng, sim, 1);
    sim.step([a0, a1]);
    if (k % 50 === 0) {
      const obs = sim.encodeObs();
      assert(obs.length === 14 * 169, '观测通道数不对');
      let any = false;
      for (let i = 0; i < obs.length; i++) {
        assert(obs[i] >= 0 && obs[i] <= 1.001, `obs 值越界 @${i}=${obs[i]}`);
        if (obs[i] > 0) any = true;
      }
      assert(any, '观测全零（异常）');
      const d = sim.dangerMap();
      for (let i = 0; i < d.length; i++) {
        assert(d[i] >= 0 && d[i] <= 1.001, `danger 越界 @${i}=${d[i]}`);
      }
    }
  }
  console.log('  [ok]  观测/危险图数值合法（corridor 300 ticks 抽样）');

  console.log('\n冒烟测试全部通过 ✔');
}

main();
