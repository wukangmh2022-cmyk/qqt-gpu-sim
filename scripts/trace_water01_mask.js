#!/usr/bin/env node
/* Trace masked model actions against water01 collision movement. */
'use strict';

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
global.ort = ort;
const Q = require(path.join(__dirname, '..', 'web', 'sim.js'));

const ROOT = path.join(__dirname, '..');
const MODEL_NAME = 'params_it00000889';
const LEVEL_SOURCE = 'water01_4.map';
const TICKS = Number(process.env.TICKS || 900);
const SEED = Number(process.env.SEED || 7);
const TRACE_SEED = Number(process.env.TRACE_SEED || 123);
const RADII = [0.45, 0.49];
const DIR_NAMES = ['up', 'down', 'left', 'right', 'idle'];
const MOVE_DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];

function add(a, b) { return a + b; }
function zeroStats() {
  return {
    ticks: 0,
    aliveTicks: 0,
    sampledMaskInvalid: 0,
    sampledMoveMaskInvalid: 0,
    sampledBombMaskInvalid: 0,
    maskLegalButNoop: 0,
    partialCollision: 0,
    fullStep: 0,
    wallHit: 0,
    brickHit: 0,
    bombHit: 0,
    boundaryHit: 0,
    boundaryAction: 0,
    boundaryMaskBlocked: 0,
    repeatDirection: 0,
    actionCounts: [0, 0, 0, 0, 0],
    legalCounts: [0, 0, 0, 0, 0],
    displacement: 0,
  };
}

function moveBlocked(pre, mv, expected, blocked) {
  const [dy, dx] = MOVE_DIRS[mv];
  let coord;
  if (dy !== 0) {
    coord = Q.resolveAxis(pre[0] + dy * expected, dy * expected,
                          pre[1], pre[0], pre[1], blocked, Q.CFG.radius,
                          Q.H, Q.W, true);
    if (Math.abs(coord - pre[0]) < expected - 1e-5) return true;
    const startR = Math.max(0, Math.min(Q.H - 1, Math.floor(pre[0])));
    const startC = Math.max(0, Math.min(Q.W - 1, Math.floor(pre[1])));
    const lo = Math.max(0, Math.min(Q.H - 1, Math.floor(Math.min(pre[0], coord))));
    const hi = Math.max(0, Math.min(Q.H - 1, Math.floor(Math.max(pre[0], coord))));
    for (let r = lo; r <= hi; r++) {
      if (r !== startR && blocked[r * Q.W + startC]) return true;
    }
  } else {
    coord = Q.resolveAxis(pre[1] + dx * expected, dx * expected,
                          pre[0], pre[0], pre[1], blocked, Q.CFG.radius,
                          Q.H, Q.W, false);
    if (Math.abs(coord - pre[1]) < expected - 1e-5) return true;
    const startR = Math.max(0, Math.min(Q.H - 1, Math.floor(pre[0])));
    const startC = Math.max(0, Math.min(Q.W - 1, Math.floor(pre[1])));
    const lo = Math.max(0, Math.min(Q.W - 1, Math.floor(Math.min(pre[1], coord))));
    const hi = Math.max(0, Math.min(Q.W - 1, Math.floor(Math.max(pre[1], coord))));
    for (let c = lo; c <= hi; c++) {
      if (c !== startC && blocked[startR * Q.W + c]) return true;
    }
  }
  return false;
}

function boundaryBlocked(pre, mv, expected) {
  const [dy, dx] = MOVE_DIRS[mv];
  const target = dy !== 0 ? pre[0] + dy * expected : pre[1] + dx * expected;
  const min = Q.CFG.radius;
  const max = dy !== 0 ? Q.H - Q.CFG.radius : Q.W - Q.CFG.radius;
  return target < min - 1e-9 || target > max + 1e-9;
}

function classify(stats, pre, post, action, mask, sim, pid, before) {
  if (!sim.alive[pid]) return;
  stats.aliveTicks++;
  const mv = action[0];
  const dy = post[0] - pre[0];
  const dx = post[1] - pre[1];
  const dist = Math.hypot(dy, dx);
  stats.displacement += dist;
  stats.actionCounts[mv]++;
  for (let i = 0; i < 5; i++) stats.legalCounts[i] += mask.mm[pid][i] ? 1 : 0;

  const moveInvalid = !mask.mm[pid][mv];
  const bombInvalid = !mask.bm[pid][action[1]];
  if (moveInvalid || bombInvalid) stats.sampledMaskInvalid++;
  if (moveInvalid) stats.sampledMoveMaskInvalid++;
  if (bombInvalid) stats.sampledBombMaskInvalid++;
  if (mv < 4) {
    const expected = Q.CFG.stepLen * sim.spdG[pid];
    if (dist < expected * 0.05) {
      stats.maskLegalButNoop += mask.mm[pid][mv] ? 1 : 0;
      if (mask.mm[pid][mv]) {
        const wallBlocked = moveBlocked(pre, mv, expected, before.wall);
        const brickBlocked = moveBlocked(pre, mv, expected, before.brick);
        const bombBlocked = moveBlocked(pre, mv, expected, before.fuse);
        if (wallBlocked) stats.wallHit++;
        if (brickBlocked) stats.brickHit++;
        if (bombBlocked) stats.bombHit++;
        if (boundaryBlocked(pre, mv, expected)) stats.boundaryHit++;
      }
    } else if (dist < expected - 1e-5) {
      stats.partialCollision++;
      if (boundaryBlocked(pre, mv, expected)) stats.boundaryHit++;
      if (moveBlocked(pre, mv, expected, before.wall)) stats.wallHit++;
      if (moveBlocked(pre, mv, expected, before.brick)) stats.brickHit++;
      if (moveBlocked(pre, mv, expected, before.fuse)) stats.bombHit++;
    } else {
      stats.fullStep++;
    }
    const [dr, dc] = MOVE_DIRS[mv];
    const nr = Math.floor(pre[0] + dr * Q.CFG.radius + 1e-5);
    const nc = Math.floor(pre[1] + dc * Q.CFG.radius + 1e-5);
    if (nr < 0 || nr >= Q.H || nc < 0 || nc >= Q.W) stats.boundaryAction++;
  }
}

function runOne(sim, model, radius, rng) {
  Q.CFG.radius = radius;
  sim.reset(sim.level);
  const stats = [zeroStats(), zeroStats()];
  const previous = [null, null];
  for (let t = 0; t < TICKS && !sim.done; t++) {
    const mask = sim.legalMask();
    return Promise.all([model.act(sim, 0, rng), model.act(sim, 1, rng)]).then(([a0, a1]) => {
      const actions = [a0, a1];
      const pre = [[sim.pos[0], sim.pos[1]], [sim.pos[2], sim.pos[3]]];
      for (let p = 0; p < 2; p++) {
        if (previous[p] != null && previous[p] === actions[p][0] && actions[p][0] < 4) stats[p].repeatDirection++;
        previous[p] = actions[p][0];
      }
      sim.step(actions);
      const post = [[sim.pos[0], sim.pos[1]], [sim.pos[2], sim.pos[3]]];
      for (let p = 0; p < 2; p++) {
        stats[p].ticks++;
        classify(stats[p], pre[p], post[p], actions[p], mask, sim, p);
      }
      return t + 1 < TICKS && !sim.done ? runLoop() : null;
    });
    function runLoop() {
      return runOneTick();
    }
    function runOneTick() {
      return Promise.resolve();
    }
  }
  return Promise.resolve(stats);
}

async function runTrace(level, model, radius) {
  Q.CFG.radius = radius;
  const sim = new Q.Sim(SEED);
  sim.reset(level);
  const stats = [zeroStats(), zeroStats()];
  const previous = [null, null];
  const rng = Q.mulberry32(TRACE_SEED);
  for (let t = 0; t < TICKS && !sim.done; t++) {
    const mask = sim.legalMask();
    const actions = [await model.act(sim, 0, rng), await model.act(sim, 1, rng)];
    const pre = [[sim.pos[0], sim.pos[1]], [sim.pos[2], sim.pos[3]]];
    const before = {
      wall: Uint8Array.from(sim.wall),
      brick: Uint8Array.from(sim.brick),
      fuse: Uint8Array.from(sim.fuse, (v) => v > 0 ? 1 : 0),
    };
    for (let p = 0; p < 2; p++) {
      if (previous[p] != null && previous[p] === actions[p][0] && actions[p][0] < 4) stats[p].repeatDirection++;
      previous[p] = actions[p][0];
    }
    sim.step(actions);
    const post = [[sim.pos[0], sim.pos[1]], [sim.pos[2], sim.pos[3]]];
    for (let p = 0; p < 2; p++) {
      stats[p].ticks++;
      classify(stats[p], pre[p], post[p], actions[p], mask, sim, p);
    }
  }
  return { radius, ticks: sim.t, done: sim.done, winner: sim.winner, stats };
}

(async () => {
  const dir = path.join(ROOT, 'web', 'models');
  const doc = JSON.parse(fs.readFileSync(path.join(dir, `${MODEL_NAME}.json`), 'utf8'));
  const sess = await ort.InferenceSession.create(path.join(dir, `${MODEL_NAME}.onnx`), { executionProviders: ['cpu'] });
  const model = new Q.ORTTransformerModel(doc, sess);
  model.inferEvery = 1;
  const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'assets', 'maps', 'levels.json'), 'utf8'));
  const level = levels.find((x) => x.source === LEVEL_SOURCE);
  if (!level) throw new Error(`${LEVEL_SOURCE} not found`);
  for (const radius of RADII) console.log(JSON.stringify(await runTrace(level, model, radius)));
})().catch((err) => { console.error(err.stack || err); process.exit(1); });
