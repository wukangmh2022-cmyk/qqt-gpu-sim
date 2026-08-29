#!/usr/bin/env node
/* Verify Sim replay snapshots restore every serialized logical field. */
'use strict';

const fs = require('fs');
const path = require('path');
const Q = require(path.join(__dirname, '..', 'web', 'sim.js'));

const levels = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'web', 'assets', 'maps', 'levels.json'), 'utf8'));
const cases = ['water01_4.map', 'box01_8.map'];
const MOVE = [Q.MOVE_UP, Q.MOVE_DOWN, Q.MOVE_LEFT, Q.MOVE_RIGHT, Q.MOVE_IDLE];

function equal(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}
function actionsFor(sim, t) {
  const m = sim.legalMask();
  const out = [];
  for (let p = 0; p < 2; p++) {
    const legal = m.mm[p].map((v, i) => v ? i : -1).filter((i) => i >= 0);
    const mv = legal[(t * 3 + p) % legal.length];
    const bomb = m.bm[p][1] && ((t + p) % 17 === 0) ? 1 : 0;
    out.push([mv, bomb]);
  }
  return out;
}
function assertFrame(a, b, label) {
  for (const k of Object.keys(a)) {
    if (!equal(a[k], b[k])) throw new Error(`${label}: field ${k} differs`);
  }
}

let total = 0;
for (const source of cases) {
  const level = levels.find((l) => l.source === source);
  if (!level) throw new Error(`missing level ${source}`);
  const original = new Q.Sim(20260822);
  const restored = new Q.Sim(20260822);
  original.reset(level);
  restored.reset(level);
  assertFrame(original.snapshotReplay(), restored.snapshotReplay(), `${source} initial`);

  for (let t = 0; t < 180 && !original.done; t++) {
    const actions = actionsFor(original, t);
    const info = original.step(actions);
    const frame = original.snapshotReplay(info);
    restored.restoreReplay(frame);
    assertFrame(frame, restored.snapshotReplay(), `${source} tick ${t + 1}`);
    total++;
  }
}
console.log(`replay round-trip passed: ${total} logical frames across ${cases.length} maps`);
