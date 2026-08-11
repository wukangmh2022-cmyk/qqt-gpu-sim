#!/usr/bin/env node
// Python 参考实现 ↔ JS 移植 对拍：用 deploy/parity_ref.py 生成的固定状态，
// 在 JS 里重建同一状态，比较 danger_map / encode_obs / legal_mask /
// resolve_explosions 四个输出（逐元素，float 容差 1e-4）。

'use strict';

const fs = require('fs');
const QQT = require('./sim.js');

const { Sim, CFG } = QQT;

function maxdiff(a, b) {
  let md = 0, idx = -1;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > md) { md = d; idx = i; }
  }
  return { md, idx, n, na: a.length, nb: b.length };
}

function main() {
  const argv = process.argv.slice(2);
  const sweepMode = argv.includes('--sweep');
  const refPath = argv[argv.length - 1] || 'deploy/ref_state.json';
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));

  if (sweepMode) {
    // 批量对拍：ref.states 是随机状态数组
    let fails = 0;
    ref.states.forEach((entry, k) => {
      const sim = buildSim(entry.state);
      const r = checkState(sim, entry, '');
      if (!r.ok) {
        fails++;
        console.log(`  [FAIL] state#${k}: ${r.msg}`);
      }
    });
    console.log(fails === 0
      ? `\n扫描 ${ref.states.length} 个随机状态，全部一致 ✔`
      : `\n扫描 ${ref.states.length} 个随机状态，${fails} 个不一致 ✘`);
    process.exit(fails === 0 ? 0 : 1);
  }

  const sim = buildSim(ref.state);
  const r = checkState(sim, ref, '');
  console.log(r.ok ? '\n对拍全部一致 ✔' : '\n对拍存在差异 ✘');
  process.exit(r.ok ? 0 : 1);
}

function buildSim(st) {
  const sim = new Sim(0);
  sim.reset('corridor');
  sim.wall.set(st.wall);
  sim.brick.set(st.brick);
  sim.fuse.set(st.fuse);
  sim.owner.set(st.owner);
  sim.bombBlast.set(st.bomb_blast);
  sim.pos.set(st.pos);
  sim.alive = st.alive;
  sim.t = st.t;
  sim.crate.set(st.crate);
  sim.invuln = st.invuln;
  sim.bombsCap = st.bombs_cap;
  sim.blastCap = st.blast_cap;
  return sim;
}

function checkState(sim, ref, _tag) {
  let fail = false;
  const out = { ok: true, msg: '' };
  const check = (name, js, py, tol) => {
    const d = maxdiff(js, py);
    const ok = d.md <= tol && d.na === d.nb;
    if (!ok) {
      fail = true;
      out.ok = false;
      out.msg += `${name} maxdiff=${d.md.toExponential(3)} @${d.idx} ` +
        `(js=${js[d.idx]}, py=${py[d.idx]}) `;
    }
  };
  check('danger', sim.dangerMap(), ref.danger, 1e-4);
  check('obs', sim.encodeObs(), ref.obs, 1e-4);
  const { mm, bm } = sim.legalMask();
  check('mm', mm.flat(), ref.mm.flat(), 0);
  check('bm', bm.flat(), ref.bm.flat(), 0);
  const { covered, triggered } = sim._resolveExplosions();
  check('covered', covered, ref.covered, 0);
  check('triggered', triggered, ref.triggered, 0);
  if (fail) {
    console.log(`  [INFO] 不一致状态: pos=${Array.from(sim.pos)} bombs=` +
      sim.fuse.filter(v => v > 0).length);
  }
  return out;
}

main();
