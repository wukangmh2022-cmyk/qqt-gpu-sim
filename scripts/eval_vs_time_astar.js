#!/usr/bin/env node
'use strict';
const path = require('path');
const fs = require('fs');
const ort = require('onnxruntime-node');
global.ort = ort;

const ROOT = path.join(__dirname, '..');
const QQT = require(path.join(ROOT, 'web', 'sim.js'));
const TimeAStarAI = require(path.join(ROOT, 'web', 'time_astar_ai.js'));
const { Sim, ORTTransformerModel, mulberry32, W, H } = QQT;

async function runMatch(model, modelName, domain, numGames = 16) {
  const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'assets', 'maps', 'levels.json'), 'utf8'));
  const lvList = Array.isArray(levels) ? levels : (levels.levels || levels.maps);

  let wins = 0, losses = 0, timeouts = 0, mutuals = 0, suicides = 0;
  let totalTicks = 0, totalBombs = 0;

  console.log(`\n⚔️ 开始对战: [${modelName}] vs [TimeAStarAI (竞技追猎版)] | 场景: ${domain} | 场次: ${numGames}`);

  for (let g = 0; g < numGames; g++) {
    const seed = 42 + g * 31337;
    const sim = new Sim(seed);
    if (domain === 'open') {
      sim.reset('open');
    } else {
      sim.reset(lvList[g % lvList.length]);
    }

    const ai = new TimeAStarAI({ mode: 'hunt' });
    const rng = mulberry32(seed ^ 0x9e3779b9);

    let p0Bombs = 0;
    let p0Suicide = false;

    while (!sim.done && sim.t < 1800) {
      const hpBefore = [sim.hp[0], sim.hp[1]];
      const aliveBefore = [sim.alive[0], sim.alive[1]];

      const a0 = await model.act(sim, 0, rng);
      const a1 = ai.act(sim, 1);

      if (a0[1] === 1) p0Bombs++;
      sim.step([a0, a1]);

      if (aliveBefore[0] && !sim.alive[0] && aliveBefore[1] && sim.alive[1]) {
        // 如果己方阵亡而对方未直接接触伤害
      }
    }

    totalTicks += sim.t;
    totalBombs += p0Bombs;

    if (!sim.alive[0] && !sim.alive[1]) {
      mutuals++;
    } else if (sim.alive[0] && !sim.alive[1]) {
      wins++;
    } else if (!sim.alive[0] && sim.alive[1]) {
      losses++;
    } else {
      if (sim.hp[0] > sim.hp[1]) wins++;
      else if (sim.hp[1] > sim.hp[0]) losses++;
      else timeouts++;
    }

    process.stdout.write(`  局 #${g+1}/${numGames}: ${sim.alive[0] ? (sim.alive[1] ? '平' : '胜') : (sim.alive[1] ? '负' : '同归')} (ticks: ${sim.t}, hp: ${sim.hp[0]}:${sim.hp[1]})\n`);
  }

  const wr = ((wins / numGames) * 100).toFixed(1);
  const avgTicks = (totalTicks / numGames).toFixed(1);
  const avgBombs = (totalBombs / numGames).toFixed(1);
  console.log(`📊 战果: 胜=${wins} 负=${losses} 平/超时=${timeouts} 同归=${mutuals} | 胜率=${wr}% | 均长=${avgTicks} ticks | 均放炮=${avgBombs}`);
  return { domain, wins, losses, timeouts, mutuals, wr, avgTicks, avgBombs };
}

async function main() {
  const modelName = process.argv[2] || 'params_it00000068';
  const numGames = parseInt(process.argv[3] || '16', 10);
  const doc = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'models', `${modelName}.json`), 'utf8'));
  const sess = await ort.InferenceSession.create(path.join(ROOT, 'web', 'models', `${modelName}.onnx`), {
    executionProviders: ['cpu'],
    intraOpNumThreads: 2,
    interOpNumThreads: 2,
  });
  const model = new ORTTransformerModel(doc, sess);
  model.inferEvery = 1;

  await runMatch(model, modelName, 'open', numGames);
  await runMatch(model, modelName, 'full241', numGames);
}

main().catch(console.error);
