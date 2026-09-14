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

  let wins = 0, cleanWins = 0, tradeWins = 0, losses = 0, timeouts = 0, mutuals = 0;
  let totalTicks = 0, totalBombs = 0, totalMutualHits = 0;

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
    let matchMutualHits = 0;
    let lastP0TookDmg = false;

    while (!sim.done && sim.t < 1800) {
      const hpBefore = [sim.hp[0], sim.hp[1]];
      const aliveBefore = [sim.alive[0], sim.alive[1]];

      const a0 = await model.act(sim, 0, rng);
      const a1 = ai.act(sim, 1);

      if (a0[1] === 1) p0Bombs++;
      sim.step([a0, a1]);

      const p0TookDmg = aliveBefore[0] && sim.hp[0] < hpBefore[0];
      const p1TookDmg = aliveBefore[1] && sim.hp[1] < hpBefore[1];
      if (p0TookDmg && p1TookDmg) matchMutualHits++;
      lastP0TookDmg = p0TookDmg;
    }

    totalTicks += sim.t;
    totalBombs += p0Bombs;
    totalMutualHits += matchMutualHits;

    let resStr = '平';
    if (!sim.alive[0] && !sim.alive[1]) {
      mutuals++;
      resStr = '同归双亡';
    } else if (sim.alive[0] && !sim.alive[1]) {
      wins++;
      if (lastP0TookDmg) {
        tradeWins++;
        resStr = '换血同归胜';
      } else {
        cleanWins++;
        resStr = '纯无伤胜';
      }
    } else if (!sim.alive[0] && sim.alive[1]) {
      losses++;
      resStr = '负';
    } else {
      if (sim.hp[0] > sim.hp[1]) {
        wins++;
        resStr = '超时血多胜';
      } else if (sim.hp[1] > sim.hp[0]) {
        losses++;
        resStr = '超时血少负';
      } else {
        timeouts++;
        resStr = '超时平局';
      }
    }

    process.stdout.write(`  局 #${g+1}/${numGames}: ${resStr} (ticks: ${sim.t}, hp: ${sim.hp[0]}:${sim.hp[1]}, 互损: ${matchMutualHits})\n`);
  }

  const wr = ((wins / numGames) * 100).toFixed(1);
  const cleanWr = ((cleanWins / numGames) * 100).toFixed(1);
  const tradeRatio = wins > 0 ? ((tradeWins / wins) * 100).toFixed(1) : '0.0';
  const avgTicks = (totalTicks / numGames).toFixed(1);
  const avgBombs = (totalBombs / numGames).toFixed(1);
  const avgMutual = (totalMutualHits / numGames).toFixed(2);
  console.log(`📊 战果: 纯胜=${cleanWins}(${cleanWr}%) 换血胜=${tradeWins}(占胜场${tradeRatio}%) 负=${losses} 平/超时=${timeouts} 双亡=${mutuals} | 总胜率=${wr}% | 局均互损=${avgMutual} | 均长=${avgTicks} ticks | 均放炮=${avgBombs}`);
  return { domain, wins, cleanWins, tradeWins, losses, timeouts, mutuals, wr, cleanWr, tradeRatio, avgTicks, avgBombs, avgMutual };
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
