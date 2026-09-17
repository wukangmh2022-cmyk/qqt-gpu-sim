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

async function createModel(modelName) {
  const doc = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'models', `${modelName}.json`), 'utf8'));
  const sess = await ort.InferenceSession.create(path.join(ROOT, 'web', 'models', `${modelName}.onnx`), {
    executionProviders: ['cpu'],
    intraOpNumThreads: 2,
    interOpNumThreads: 2,
  });
  const model = new ORTTransformerModel(doc, sess);
  model.inferEvery = 1;
  return model;
}

async function runSeries({
  name,
  model0,
  opp, // 'hunt' | 'roam' | ORTTransformerModel
  oppName,
  domains = ['open', 'full241'],
  gamesPerDomain = 8,
}) {
  const levels = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'assets', 'maps', 'levels.json'), 'utf8'));
  const lvList = Array.isArray(levels) ? levels : (levels.levels || levels.maps);

  console.log(`\n======================================================`);
  console.log(`⚔️ 对决系列: [${name}] vs [${oppName}] (每地图 ${gamesPerDomain} 局)`);
  console.log(`======================================================`);

  const results = {};

  for (const domain of domains) {
    let wins = 0, cleanWins = 0, tradeWins = 0, losses = 0, timeouts = 0, mutuals = 0;
    let totalTicks = 0, totalBombs0 = 0, totalBombs1 = 0, totalIdle0 = 0, totalMutualHits = 0;

    for (let g = 0; g < gamesPerDomain; g++) {
      const seed = 1000 + g * 31337;
      const sim = new Sim(seed);
      if (domain === 'open') {
        sim.reset('open');
      } else {
        sim.reset(lvList[g % lvList.length]);
      }

      let aiOpp = null;
      let isModelOpp = false;
      if (opp === 'hunt' || opp === 'roam') {
        aiOpp = new TimeAStarAI({ mode: opp });
      } else if (opp === 'idle') {
        aiOpp = { act: () => [4, 0] };
      } else {
        isModelOpp = true;
      }

      const rng = mulberry32(seed ^ 0x9e3779b9);
      let p0Bombs = 0, p1Bombs = 0, p0Idle = 0;
      let matchMutualHits = 0;
      let lastP0TookDmg = false;

      while (!sim.done && sim.t < 1800) {
        const hpBefore = [sim.hp[0], sim.hp[1]];
        const aliveBefore = [sim.alive[0], sim.alive[1]];

        const a0 = await model0.act(sim, 0, rng);
        let a1;
        if (isModelOpp) {
          a1 = await opp.act(sim, 1, rng);
        } else {
          a1 = aiOpp.act(sim, 1);
        }

        if (a0[0] === 4) p0Idle++;
        if (a0[1] === 1) p0Bombs++;
        if (a1[1] === 1) p1Bombs++;

        sim.step([a0, a1]);

        const p0TookDmg = aliveBefore[0] && sim.hp[0] < hpBefore[0];
        const p1TookDmg = aliveBefore[1] && sim.hp[1] < hpBefore[1];
        if (p0TookDmg && p1TookDmg) matchMutualHits++;
        lastP0TookDmg = p0TookDmg;
      }

      totalTicks += sim.t;
      totalBombs0 += p0Bombs;
      totalBombs1 += p1Bombs;
      totalIdle0 += p0Idle;
      totalMutualHits += matchMutualHits;

      let resStr = '平';
      if (!sim.alive[0] && !sim.alive[1]) {
        mutuals++;
        resStr = '同归双亡';
      } else if (sim.alive[0] && !sim.alive[1]) {
        wins++;
        if (lastP0TookDmg) {
          tradeWins++;
          resStr = '换血胜';
        } else {
          cleanWins++;
          resStr = '无伤胜';
        }
      } else if (!sim.alive[0] && sim.alive[1]) {
        losses++;
        resStr = '负';
      } else {
        if (sim.hp[0] > sim.hp[1]) {
          wins++;
          resStr = '血优胜';
        } else if (sim.hp[1] > sim.hp[0]) {
          losses++;
          resStr = '血劣负';
        } else {
          timeouts++;
          resStr = '超时平';
        }
      }

      const idlePct = ((p0Idle / Math.max(1, sim.t)) * 100).toFixed(0);
      process.stdout.write(`  [${domain} #${g+1}/${gamesPerDomain}] ${resStr} | ticks: ${sim.t} | hp: ${sim.hp[0]}:${sim.hp[1]} | p0放炮: ${p0Bombs} | 发呆率: ${idlePct}%\n`);
    }

    const n = gamesPerDomain;
    const wr = ((wins / n) * 100).toFixed(1);
    const cleanWr = ((cleanWins / n) * 100).toFixed(1);
    const avgTicks = (totalTicks / n).toFixed(0);
    const avgBombs0 = (totalBombs0 / n).toFixed(1);
    const avgIdlePct = ((totalIdle0 / Math.max(1, totalTicks)) * 100).toFixed(1);
    const avgMutual = (totalMutualHits / n).toFixed(2);

    console.log(`--- [${domain}] 战果: 胜=${wins}/${n} (${wr}%) [无伤=${cleanWins}, 换血=${tradeWins}], 负=${losses}, 双亡/平=${mutuals + timeouts} | 均长=${avgTicks}t | 均炮=${avgBombs0} | 平均发呆率=${avgIdlePct}% | 局均互损=${avgMutual}`);
    results[domain] = { wins, cleanWins, tradeWins, losses, timeouts, mutuals, wr, cleanWr, avgTicks, avgBombs0, avgIdlePct, avgMutual };
  }

  return results;
}

async function main() {
  const modelName = process.argv[2] || 'params_it00000065_ema';
  const baselineName = process.argv[3] || 'params_it00000349_ema';
  const games = parseInt(process.argv[4] || '8', 10);

  console.log(`加载模型 [${modelName}] ...`);
  const mCurrent = await createModel(modelName);

  console.log(`加载底模 [${baselineName}] ...`);
  const mBaseline = await createModel(baselineName);

  // 0. vs 纯静止死靶 (不动不放泡，专门检测原地发呆率)
  await runSeries({
    name: modelName,
    model0: mCurrent,
    opp: 'idle',
    oppName: '纯静止死靶 (不动不放泡)',
    gamesPerDomain: games,
  });

  // 1. vs TimeAStarAI (hunt - 竞技追猎版)
  const resHunt = await runSeries({
    name: modelName,
    model0: mCurrent,
    opp: 'hunt',
    oppName: 'TimeAStarAI (竞技追猎版)',
    gamesPerDomain: games,
  });

  // 2. vs TimeAStarAI (roam - 漫游连炮版)
  const resRoam = await runSeries({
    name: modelName,
    model0: mCurrent,
    opp: 'roam',
    oppName: 'TimeAStarAI (漫游连炮版)',
    gamesPerDomain: games,
  });

  // 3. vs 底模 baseline (params_it00000349_ema)
  const resVsBaseline = await runSeries({
    name: modelName,
    model0: mCurrent,
    opp: mBaseline,
    oppName: baselineName,
    gamesPerDomain: games,
  });

  // 4. vs 上一代模型 (params_it00000065_ema)
  const prevName = process.argv[5] || 'params_it00000065_ema';
  if (prevName && fs.existsSync(path.join(ROOT, 'web', 'models', `${prevName}.onnx`))) {
    console.log(`加载上一代模型 [${prevName}] ...`);
    const mPrev = await createModel(prevName);
    await runSeries({
      name: modelName,
      model0: mCurrent,
      opp: mPrev,
      oppName: prevName,
      gamesPerDomain: games,
    });
  }

  console.log(`\n======================================================`);
  console.log(`🏆 综合测试完成！全部对决数据已统计完毕。`);
  console.log(`======================================================`);
}

main().catch(console.error);
