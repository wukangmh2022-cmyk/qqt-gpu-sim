const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevGridAI = require('../web/jev_grid_ai.js');
const JevAutonomousAI = require('../web/jev_autonomous_ai.js');
const JevAI = require('../web/jev_ai.js');

const { Sim } = QQT;
const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));

function getApiKey() {
  if (process.env.TYPESAFE_API_KEY) return process.env.TYPESAFE_API_KEY;
  const envPath = path.join(process.env.HOME || '', 'Documents/杂项/jev_lab/jev-browser/.env');
  if (fs.existsSync(envPath)) {
    const lines = fs.readFileSync(envPath, 'utf8').split('\n');
    for (const l of lines) {
      if (l.startsWith('TYPESAFE_API_KEY=')) {
        return l.split('=')[1].trim().replace(/['"]/g, '');
      }
    }
  }
  return '';
}
process.env.TYPESAFE_API_KEY = getApiKey();

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function runStationaryTest(levelSource = 'contest01_8.map', maxTicks = 150) {
  console.log(`\n=== 🥊 对战静止敌人实测 (${levelSource}) ===`);
  const level = levels.find(l => l.source === levelSource) || levels[0];
  const sim = new Sim(2026);
  sim.reset(level);
  sim.hp[0] = 1;
  sim.hp[1] = 1;

  const jev = new JevGridAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 8
  });

  const p1Pos = sim.centerCell(1);
  console.log(`静止敌人位于: (${p1Pos[0]}, ${p1Pos[1]}), HP=1 (一血模式)`);
  console.log(`Jev 初始位置: (${sim.centerCell(0)[0]}, ${sim.centerCell(0)[1]})\n`);

  let firstAttackTick = -1;
  let killTick = -1;

  for (let t = 0; t < maxTicks; t++) {
    if (jev.isInferring) {
      await sleep(200);
    }

    const a0 = jev.act(sim, 0);
    const a1 = [4, 0]; // 静止敌人：完全不移动、不放泡

    sim.step([a0, a1]);

    const own = sim.centerCell(0);
    const distToOpp = Math.abs(own[0] - p1Pos[0]) + Math.abs(own[1] - p1Pos[1]);

    if (jev.lastDecision && t % 8 === 0) {
      const dec = jev.lastDecision;
      const bombTag = a0[1] === 1 ? ' [💣 本Tick落子]' : '';
      console.log(`[t=${t}] 意图: ${dec.intent.padEnd(16)} | 目标: (${dec.row}, ${dec.col}) -> 投影: [${jev.targetPos}] | 距敌: ${distToOpp}格 | 当前位: (${own[0]}, ${own[1]})${bombTag}`);
    }

    if (distToOpp <= 2 && firstAttackTick === -1) {
      firstAttackTick = t;
      console.log(`🎯 [t=${t}] Jev 首次接近静止敌人至 2 格距离！`);
    }

    if (sim.done || sim.hp[1] <= 0 || !sim.alive[1]) {
      killTick = t;
      console.log(`\n💥 [t=${t}] 静止敌人已被击败！胜者: P${sim.winner} (0=Jev, 1=对手)`);
      break;
    }
  }

  console.log(`\n📊 统计:`);
  console.log(`- 首次逼近敌方耗时: ${firstAttackTick !== -1 ? firstAttackTick + ' ticks (' + (firstAttackTick*0.1).toFixed(1) + 's)' : '未能逼近'}`);
  console.log(`- 击杀敌方耗时: ${killTick !== -1 ? killTick + ' ticks (' + (killTick*0.1).toFixed(1) + 's)' : '未能击杀'}`);
  console.log(`- Jev 存活: ${sim.alive[0]}, 敌人存活: ${sim.alive[1]}`);
  console.log(`- Jev 总落子数: ${jev.stats.bombsPlaced}`);
}

runStationaryTest('contest01_8.map', 130).catch(console.error);
