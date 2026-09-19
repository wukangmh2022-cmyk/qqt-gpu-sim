const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevGridAI = require('../web/jev_grid_ai.js');

const { Sim, FleeBotAI } = QQT;
const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));

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

async function testGridAI(levelSource = 'contest01_8.map', maxTicks = 80) {
  console.log(`\n=== 🧪 测试 TypeSafe Jev (高维坐标全图版 · 15x13网格+时序上下文) ===`);
  console.log(`地图: ${levelSource}, 最大 Ticks: ${maxTicks}`);

  const level = levels.find(l => l.source === levelSource) || levels[0];
  const sim = new Sim(2026);
  sim.reset(level);
  sim.hp[0] = 1;
  sim.hp[1] = 1;

  const jev = new JevGridAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 8
  });

  const oppBot = new FleeBotAI();

  for (let t = 0; t < maxTicks; t++) {
    if (jev.isInferring) {
      await sleep(250);
    }

    const a0 = jev.act(sim, 0);
    const a1 = oppBot.act(sim, 1);
    sim.step([a0, a1]);

    if (jev.lastDecision && (t % 16 === 0 || a0[1] === 1)) {
      const dec = jev.lastDecision;
      const bombTag = a0[1] === 1 ? ' [💣 本Tick落子]' : '';
      console.log(`[t=${t}] 意图: ${dec.intent} | 目标网格: (${dec.row}, ${dec.col}) | 动作: ${dec.bombChoice}${bombTag} | 当前位: (${sim.pos[0].toFixed(1)}, ${sim.pos[1].toFixed(1)})`);
    }

    if (sim.done) {
      console.log(`\n🎉 [t=${t}] 对局结束！胜者: P${sim.winner} (0=Jev, 1=对手, -1=平局)`);
      break;
    }
  }

  console.log(`\n⏱️ 仿真测试结束。`);
  console.log(`Jev 存活: ${sim.alive[0]}, 对手存活: ${sim.alive[1]}`);
  console.log(`Jev 总放泡数: ${jev.stats.bombsPlaced}, 总推理次数: ${jev.stats.totalCalls}`);
  console.log(`时序上下文历史条数: ${jev.temporalHistory.length}`);
  if (jev.temporalHistory.length > 0) {
    console.log(`最新一条时序历史:`, JSON.stringify(jev.temporalHistory[jev.temporalHistory.length - 1]));
  }
}

testGridAI('contest01_8.map', 70).catch(console.error);
