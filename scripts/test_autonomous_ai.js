const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevAutonomousAI = require('../web/jev_autonomous_ai.js');

const { Sim, FleeBotAI } = QQT;
const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));

// 从环境变量或 .env 读取 API Key
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

async function testAutonomousAI(levelSource = 'contest01_8.map', maxTicks = 150) {
  console.log(`\n=== 🧪 测试 TypeSafe Jev (完全自主版 GPT-5.6 设想) ===`);
  console.log(`地图: ${levelSource}, 最大 Ticks: ${maxTicks}`);

  const level = levels.find(l => l.source === levelSource) || levels[0];
  const sim = new Sim(2026);
  sim.reset(level);
  sim.hp[0] = 1;
  sim.hp[1] = 1;

  const jev = new JevAutonomousAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 10
  });

  const oppBot = new FleeBotAI();

  for (let t = 0; t < maxTicks; t++) {
    if (jev.isInferring) {
      await sleep(300);
    }

    const a0 = jev.act(sim, 0);
    const a1 = oppBot.act(sim, 1);
    sim.step([a0, a1]);

    if (jev.lastDecision && (t % 20 === 0 || a0[1] === 1)) {
      const dec = jev.lastDecision;
      const bombTag = a0[1] === 1 ? ' [💣 本Tick落子]' : '';
      console.log(`[t=${t}] 姿态: ${dec.posture} | 目标: ${dec.targetKey} | 动作: ${dec.bombAction} (conf=${dec.bombConf.toFixed(2)})${bombTag} | 放炮数: ${jev.stats.bombsPlaced}`);
    }

    if (sim.done) {
      console.log(`\n🎉 [t=${t}] 对局结束！胜者: P${sim.winner} (0=Jev自主版, 1=对手, -1=平局)`);
      console.log(`Jev 存活: ${sim.alive[0]}, 对手存活: ${sim.alive[1]}`);
      console.log(`Jev 总放泡数: ${jev.stats.bombsPlaced}, 总推理次数: ${jev.stats.totalCalls}, 平均延迟: ${jev.stats.lastLatencyMs}ms`);
      return sim.winner;
    }
  }

  console.log(`\n⏱️ 仿真达到 ${maxTicks} Ticks。`);
  console.log(`Jev 存活: ${sim.alive[0]}, 对手存活: ${sim.alive[1]}`);
  console.log(`Jev 总放泡数: ${jev.stats.bombsPlaced}, 总推理次数: ${jev.stats.totalCalls}`);
  return -1;
}

testAutonomousAI('contest01_8.map', 120).catch(console.error);
