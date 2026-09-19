const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevAI = require('../web/jev_ai.js');

const { Sim } = QQT;
const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function testStationary() {
  console.log('=== 测试 Jev vs 静止死靶 (empty_scene, 1 HP) ===');
  const level = levels.find(l => l.source === 'empty_scene') || levels[0];
  const sim = new Sim(12345);
  sim.reset(level);
  sim.hp[0] = 1;
  sim.hp[1] = 1;

  const jev = new JevAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 8
  });

  for (let t = 0; t < 150; t++) {
    if (jev.isInferring) {
      await sleep(250);
    }
    const a0 = jev.act(sim, 0);
    const a1 = [4, 0]; // 静止靶
    sim.step([a0, a1]);

    if (t % 10 === 0) {
      const dec = jev.lastTacticalDecision;
      console.log(`[t=${t}] P0: (${sim.pos[0].toFixed(1)}, ${sim.pos[1].toFixed(1)}), P1: (${sim.pos[2].toFixed(1)}, ${sim.pos[3].toFixed(1)}), 决策: ${dec?.priority}, 目标: ${dec?.targetKey}, 放炮数: ${jev.stats.bombsPlaced}`);
    }

    if (sim.done) {
      console.log(`[t=${t}] 对局结束！胜者: P${sim.winner} (0=Jev, 1=Target, -1=平局)`);
      return sim.winner;
    }
  }
  console.log('超时未结束！');
  return -1;
}

testStationary().catch(console.error);
