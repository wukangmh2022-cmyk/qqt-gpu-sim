const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const JevGridAI = require('../web/jev_grid_ai.js');
const { Sim } = QQT;

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

const JevAutonomousAI = require('../web/jev_autonomous_ai.js');
const JevAI = require('../web/jev_ai.js');

async function testHumanBombChain() {
  const models = [
    { name: 'JevGridAI (高维坐标网格版)', cls: JevGridAI },
    { name: 'JevAutonomousAI (完全自主版)', cls: JevAutonomousAI },
    { name: 'JevAI (经典版)', cls: JevAI }
  ];

  const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));
  const level = levels.find(l => l.name === '空场景') || levels[levels.length - 1];
  const W = level.w || level.width || 15;

  for (const m of models) {
    console.log(`\n==================================================`);
    console.log(`=== 🏃 测试 ${m.name} 应对人类连环放泡 ===`);
    console.log(`==================================================`);

    const sim = new Sim(2026);
    sim.reset(level);
    sim.hp[0] = 5;
    sim.hp[1] = 5;

    const jev = new m.cls({
      apiUrl: 'https://api.typesafe.ai/v1/systemone',
      inferIntervalTicks: 8
    });

    // Jev at (6, 6), Opponent at (6, 8)
    sim.pos[0] = 6.5; sim.pos[1] = 6.5;
    sim.pos[2] = 6.5; sim.pos[3] = 8.5;

    // Opponent drops a chain of 4 bombs chasing Jev:
    // t=0: opponent at (6, 8) drops bomb, moves right to (6, 9)
    // t=2: opponent at (6, 9) drops bomb, moves up to (5, 9)
    // t=4: opponent at (5, 9) drops bomb, moves left to (5, 8)
    // t=6: opponent at (5, 8) drops bomb, moves left to (5, 7)
    let tookDamage = false;
    for (let t = 0; t < 40; t++) {
      let a1 = [4, 0];
      if (t === 0) a1 = [3, 1]; // drop bomb at (6, 8), move right
      if (t === 2) a1 = [0, 1]; // drop bomb at (6, 9), move up
      if (t === 4) a1 = [2, 1]; // drop bomb at (5, 9), move left
      if (t === 6) a1 = [2, 1]; // drop bomb at (5, 8), move left

      const a0 = jev.act(sim, 0);
      sim.step([a0, a1]);

      const cell = sim.centerCell(0);
      const danger = jev.helperAi.buildDangerMap(sim, t * 100);
      const inDanger = danger.hasFutureDanger(cell[0] * W + cell[1], t * 100);
      const nextStart = danger.nextDangerStart(cell[0] * W + cell[1], t * 100);
      const ttb = nextStart !== null ? (nextStart - t * 100) : 'none';

      if (t % 5 === 0 || inDanger || sim.hp[0] < 5) {
        console.log(`[t=${t}] Jev: cell=(${cell[0]},${cell[1]}) act=[${a0}] inDanger=${inDanger} timeToBoom=${ttb}ms hp=${sim.hp[0]}`);
      }

      if (sim.hp[0] < 5) {
        console.log(`❌ [t=${t}] ${m.name} 受到伤害！当前 HP=${sim.hp[0]}`);
        tookDamage = true;
        break;
      }
    }

    console.log(`结果: ${m.name} 最终 HP = ${sim.hp[0]} / 5, 受伤: ${tookDamage ? '是' : '否'}`);
  }
}

testHumanBombChain().catch(console.error);
