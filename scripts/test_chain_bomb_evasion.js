const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
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

async function testChainBomb() {
  console.log('=== 💣 测试对手连环放炮时 Jev 的反应 ===');
  // Open map / corridor
  const level = levels.find(l => l.source === 'contest01_8.map') || levels[0];
  const sim = new Sim(2026);
  sim.reset(level);
  sim.hp[0] = 5;
  sim.hp[1] = 5;

  const jev = new JevGridAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 8
  });

  const W = sim.W || 15;

  // Jev starts at (6, 5)
  // P1 drops a bomb right at (6, 6), then (6, 4)...
  sim.pos[0] = 6.5; sim.pos[1] = 5.5; // Jev at (6, 5)
  sim.pos[2] = 6.5; sim.pos[3] = 6.5; // Opponent at (6, 6)

  console.log(`Jev 初始位置: (${sim.centerCell(0)})`);
  console.log(`对手初始位置: (${sim.centerCell(1)})`);

  for (let t = 0; t < 40; t++) {
    if (jev.isInferring) await sleep(200);

    const ownBefore = sim.centerCell(0);

    // Opponent drops bomb at (6, 6) at t=0, then moves away
    let a1 = [4, 0];
    if (t === 0) a1 = [3, 1]; // drop bomb at (6, 6), move right to (6, 7)
    if (t === 3) a1 = [3, 1]; // drop bomb at (6, 7), move right to (6, 8)

    const a0 = jev.act(sim, 0);

    sim.step([a0, a1]);

    const ownAfter = sim.centerCell(0);
    const danger = jev.helperAi.buildDangerMap(sim, t * 100);
    const inDanger = danger.hasFutureDanger(ownAfter[0] * W + ownAfter[1], t * 100);
    const nextStart = danger.nextDangerStart(ownAfter[0] * W + ownAfter[1], t * 100);
    const timeToBoom = nextStart !== null ? (nextStart - t * 100) : 'none';

    console.log(`[t=${t}] Jev: cell=(${ownBefore[0]},${ownBefore[1]}) act=[${a0}] inFutureDanger=${inDanger} timeToBoom=${timeToBoom}ms hp=${sim.hp[0]}`);

    if (sim.hp[0] <= 0 || !sim.alive[0]) {
      console.log(`\n💀 [t=${t}] Jev 被炸死！`);
      break;
    }
  }
}

testChainBomb().catch(console.error);
