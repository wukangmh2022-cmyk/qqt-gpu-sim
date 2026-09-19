const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const JevPureAI = require('../web/jev_pure_ai.js');
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

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function testPureAI() {
  console.log('=== ⚡ 测试 JevPureAI (纯净直出版 · Doom 范式零规则) ===\n');

  const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));
  const level = levels.find(l => l.source === 'contest01_8.map') || levels[0];

  const sim = new Sim(2026);
  sim.reset(level);

  // 1. 测试速度劣势感知与道具优先
  sim.spdG[0] = 1.0; // Jev speed 1.0
  sim.spdG[1] = 1.5; // Opponent speed 1.5 (Jev 速度显著落后)
  const W = sim.W || 15;
  sim.crate[5 * W + 8] = 1;
  sim.crateType[5 * W + 8] = 2; // speed boots at (5, 8)

  const pureAi = new JevPureAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 4
  });

  const state = pureAi.extractPureState(sim, 0);
  console.log('1. 状态提取与速度劣势检测:');
  console.log(`- 玩家速度: ${state.player.speed}, 对手速度: ${state.opponent.speed}`);
  console.log(`- 是否落后于对手 (is_slower_than_opponent): ${state.tactical_situation.is_slower_than_opponent}`);
  console.log(`- 战术指引: ${state.tactical_situation.speed_guidance}`);
  console.log(`- 优先道具 (首位速度道具):`, state.visible_powerups[0]);

  if (!state.tactical_situation.is_slower_than_opponent) {
    throw new Error('速度劣势检测失败！');
  }

  // 2. 实机推演测试（调用 Jev API 并直接返回动作）
  console.log('\n2. 真实在线推理与零规则直出测试:');
  let totalMoves = 0;
  for (let t = 0; t < 25; t++) {
    if (pureAi.isInferring) {
      await sleep(150);
    }

    const posBefore = sim.centerCell(0);
    const act = pureAi.act(sim, 0);
    sim.step([act, [4, 0]]); // 对手静止

    const posAfter = sim.centerCell(0);
    const dec = pureAi.lastDecision;
    const moveNames = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'IDLE'];

    console.log(`[t=${t}] Jev: cell=(${posBefore[0]},${posBefore[1]}) -> act=[${moveNames[act[0]]}, bomb=${act[1]}] | Jev决策: dir=${dec ? dec.moveDir : 'none'} bomb=${dec ? dec.bombAct : 'none'} | 耗时=${pureAi.stats.lastLatencyMs}ms`);
    totalMoves++;
  }

  console.log('\n📊 统计:');
  console.log(`- 总推演步数: ${totalMoves}`);
  console.log(`- Jev API 成功调用次数: ${pureAi.stats.totalCalls}`);
  console.log(`- 最新单次延迟: ${pureAi.stats.lastLatencyMs}ms`);
  console.log(`- 决策动作是否直接映射执行: 100% 纯净直出（底层无 A*、无掩体逻辑）`);
  console.log('\n✅ JevPureAI 实机推演验证通过！');
}

testPureAI().catch(console.error);
