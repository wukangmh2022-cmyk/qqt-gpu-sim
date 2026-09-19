const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const JevPureAI = require('../web/jev_pure_ai.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const { Sim, StationaryDefenseAI, FleeBotAI, RoamBotAI, HunterAI } = QQT;

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

const sleep = ms => new Promise(r => setTimeout(r, ms));

const MOVE_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'IDLE'];

async function runMatch(enemyConfig, maxTicks = 60) {
  const { name, levelName, botFactory } = enemyConfig;
  console.log(`\n======================================================`);
  console.log(`⚔️  对战测试: JevPureAI (P0) VS ${name} (${levelName})`);
  console.log(`======================================================`);

  const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));
  const level = levels.find(l => l.source === 'contest01_8.map') || levels[0];

  const sim = new Sim(2026);
  sim.reset(level);

  const pureAi = new JevPureAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 4
  });

  const enemyBot = botFactory();
  const rng = QQT.mulberry32 ? QQT.mulberry32(12345) : Math.random;

  const moveCounts = { UP: 0, DOWN: 0, LEFT: 0, RIGHT: 0, IDLE: 0 };
  let bombsPlanted = 0;
  let jevDecisions = [];
  let tick = 0;

  const startTime = Date.now();

  for (tick = 0; tick < maxTicks; tick++) {
    if (!sim.alive[0] || !sim.alive[1]) break;

    const act0 = pureAi.act(sim, 0, rng);
    let act1 = [4, 0];
    if (enemyBot) {
      act1 = enemyBot.act(sim, 1, rng);
    }

    moveCounts[MOVE_NAMES[act0[0]]] = (moveCounts[MOVE_NAMES[act0[0]]] || 0) + 1;
    if (act0[1] === 1) bombsPlanted++;

    if (pureAi.lastDecision && (!jevDecisions.length || jevDecisions[jevDecisions.length - 1].tick !== pureAi.lastInferTick)) {
      jevDecisions.push({
        tick: pureAi.lastInferTick,
        decision: pureAi.lastDecision,
        pos0: sim.centerCell(0),
        pos1: sim.centerCell(1)
      });
    }

    sim.step([act0, act1]);

    // 模拟真实时钟（100ms/tick），为异步在线推演提供往返窗口
    await sleep(90);
  }

  const durationSec = ((Date.now() - startTime) / 1000).toFixed(1);
  const p0Alive = Boolean(sim.alive[0]);
  const p1Alive = Boolean(sim.alive[1]);
  const p0Hp = sim.hp ? sim.hp[0] : (p0Alive ? 5 : 0);
  const p1Hp = sim.hp ? sim.hp[1] : (p1Alive ? 5 : 0);

  let result = 'DRAW (超时存活)';
  if (p0Alive && !p1Alive) result = '🏆 JevPureAI 胜利 (对手阵亡)';
  else if (!p0Alive && p1Alive) result = `💀 JevPureAI 战败 (${name} 击杀)`;
  else if (!p0Alive && !p1Alive) result = '💥 同归于尽 (双方阵亡)';

  console.log(`\n【对局结果】: ${result}`);
  console.log(`- 运行步数: ${tick} ticks (耗时 ${durationSec}s)`);
  console.log(`- JevPureAI (P0): ${p0Alive ? '🟢 存活' : '🔴 阵亡'} (HP: ${p0Hp})`);
  console.log(`- ${name} (P1): ${p1Alive ? '🟢 存活' : '🔴 阵亡'} (HP: ${p1Hp})`);
  console.log(`- Jev 动作分布: UP=${moveCounts.UP}, DOWN=${moveCounts.DOWN}, LEFT=${moveCounts.LEFT}, RIGHT=${moveCounts.RIGHT}, IDLE=${moveCounts.IDLE}`);
  console.log(`- Jev 放泡次数: ${bombsPlanted}`);
  console.log(`- Jev API 调用总数: ${pureAi.stats.totalCalls}, 平均延迟: ${pureAi.stats.lastLatencyMs}ms`);

  console.log(`\n【关键决策采样】:`);
  for (let i = 0; i < Math.min(jevDecisions.length, 5); i++) {
    const d = jevDecisions[i];
    console.log(`  [Tick ${d.tick}] P0@(${d.pos0[0]},${d.pos0[1]}) vs P1@(${d.pos1[0]},${d.pos1[1]}) -> dir=${d.decision.moveDir}, bomb=${d.decision.bombAct}, threat=${d.decision.currentThreat}`);
  }
  if (jevDecisions.length > 5) {
    const d = jevDecisions[jevDecisions.length - 1];
    console.log(`  ... [Tick ${d.tick}] P0@(${d.pos0[0]},${d.pos0[1]}) vs P1@(${d.pos1[0]},${d.pos1[1]}) -> dir=${d.decision.moveDir}, bomb=${d.decision.bombAct}, threat=${d.decision.currentThreat}`);
  }

  return {
    enemyName: name,
    levelName,
    ticks: tick,
    durationSec,
    p0Alive,
    p1Alive,
    p0Hp,
    p1Hp,
    result,
    totalCalls: pureAi.stats.totalCalls,
    bombsPlanted,
    moveCounts
  };
}

async function runAllTests() {
  console.log('======================================================');
  console.log('🚀 TypeSafe Jev (Doom 纯净版零规则) 全水平规则敌人实机评测');
  console.log('======================================================');

  const enemies = [
    {
      name: '静态木桩 (Idle Target)',
      levelName: 'Level 0: 零威胁靶子',
      botFactory: () => ({ act: () => [4, 0] })
    },
    {
      name: '原地守备 (StationaryDefenseAI)',
      levelName: 'Level 1: 遇险避险+敌近放炮反击',
      botFactory: () => new StationaryDefenseAI()
    },
    {
      name: '纯漫游 (RoamBotAI)',
      levelName: 'Level 2: 全图漫游+避险+吃道具',
      botFactory: () => new RoamBotAI()
    },
    {
      name: '逃跑风筝 (FleeBotAI)',
      levelName: 'Level 3: 全图时空规避+拉扯风筝+贴身落雷',
      botFactory: () => new FleeBotAI()
    },
    {
      name: '规则猎人 (HunterAI)',
      levelName: 'Level 4: 全图 Dijkstra 破砖开路+追杀压迫',
      botFactory: () => new HunterAI()
    },
    {
      name: '顶级时空 (TimeAStarAI)',
      levelName: 'Level 5: 四维时空预判+连环炮竞技追猎',
      botFactory: () => new TimeAStarAI({ mode: 'hunt' })
    }
  ];

  const results = [];
  for (const enemy of enemies) {
    try {
      const res = await runMatch(enemy, 50); // 50 ticks 约 5 秒对局
      results.push(res);
    } catch (err) {
      console.error(`测试 ${enemy.name} 出错:`, err);
      results.push({
        enemyName: enemy.name,
        levelName: enemy.levelName,
        result: `ERROR: ${err.message}`,
        p0Alive: false,
        p1Alive: false
      });
    }
  }

  console.log('\n\n======================================================');
  console.log('📊 全水平规则敌人评测汇总');
  console.log('======================================================');
  console.table(results.map(r => ({
    '对手': r.enemyName,
    '难度等级': r.levelName,
    '战局结果': r.result,
    'Jev存活/HP': `${r.p0Alive ? '存活' : '阵亡'} (${r.p0Hp})`,
    '对手存活/HP': `${r.p1Alive ? '存活' : '阵亡'} (${r.p1Hp})`,
    '总步数': r.ticks,
    'Jev放泡': r.bombsPlanted,
    'API调用数': r.totalCalls
  })));
}

runAllTests().catch(console.error);
