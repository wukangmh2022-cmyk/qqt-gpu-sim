const path = require('path');
const fs = require('fs');
global.ort = require('onnxruntime-node');
const QQT = require(path.join(__dirname, '..', 'web', 'sim.js'));
const TimeAStarAI = require(path.join(__dirname, '..', 'web', 'time_astar_ai.js'));
const { Sim, ORTTransformerModel, mulberry32, W, H } = QQT;

console.log("==============================================================================");
console.log("🔍 PART 1: 规则敌人行为逻辑、轨迹热力分布与自杀 Bug 深度诊断");
console.log("==============================================================================");

// 1.1 FleeBot 轨迹分布检测
{
  const sim = new Sim({ level: "empty", seed: 42 });
  sim.reset();
  const flee = new QQT.FleeBotAI();
  const hunter = new QQT.HunterAI();
  const posHistory = [];
  
  for (let t = 0; t < 500; t++) {
    const act0 = hunter.act(sim, 0);
    const act1 = flee.act(sim, 1);
    sim.step([act0, act1]);
    posHistory.push([sim.pos[2], sim.pos[3]]);
    if (!sim.alive[1] || !sim.alive[0]) break;
  }
  
  let sumR = 0, sumC = 0;
  for (const [r, c] of posHistory) { sumR += r; sumC += c; }
  const avgR = (sumR / posHistory.length).toFixed(2);
  const avgC = (sumC / posHistory.length).toFixed(2);
  const maxR = Math.max(...posHistory.map(p => p[0])).toFixed(2);
  const maxC = Math.max(...posHistory.map(p => p[1])).toFixed(2);
  const cornerStuck = posHistory.filter(p => p[0] > 11 && p[1] > 13).length;
  
  console.log(`\n[FleeBotAI 轨迹分布] (500 ticks 面对追击):`);
  console.log(`  - 存活步数: ${posHistory.length} ticks | 角色是否存活: ${sim.alive[1]}`);
  console.log(`  - 运动重心: [row: ${avgR}, col: ${avgC}] (地图中心为 [6.5, 7.5])`);
  console.log(`  - 最大到达坐标: [max_row: ${maxR}, max_col: ${maxC}] (右下角极值为 [12, 14])`);
  console.log(`  - 滞留右下死角帧数: ${cornerStuck} / ${posHistory.length} (${(cornerStuck/posHistory.length*100).toFixed(1)}%) -> ${cornerStuck === 0 ? "✓ 无右下角缩角 Bug" : "⚠️ 仍有缩角"}`);
}

// 1.2 StationaryDefenseAI 自杀逻辑 Bug 溯源
{
  const sim = new Sim({ level: "empty" });
  sim.reset();
  const bot = new QQT.StationaryDefenseAI();
  sim.hp[1] = 1;
  sim.pos[2] = 6.5; sim.pos[3] = 7.5;
  sim.pos[0] = 6.5; sim.pos[1] = 6.5;
  
  let diedTick = -1;
  let moveLogged = [];
  for (let t = 0; t < 35; t++) {
    const act1 = bot.act(sim, 1);
    moveLogged.push(`t=${t}: mv=${act1[0]}, bomb=${act1[1]}, fuse=${sim.fuse[6*15+7]}, dng=${sim.dangerMap()[6*15+7].toFixed(3)}`);
    sim.step([[4, 0], act1]);
    if (!sim.alive[1]) { diedTick = t; break; }
  }
  console.log(`\n[StationaryDefenseAI 自爆 Bug 溯源]:`);
  console.log(`  - 结果: ${diedTick >= 0 ? `❌ 在 tick ${diedTick} 自爆死亡` : "✓ 存活"}`);
  console.log(`  - 关键过程抽样:`);
  for (let i = 0; i < Math.min(3, moveLogged.length); i++) console.log(`    ${moveLogged[i]}`);
  console.log(`    ...`);
  for (let i = Math.max(0, moveLogged.length - 3); i < moveLogged.length; i++) console.log(`    ${moveLogged[i]}`);
  console.log(`  - 根因分析: 炸弹放置后，爆炸十字范围内所有邻居格与中心格的 dangerMap() 完全相同，nDng < minDng 恒为假，导致 Bot 每一帧都无法决策走位，只能原地发呆等死。`);
}

// 1.3 HunterAI 狭窄死角/单格闭门自爆 Bug
{
  const sim = new Sim({ level: "empty" });
  sim.reset();
  const hunter = new QQT.HunterAI();
  sim.hp[1] = 1;
  sim.pos[2] = 2.5; sim.pos[3] = 2.5;
  sim.wall[1 * 15 + 2] = 1;
  sim.wall[3 * 15 + 2] = 1;
  sim.wall[2 * 15 + 1] = 1;
  sim.wall[2 * 15 + 3] = 1;
  sim.pos[0] = 0.5; sim.pos[1] = 2.5;
  const act = hunter.act(sim, 1);
  console.log(`\n[HunterAI 4面封死死锁测试]:`);
  console.log(`  - 输入: 上下左右 4 方向全墙 (仅原地 MOVE_IDLE 合法)`);
  console.log(`  - 输出动作: [move=${act[0]}, bomb=${act[1]}]`);
  console.log(`  - 判定: ${act[1] === 1 ? "❌ 存在死角下雷自杀 Bug (canEscape 循环了 0..4，把原地 MOVE_IDLE 误判为 escape 退路)" : "✓ 安全不放雷"}`);
}

// ==============================================================================
// 2. 跨代核心 Checkpoint 多维 Metric 横向全景评测
// ==============================================================================
console.log("\n==============================================================================");
console.log("📊 PART 2: 跨代模型多维 Metric 横向实测 (进攻风格、终结效率、微操防守)");
console.log("==============================================================================");

async function loadModel(name) {
  const doc = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'web', 'models', `${name}.json`), 'utf8'));
  const sess = await ort.InferenceSession.create(path.join(__dirname, '..', 'web', 'models', `${name}.onnx`), {
    executionProviders: ['cpu'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });
  const m = new ORTTransformerModel(doc, sess);
  m.inferEvery = 1;
  return m;
}

const MODELS = [
  { id: "params_aggr_it00000419_ema", label: "aggr419 (极速进攻肉搏王)" },
  { id: "params_it00000782_ema",     label: "it782 (昨日最新防守大师)" },
  { id: "params_it00000279_ema",     label: "it279 (优雅纯胜平衡版)" },
  { id: "ViTModel2_31.9B",          label: "ViT2_31.9B (原始高频基模)" }
];

async function runBenchmark() {
  const loaded = {};
  for (const m of MODELS) {
    loaded[m.id] = await loadModel(m.id);
  }

  const seeds = [42, 100, 2026, 7, 99];
  const summary = [];

  for (const m of MODELS) {
    const model = loaded[m.id];

    // --- Metric 维度 1: 面对静止靶 [4, 0] (开阔地 1000t 上限) ---
    let idleKills = 0, idleTimeouts = 0, idleTicks = 0, idleBombs = 0, idleMoves = 0, idleMinDistSum = 0;
    for (const s of seeds) {
      const sim = new Sim({ level: "empty", seed: s });
      sim.reset();
      const rng = mulberry32(s ^ 0x1234);
      let t = 0, minDist = 999;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, rng);
        if (a0[0] === 4 && a0[1] === 0) idleMoves++;
        if (a0[1] === 1) idleBombs++;
        const dist = Math.hypot(sim.pos[0] - sim.pos[2], sim.pos[1] - sim.pos[3]);
        if (dist < minDist) minDist = dist;
        sim.step([a0, [4, 0]]);
        if (!sim.alive[1] || !sim.alive[0]) break;
      }
      if (!sim.alive[1] && sim.alive[0]) idleKills++;
      else if (t >= 1000) idleTimeouts++;
      idleTicks += Math.min(t, 1000);
      idleMinDistSum += minDist;
    }

    // --- Metric 维度 2: 面对时空 A* 追猎者 (TimeAStarAI hunt) ---
    let astarWins = 0, astarLosses = 0, astarDraws = 0, astarTicks = 0, astarBombs = 0;
    for (const s of seeds) {
      const sim = new Sim({ level: "empty", seed: s });
      sim.reset();
      const bot = new TimeAStarAI({ mode: "hunt" });
      const rng = mulberry32(s ^ 0x5678);
      let t = 0;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, rng);
        const a1 = bot.act(sim, 1);
        if (a0[1] === 1) astarBombs++;
        sim.step([a0, a1]);
        if (!sim.alive[0] || !sim.alive[1]) break;
      }
      if (sim.alive[0] && !sim.alive[1]) astarWins++;
      else if (!sim.alive[0] && sim.alive[1]) astarLosses++;
      else astarDraws++;
      astarTicks += Math.min(t, 1000);
    }

    summary.push({
      label: m.label,
      idleKillRate: (idleKills / seeds.length * 100).toFixed(0) + "%",
      idleTimeout: (idleTimeouts / seeds.length * 100).toFixed(0) + "%",
      idleAvgTicks: (idleTicks / seeds.length).toFixed(0) + "t",
      idleAvgBombs: (idleBombs / seeds.length).toFixed(1),
      idleMovePct: (idleMoves / Math.max(1, idleTicks) * 100).toFixed(1) + "%",
      idleMinDist: (idleMinDistSum / seeds.length).toFixed(1) + "格",
      astarWinRate: (astarWins / seeds.length * 100).toFixed(0) + "%",
      astarLossRate: (astarLosses / seeds.length * 100).toFixed(0) + "%",
      astarDrawRate: (astarDraws / seeds.length * 100).toFixed(0) + "%",
      astarAvgTicks: (astarTicks / seeds.length).toFixed(0) + "t",
      astarBombs: (astarBombs / seeds.length).toFixed(1),
    });
  }

  console.log("\n【表 1：主动终结能力 (vs 静止死靶 [4, 0]，5 Seed 实测)】");
  console.table(summary.map(s => ({
    "模型版本与风格": s.label,
    "击杀胜率": s.idleKillRate,
    "1000t超时率": s.idleTimeout,
    "平均耗时": s.idleAvgTicks,
    "局均放炮": s.idleAvgBombs,
    "发呆率": s.idleMovePct,
    "最近逼近距离": s.idleMinDist
  })));

  console.log("\n【表 2：极限抗压与反杀能力 (vs 时空 A* 竞技猎人，5 Seed 实测)】");
  console.table(summary.map(s => ({
    "模型版本与风格": s.label,
    "反杀胜率": s.astarWinRate,
    "战败率": s.astarLossRate,
    "平局率": s.astarDrawRate,
    "局均耗时": s.astarAvgTicks,
    "局均放炮": s.astarBombs
  })));

  // --- Metric 维度 3: 模型间闭环互搏矩阵 (Duel Matrix) ---
  console.log("\n【表 3：模型间直接闭环互搏矩阵 (P0 行 vs P1 列，5 Seed 胜-平-负)】");
  const duelMatrix = {};
  for (const m0 of MODELS) {
    duelMatrix[m0.label] = {};
    for (const m1 of MODELS) {
      if (m0.id === m1.id) {
        duelMatrix[m0.label][m1.label] = "-";
        continue;
      }
      let w = 0, d = 0, l = 0;
      for (const s of seeds) {
        const sim = new Sim({ level: "empty", seed: s });
        sim.reset();
        const r0 = mulberry32(s ^ 0xAAAA);
        const r1 = mulberry32(s ^ 0xBBBB);
        for (let t = 0; t < 1000; t++) {
          const a0 = await loaded[m0.id].act(sim, 0, r0);
          const a1 = await loaded[m1.id].act(sim, 1, r1);
          sim.step([a0, a1]);
          if (!sim.alive[0] || !sim.alive[1]) break;
        }
        if (sim.alive[0] && !sim.alive[1]) w++;
        else if (!sim.alive[0] && sim.alive[1]) l++;
        else d++;
      }
      duelMatrix[m0.label][m1.label] = `${w}胜-${d}平-${l}负`;
    }
  }
  console.table(duelMatrix);
}

runBenchmark().catch(console.error);
