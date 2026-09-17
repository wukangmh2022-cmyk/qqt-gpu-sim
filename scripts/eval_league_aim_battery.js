const path = require("path");
const fs = require("fs");
global.ort = require("onnxruntime-node");
const QQT = require(path.join(__dirname, "..", "web", "sim.js"));
const TimeAStarAI = require(path.join(__dirname, "..", "web", "time_astar_ai.js"));
const { Sim, ORTTransformerModel, mulberry32, W, H } = QQT;

async function loadModel(name) {
  const jsonPath = path.join(__dirname, "..", "web", "models", `${name}.json`);
  const onnxPath = path.join(__dirname, "..", "web", "models", `${name}.onnx`);
  if (!fs.existsSync(jsonPath) || !fs.existsSync(onnxPath)) {
    throw new Error(`Model file not found: ${name}`);
  }
  const doc = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const sess = await ort.InferenceSession.create(onnxPath, {
    executionProviders: ["cpu"],
    intraOpNumThreads: 2,
    interOpNumThreads: 2,
  });
  const m = new ORTTransformerModel(doc, sess);
  m.inferEvery = 1;
  return m;
}

const SEEDS_10 = [42, 100, 2026, 7, 99, 314, 555, 777, 888, 1234];
const SEEDS_5  = [42, 100, 2026, 7, 99];

async function evaluateModel(modelId, modelLabel) {
  console.log(`\n==============================================================================`);
  console.log(`🚀 开始深度评测: ${modelLabel} (${modelId})`);
  console.log(`==============================================================================`);
  const model = await loadModel(modelId);

  // Test 1: Static Dummy [4, 0] (Domain B: 终结破局防退化)
  let dummyWins = 0, dummyTimeouts = 0, dummyTicks = 0, dummyBombs = 0, dummyIdleMoves = 0, dummyMinDistSum = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const rng = mulberry32(s ^ 0x1111);
    let t = 0, minDist = 999;
    for (t = 0; t < 1000; t++) {
      const a0 = await model.act(sim, 0, rng);
      if (a0[0] === 4 && a0[1] === 0) dummyIdleMoves++;
      if (a0[1] === 1) dummyBombs++;
      const dist = Math.hypot(sim.pos[0] - sim.pos[2], sim.pos[1] - sim.pos[3]);
      if (dist < minDist) minDist = dist;
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) break;
    }
    if (!sim.alive[1] && sim.alive[0]) dummyWins++;
    else if (t >= 1000) dummyTimeouts++;
    dummyTicks += Math.min(t, 1000);
    dummyMinDistSum += minDist;
  }

  // Test 2: TimeAStarAI Hunt (Domain C: 高压微操与防反杀)
  let astarWins = 0, astarLosses = 0, astarDraws = 0, astarTicks = 0, astarBombs = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const bot = new TimeAStarAI({ mode: "hunt" });
    const rng = mulberry32(s ^ 0x2222);
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

  // Test 3: FleeBotAI (Domain D: 反风筝与中距离追围)
  let fleeWins = 0, fleeLosses = 0, fleeDraws = 0, fleeTicks = 0, fleeBombs = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const flee = new QQT.FleeBotAI();
    const rng = mulberry32(s ^ 0x3333);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await model.act(sim, 0, rng);
      const a1 = flee.act(sim, 1);
      if (a0[1] === 1) fleeBombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) fleeWins++;
    else if (!sim.alive[0] && sim.alive[1]) fleeLosses++;
    else fleeDraws++;
    fleeTicks += Math.min(t, 1000);
  }

  // Test 4: RoamBotAI (游走靶测试)
  let roamWins = 0, roamLosses = 0, roamDraws = 0, roamTicks = 0;
  for (const s of SEEDS_5) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const roam = new QQT.RoamBotAI();
    const rng = mulberry32(s ^ 0x4444);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await model.act(sim, 0, rng);
      const a1 = roam.act(sim, 1);
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) roamWins++;
    else if (!sim.alive[0] && sim.alive[1]) roamLosses++;
    else roamDraws++;
    roamTicks += Math.min(t, 1000);
  }

  // Test 5: Duel vs it782_ema (昨日防守大师：进攻博弈与对峙检测)
  const bossModel = await loadModel("params_it00000782_ema");
  let duelWins = 0, duelLosses = 0, duelDraws = 0, duelTicks = 0, duelBombs0 = 0, duelBombs1 = 0, zeroBombCount = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x5555);
    const r1 = mulberry32(s ^ 0x6666);
    let t = 0, b0InGame = 0, b1InGame = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await model.act(sim, 0, r0);
      const a1 = await bossModel.act(sim, 1, r1);
      if (a0[1] === 1) { duelBombs0++; b0InGame++; }
      if (a1[1] === 1) { duelBombs1++; b1InGame++; }
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (b0InGame < 5) zeroBombCount++;
    if (sim.alive[0] && !sim.alive[1]) duelWins++;
    else if (!sim.alive[0] && sim.alive[1]) duelLosses++;
    else duelDraws++;
    duelTicks += Math.min(t, 1000);
  }

  const res = {
    modelId,
    label: modelLabel,
    dummy: {
      winRate: (dummyWins / SEEDS_10.length * 100).toFixed(0) + "%",
      timeoutRate: (dummyTimeouts / SEEDS_10.length * 100).toFixed(0) + "%",
      avgTicks: (dummyTicks / SEEDS_10.length).toFixed(0) + "t",
      avgBombs: (dummyBombs / SEEDS_10.length).toFixed(1),
      idleMoveRate: (dummyIdleMoves / Math.max(1, dummyTicks) * 100).toFixed(1) + "%",
      minDist: (dummyMinDistSum / SEEDS_10.length).toFixed(1) + "格"
    },
    astar: {
      winRate: (astarWins / SEEDS_10.length * 100).toFixed(0) + "%",
      lossRate: (astarLosses / SEEDS_10.length * 100).toFixed(0) + "%",
      drawRate: (astarDraws / SEEDS_10.length * 100).toFixed(0) + "%",
      avgTicks: (astarTicks / SEEDS_10.length).toFixed(0) + "t",
      avgBombs: (astarBombs / SEEDS_10.length).toFixed(1)
    },
    flee: {
      winRate: (fleeWins / SEEDS_10.length * 100).toFixed(0) + "%",
      timeoutRate: (fleeDraws / SEEDS_10.length * 100).toFixed(0) + "%",
      avgTicks: (fleeTicks / SEEDS_10.length).toFixed(0) + "t",
      avgBombs: (fleeBombs / SEEDS_10.length).toFixed(1)
    },
    roam: {
      winRate: (roamWins / SEEDS_5.length * 100).toFixed(0) + "%",
      avgTicks: (roamTicks / SEEDS_5.length).toFixed(0) + "t"
    },
    boss782: {
      record: `${duelWins}W-${duelDraws}D-${duelLosses}L`,
      winRate: (duelWins / SEEDS_10.length * 100).toFixed(0) + "%",
      drawRate: (duelDraws / SEEDS_10.length * 100).toFixed(0) + "%",
      avgTicks: (duelTicks / SEEDS_10.length).toFixed(0) + "t",
      avgBombsAgent: (duelBombs0 / SEEDS_10.length).toFixed(1),
      avgBombs782: (duelBombs1 / SEEDS_10.length).toFixed(1),
      zeroBombGames: zeroBombCount
    }
  };

  return res;
}

async function main() {
  const models = [
    { id: "params_aggr_it00000419_ema", label: "aggr419 (基线起点)" },
    { id: "params_it00000128_ema",     label: "it128 (长训+1h10m)" },
    { id: "params_it00000192_ema",     label: "it192 (长训+1h40m最新)" },
  ];

  const results = [];
  for (const m of models) {
    const r = await evaluateModel(m.id, m.label);
    results.push(r);
  }

  console.log("\n==============================================================================");
  console.log("🏆 破局长训巡检报告: 跨阶段多维 Aim 战力演化对比");
  console.log("==============================================================================");

  console.log("\n【表 1：Domain B 破局终结力 (vs 静止死靶 [4, 0] 10局实测 - Aim 1)】");
  console.table(results.map(r => ({
    "模型版本": r.label,
    "终结击杀率": r.dummy.winRate,
    "超时率": r.dummy.timeoutRate,
    "平均终结耗时": r.dummy.avgTicks,
    "局均放炮": r.dummy.avgBombs,
    "发呆等待率": r.dummy.idleMoveRate,
    "逼近最近距离": r.dummy.minDist,
  })));

  console.log("\n【表 2：Domain C 极限微操力 (vs 时空 A* 竞技猎人 10局实测 - Aim 2)】");
  console.table(results.map(r => ({
    "模型版本": r.label,
    "反杀胜率": r.astar.winRate,
    "败率": r.astar.lossRate,
    "平局超时率": r.astar.drawRate,
    "平均耗时": r.astar.avgTicks,
    "局均放炮": r.astar.avgBombs,
  })));

  console.log("\n【表 3：反风筝与中距离追击 (vs FleeBot 10局 / RoamBot 5局实测)】");
  console.table(results.map(r => ({
    "模型版本": r.label,
    "FleeBot 胜率": r.flee.winRate,
    "FleeBot 耗时": r.flee.avgTicks,
    "RoamBot 胜率": r.roam.winRate,
    "RoamBot 耗时": r.roam.avgTicks,
  })));

  console.log("\n【表 4：宗师防反抗性与攻防对峙检测 (vs it782_ema 10局巅峰互搏)】");
  console.table(results.map(r => ({
    "模型版本": r.label,
    "对决战绩 (W-D-L)": r.boss782.record,
    "胜率": r.boss782.winRate,
    "平局率": r.boss782.drawRate,
    "平均耗时": r.boss782.avgTicks,
    "我方放炮": r.boss782.avgBombsAgent,
    "it782放炮": r.boss782.avgBombs782,
    "避战(<5炮)局数": r.boss782.zeroBombGames,
  })));
}

main().catch(console.error);
