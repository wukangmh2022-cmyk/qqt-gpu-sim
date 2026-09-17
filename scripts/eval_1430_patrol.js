const path = require("path");
const fs = require("fs");
global.ort = require("onnxruntime-node");
const QQT = require(path.join(__dirname, "..", "web", "sim.js"));
const TimeAStarAI = require(path.join(__dirname, "..", "web", "time_astar_ai.js"));
const { Sim, ORTTransformerModel, mulberry32 } = QQT;

async function loadModel(name) {
  const jsonPath = path.join(__dirname, "..", "web", "models", `${name}.json`);
  const onnxPath = path.join(__dirname, "..", "web", "models", `${name}.onnx`);
  const doc = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
  const sess = await ort.InferenceSession.create(onnxPath, { executionProviders: ["cpu"] });
  const m = new ORTTransformerModel(doc, sess);
  m.inferEvery = 1;
  return m;
}

const SEEDS_10 = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022];
const SEEDS_5  = [42, 100, 2026, 7, 99];

async function runAll(mTarget, labelTarget, mBase, m782) {
  console.log(`=== 开始全量评测: ${labelTarget} ===`);

  // 1. 静止死靶
  let dWins = 0, dTimeouts = 0, dTicks = 0, dIdleMoves = 0, dBombs = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const rng = mulberry32(s ^ 0x1111);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mTarget.act(sim, 0, rng);
      if (a0[0] === 4 && a0[1] === 0) dIdleMoves++;
      if (a0[1] === 1) dBombs++;
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) break;
    }
    if (!sim.alive[1] && sim.alive[0]) dWins++;
    else if (t >= 1000) dTimeouts++;
    dTicks += Math.min(t, 1000);
  }

  // 2. 时空 A*
  let aWins = 0, aLoss = 0, aDraws = 0, aTicks = 0, aBombs = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const bot = new TimeAStarAI({ mode: "hunt" });
    const rng = mulberry32(s ^ 0x2222);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mTarget.act(sim, 0, rng);
      const a1 = bot.act(sim, 1);
      if (a0[1] === 1) aBombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) aWins++;
    else if (!sim.alive[0] && sim.alive[1]) aLoss++;
    else aDraws++;
    aTicks += Math.min(t, 1000);
  }

  // 3. FleeBot
  let fWins = 0, fLoss = 0, fDraws = 0, fTicks = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const flee = new QQT.FleeBotAI();
    const rng = mulberry32(s ^ 0x3333);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mTarget.act(sim, 0, rng);
      const a1 = flee.act(sim, 1);
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) fWins++;
    else if (!sim.alive[0] && sim.alive[1]) fLoss++;
    else fDraws++;
    fTicks += Math.min(t, 1000);
  }

  // 4. Duel vs 基模 aggr419
  let bWins = 0, bLoss = 0, bDraws = 0, bTicks = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0xAAAA);
    const r1 = mulberry32(s ^ 0xBBBB);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mTarget.act(sim, 0, r0);
      const a1 = await mBase.act(sim, 1, r1);
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) bWins++;
    else if (!sim.alive[0] && sim.alive[1]) bLoss++;
    else bDraws++;
    bTicks += Math.min(t, 1000);
  }

  // 5. Duel vs 防守大师 it782
  let sWins = 0, sLoss = 0, sDraws = 0, sTicks = 0, sBombs0 = 0, sBombs1 = 0;
  for (const s of SEEDS_10) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x5555);
    const r1 = mulberry32(s ^ 0x6666);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mTarget.act(sim, 0, r0);
      const a1 = await m782.act(sim, 1, r1);
      if (a0[1] === 1) sBombs0++;
      if (a1[1] === 1) sBombs1++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) sWins++;
    else if (!sim.alive[0] && sim.alive[1]) sLoss++;
    else sDraws++;
    sTicks += Math.min(t, 1000);
  }

  return {
    label: labelTarget,
    dummyKill: `${(dWins/10*100).toFixed(0)}% (${(dTicks/10).toFixed(0)}t, idle ${(dIdleMoves/dTicks*100).toFixed(1)}%)`,
    astarWin: `${(aWins/10*100).toFixed(0)}% W - ${(aLoss/10*100).toFixed(0)}% L (${(aTicks/10).toFixed(0)}t)`,
    fleeWin: `${(fWins/10*100).toFixed(0)}% (${(fTicks/10).toFixed(0)}t)`,
    vsBase419: `${bWins}W-${bDraws}D-${bLoss}L (胜率 ${(bWins/10*100).toFixed(0)}%)`,
    vsBoss782: `${sWins}W-${sDraws}D-${sLoss}L (放炮 ${(sBombs0/10).toFixed(1)} vs ${(sBombs1/10).toFixed(1)})`
  };
}

(async () => {
  const mBase = await loadModel("params_aggr_it00000419_ema");
  const m782  = await loadModel("params_it00000782_ema");
  const m192  = await loadModel("params_it00000192_ema");
  const m256  = await loadModel("params_it00000256_ema");

  const r192 = await runAll(m192, "it192 (1h40m)", mBase, m782);
  const r256 = await runAll(m256, "it256 (2h10m 最新)", mBase, m782);

  console.log("\n==============================================================================");
  console.log("🏆 14:30 巡检核心对照报表 (去美化、硬指标直面)");
  console.log("==============================================================================");
  console.table([r192, r256]);
})();
