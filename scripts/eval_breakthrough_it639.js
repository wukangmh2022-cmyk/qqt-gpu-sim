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
  const sess = await ort.InferenceSession.create(onnxPath, { executionProviders: ["cpu"], intraOpNumThreads: 2 });
  const m = new ORTTransformerModel(doc, sess);
  m.inferEvery = 1;
  return m;
}

const SEEDS_20 = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022, 314, 555, 777, 888, 999, 1314, 1688, 2048, 4096, 8192];

async function main() {
  console.log("正在加载 params_it00000639_ema 模型...");
  const mEval = await loadModel("params_it00000639_ema");
  const mBase = await loadModel("params_aggr_it00000419_ema");

  // 1. Domain B: 静止木桩 (20局)
  let dWins = 0, dTimeouts = 0, dTicks = 0, dIdle = 0, dBombs = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const rng = mulberry32(s ^ 0x1111);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mEval.act(sim, 0, rng);
      if (a0[0] === 4 && a0[1] === 0) dIdle++;
      if (a0[1] === 1) dBombs++;
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) break;
    }
    if (!sim.alive[1] && sim.alive[0]) dWins++;
    else if (t >= 1000) dTimeouts++;
    dTicks += Math.min(t, 1000);
  }
  const dAvgTicks = (dTicks / 20).toFixed(1);
  const dIdlePct = ((dIdle / dTicks) * 100).toFixed(2);
  const dWinPct = (dWins / 20 * 100).toFixed(1);
  console.log(`[1/5 Domain B 静止靶] 胜率: ${dWinPct}%, 均长: ${dAvgTicks} 帧, 发呆率: ${dIdlePct}%`);

  // 2. 逃跑风筝 Bot (20局)
  let fWins = 0, fTimeouts = 0, fTicks = 0, fIdle = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const flee = new QQT.FleeBotAI();
    const rng = mulberry32(s ^ 0x2222);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mEval.act(sim, 0, rng);
      const a1 = flee.act(sim, 1);
      if (a0[0] === 4 && a0[1] === 0) fIdle++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) fWins++;
    else if (t >= 1000) fTimeouts++;
    fTicks += Math.min(t, 1000);
  }
  const fAvgTicks = (fTicks / 20).toFixed(1);
  const fIdlePct = ((fIdle / fTicks) * 100).toFixed(2);
  const fWinPct = (fWins / 20 * 100).toFixed(1);
  const fTimeoutPct = (fTimeouts / 20 * 100).toFixed(1);
  console.log(`[2/5 逃跑风筝怪] 斩杀率: ${fWinPct}%, 超时率: ${fTimeoutPct}%, 均长: ${fAvgTicks} 帧, 发呆率: ${fIdlePct}%`);

  // 3. Domain C: 高级时空 A* (竞技追猎版, 10局)
  let aWins = 0, aLoss = 0, aDraws = 0, aTicks = 0;
  for (let i = 0; i < 10; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const astar = new TimeAStarAI({ mode: "hunt" });
    const rng = mulberry32(s ^ 0x3333);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mEval.act(sim, 0, rng);
      const a1 = astar.act(sim, 1);
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) aWins++;
    else if (!sim.alive[0] && sim.alive[1]) aLoss++;
    else aDraws++;
    aTicks += Math.min(t, 1000);
  }
  const aWinPct = (aWins / 10 * 100).toFixed(1);
  const aLossPct = (aLoss / 10 * 100).toFixed(1);
  const aAvgTicks = (aTicks / 10).toFixed(1);
  console.log(`[3/5 时空 A* 追猎] 胜率: ${aWinPct}%, 负率: ${aLossPct}%, 均长: ${aAvgTicks} 帧`);

  // 4. 对战基模 params_aggr_it00000419_ema (20局)
  let bWins = 0, bDraws = 0, bLoss = 0, bTicks = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x5555);
    const r1 = mulberry32(s ^ 0x6666);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mEval.act(sim, 0, r0);
      const a1 = await mBase.act(sim, 1, r1);
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) bWins++;
    else if (!sim.alive[0] && sim.alive[1]) bLoss++;
    else bDraws++;
    bTicks += Math.min(t, 1000);
  }
  const bWinPct = (bWins / 20 * 100).toFixed(1);
  const bLossPct = (bLoss / 20 * 100).toFixed(1);
  const bDrawPct = (bDraws / 20 * 100).toFixed(1);
  const bAvgTicks = (bTicks / 20).toFixed(1);
  console.log(`[4/5 对战基模] 胜率: ${bWinPct}%, 负率: ${bLossPct}%, 平局: ${bDrawPct}%, 均长: ${bAvgTicks} 帧`);

  // 5. 自对弈镜像局 (20局)
  let sBothIdle = 0, sTicks = 0, sTimeouts = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x7777);
    const r1 = mulberry32(s ^ 0x8888);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await mEval.act(sim, 0, r0);
      const a1 = await mEval.act(sim, 1, r1);
      if (a0[0] === 4 && a0[1] === 0 && a1[0] === 4 && a1[1] === 0) sBothIdle++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (t >= 1000) sTimeouts++;
    sTicks += Math.min(t, 1000);
  }
  const sTimeoutPct = (sTimeouts / 20 * 100).toFixed(1);
  const sBothIdlePct = ((sBothIdle / sTicks) * 100).toFixed(2);
  const sAvgTicks = (sTicks / 20).toFixed(1);
  console.log(`[5/5 自对弈镜像] 超时率: ${sTimeoutPct}%, 双闲置率: ${sBothIdlePct}%, 均长: ${sAvgTicks} 帧`);

  console.log("RESULT_JSON:" + JSON.stringify({
    checkpoint: "params_it00000639_ema",
    staticTarget: { winPct: dWinPct, avgTicks: dAvgTicks, idlePct: dIdlePct },
    fleeBot: { winPct: fWinPct, timeoutPct: fTimeoutPct, avgTicks: fAvgTicks, idlePct: fIdlePct },
    timeAStar: { winPct: aWinPct, lossPct: aLossPct, avgTicks: aAvgTicks },
    vsBaseModel: { winPct: bWinPct, lossPct: bLossPct, drawPct: bDrawPct, avgTicks: bAvgTicks },
    selfPlay: { timeoutPct: sTimeoutPct, bothIdlePct: sBothIdlePct, avgTicks: sAvgTicks }
  }));
}

main().catch(console.error);
