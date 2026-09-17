const path = require("path");
const fs = require("fs");
global.ort = require("onnxruntime-node");
const QQT = require(path.join(__dirname, "..", "web", "sim.js"));
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

class PureFleeBot extends QQT.FleeBotAI {
  act(sim, pid) {
    const act = super.act(sim, pid);
    return [act[0], 0];
  }
}

const SEEDS_20 = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022, 314, 555, 777, 888, 999, 1314, 1688, 2048, 4096, 8192];

async function runCheckpointEval(m, label, mBase, m782) {
  console.log(`=== 开始全量评测: ${label} ===`);

  // 1. 静止木桩 (20局)
  let dWins = 0, dTimeouts = 0, dTicks = 0, dIdle = 0, dBombs = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const rng = mulberry32(s ^ 0x1111);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, rng);
      if (a0[0] === 4 && a0[1] === 0) dIdle++;
      if (a0[1] === 1) dBombs++;
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) break;
    }
    if (!sim.alive[1] && sim.alive[0]) dWins++;
    else if (t >= 1000) dTimeouts++;
    dTicks += Math.min(t, 1000);
  }

  // 2. 纯逃跑木桩 (20局)
  let fWins = 0, fTimeouts = 0, fTicks = 0, fIdle = 0, fBombs = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const flee = new PureFleeBot();
    const rng = mulberry32(s ^ 0x2222);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, rng);
      const a1 = flee.act(sim, 1);
      if (a0[0] === 4 && a0[1] === 0) fIdle++;
      if (a0[1] === 1) fBombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) fWins++;
    else if (t >= 1000) fTimeouts++;
    fTicks += Math.min(t, 1000);
  }

  // 3. 自对弈镜像局 (20局)
  let sBothIdle = 0, sTicks = 0, sTimeouts = 0, sBombs0 = 0, sBombs1 = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x3333);
    const r1 = mulberry32(s ^ 0x4444);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, r0);
      const a1 = await m.act(sim, 1, r1);
      if (a0[0] === 4 && a0[1] === 0 && a1[0] === 4 && a1[1] === 0) sBothIdle++;
      if (a0[1] === 1) sBombs0++;
      if (a1[1] === 1) sBombs1++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (t >= 1000) sTimeouts++;
    sTicks += Math.min(t, 1000);
  }

  // 4. 对战自身基模 aggr419 (20局)
  let bWins = 0, bDraws = 0, bLoss = 0, bTicks = 0, b0Bombs = 0, b1Bombs = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x5555);
    const r1 = mulberry32(s ^ 0x6666);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, r0);
      const a1 = await mBase.act(sim, 1, r1);
      if (a0[1] === 1) b0Bombs++;
      if (a1[1] === 1) b1Bombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) bWins++;
    else if (!sim.alive[0] && sim.alive[1]) bLoss++;
    else bDraws++;
    bTicks += Math.min(t, 1000);
  }

  // 5. 对战昨日防守大师 it782 (20局)
  let kWins = 0, kDraws = 0, kLoss = 0, kTicks = 0, k0Bombs = 0, k1Bombs = 0;
  for (let i = 0; i < 20; i++) {
    const s = SEEDS_20[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x7777);
    const r1 = mulberry32(s ^ 0x8888);
    let t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, r0);
      const a1 = await m782.act(sim, 1, r1);
      if (a0[1] === 1) k0Bombs++;
      if (a1[1] === 1) k1Bombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) kWins++;
    else if (!sim.alive[0] && sim.alive[1]) kLoss++;
    else kDraws++;
    kTicks += Math.min(t, 1000);
  }

  return {
    label,
    静止木桩: `${(dWins/20*100).toFixed(0)}% 胜 (${(dTicks/20).toFixed(0)}t, 发呆 ${(dIdle/dTicks*100).toFixed(1)}%)`,
    纯逃跑木桩: `${(fWins/20*100).toFixed(0)}% 胜 / ${(fTimeouts/20*100).toFixed(0)}% 超时 (${(fTicks/20).toFixed(0)}t, 发呆 ${(fIdle/fTicks*100).toFixed(1)}%)`,
    自对弈镜像: `平局率 ${(sTimeouts/20*100).toFixed(0)}% | 双挂机 ${(sBothIdle/sTicks*100).toFixed(1)}% (${(sTicks/20).toFixed(0)}t, 炮 ${((sBombs0+sBombs1)/2/20).toFixed(1)})`,
    vs基模419: `${bWins}W-${bDraws}D-${bLoss}L (胜率 ${(bWins/20*100).toFixed(0)}%, 耗时 ${(bTicks/20).toFixed(0)}t, 炮 ${(b0Bombs/20).toFixed(1)} vs ${(b1Bombs/20).toFixed(1)})`,
    vs防守782: `${kWins}W-${kDraws}D-${kLoss}L (胜率 ${(kWins/20*100).toFixed(0)}%, 耗时 ${(kTicks/20).toFixed(0)}t, 炮 ${(k0Bombs/20).toFixed(1)} vs ${(k1Bombs/20).toFixed(1)})`
  };
}

(async () => {
  const mBase = await loadModel("params_aggr_it00000419_ema");
  const m782  = await loadModel("params_it00000782_ema");
  const mBreakthrough223 = await loadModel("params_it00000223_ema");

  const rBreakthrough223 = await runCheckpointEval(mBreakthrough223, "Breakthrough it223 (1h45m 进阶版)", mBase, m782);
  console.log("\n==============================================================================");
  console.log("🏆 破局新训 1小时45分快照 it223 核心五维攻防实测报表 (各20局全量实测)");
  console.log("==============================================================================");
  console.log(JSON.stringify(rBreakthrough223, null, 2));
})();
