const path = require("path");
const fs = require("fs");
global.ort = require("onnxruntime-node");
const QQT = require(path.join(__dirname, "..", "web", "sim.js"));
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

async function duel(m0, m1, label0, label1, nGames = 10) {
  const seeds = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022];
  let w0 = 0, w1 = 0, draws = 0, ticksSum = 0, b0Sum = 0, b1Sum = 0;
  for (let i = 0; i < nGames; i++) {
    const s = seeds[i];
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0xAAAA);
    const r1 = mulberry32(s ^ 0xBBBB);
    let t = 0, b0 = 0, b1 = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m0.act(sim, 0, r0);
      const a1 = await m1.act(sim, 1, r1);
      if (a0[1] === 1) b0++;
      if (a1[1] === 1) b1++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    if (sim.alive[0] && !sim.alive[1]) w0++;
    else if (!sim.alive[0] && sim.alive[1]) w1++;
    else draws++;
    ticksSum += Math.min(t, 1000);
    b0Sum += b0;
    b1Sum += b1;
  }
  return {
    match: `${label0} vs ${label1}`,
    score: `${w0}W-${draws}D-${w1}L (胜率 ${(w0/nGames*100).toFixed(0)}%)`,
    avgTicks: (ticksSum / nGames).toFixed(0) + "t",
    b0: (b0Sum / nGames).toFixed(1),
    b1: (b1Sum / nGames).toFixed(1)
  };
}

(async () => {
  const mBase = await loadModel("params_aggr_it00000419_ema");
  const m128 = await loadModel("params_it00000128_ema");
  const m192 = await loadModel("params_it00000192_ema");
  const m782 = await loadModel("params_it00000782_ema");

  console.log("=== 闭环对决实测：当前新模型 vs 自身基模 (aggr419) ===");
  const r1 = await duel(m128, mBase, "it128", "aggr419(基模)");
  const r2 = await duel(m192, mBase, "it192", "aggr419(基模)");
  const r3 = await duel(m782, mBase, "it782", "aggr419(基模)");
  console.table([r1, r2, r3]);
})();
