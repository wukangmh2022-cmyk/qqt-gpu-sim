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

(async () => {
  const m32 = await loadModel("params_it00000032_ema");
  const m782 = await loadModel("params_it00000782_ema");

  console.log("=== Diagnostic 1: it32 vs Dummy (Seed 42) ===");
  {
    const sim = new Sim({ level: "empty", seed: 42 });
    sim.reset();
    const rng = mulberry32(42 ^ 0x1111);
    for (let t = 0; t < 500; t++) {
      const a0 = await m32.act(sim, 0, rng);
      if (t % 30 === 0 || a0[1] === 1 || !sim.alive[1] || !sim.alive[0]) {
        console.log(`t=${t} P0=(${sim.pos[0].toFixed(1)}, ${sim.pos[1].toFixed(1)}) act=[move=${a0[0]}, bomb=${a0[1]}] P1=(${sim.pos[2].toFixed(1)}, ${sim.pos[3].toFixed(1)}) bombs=${sim.liveBombs(0)}/${sim.liveBombs(1)}`);
      }
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) {
        console.log(`Finished at t=${t}, alive=[${sim.alive[0]}, ${sim.alive[1]}]`);
        break;
      }
    }
  }

  console.log("\n=== Diagnostic 2: it32 vs 782 (Seed 42) ===");
  {
    const sim = new Sim({ level: "empty", seed: 42 });
    sim.reset();
    const r0 = mulberry32(42 ^ 0x7777);
    const r1 = mulberry32(42 ^ 0x8888);
    for (let t = 0; t < 500; t++) {
      const a0 = await m32.act(sim, 0, r0);
      const a1 = await m782.act(sim, 1, r1);
      if (t % 50 === 0 || a0[1] === 1 || a1[1] === 1) {
        console.log(`t=${t} P0 act=[${a0}] P1 act=[${a1}] dist=${Math.hypot(sim.pos[0] - sim.pos[2], sim.pos[1] - sim.pos[3]).toFixed(2)} bombs=${sim.liveBombs(0)}/${sim.liveBombs(1)}`);
      }
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) {
        console.log(`Finished at t=${t}, alive=[${sim.alive[0]}, ${sim.alive[1]}]`);
        break;
      }
    }
  }
})();
