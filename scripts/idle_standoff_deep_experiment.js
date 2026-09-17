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

class PureFleeBot extends QQT.FleeBotAI {
  act(sim, pid) {
    const act = super.act(sim, pid);
    return [act[0], 0]; // 强制 0 炮，纯逃跑拉扯
  }
}

const SEEDS = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022];

async function testModelIdle(m, label) {
  // -------------------------------------------------------------
  // 场景 1: vs 纯静止不放炮木桩 [4, 0]
  // -------------------------------------------------------------
  let s1_ticks = 0, s1_idle = 0, s1_bombs = 0, s1_maxConsec = 0, s1_timeouts = 0;
  for (const s of SEEDS) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const rng = mulberry32(s ^ 0x1111);
    let curConsec = 0, maxConsec = 0, t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, rng);
      const isIdle = (a0[0] === 4 && a0[1] === 0);
      if (isIdle) {
        s1_idle++;
        curConsec++;
        if (curConsec > maxConsec) maxConsec = curConsec;
      } else {
        curConsec = 0;
      }
      if (a0[1] === 1) s1_bombs++;
      sim.step([a0, [4, 0]]);
      if (!sim.alive[1] || !sim.alive[0]) break;
    }
    s1_ticks += Math.min(t, 1000);
    if (t >= 1000) s1_timeouts++;
    s1_maxConsec += maxConsec;
  }

  // -------------------------------------------------------------
  // 场景 2: vs 纯逃跑不放炮 FleeBot
  // -------------------------------------------------------------
  let s2_ticks = 0, s2_idle = 0, s2_bombs = 0, s2_maxConsec = 0, s2_timeouts = 0, s2_wins = 0;
  for (const s of SEEDS) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const flee = new PureFleeBot();
    const rng = mulberry32(s ^ 0x2222);
    let curConsec = 0, maxConsec = 0, t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, rng);
      const a1 = flee.act(sim, 1);
      const isIdle = (a0[0] === 4 && a0[1] === 0);
      if (isIdle) {
        s2_idle++;
        curConsec++;
        if (curConsec > maxConsec) maxConsec = curConsec;
      } else {
        curConsec = 0;
      }
      if (a0[1] === 1) s2_bombs++;
      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    s2_ticks += Math.min(t, 1000);
    if (t >= 1000) s2_timeouts++;
    if (sim.alive[0] && !sim.alive[1]) s2_wins++;
    s2_maxConsec += maxConsec;
  }

  // -------------------------------------------------------------
  // 场景 3: 纯自对弈镜像局 (Self-Play: m vs m)
  // -------------------------------------------------------------
  let s3_ticks = 0, s3_p0Idle = 0, s3_p1Idle = 0, s3_bothIdle = 0;
  let s3_p0Bombs = 0, s3_p1Bombs = 0, s3_timeouts = 0, s3_maxZeroBombStreak = 0;
  for (const s of SEEDS) {
    const sim = new Sim({ level: "empty", seed: s });
    sim.reset();
    const r0 = mulberry32(s ^ 0x3333);
    const r1 = mulberry32(s ^ 0x4444);
    let curZeroBomb = 0, maxZeroBomb = 0, t = 0;
    for (t = 0; t < 1000; t++) {
      const a0 = await m.act(sim, 0, r0);
      const a1 = await m.act(sim, 1, r1);
      const p0Idle = (a0[0] === 4 && a0[1] === 0);
      const p1Idle = (a1[0] === 4 && a1[1] === 0);
      if (p0Idle) s3_p0Idle++;
      if (p1Idle) s3_p1Idle++;
      if (p0Idle && p1Idle) s3_bothIdle++;

      if (a0[1] === 1) s3_p0Bombs++;
      if (a1[1] === 1) s3_p1Bombs++;

      if (a0[1] === 0 && a1[1] === 0) {
        curZeroBomb++;
        if (curZeroBomb > maxZeroBomb) maxZeroBomb = curZeroBomb;
      } else {
        curZeroBomb = 0;
      }

      sim.step([a0, a1]);
      if (!sim.alive[0] || !sim.alive[1]) break;
    }
    s3_ticks += Math.min(t, 1000);
    if (t >= 1000) s3_timeouts++;
    s3_maxZeroBombStreak += maxZeroBomb;
  }

  return {
    label,
    // 场景 1: 静止木桩
    idle_dummy: {
      idlePct: `${(s1_idle / s1_ticks * 100).toFixed(1)}%`,
      maxIdleStreak: `${(s1_maxConsec / SEEDS.length).toFixed(1)}t`,
      avgTicks: `${(s1_ticks / SEEDS.length).toFixed(0)}t`,
      avgBombs: `${(s1_bombs / SEEDS.length).toFixed(1)}`,
      timeoutPct: `${(s1_timeouts / SEEDS.length * 100).toFixed(0)}%`
    },
    // 场景 2: 逃跑不放炮
    idle_flee: {
      winRate: `${(s2_wins / SEEDS.length * 100).toFixed(0)}%`,
      idlePct: `${(s2_idle / s2_ticks * 100).toFixed(1)}%`,
      maxIdleStreak: `${(s2_maxConsec / SEEDS.length).toFixed(1)}t`,
      avgTicks: `${(s2_ticks / SEEDS.length).toFixed(0)}t`,
      avgBombs: `${(s2_bombs / SEEDS.length).toFixed(1)}`,
      timeoutPct: `${(s2_timeouts / SEEDS.length * 100).toFixed(0)}%`
    },
    // 场景 3: 自对弈镜像局
    self_play: {
      bothIdlePct: `${(s3_bothIdle / s3_ticks * 100).toFixed(1)}%`,
      singleIdlePct: `${((s3_p0Idle + s3_p1Idle) / 2 / s3_ticks * 100).toFixed(1)}%`,
      maxZeroBombStreak: `${(s3_maxZeroBombStreak / SEEDS.length).toFixed(0)}t`,
      avgBombs: `${((s3_p0Bombs + s3_p1Bombs) / 2 / SEEDS.length).toFixed(1)}`,
      timeoutPct: `${(s3_timeouts / SEEDS.length * 100).toFixed(0)}%`,
      avgTicks: `${(s3_ticks / SEEDS.length).toFixed(0)}t`
    }
  };
}

(async () => {
  const mBase = await loadModel("params_aggr_it00000419_ema");
  const m782  = await loadModel("params_it00000782_ema");
  const m256  = await loadModel("params_it00000256_ema");
  const m320  = await loadModel("params_it00000320_ema");

  const rBase = await testModelIdle(mBase, "aggr419 (基模)");
  const r782  = await testModelIdle(m782,  "it782 (昨日防守版)");
  const r256  = await testModelIdle(m256,  "it256 (长训 2h10m)");
  const r320  = await testModelIdle(m320,  "it320 (长训 2h40m 最新)");

  console.log("\n【表 1：对战纯静止不放炮木桩 —— 发呆时长占比与连续发呆最大帧数】");
  console.table([rBase, r782, r256, r320].map(r => ({
    "模型版本": r.label,
    "发呆时长占比": r.idle_dummy.idlePct,
    "单次最长连续发呆": r.idle_dummy.maxIdleStreak,
    "平均终结耗时": r.idle_dummy.avgTicks,
    "局均放炮数": r.idle_dummy.avgBombs,
    "超时率": r.idle_dummy.timeoutPct
  })));

  console.log("\n【表 2：对战逃跑不放炮 FleeBot —— 追击中发呆时长占比与围剿耗时】");
  console.table([rBase, r782, r256, r320].map(r => ({
    "模型版本": r.label,
    "胜率": r.idle_flee.winRate,
    "追击发呆占比": r.idle_flee.idlePct,
    "单次最长连续发呆": r.idle_flee.maxIdleStreak,
    "围剿耗时": r.idle_flee.avgTicks,
    "局均放炮数": r.idle_flee.avgBombs,
    "超时率": r.idle_flee.timeoutPct
  })));

  console.log("\n【表 3：自对弈镜像局 (Self-Play) —— 双方同时对峙发呆与零放炮死锁检测】");
  console.table([rBase, r782, r256, r320].map(r => ({
    "模型版本": r.label,
    "双方同时发呆占比": r.self_play.bothIdlePct,
    "单方发呆占比": r.self_play.singleIdlePct,
    "最长连续零放炮帧": r.self_play.maxZeroBombStreak,
    "局均放炮数": r.self_play.avgBombs,
    "超时平局率": r.self_play.timeoutPct,
    "平均对局时长": r.self_play.avgTicks
  })));
})();
