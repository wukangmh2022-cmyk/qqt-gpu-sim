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

class PureFleeBot extends QQT.FleeBotAI {
  act(sim, pid) {
    const act = super.act(sim, pid);
    return [act[0], 0]; // 纯逃跑不放雷
  }
}

// 20 组固定确定性种子，保障跨代对比严密
const SEEDS_20 = [42, 100, 2026, 7, 99, 123, 456, 789, 1011, 2022, 314, 555, 777, 888, 999, 1314, 1688, 2048, 4096, 8192];

async function evaluateComprehensiveSuite(model, modelLabel, extraModels = {}) {
  console.log(`\n==============================================================================`);
  console.log(`🚀 开始 10 分钟全域泛化大回测: ${modelLabel}`);
  console.log(`==============================================================================`);

  const results = {};

  // 通用单方评测循环
  async function runPreset(botFactory, styleName, nGames = 20, isLevel0 = false) {
    let wins = 0, loss = 0, draws = 0, ticksSum = 0, idleFrames = 0, bombsSum = 0, maxConsecIdleSum = 0;
    for (let i = 0; i < nGames; i++) {
      const s = SEEDS_20[i % SEEDS_20.length];
      const sim = new Sim({ level: isLevel0 ? 0 : "empty", seed: s });
      sim.reset();
      const bot = botFactory ? botFactory() : null;
      const rng = mulberry32(s ^ 0xABCD);
      let curIdle = 0, maxIdle = 0, t = 0;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, rng);
        const a1 = bot ? bot.act(sim, 1) : [4, 0];
        const isIdle = (a0[0] === 4 && a0[1] === 0);
        if (isIdle) {
          idleFrames++;
          curIdle++;
          if (curIdle > maxIdle) maxIdle = curIdle;
        } else {
          curIdle = 0;
        }
        if (a0[1] === 1) bombsSum++;
        sim.step([a0, a1]);
        if (!sim.alive[0] || !sim.alive[1]) break;
      }
      if (sim.alive[0] && !sim.alive[1]) wins++;
      else if (!sim.alive[0] && sim.alive[1]) loss++;
      else draws++;
      ticksSum += Math.min(t, 1000);
      maxConsecIdleSum += maxIdle;
    }
    const res = {
      胜率: `${(wins / nGames * 100).toFixed(0)}%`,
      超时平局率: `${(draws / nGames * 100).toFixed(0)}%`,
      战败率: `${(loss / nGames * 100).toFixed(0)}%`,
      平均终结耗时: `${(ticksSum / nGames).toFixed(0)}t`,
      发呆时长占比: `${(idleFrames / ticksSum * 100).toFixed(1)}%`,
      单次最长发呆: `${(maxConsecIdleSum / nGames).toFixed(0)}t`,
      局均主动放炮: `${(bombsSum / nGames).toFixed(1)}`
    };
    console.log(`  [已完成] 风格: ${styleName} (20局) -> 胜率 ${res.胜率} | 发呆率 ${res.发呆时长占比} | 耗时 ${res.平均终结耗时}`);
    return res;
  }

  // 1. 纯静止不放炮木桩 (开阔地)
  results["1. 纯静止木桩(开阔地)"] = await runPreset(null, "纯静止木桩(开阔地)");
  // 2. 纯静止木桩 (比赛01 地图)
  results["2. 纯静止木桩(比赛01)"] = await runPreset(null, "纯静止木桩(比赛01)", 20, true);
  // 3. 漫游不放炮 (RoamBot)
  results["3. 纯漫游不放炮(RoamBot)"] = await runPreset(() => new QQT.RoamBotAI(), "纯漫游不放炮(RoamBot)");
  // 4. 纯逃跑不放炮 (PureFleeBot)
  results["4. 纯逃跑不放炮(PureFlee)"] = await runPreset(() => new PureFleeBot(), "纯逃跑不放炮(PureFlee)");
  // 5. 智能拉扯反击 (SmartKiteBot, 边退边留雷)
  results["5. 智能拉扯反击(KiteBot)"] = await runPreset(() => new QQT.FleeBotAI(), "智能拉扯反击(KiteBot)");
  // 6. 经典全图追杀 (HunterAI)
  results["6. 经典规则追猎(HunterAI)"] = await runPreset(() => new QQT.HunterAI(), "经典规则追猎(HunterAI)");
  // 7. 时空 A* 极限微操 (TimeAStarAI)
  results["7. 时空A*微操(TimeAStar)"] = await runPreset(() => new TimeAStarAI({ mode: "hunt" }), "时空A*微操(TimeAStar)");

  // 8. 自对弈镜面对局 (Self-Play)
  {
    let p0Wins = 0, p1Wins = 0, draws = 0, ticksSum = 0, bothIdleFrames = 0, singleIdleFrames = 0, zeroBombFrames = 0, p0Bombs = 0, p1Bombs = 0;
    for (let i = 0; i < 20; i++) {
      const s = SEEDS_20[i];
      const sim = new Sim({ level: "empty", seed: s });
      sim.reset();
      const r0 = mulberry32(s ^ 0x1111);
      const r1 = mulberry32(s ^ 0x2222);
      let t = 0;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, r0);
        const a1 = await model.act(sim, 1, r1);
        const p0Idle = (a0[0] === 4 && a0[1] === 0);
        const p1Idle = (a1[0] === 4 && a1[1] === 0);
        if (p0Idle && p1Idle) bothIdleFrames++;
        if (p0Idle) singleIdleFrames++;
        if (p1Idle) singleIdleFrames++;
        if (a0[1] === 0 && a1[1] === 0) zeroBombFrames++;
        if (a0[1] === 1) p0Bombs++;
        if (a1[1] === 1) p1Bombs++;
        sim.step([a0, a1]);
        if (!sim.alive[0] || !sim.alive[1]) break;
      }
      if (sim.alive[0] && !sim.alive[1]) p0Wins++;
      else if (!sim.alive[0] && sim.alive[1]) p1Wins++;
      else draws++;
      ticksSum += Math.min(t, 1000);
    }
    results["8. 自对弈镜像局(Self-Play)"] = {
      胜率: `P0:${(p0Wins/20*100).toFixed(0)}% / P1:${(p1Wins/20*100).toFixed(0)}%`,
      超时平局率: `${(draws / 20 * 100).toFixed(0)}%`,
      战败率: "-",
      平均终结耗时: `${(ticksSum / 20).toFixed(0)}t`,
      发呆时长占比: `双方同挂机:${(bothIdleFrames/ticksSum*100).toFixed(1)}% (单方:${(singleIdleFrames/2/ticksSum*100).toFixed(1)}%)`,
      单次最长发呆: `零放炮帧占比:${(zeroBombFrames/ticksSum*100).toFixed(1)}%`,
      局均主动放炮: `${((p0Bombs+p1Bombs)/2/20).toFixed(1)}`
    };
    console.log(`  [已完成] 风格: 自对弈镜像局 (20局) -> 双方同挂机占比 ${results["8. 自对弈镜像局(Self-Play)"].发呆时长占比}`);
  }

  // 9. 对战自身基模 aggr419
  if (extraModels.base) {
    let w = 0, d = 0, l = 0, ticks = 0, b0 = 0, b1 = 0;
    for (let i = 0; i < 20; i++) {
      const s = SEEDS_20[i];
      const sim = new Sim({ level: "empty", seed: s });
      sim.reset();
      const r0 = mulberry32(s ^ 0x3333);
      const r1 = mulberry32(s ^ 0x4444);
      let t = 0;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, r0);
        const a1 = await extraModels.base.act(sim, 1, r1);
        if (a0[1] === 1) b0++;
        if (a1[1] === 1) b1++;
        sim.step([a0, a1]);
        if (!sim.alive[0] || !sim.alive[1]) break;
      }
      if (sim.alive[0] && !sim.alive[1]) w++;
      else if (!sim.alive[0] && sim.alive[1]) l++;
      else d++;
      ticks += Math.min(t, 1000);
    }
    results["9. 对战基模(aggr419)"] = {
      胜率: `${(w / 20 * 100).toFixed(0)}% (${w}W-${d}D-${l}L)`,
      超时平局率: `${(d / 20 * 100).toFixed(0)}%`,
      战败率: `${(l / 20 * 100).toFixed(0)}%`,
      平均终结耗时: `${(ticks / 20).toFixed(0)}t`,
      发呆时长占比: "-",
      单次最长发呆: "-",
      局均主动放炮: `我:${(b0/20).toFixed(1)} vs 基模:${(b1/20).toFixed(1)}`
    };
    console.log(`  [已完成] 风格: 对战基模 aggr419 (20局) -> 胜率 ${results["9. 对战基模(aggr419)"].胜率}`);
  }

  // 10. 对战昨日防守大师 it782
  if (extraModels.boss) {
    let w = 0, d = 0, l = 0, ticks = 0, b0 = 0, b1 = 0;
    for (let i = 0; i < 20; i++) {
      const s = SEEDS_20[i];
      const sim = new Sim({ level: "empty", seed: s });
      sim.reset();
      const r0 = mulberry32(s ^ 0x5555);
      const r1 = mulberry32(s ^ 0x6666);
      let t = 0;
      for (t = 0; t < 1000; t++) {
        const a0 = await model.act(sim, 0, r0);
        const a1 = await extraModels.boss.act(sim, 1, r1);
        if (a0[1] === 1) b0++;
        if (a1[1] === 1) b1++;
        sim.step([a0, a1]);
        if (!sim.alive[0] || !sim.alive[1]) break;
      }
      if (sim.alive[0] && !sim.alive[1]) w++;
      else if (!sim.alive[0] && sim.alive[1]) l++;
      else d++;
      ticks += Math.min(t, 1000);
    }
    results["10. 对战防守大师(it782)"] = {
      胜率: `${(w / 20 * 100).toFixed(0)}% (${w}W-${d}D-${l}L)`,
      超时平局率: `${(d / 20 * 100).toFixed(0)}%`,
      战败率: `${(l / 20 * 100).toFixed(0)}%`,
      平均终结耗时: `${(ticks / 20).toFixed(0)}t`,
      发呆时长占比: "-",
      单次最长发呆: "-",
      局均主动放炮: `我:${(b0/20).toFixed(1)} vs 782:${(b1/20).toFixed(1)}`
    };
    console.log(`  [已完成] 风格: 对战防守大师 it782 (20局) -> 胜率 ${results["10. 对战防守大师(it782)"].胜率}`);
  }

  return results;
}

(async () => {
  const mBase = await loadModel("params_aggr_it00000419_ema");
  const m782  = await loadModel("params_it00000782_ema");
  const m320  = await loadModel("params_it00000320_ema");

  console.log("==============================================================================");
  console.log("🏆 10分钟大回测：最新模型 it320 vs 基模 aggr419 vs 昨日防守版 it782");
  console.log("==============================================================================");

  console.log("\n>>> [第一阶段] 评测最新快照 it320 (2h40m / 40.7 亿步)...");
  const res320 = await evaluateComprehensiveSuite(m320, "it320 (最新)", { base: mBase, boss: m782 });

  console.log("\n>>> [第二阶段] 评测昨日防守版 it782 (历史参照基准)...");
  const res782 = await evaluateComprehensiveSuite(m782, "it782 (昨日版)", { base: mBase });

  console.log("\n>>> [第三阶段] 评测长训底模 aggr419 (起点基准)...");
  const resBase = await evaluateComprehensiveSuite(mBase, "aggr419 (底模)");

  const out = {
    it320: res320,
    it782: res782,
    aggr419: resBase
  };
  fs.writeFileSync(path.join(__dirname, "..", "web", "10m_eval_summary.json"), JSON.stringify(out, null, 2));

  console.log("\n==============================================================================");
  console.log("📊 最终对比汇总：10 大风格综合指标对照表");
  console.log("==============================================================================");
  const styles = Object.keys(res320);
  for (const st of styles) {
    console.log(`\n【风格: ${st}】`);
    console.table([
      { "版本": "aggr419 (底模)", ...resBase[st] },
      { "版本": "it782 (昨日防守版)", ...res782[st] },
      { "版本": "it320 (长训最新)", ...res320[st] },
    ]);
  }
})();
