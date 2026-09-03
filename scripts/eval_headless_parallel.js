#!/usr/bin/env node
/**
 * eval_headless_parallel.js - 高性能 WebJS Headless 真实对战评测套件
 * 
 * 保证与 WebJS 运行时 100% 一致：
 * 1. 采用 web/sim.js 核心模拟器（含 1800 tick 上限、0.3s 余威、3 tick 残骸延迟阻挡、相同浮点步进）
 * 2. 采用真实 HunterAI（多源 Dijkstra 逃生、全局威胁场）与 IDLE 规则基准
 * 3. 采用 onnxruntime-node 真实 ONNX 推理（单线程绑定，杜绝核争用）
 * 4. 默认 4 核心并发（对齐 4 个高性能物理 P-Core），每场景默认 32 局（总计 128 局，耗时 ~3.1 分钟）
 */
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const { Worker, isMainThread, parentPort } = require('worker_threads');

const ROOT = path.join(__dirname, '..');
const MODELS_DIR = path.join(ROOT, 'web', 'models');
const MAPS_JSON = path.join(ROOT, 'web', 'assets', 'maps', 'levels.json');

// --------------------------- CLI 参数解析 ---------------------------
function parseArgs() {
  const argv = process.argv.slice(2);
  const getArg = (name, dflt) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
  };
  return {
    model: getArg('--model', 'params_it00000068_hlgauss_top25foractor_patch3_k32'),
    oppModel: getArg('--opp-model', ''),              // 可选：对手 ONNX 模型代号 (开启 Model vs Model 跨代对决)
    workers: parseInt(getArg('--workers', '4'), 10), // 默认 4 核心并发（对齐物理大核）
    games: parseInt(getArg('--games', '32'), 10),    // 默认 4 场景各 32 局 = 总共 128 局（~3 分钟）
    maxTicks: parseInt(getArg('--max-ticks', '1800'), 10), // 默认 1800 tick 真实上限
    seed: parseInt(getArg('--seed', '42'), 10),
    domain: getArg('--domain', 'all'), // all | open_idle | open_hunter | full_idle | full_hunter
  };
}

// =========================================================================
// Worker 线程工作逻辑
// =========================================================================
if (!isMainThread) {
  const ort = require('onnxruntime-node');
  global.ort = ort;
  const QQT = require(path.join(ROOT, 'web', 'sim.js'));
  const { Sim, ORTTransformerModel, HunterAI, mulberry32, W, H } = QQT;

  let model0 = null;
  let model1 = null;
  let hunter = null;
  let levels = null;

  function isBombExplodingNow(sim, i) {
    return sim.fuse[i] === 1;
  }

  function snapshotBombs(sim) {
    const bs = [];
    for (let i = 0; i < W * H; i++) {
      if (sim.fuse[i] > 0) {
        bs.push({ i, owner: sim.owner[i], blast: sim.bombBlast[i], fuse: sim.fuse[i] });
      }
    }
    return bs;
  }

  async function initWorker(modelName, oppModelName) {
    const loadModel = async (name) => {
      const docPath = path.join(MODELS_DIR, `${name}.json`);
      const onnxPath = path.join(MODELS_DIR, `${name}.onnx`);
      const doc = JSON.parse(fs.readFileSync(docPath, 'utf8'));
      const sess = await ort.InferenceSession.create(onnxPath, {
        executionProviders: ['cpu'],
        intraOpNumThreads: 1,
        interOpNumThreads: 1,
      });
      const m = new ORTTransformerModel(doc, sess);
      m.inferEvery = 1;
      return m;
    };

    model0 = await loadModel(modelName);
    if (oppModelName) {
      model1 = await loadModel(oppModelName);
    }
    hunter = new HunterAI();
    levels = JSON.parse(fs.readFileSync(MAPS_JSON, 'utf8'));
    if (!Array.isArray(levels)) levels = levels.levels || levels.maps;
  }

  async function runTask(task) {
    const { domain, gameIdx, seed, maxTicks, oppType, swapSide } = task;
    const sim = new Sim(seed);
    if (domain.startsWith('open')) {
      sim.reset('open');
    } else {
      const lv = levels[gameIdx % levels.length];
      sim.reset(lv);
    }

    const rng = mulberry32(seed ^ 0x9e3779b9);
    const isHunter = domain.endsWith('hunter');
    const isModelOpp = oppType === 'model' && model1 !== null;

    let p0Bombs = 0;
    let p0Hits = 0;
    let p0Suicide = false;
    const p0Moves = [0, 0, 0, 0, 0];
    const visitedCells = new Set();
    const model0PlayerIdx = swapSide ? 1 : 0;
    const [initR, initC] = sim.centerCell(model0PlayerIdx);
    visitedCells.add(initR * W + initC);

    while (!sim.done && sim.t < maxTicks) {
      const preBombs = snapshotBombs(sim);
      const hpBefore = [sim.hp[0], sim.hp[1]];
      const aliveBefore = [sim.alive[0], sim.alive[1]];

      let a0, a1;
      if (isModelOpp) {
        if (!swapSide) {
          a0 = await model0.act(sim, 0, rng);
          a1 = await model1.act(sim, 1, rng);
        } else {
          a0 = await model1.act(sim, 0, rng);
          a1 = await model0.act(sim, 1, rng);
        }
      } else {
        a0 = await model0.act(sim, 0, rng);
        a1 = isHunter ? hunter.act(sim, 1) : [4, 0]; // 4=IDLE, 0=NO_BOMB
      }

      const model0Action = swapSide ? a1 : a0;

      if (aliveBefore[model0PlayerIdx]) {
        if (model0Action[1] === 1) p0Bombs++;
        p0Moves[model0Action[0]]++;
        const [cr, cc] = sim.centerCell(model0PlayerIdx);
        visitedCells.add(cr * W + cc);
      }

      sim.step([a0, a1]);

      // 检查 model0 炸弹造成的命中或自杀
      for (const b of preBombs) {
        if (b.owner === model0PlayerIdx && isBombExplodingNow(sim, b.i)) {
          const oppIdx = 1 - model0PlayerIdx;
          if (aliveBefore[oppIdx] && sim.hp[oppIdx] < hpBefore[oppIdx]) {
            p0Hits++;
          }
          if (aliveBefore[model0PlayerIdx] && sim.hp[model0PlayerIdx] < hpBefore[model0PlayerIdx]) {
            p0Suicide = true;
          }
        }
      }
    }

    const oppIdx = 1 - model0PlayerIdx;
    const finalAlive = [sim.alive[0], sim.alive[1]];
    const finalHp = [sim.hp[0], sim.hp[1]];

    let outcome = 'timeout';
    if (!finalAlive[0] && !finalAlive[1]) {
      outcome = 'mutual';
    } else if (finalAlive[model0PlayerIdx] && !finalAlive[oppIdx]) {
      outcome = 'win';
    } else if (!finalAlive[model0PlayerIdx] && finalAlive[oppIdx]) {
      outcome = 'loss';
    } else {
      if (finalHp[model0PlayerIdx] > finalHp[oppIdx]) outcome = 'win';
      else if (finalHp[oppIdx] > finalHp[model0PlayerIdx]) outcome = 'loss';
      else outcome = 'timeout';
    }

    // 计算发呆率与动作经验熵
    const totalMoves = p0Moves.reduce((a, b) => a + b, 0);
    const idleRatio = totalMoves > 0 ? Number(((p0Moves[4] / totalMoves) * 100).toFixed(1)) : 0;
    let moveEntropy = 0;
    if (totalMoves > 0) {
      for (let i = 0; i < 5; i++) {
        if (p0Moves[i] > 0) {
          const p = p0Moves[i] / totalMoves;
          moveEntropy -= p * Math.log2(p);
        }
      }
    }
    const exploredRatio = Number(((visitedCells.size / (W * H)) * 100).toFixed(1));

    return {
      domain,
      gameIdx,
      ticks: sim.t,
      outcome,
      p0Bombs,
      p0Hits,
      p0Suicide,
      idleRatio,
      exploredRatio,
      moveEntropy: Number(moveEntropy.toFixed(3)),
      hp: [sim.hp[0], sim.hp[1]],
    };
  }

  parentPort.on('message', async (msg) => {
    if (msg.type === 'init') {
      await initWorker(msg.modelName, msg.oppModelName);
      parentPort.postMessage({ type: 'ready' });
    } else if (msg.type === 'task') {
      const result = await runTask(msg.task);
      parentPort.postMessage({ type: 'result', result });
    }
  });
  return;
}

// =========================================================================
// 主线程：任务分发与聚合统计
// =========================================================================
async function main() {
  const args = parseArgs();
  console.log(`\n==========================================================================`);
  const totalGames = args.games * (args.domain === 'all' ? 4 : 1);
  console.log(`🚀 QQ堂 WebJS 真实对战评测套件 (Headless 1800-tick)`);
  console.log(`   模型代号: [${args.model}]`);
  console.log(`   并发核心: ${args.workers} Workers (Host Cores: ${os.cpus().length})`);
  console.log(`   评测规模: 4 场景各 ${args.games} 局 (总计 ${totalGames} 局) | 物理上限: ${args.maxTicks} ticks`);
  console.log(`   一致性保证: 100% web/sim.js 真实物理 + 全局 Dijkstra HunterAI + 单线程绑定 ONNX`);
  console.log(`==========================================================================\n`);

  let domains = [];
  if (args.oppModel) {
    domains = [
      { id: 'open_vs_opp', name: `模型对决 / 空道场 vs ${args.oppModel}`, oppType: 'model' },
      { id: 'full_vs_opp', name: `模型对决 / 241复杂图 vs ${args.oppModel}`, oppType: 'model' },
    ];
  } else if (args.domain === 'all') {
    domains = [
      { id: 'open_hunter', name: 'AI Hunter / 空场景道场' },
      { id: 'full_hunter', name: 'AI Hunter / 全池241复杂地图' },
      { id: 'open_idle',   name: '静止木桩  / 空场景道场' },
      { id: 'full_idle',   name: '静止木桩  / 全池241复杂地图' },
    ];
  } else {
    domains = [{ id: args.domain, name: args.domain }];
  }

  // 1. 初始化 Worker 池
  const workers = [];
  const readyPromises = [];

  for (let i = 0; i < args.workers; i++) {
    const w = new Worker(__filename);
    workers.push(w);
    const p = new Promise((resolve) => {
      const onMsg = (msg) => {
        if (msg.type === 'ready') {
          w.off('message', onMsg);
          resolve();
        }
      };
      w.on('message', onMsg);
    });
    readyPromises.push(p);
    w.postMessage({ type: 'init', modelName: args.model, oppModelName: args.oppModel });
  }

  const tInitStart = Date.now();
  process.stdout.write(`⏳ 正在初始化 ${args.workers} 个 Worker 进程与 ONNX Session... `);
  await Promise.all(readyPromises);
  console.log(`完成 (${((Date.now() - tInitStart) / 1000).toFixed(2)}s)`);

  // 2. 按领域依次执行测试
  const domainReports = [];
  const tTotalStart = Date.now();

  for (const dom of domains) {
    const tDomStart = Date.now();
    const tasks = [];
    for (let i = 0; i < args.games; i++) {
      tasks.push({
        domain: dom.id,
        oppType: dom.oppType || 'rule',
        swapSide: dom.oppType === 'model' ? (i % 2 === 1) : false,
        gameIdx: i,
        seed: args.seed + i * 31337,
        maxTicks: args.maxTicks,
      });
    }

    const results = [];
    let completed = 0;

    await new Promise((resolve) => {
      let taskIdx = 0;

      function dispatch(w) {
        if (taskIdx >= tasks.length) return;
        const task = tasks[taskIdx++];
        w.postMessage({ type: 'task', task });
      }

      for (const w of workers) {
        w.removeAllListeners('message');
        w.on('message', (msg) => {
          if (msg.type === 'result') {
            results.push(msg.result);
            completed++;
            if (completed % 16 === 0 || completed === args.games) {
              const el = ((Date.now() - tDomStart) / 1000).toFixed(1);
              process.stdout.write(`\r  [${dom.id}] 进度: ${completed}/${args.games} (${el}s)`);
            }
            if (completed === args.games) {
              resolve();
            } else {
              dispatch(w);
            }
          }
        });
        dispatch(w);
      }
    });

    const domTime = ((Date.now() - tDomStart) / 1000).toFixed(2);
    process.stdout.write(`\r  [${dom.id}] 进度: ${args.games}/${args.games} 完成！耗时 ${domTime}s\n`);

    // 统计聚合
    const wins = results.filter((r) => r.outcome === 'win').length;
    const losses = results.filter((r) => r.outcome === 'loss').length;
    const mutuals = results.filter((r) => r.outcome === 'draw_mutual').length;
    const timeouts = results.filter((r) => r.outcome === 'draw_timeout').length;
    const suicides = results.filter((r) => r.p0Suicide).length;
    const avgBombs = (results.reduce((a, b) => a + b.p0Bombs, 0) / args.games).toFixed(1);
    const avgHits = (results.reduce((a, b) => a + b.p0Hits, 0) / args.games).toFixed(2);
    const avgTicks = (results.reduce((a, b) => a + b.ticks, 0) / args.games).toFixed(1);
    const avgExplored = (results.reduce((a, b) => a + b.exploredRatio, 0) / args.games).toFixed(1);
    const avgIdle = (results.reduce((a, b) => a + b.idleRatio, 0) / args.games).toFixed(1);
    const avgEntropy = (results.reduce((a, b) => a + b.moveEntropy, 0) / args.games).toFixed(2);
    const winRate = ((wins / args.games) * 100).toFixed(1);

    domainReports.push({
      id: dom.id,
      name: dom.name,
      games: args.games,
      wins,
      losses,
      mutuals,
      timeouts,
      suicides,
      avgBombs,
      avgHits,
      avgTicks,
      avgExplored,
      avgIdle,
      avgEntropy,
      winRate,
      domTime,
    });
  }

  // 关闭 Worker
  for (const w of workers) w.terminate();

  const totalTime = ((Date.now() - tTotalStart) / 1000).toFixed(2);

  // 3. 输出汇总 Markdown 表格
  console.log(`\n==========================================================================`);
  console.log(`📊 评测汇总成果 (总对局=${args.games * domains.length}，总耗时=${totalTime}s)`);
  console.log(`==========================================================================\n`);

  console.log(`| 对手 / 地图场景 | 胜 | 负 | 同归 | 超时 | 胜率(%) | 自杀 | 炮/局 | 命中/局 | 平均局长 | 控图率(%) | 发呆率(%) | 动作熵 |`);
  console.log(`| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |`);
  for (const r of domainReports) {
    console.log(`| ${r.name} | **${r.wins}** | ${r.losses} | ${r.mutuals} | ${r.timeouts} | **${r.winRate}%** | ${r.suicides} | ${r.avgBombs} | ${r.avgHits} | ${r.avgTicks} | ${r.avgExplored}% | ${r.avgIdle}% | ${r.avgEntropy} |`);
  }

  console.log(`\n✨ 验收完毕！可直接将表格写入评估复盘文档。\n`);
}

main().catch((err) => {
  console.error('评测异常中断:', err);
  process.exit(1);
});
