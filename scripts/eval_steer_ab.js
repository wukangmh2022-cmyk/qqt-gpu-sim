#!/usr/bin/env node
/**
 * eval_steer_ab.js - Headless A/B 真实同局对战评测套件
 * 
 * 严格控制变量对比：
 * 同一局对抗中：
 *   - Player New: 新版 legal_mask（物理位移探针 _tryMove）+ 新版 _steer（移除开阔地/平墙归中被动切向滑移）
 *   - Player Old: 老版 legal_mask（整格静态查表）+ 老版 _steer（撞障主动向中线归中）
 * 双方使用完全一致的模型权重（Self-Play），严格对调红蓝座位（P0/P1 swap）消除位置先后手偏置。
 * 
 * 评测场景：
 *   1. 空场景（Level 240 / 45度X形十字道具排布 / 50% 起步域随机化）
 *   2. 综合场景第二阶（curriculum.json Stage 2 全量 22 张经典对战地图）
 */
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const { Worker, isMainThread, parentPort } = require('worker_threads');

const ROOT = path.join(__dirname, '..');
const MODELS_DIR = path.join(ROOT, 'web', 'models');
const MAPS_JSON = path.join(ROOT, 'web', 'assets', 'maps', 'levels.json');
const CURR_JSON = path.join(ROOT, 'web', 'assets', 'maps', 'curriculum.json');

function parseArgs() {
  const argv = process.argv.slice(2);
  const getArg = (name, dflt) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
  };
  return {
    model: getArg('--model', 'params_it00000346_ema'),
    workers: parseInt(getArg('--workers', '6'), 10),
    openGames: parseInt(getArg('--open-games', '50'), 10),
    stage2Rounds: parseInt(getArg('--stage2-rounds', '4'), 10), // 每张地图对战局数（必须偶数，保证对称互换）
    maxTicks: parseInt(getArg('--max-ticks', '1800'), 10),
    seed: parseInt(getArg('--seed', '42'), 10),
  };
}

// =========================================================================
// Worker 线程工作逻辑
// =========================================================================
if (!isMainThread) {
  const ort = require('onnxruntime-node');
  global.ort = ort;
  const QQT = require(path.join(ROOT, 'web', 'sim.js'));
  const { Sim, ORTTransformerModel, mulberry32, W, H, DIRS } = QQT;

  let model = null;
  let levelsById = new Map();

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

  async function initWorker(modelName) {
    const docPath = path.join(MODELS_DIR, `${modelName}.json`);
    const onnxPath = path.join(MODELS_DIR, `${modelName}.onnx`);
    const doc = JSON.parse(fs.readFileSync(docPath, 'utf8'));
    const sess = await ort.InferenceSession.create(onnxPath, {
      executionProviders: ['cpu'],
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
    });
    model = new ORTTransformerModel(doc, sess);
    model.inferEvery = 1;

    const allLevels = JSON.parse(fs.readFileSync(MAPS_JSON, 'utf8'));
    const list = Array.isArray(allLevels) ? allLevels : (allLevels.levels || allLevels.maps);
    for (const lv of list) {
      levelsById.set(lv.id, lv);
    }
  }

  async function runTask(task) {
    const { domain, mapId, mapName, gameIdx, seed, maxTicks, swapSide } = task;
    const rng = mulberry32(seed ^ 0x9e3779b9);

    // swapSide=false: P0=new, P1=old
    // swapSide=true:  P0=old, P1=new
    const playerModes = swapSide ? ['old', 'new'] : ['new', 'old'];
    const newIdx = swapSide ? 1 : 0;
    const oldIdx = swapSide ? 0 : 1;

    const sim = new Sim(seed, { playerModes });
    const levelObj = levelsById.get(mapId);
    if (levelObj) {
      sim.reset(levelObj, { playerModes });
    } else {
      sim.reset('open', { playerModes });
    }

    let newBombs = 0, oldBombs = 0;
    let newHits = 0, oldHits = 0;
    let newSuicide = false, oldSuicide = false;
    const newMoves = [0, 0, 0, 0, 0];
    const oldMoves = [0, 0, 0, 0, 0];
    const newVisited = new Set();
    const oldVisited = new Set();

    const [initR0, initC0] = sim.centerCell(newIdx);
    newVisited.add(initR0 * W + initC0);
    const [initR1, initC1] = sim.centerCell(oldIdx);
    oldVisited.add(initR1 * W + initC1);

    while (!sim.done && sim.t < maxTicks) {
      const preBombs = snapshotBombs(sim);
      const hpBefore = [sim.hp[0], sim.hp[1]];
      const aliveBefore = [sim.alive[0], sim.alive[1]];

      const [a0, a1] = await model.bothAct(sim, rng);
      const aNew = swapSide ? a1 : a0;
      const aOld = swapSide ? a0 : a1;

      if (aliveBefore[newIdx]) {
        if (aNew[1] === 1) newBombs++;
        newMoves[aNew[0]]++;
        const [cr, cc] = sim.centerCell(newIdx);
        newVisited.add(cr * W + cc);
      }
      if (aliveBefore[oldIdx]) {
        if (aOld[1] === 1) oldBombs++;
        oldMoves[aOld[0]]++;
        const [cr, cc] = sim.centerCell(oldIdx);
        oldVisited.add(cr * W + cc);
      }

      sim.step([a0, a1]);

      // 跟踪伤害与自杀
      for (const b of preBombs) {
        if (isBombExplodingNow(sim, b.i)) {
          if (b.owner === newIdx) {
            if (aliveBefore[oldIdx] && sim.hp[oldIdx] < hpBefore[oldIdx]) newHits++;
            if (aliveBefore[newIdx] && sim.hp[newIdx] < hpBefore[newIdx]) newSuicide = true;
          } else if (b.owner === oldIdx) {
            if (aliveBefore[newIdx] && sim.hp[newIdx] < hpBefore[newIdx]) oldHits++;
            if (aliveBefore[oldIdx] && sim.hp[oldIdx] < hpBefore[oldIdx]) oldSuicide = true;
          }
        }
      }
    }

    const finalAlive = [sim.alive[0], sim.alive[1]];
    const finalHp = [sim.hp[0], sim.hp[1]];

    let outcome = 'timeout';
    if (!finalAlive[0] && !finalAlive[1]) {
      outcome = 'mutual_death';
    } else if (sim.winner === newIdx) {
      outcome = 'new_win';
    } else if (sim.winner === oldIdx) {
      outcome = 'old_win';
    } else if (finalAlive[newIdx] && !finalAlive[oldIdx]) {
      outcome = 'new_win';
    } else if (!finalAlive[newIdx] && finalAlive[oldIdx]) {
      outcome = 'old_win';
    } else {
      if (finalHp[newIdx] > finalHp[oldIdx]) outcome = 'new_win';
      else if (finalHp[oldIdx] > finalHp[newIdx]) outcome = 'old_win';
      else outcome = 'draw';
    }

    const newTotalMoves = newMoves.reduce((a, b) => a + b, 0);
    const oldTotalMoves = oldMoves.reduce((a, b) => a + b, 0);
    const newIdlePct = newTotalMoves > 0 ? (newMoves[4] / newTotalMoves) * 100 : 0;
    const oldIdlePct = oldTotalMoves > 0 ? (oldMoves[4] / oldTotalMoves) * 100 : 0;

    return {
      domain,
      mapId,
      mapName,
      gameIdx,
      swapSide,
      ticks: sim.t,
      outcome,
      winner: sim.winner,
      newBombs, oldBombs,
      newHits, oldHits,
      newSuicide, oldSuicide,
      newVisited: newVisited.size,
      oldVisited: oldVisited.size,
      newIdlePct, oldIdlePct,
      hp: [sim.hp[newIdx], sim.hp[oldIdx]],
    };
  }

  parentPort.on('message', async (msg) => {
    if (msg.type === 'init') {
      await initWorker(msg.modelName);
      parentPort.postMessage({ type: 'ready' });
    } else if (msg.type === 'task') {
      const result = await runTask(msg.task);
      parentPort.postMessage({ type: 'result', result });
    }
  });
  return;
}

// =========================================================================
// 主线程：任务生成、并发分发与统计报告
// =========================================================================
async function main() {
  const args = parseArgs();
  console.log(`\n================================================================================`);
  console.log(`⚔️  QQ堂 Headless A/B 真实同局对战评测套件`);
  console.log(`   模型: [${args.model}] (双方同权镜像互搏)`);
  console.log(`   核心对抗模式:`);
  console.log(`     [New] 物理位移探针 legal_mask + 移除开阔地/平墙归中被动切向滑移 _steer`);
  console.log(`     [Old] 整格静态查表 legal_mask + 撞障主动向中线归中 _steer`);
  console.log(`   并发 Worker: ${args.workers} (主机核数: ${os.cpus().length})`);
  console.log(`   物理上限: ${args.maxTicks} ticks | 严格座位互换消除偏置`);
  console.log(`================================================================================\n`);

  // 1. 读取 Stage 2 地图清单与 levels 元数据
  const currData = JSON.parse(fs.readFileSync(CURR_JSON, 'utf8'));
  const stage2Ids = currData.stages[1]; // Stage 2 map id list (22 maps)
  const allLevels = JSON.parse(fs.readFileSync(MAPS_JSON, 'utf8'));
  const list = Array.isArray(allLevels) ? allLevels : (allLevels.levels || allLevels.maps);
  const levelsById = new Map();
  list.forEach((lv) => levelsById.set(lv.id, lv));

  // 2. 初始化 Worker 池
  const workers = [];
  const readyPromises = [];
  for (let i = 0; i < args.workers; i++) {
    const w = new Worker(__filename);
    workers.push(w);
    readyPromises.push(new Promise((resolve) => {
      const onMsg = (msg) => {
        if (msg.type === 'ready') {
          w.off('message', onMsg);
          resolve();
        }
      };
      w.on('message', onMsg);
    }));
    w.postMessage({ type: 'init', modelName: args.model });
  }

  process.stdout.write(`⏳ 正在初始化 ${args.workers} 个 Worker 进程与 ONNX Session... `);
  const tInitStart = Date.now();
  await Promise.all(readyPromises);
  console.log(`完成 (${((Date.now() - tInitStart) / 1000).toFixed(2)}s)\n`);

  // 3. 构建评测域任务
  const domains = [
    {
      id: 'open_scene',
      name: '空场景 (Level 240 / 45°对角十字道具 / 50% 域随机化)',
      generateTasks: () => {
        const tasks = [];
        const openMap = levelsById.get(240);
        for (let i = 0; i < args.openGames; i++) {
          tasks.push({
            domain: 'open_scene',
            mapId: 240,
            mapName: openMap ? openMap.name : '空场景',
            gameIdx: i,
            seed: args.seed + i * 10007,
            maxTicks: args.maxTicks,
            swapSide: i % 2 === 1, // 严格半数 P0=New，半数 P1=New
          });
        }
        return tasks;
      },
    },
    {
      id: 'stage_2',
      name: `综合场景第二阶 (Stage 2 / 共 ${stage2Ids.length} 张经典地图)`,
      generateTasks: () => {
        const tasks = [];
        let gIdx = 0;
        for (const mapId of stage2Ids) {
          const mapObj = levelsById.get(mapId);
          const mapName = mapObj ? mapObj.name : `Level_${mapId}`;
          for (let r = 0; r < args.stage2Rounds; r++) {
            tasks.push({
              domain: 'stage_2',
              mapId,
              mapName,
              gameIdx: gIdx++,
              seed: args.seed + gIdx * 10007 + mapId * 37,
              maxTicks: args.maxTicks,
              swapSide: r % 2 === 1, // 严格 1:1 座位对调
            });
          }
        }
        return tasks;
      },
    },
  ];

  // 4. 执行测试分发与统计
  async function runDomainSuite(dom) {
    const tasks = dom.generateTasks();
    console.log(`--------------------------------------------------------------------------------`);
    console.log(`🎯 开始评测域: ${dom.name}`);
    console.log(`   规划局数: ${tasks.length} 局 (New 作为 P0: ${tasks.filter(t => !t.swapSide).length} 局, New 作为 P1: ${tasks.filter(t => t.swapSide).length} 局)`);
    console.log(`--------------------------------------------------------------------------------`);

    const results = [];
    const tStart = Date.now();
    let completed = 0;

    await new Promise((resolve) => {
      let taskIdx = 0;

      function dispatch(w) {
        if (taskIdx >= tasks.length) return;
        const task = tasks[taskIdx++];
        w.postMessage({ type: 'task', task });
      }

      workers.forEach((w) => {
        const onMsg = (msg) => {
          if (msg.type === 'result') {
            results.push(msg.result);
            completed++;
            const pct = Math.floor((completed / tasks.length) * 100);
            const res = msg.result;
            const resTag = res.outcome === 'new_win' ? '\x1b[32m[NEW 胜]\x1b[0m'
                         : (res.outcome === 'old_win' ? '\x1b[31m[OLD 胜]\x1b[0m' : '\x1b[33m[平 局]\x1b[0m');
            process.stdout.write(`\r[${pct.toString().padStart(3)}%] 已完成 ${completed}/${tasks.length} 局 | 最新: ${res.mapName} (t=${res.ticks}) ${resTag}   `);

            if (completed >= tasks.length) {
              workers.forEach((worker) => worker.off('message', onMsg));
              resolve();
            } else {
              dispatch(w);
            }
          }
        };
        w.on('message', onMsg);
        dispatch(w);
      });
    });

    const elapsed = ((Date.now() - tStart) / 1000).toFixed(2);
    console.log(`\n✅ 评测域完成，总耗时 ${elapsed}s (${(tasks.length / elapsed).toFixed(2)} 局/秒)\n`);

    // 统计分析
    const total = results.length;
    const newWins = results.filter((r) => r.outcome === 'new_win').length;
    const oldWins = results.filter((r) => r.outcome === 'old_win').length;
    const draws = results.filter((r) => r.outcome !== 'new_win' && r.outcome !== 'old_win').length;
    const mutuals = results.filter((r) => r.outcome === 'mutual_death').length;
    const timeouts = results.filter((r) => r.outcome === 'timeout' || r.outcome === 'draw').length;

    // 座位分项胜率 (P0 vs P1)
    const asP0 = results.filter((r) => !r.swapSide);
    const asP1 = results.filter((r) => r.swapSide);
    const p0NewWins = asP0.filter((r) => r.outcome === 'new_win').length;
    const p0OldWins = asP0.filter((r) => r.outcome === 'old_win').length;
    const p1NewWins = asP1.filter((r) => r.outcome === 'new_win').length;
    const p1OldWins = asP1.filter((r) => r.outcome === 'old_win').length;

    // 行为指标均值
    const avgTicks = (results.reduce((a, b) => a + b.ticks, 0) / total).toFixed(1);
    const avgNewBombs = (results.reduce((a, b) => a + b.newBombs, 0) / total).toFixed(1);
    const avgOldBombs = (results.reduce((a, b) => a + b.oldBombs, 0) / total).toFixed(1);
    const avgNewHits = (results.reduce((a, b) => a + b.newHits, 0) / total).toFixed(2);
    const avgOldHits = (results.reduce((a, b) => a + b.oldHits, 0) / total).toFixed(2);
    const newSuicideCount = results.filter((r) => r.newSuicide).length;
    const oldSuicideCount = results.filter((r) => r.oldSuicide).length;
    const avgNewVisited = (results.reduce((a, b) => a + b.newVisited, 0) / total).toFixed(1);
    const avgOldVisited = (results.reduce((a, b) => a + b.oldVisited, 0) / total).toFixed(1);
    const avgNewIdle = (results.reduce((a, b) => a + b.newIdlePct, 0) / total).toFixed(1);
    const avgOldIdle = (results.reduce((a, b) => a + b.oldIdlePct, 0) / total).toFixed(1);

    const newWinRateTotal = ((newWins / total) * 100).toFixed(1);
    const oldWinRateTotal = ((oldWins / total) * 100).toFixed(1);
    const nonDrawGames = newWins + oldWins;
    const newNetWinRate = nonDrawGames > 0 ? ((newWins / nonDrawGames) * 100).toFixed(1) : '50.0';

    console.log(`📊 【${dom.name} 对战总结】:`);
    console.log(`   总对局数: ${total} | 平均局长: ${avgTicks} ticks`);
    console.log(`   ┌───────────────┬──────────────┬──────────────┬──────────────┐`);
    console.log(`   │    阵营/指标  │  全局总胜率  │  扣除平局净胜│  总胜场/局数 │`);
    console.log(`   ├───────────────┼──────────────┼──────────────┼──────────────┤`);
    console.log(`   │  New (新动力) │    ${newWinRateTotal.padStart(6)}%   │    ${newNetWinRate.padStart(6)}%   │    ${newWins.toString().padStart(4)} / ${total}   │`);
    console.log(`   │  Old (原动力) │    ${oldWinRateTotal.padStart(6)}%   │    ${(100 - parseFloat(newNetWinRate)).toFixed(1).padStart(6)}%   │    ${oldWins.toString().padStart(4)} / ${total}   │`);
    console.log(`   │  平局/同归    │    ${((draws / total) * 100).toFixed(1).padStart(6)}%   │      -       │    ${draws.toString().padStart(4)} / ${total}   │`);
    console.log(`   └───────────────┴──────────────┴──────────────┴──────────────┘`);
    console.log(`   座位对称检验:`);
    console.log(`     - New 在 P0 座位 (先手/左): 胜 ${p0NewWins} / 负 ${p0OldWins} / 平 ${asP0.length - p0NewWins - p0OldWins} (胜率 ${((p0NewWins / asP0.length) * 100).toFixed(1)}%)`);
    console.log(`     - New 在 P1 座位 (后手/右): 胜 ${p1NewWins} / 负 ${p1OldWins} / 平 ${asP1.length - p1NewWins - p1OldWins} (胜率 ${((p1NewWins / asP1.length) * 100).toFixed(1)}%)`);
    console.log(`   关键战斗与行为对比:`);
    console.log(`     - 命中对手炸弹数: New 均 ${avgNewHits} 次 vs Old 均 ${avgOldHits} 次`);
    console.log(`     - 自身失误自杀数: New ${newSuicideCount} 次 vs Old ${oldSuicideCount} 次`);
    console.log(`     - 场均放泡量:     New ${avgNewBombs} 颗 vs Old ${avgOldBombs} 颗`);
    console.log(`     - 地图探索步履:   New 均走过 ${avgNewVisited} 格 vs Old 均走过 ${avgOldVisited} 格`);
    console.log(`     - 发呆静止率:     New ${avgNewIdle}% vs Old ${avgOldIdle}%\n`);

    return {
      domainId: dom.id,
      domainName: dom.name,
      total,
      newWins, oldWins, draws, mutuals, timeouts,
      newWinRateTotal, oldWinRateTotal, newNetWinRate,
      p0NewWins, p0OldWins, p1NewWins, p1OldWins, asP0Count: asP0.length, asP1Count: asP1.length,
      avgTicks, avgNewHits, avgOldHits, newSuicideCount, oldSuicideCount,
      avgNewBombs, avgOldBombs, avgNewVisited, avgOldVisited, avgNewIdle, avgOldIdle,
      results,
    };
  }

  const reports = [];
  for (const dom of domains) {
    const rep = await runDomainSuite(dom);
    reports.push(rep);
  }

  // 关闭 workers
  for (const w of workers) w.terminate();

  console.log(`================================================================================`);
  console.log(`🏁 全部 A/B 对战评测圆满交付！`);
  console.log(`================================================================================\n`);
  return reports;
}

if (isMainThread) {
  main().catch((err) => {
    console.error('评测运行失败:', err);
    process.exit(1);
  });
}
