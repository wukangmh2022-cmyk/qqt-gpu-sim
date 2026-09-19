const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const JevPureAI = require('../web/jev_pure_ai.js');
const JevGridAI = require('../web/jev_grid_ai.js');
const { Sim } = QQT;

const sleep = ms => new Promise(r => setTimeout(r, ms));

function computePercentile(sortedArr, p) {
  if (!sortedArr.length) return 0;
  const index = (p / 100) * (sortedArr.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  if (lower === upper) return sortedArr[lower];
  return Math.round(sortedArr[lower] * (1 - weight) + sortedArr[upper] * weight);
}

function summarizeLatencies(latencies) {
  const sorted = [...latencies].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const mean = Math.round(sum / sorted.length);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const p50 = computePercentile(sorted, 50);
  const p90 = computePercentile(sorted, 90);
  const p95 = computePercentile(sorted, 95);
  const p99 = computePercentile(sorted, 99);

  return { count: sorted.length, min, mean, p50, p90, p95, p99, max, samples: sorted };
}

async function benchmarkModel(name, aiFactory, numSamples = 25) {
  console.log(`\n======================================================`);
  console.log(`⏱️  开始压测: ${name} (样本数: ${numSamples})`);
  console.log(`======================================================`);

  const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));
  const level = levels.find(l => l.source === 'contest01_8.map') || levels[0];

  const latencies = [];

  for (let i = 0; i < numSamples; i++) {
    const sim = new Sim(2026 + i);
    sim.reset(level);

    // 制造不同游戏步态与位置变化
    sim.t = i * 4;
    sim.pos[0] = 6 + ((i % 3) - 1) * 0.2;
    sim.pos[1] = 9 + ((i % 2) - 1) * 0.3;

    const ai = aiFactory();
    const t0 = performance.now();
    try {
      await ai.callJevAsync(sim, 0);
      const latency = Math.round(performance.now() - t0);
      latencies.push(latency);
      process.stdout.write(`[${i + 1}/${numSamples}] ${latency}ms | `);
      if ((i + 1) % 5 === 0) console.log();
    } catch (err) {
      console.error(`\n[${i + 1}/${numSamples}] 失败:`, err.message);
    }

    // 避免过度并发触发严格频控
    await sleep(80);
  }

  const stats = summarizeLatencies(latencies);
  console.log(`\n📊 【${name}】统计结果:`);
  console.log(`- 成功样本数: ${stats.count}/${numSamples}`);
  console.log(`- 最小延迟 (Min): ${stats.min} ms`);
  console.log(`- 平均延迟 (Mean): ${stats.mean} ms`);
  console.log(`- P50 (中位数): ${stats.p50} ms`);
  console.log(`- P90 延迟: ${stats.p90} ms`);
  console.log(`- P95 延迟: ${stats.p95} ms`);
  console.log(`- P99 延迟: ${stats.p99} ms`);
  console.log(`- 最大延迟 (Max): ${stats.max} ms`);

  return { name, ...stats };
}

async function runBenchmark() {
  console.log('🚀 TypeSafe Jev API 端到端推演延迟评测 (P50/P90/P95/P99)');
  console.log('API 地址: http://localhost:8080/api/typesafe\n');

  // 1. 评测纯净直出版 (Doom 范式)
  const pureStats = await benchmarkModel(
    'JevPureAI (纯净直出版 · Doom 范式)',
    () => new JevPureAI({ apiUrl: 'http://localhost:8080/api/typesafe' }),
    25
  );

  // 2. 评测高维全图版 (15x13 网格)
  const gridStats = await benchmarkModel(
    'JevGridAI (高维坐标全图版 · 15x13 网格)',
    () => new JevGridAI({ apiUrl: 'http://localhost:8080/api/typesafe' }),
    20
  );

  console.log('\n\n======================================================');
  console.log('🏆 综合延迟对比表');
  console.log('======================================================');
  console.table([
    {
      '模型版本': pureStats.name,
      '样本数': pureStats.count,
      'Min': `${pureStats.min}ms`,
      'Mean': `${pureStats.mean}ms`,
      'P50 (中位数)': `${pureStats.p50}ms`,
      'P90': `${pureStats.p90}ms`,
      'P95': `${pureStats.p95}ms`,
      'Max': `${pureStats.max}ms`
    },
    {
      '模型版本': gridStats.name,
      '样本数': gridStats.count,
      'Min': `${gridStats.min}ms`,
      'Mean': `${gridStats.mean}ms`,
      'P50 (中位数)': `${gridStats.p50}ms`,
      'P90': `${gridStats.p90}ms`,
      'P95': `${gridStats.p95}ms`,
      'Max': `${gridStats.max}ms`
    }
  ]);
}

runBenchmark().catch(console.error);
