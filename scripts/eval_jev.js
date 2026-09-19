#!/usr/bin/env node
/**
 * scripts/eval_jev.js - TypeSafe Jev (System One) 在真实关卡中的自动化评测
 *
 * 验证目标：
 * 1. 结构化特征提取与 Jev System One 决策连通性
 * 2. Jev 炸墙破障（bricks destroyed）
 * 3. Jev 收集道具提升属性（crates collected & stat growth）
 * 4. Jev 避险与击杀对手对局表现
 */

'use strict';

const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevGridAI = require('../web/jev_grid_ai.js');

const { Sim, CFG } = QQT;
const H = QQT.H, W = QQT.W, N = QQT.N;

const levelsPath = path.join(__dirname, '../web/assets/maps/levels.json');
const levels = JSON.parse(fs.readFileSync(levelsPath, 'utf8'));

// 挑选包含丰富砖块和道具的经典竞技关卡
const testMaps = [
  'desert01_4.map',     // 沙漠普通竞技
  'contest01_8.map',    // 比武场多砖块
  'town10_8.map',       // 中国城城镇
  'treasure01_4.map'    // 夺宝地图
];

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function runMatch(levelSource, maxTicks = 120) {
  const level = levels.find(l => l.source === levelSource) || levels[0];
  console.log(`\n======================================================`);
  console.log(`开始评测关卡: [${level.name || level.source}] (${level.category || '竞技'})`);
  console.log(`初始属性: bombs=${level.initial_stats.bombs}, blast=${level.initial_stats.blast}, speed=${level.initial_stats.speed}`);
  console.log(`关卡上限: bombs_max=${level.bombs_max || 10}, blast_max=${level.blast_max || 8}, speed_max=${level.speed_max || 2.4}`);
  console.log(`======================================================`);

  const seed = 20260919;
  const sim = new Sim(seed);
  sim.reset(level);

  // P0: TypeSafe Jev (高维坐标全图版), P1: 规则 Hunter
  const jev = new JevGridAI({
    apiUrl: 'https://api.typesafe.ai/v1/systemone',
    inferIntervalTicks: 10 // 每 1s 一次 Jev 深度研判
  });
  const hunter = new QQT.HunterAI();

  let initialBricks = 0;
  for (let i = 0; i < N; i++) if (sim.brick[i]) initialBricks++;
  console.log(`地图总砖块数: ${initialBricks}, 初始宝箱数: ${sim.crate.filter(c => c > 0).length}`);

  let cratesCollected = 0;
  let prevCrates = sim.crate.slice();
  let prevBricks = sim.brick.slice();
  let bricksDestroyed = 0;

  // 预热：让 Jev 在开局第 0 tick 触发第一次决策
  jev.callJevAsync(sim, 0);
  await sleep(600); // 等待首个网络推理返回

  const startT = Date.now();

  for (let t = 0; t < maxTicks; t++) {
    // 检查 Jev 是否吃到了箱子
    const p0Cell = sim.centerCell(0);
    const p0Idx = p0Cell[0] * W + p0Cell[1];
    if (prevCrates[p0Idx] === 1 && sim.crate[p0Idx] === 0) {
      cratesCollected++;
      console.log(`[Tick ${t}] ★ Jev 成功收集到道具！当前属性: 炸弹=${sim.bombsCap[0]}, 威力=${sim.blastCap[0]}, 移速=${sim.spdG[0].toFixed(2)}`);
    }
    prevCrates = sim.crate.slice();

    // 动作生成
    const a0 = jev.act(sim, 0);
    const a1 = hunter.act(sim, 1);

    // 如果 Jev 正在推理，短暂让出事件循环允许 fetch 完成
    if (jev.isInferring) {
      await sleep(10);
    }

    const info = sim.step([a0, a1]);

    // 统计被炸毁的砖块
    for (let i = 0; i < N; i++) {
      if (prevBricks[i] === 1 && sim.brick[i] === 0) {
        bricksDestroyed++;
      }
    }
    prevBricks = sim.brick.slice();

    if (t % 20 === 0 && t > 0) {
      const dec = jev.lastTacticalDecision;
      console.log(`[Tick ${t}] Jev 战术状态: 决策=${dec ? dec.priority : 'N/A'}, 目标=${dec ? dec.targetKey : 'N/A'}, 放置炸弹数=${jev.stats.bombsPlaced}, 已破砖=${bricksDestroyed}, Jev调用=${jev.stats.totalCalls}次`);
    }

    if (sim.done) {
      console.log(`[Tick ${t}] 对局结束！胜者: P${sim.winner} (0=Jev, 1=Hunter, -1=平局)`);
      break;
    }
  }

  const durationMs = Date.now() - startT;
  console.log(`\n----------------- 关卡测试总结 -----------------`);
  console.log(`总运行时间: ${durationMs}ms`);
  console.log(`Jev API 调用次数: ${jev.stats.totalCalls}`);
  console.log(`Jev 放置炸弹次数: ${jev.stats.bombsPlaced}`);
  console.log(`破坏砖块数: ${bricksDestroyed}`);
  console.log(`收集道具数: ${cratesCollected}`);
  console.log(`最终属性: 炸弹=${sim.bombsCap[0]}/${sim.bombsMax || 10}, 威力=${sim.blastCap[0]}/${sim.blastMax || 8}, 移速=${sim.spdG[0].toFixed(2)}/${(sim.speedMax || 2.4).toFixed(2)}`);
  console.log(`存活状态: P0(Jev)=${sim.alive[0]} (HP: ${sim.hp[0]}), P1(Hunter)=${sim.alive[1]} (HP: ${sim.hp[1]})`);
  console.log(`------------------------------------------------\n`);

  return {
    level: level.source,
    ticks: sim.t,
    winner: sim.winner,
    bombsPlaced: jev.stats.bombsPlaced,
    bricksDestroyed,
    cratesCollected,
    totalCalls: jev.stats.totalCalls,
    finalStats: {
      bombs: sim.bombsCap[0],
      blast: sim.blastCap[0],
      speed: sim.spdG[0]
    }
  };
}

async function main() {
  console.log(`开始 TypeSafe Jev 跨地图评测 (共评测 ${testMaps.length} 张代表关卡)...`);
  const results = [];
  for (const mapSrc of testMaps) {
    try {
      const res = await runMatch(mapSrc, 100);
      results.push(res);
    } catch (err) {
      console.error(`评测关卡 ${mapSrc} 出错:`, err);
    }
  }

  console.log(`\n=================== 全部评测汇总 ===================`);
  console.table(results.map(r => ({
    '关卡': r.level,
    '总步数': r.ticks,
    'Jev放炮': r.bombsPlaced,
    '炸毁砖块': r.bricksDestroyed,
    '收集道具': r.cratesCollected,
    '最终炸弹/威力/移速': `${r.finalStats.bombs} / ${r.finalStats.blast} / ${r.finalStats.speed.toFixed(2)}`,
    '胜者': r.winner === 0 ? 'P0 (Jev胜)' : (r.winner === 1 ? 'P1 (Hunter胜)' : '平局/未结束')
  })));
}

main().catch(console.error);
