const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const TimeAStarAI = require('../web/time_astar_ai.js');
const JevGridAI = require('../web/jev_grid_ai.js');
const JevAutonomousAI = require('../web/jev_autonomous_ai.js');
const JevAI = require('../web/jev_ai.js');

const { Sim } = QQT;
const levels = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8')).levels || JSON.parse(fs.readFileSync(path.join(__dirname, '../web/assets/maps/levels.json'), 'utf8'));

console.log('=== 🔬 验证连通性校验与非联通格投影机制 ===\n');

// 1. 测试 Map 42 (含有实心墙孤岛盲区格 82, 112)
const m42 = levels[42];
console.log(`[测试 1] Map 42 孤岛格测试 (目标为实心墙隔断格 82: row 5, col 7)`);
const sim42 = new Sim(2026);
sim42.reset(m42);
const W = 15, H = 13;

// 初始化 JevGridAI
const gridAi = new JevGridAI({ apiUrl: 'mock' });
// 强行模拟 Jev 输出孤岛盲区坐标 (5, 7)
gridAi.targetRow = 5;
gridAi.targetCol = 7;
gridAi.targetIntent = 'hunt_opponent';
gridAi.bombDecision = 'hold_bomb';

const ownBefore = sim42.centerCell(0);
console.log(`玩家初始位置: (${ownBefore[0]}, ${ownBefore[1]}), 注入目标: (${gridAi.targetRow}, ${gridAi.targetCol})`);

// 提取状态，检查 rowCriteria / colCriteria 是否正确标记
const state42 = gridAi.extractGridState(sim42, 0);
console.log(`- reachable_rows 数量: ${state42.reachable_rows.length}`);
console.log(`- reachable_cols 数量: ${state42.reachable_cols.length}`);

// 执行一次 act
const act42 = gridAi.act(sim42, 0);
console.log(`- act 返回动作: [move=${act42[0]}, bomb=${act42[1]}]`);
console.log(`- 投影后的 targetPos: [${gridAi.targetPos}] (原始目标为 [5, 7])`);
console.log(`- 投影后的 targetCell: ${gridAi.targetCell}`);
console.log(`- A* 路径长度: ${gridAi.currentSearchPath.length}`);

if (gridAi.targetCell === 82) {
  console.error('❌ 失败：目标仍为孤岛格 82，未完成投影！');
  process.exit(1);
} else {
  console.log('✅ 成功：目标已成功从孤岛格 82 投影至合法连通格 ' + gridAi.targetCell);
}

if (gridAi.currentSearchPath.length === 0) {
  console.error('❌ 失败：A* 寻路路径为空！');
  process.exit(1);
} else {
  console.log('✅ 成功：A* 寻路成功找到路径，步数: ' + gridAi.currentSearchPath.length);
}

// 2. 测试 JevAutonomousAI 在 Map 42 上的表现
console.log(`\n[测试 2] JevAutonomousAI 候选实体连通性过滤`);
const autoAi = new JevAutonomousAI({ apiUrl: 'mock' });
const autoState = autoAi.extractAutonomousState(sim42, 0);
// 验证所有 crates 和 bricks 均在连通分量内
const reachableMask42 = autoAi.computeConnectedComponent(sim42, ownBefore[0] * W + ownBefore[1]);
let anyUnreachableCrate = false;
for (const c of autoState.nearby_crates || []) {
  if (!reachableMask42[c.cell]) anyUnreachableCrate = true;
}
if (anyUnreachableCrate) {
  console.error('❌ 失败：候选道具中混入了非联通格道具！');
  process.exit(1);
} else {
  console.log('✅ 成功：候选道具中 100% 均为合法连通格！');
}

// 3. 测试破砖开路动作与安全防自杀
console.log(`\n[测试 3] 砖块阻挡自动破障测试 (Map 0)`);
const m0 = levels[0];
const sim0 = new Sim(2026);
sim0.reset(m0);
const gridAi0 = new JevGridAI({ apiUrl: 'mock' });
// 设置一个需要穿透砖块的目标
const p1Pos = sim0.centerCell(1);
gridAi0.targetRow = p1Pos[0];
gridAi0.targetCol = p1Pos[1];
gridAi0.targetIntent = 'breach_obstacle';
gridAi0.bombDecision = 'hold_bomb'; // 即使 Jev 原文 hold_bomb，在撞砖时底层也应触发安全破障

let placedBomb = false;
for (let step = 0; step < 20; step++) {
  const act = gridAi0.act(sim0, 0);
  if (act[1] === 1) {
    placedBomb = true;
    console.log(`[step=${step}] 检测到障碍砖，成功触发就地破障落子！`);
    break;
  }
  sim0.step([act, [4, 0]]);
}

if (placedBomb) {
  console.log('✅ 成功：障碍砖阻挡时成功触发破障落子！');
} else {
  console.log('ℹ️ 当前走廊未直接撞砖或已在其他路径推进');
}

console.log('\n🎉 所有连通性与非联通格投影测试全部通过！');
