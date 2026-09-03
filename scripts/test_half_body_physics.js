// 自动化单元测试：验证 QQ堂半身位与并排连泡破半身物理判定
const assert = require('assert');
const QQT = require('../web/sim.js');
const { Sim, CFG, H, W, MOVE_IDLE } = QQT;

console.log('==============================================================');
console.log('🧪 正在测试 QQ堂半身位 (Half-Body) 与并排连泡伤害判定...');
console.log('==============================================================');

// 测试辅助函数：创建空地图环境
function createCleanSim() {
  const mapData = {
    h: 13, w: 15,
    wall: new Array(13 * 15).fill(0),
    brick: new Array(13 * 15).fill(0),
    bush: new Array(13 * 15).fill(0),
    spawns: [[3.5, 2.0], [10.5, 10.5]]
  };
  const sim = new Sim(mapData, 42);
  sim.invuln[0] = 0;
  sim.invuln[1] = 0;
  return sim;
}

// --------------------------------------------------------------------------
// 场景 1（用户图一）：单泡垂直水柱，角色在分界线半身位 (x=2.0) -> 无伤！
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  // 玩家 0 站在 (3.5, 2.0)（处于 col 1 与 col 2 分界线上，半身位）
  sim.pos[0] = 3.5;
  sim.pos[1] = 2.0;
  
  // 在 (1, 1) 放一颗泡，威力 3，fuse=1（下一 tick 爆炸，向下穿过 (2, 1) 和 (3, 1)）
  sim.bombBlast[1 * W + 1] = 3;
  sim.fuse[1 * W + 1] = 1;
  sim.owner[1 * W + 1] = 1; // 敌人放的
  
  const hpBefore = sim.hp[0];
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]); // 触发爆炸
  
  assert.strictEqual(sim.lastCovered[3 * W + 1], 1, '格 (3, 1) 应当被水柱覆盖');
  assert.strictEqual(sim.lastCovered[3 * W + 2], 0, '格 (3, 2) 不应被覆盖');
  assert.strictEqual(sim.hp[0], hpBefore, '图一单泡半身位应当完全无伤！');
  console.log('  ✓ [场景 1] 图一验证通过：单泡向下水柱，玩家在 x=2.0 半身位无伤存活！');
}

// --------------------------------------------------------------------------
// 场景 2：正中直击基线，角色站在水柱正中心 (x=1.5) -> 必伤！
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  sim.pos[0] = 3.5;
  sim.pos[1] = 1.5; // col 1 正中
  
  sim.bombBlast[1 * W + 1] = 3;
  sim.fuse[1 * W + 1] = 1;
  sim.owner[1 * W + 1] = 1;
  
  const hpBefore = sim.hp[0];
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  
  assert.strictEqual(sim.hp[0], hpBefore - 1, '直击中心应当掉血！');
  console.log('  ✓ [场景 2] 正中直击验证通过：玩家在 x=1.5 水柱正中正常受到伤害！');
}

// --------------------------------------------------------------------------
// 场景 3（用户图二）：两颗泡并排连锁向下喷发，角色在分界线半身位 (x=2.0) -> 必伤！
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  sim.pos[0] = 3.5;
  sim.pos[1] = 2.0; // 同样在 x=2.0 半身位
  
  // 在 (1, 1) 和 (1, 2) 各放一颗泡，连锁同时向下炸
  sim.bombBlast[1 * W + 1] = 3;
  sim.fuse[1 * W + 1] = 1;
  sim.owner[1 * W + 1] = 1;

  sim.bombBlast[1 * W + 2] = 3;
  sim.fuse[1 * W + 2] = 1;
  sim.owner[1 * W + 2] = 1;
  
  const hpBefore = sim.hp[0];
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  
  assert.strictEqual(sim.lastCovered[3 * W + 1], 1, '格 (3, 1) 应当被水柱覆盖');
  assert.strictEqual(sim.lastCovered[3 * W + 2], 1, '格 (3, 2) 应当被水柱覆盖');
  assert.strictEqual(sim.hp[0], hpBefore - 1, '图二双泡并排连锁，半身位必须受到伤害！');
  console.log('  ✓ [场景 3] 图二验证通过：双泡并排连锁破半身，玩家在 x=2.0 必受伤害！');
}

// --------------------------------------------------------------------------
// 场景 4：单水柱情况下，角色未卡准半身位（向着火格偏入 0.1 格，x=1.90）-> 必伤！
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  sim.pos[0] = 3.5;
  sim.pos[1] = 1.90; // 偏向 col 1: 1.90 - 0.42 = 1.48 <= 1.50
  
  sim.bombBlast[1 * W + 1] = 3;
  sim.fuse[1 * W + 1] = 1;
  sim.owner[1 * W + 1] = 1;
  
  const hpBefore = sim.hp[0];
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  
  assert.strictEqual(sim.hp[0], hpBefore - 1, '偏入着火列应当掉血！');
  console.log('  ✓ [场景 4] 边界微操验证通过：偏离绝对半身位（x=1.90）擦碰水柱轴线正常受伤！');
}

// --------------------------------------------------------------------------
// 场景 5：水平方向水柱的半身位 (y=2.0) 避伤与双泡水平并排必伤验证
// --------------------------------------------------------------------------
{
  // 5a. 单水柱水平掠过 row 1，玩家在 y=2.0 半身位 -> 无伤
  const simA = createCleanSim();
  simA.pos[0] = 2.0; simA.pos[1] = 3.5;
  simA.bombBlast[1 * W + 1] = 4;
  simA.fuse[1 * W + 1] = 1;
  simA.owner[1 * W + 1] = 1;
  const hpBeforeA = simA.hp[0];
  simA.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  assert.strictEqual(simA.hp[0], hpBeforeA, '单水平水柱，y=2.0 半身位应当无伤！');
  
  // 5b. 两道水平水柱穿过 row 1 和 row 2，玩家在 y=2.0 半身位 -> 必伤
  const simB = createCleanSim();
  simB.pos[0] = 2.0; simB.pos[1] = 3.5;
  simB.bombBlast[1 * W + 1] = 4; simB.fuse[1 * W + 1] = 1; simB.owner[1 * W + 1] = 1;
  simB.bombBlast[2 * W + 1] = 4; simB.fuse[2 * W + 1] = 1; simB.owner[2 * W + 1] = 1;
  const hpBeforeB = simB.hp[0];
  simB.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  assert.strictEqual(simB.hp[0], hpBeforeB - 1, '双水平水柱并排，y=2.0 半身位必伤！');
  console.log('  ✓ [场景 5] 水平水流验证通过：横向半身无伤与双水柱并排必伤完全对称！');
}

console.log('==============================================================');
console.log('🎉 全部 5 个半身位物理场景测试 100% 验收通过！');
console.log('==============================================================');
