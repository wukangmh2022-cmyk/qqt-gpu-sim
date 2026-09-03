// 单元测试：验证道具墓地机制、水泡炸毁道具、满属性溢出、超级道具+5档与飞鸟空投
const assert = require('assert');
const QQT = require('../web/sim.js');
const { Sim, CFG, H, W, MOVE_IDLE, MOVE_RIGHT } = QQT;

console.log('🧪 正在测试道具墓地机制、水泡炸毁道具、满属性溢出、超级道具+5档与飞鸟空投...');

function createCleanSim() {
  const mapData = {
    h: 13, w: 15,
    wall: new Array(13 * 15).fill(0),
    brick: new Array(13 * 15).fill(0),
    bush: new Array(13 * 15).fill(0),
    spawns: [[6.5, 4.5], [10.5, 10.5]]
  };
  const sim = new Sim(mapData, 42);
  sim.crate.fill(0);
  sim.crateType.fill(-1);
  sim.superCrate.fill(0);
  sim.graveyard = [];
  sim.invuln[0] = 0;
  sim.invuln[1] = 0;
  return sim;
}

// --------------------------------------------------------------------------
// 测试 1：水泡覆盖地图现有道具 -> 道具被炸毁并进入墓地
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  // 在 (3, 3) 放置一个普通速度道具 (type=2)
  sim.crate[3 * W + 3] = 1;
  sim.crateType[3 * W + 3] = 2;
  sim.superCrate[3 * W + 3] = 0;

  // 在 (3, 2) 放炸弹，威力 2，下一 tick 引爆并波及 (3, 3)
  sim.bombBlast[3 * W + 2] = 2;
  sim.fuse[3 * W + 2] = 1;
  sim.owner[3 * W + 2] = 1;

  assert.strictEqual(sim.graveyard.length, 0, '引爆前墓地应为空');

  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]); // 引爆

  assert.strictEqual(sim.crate[3 * W + 3], 0, '被水泡覆盖的道具应当被清除');
  assert.strictEqual(sim.graveyard.length, 1, '被水泡覆盖的道具应当进入墓地');
  assert.strictEqual(sim.graveyard[0].type, 2, '进入墓地的道具种类应为速度道具(2)');
  assert.strictEqual(sim.graveyard[0].isSuper, false, '普通道具 isSuper 应为 false');
  console.log('  ✓ [测试 1] 验证通过：水泡成功炸毁地图道具并将其回收进入墓地！');
}

// --------------------------------------------------------------------------
// 测试 2：满属性玩家拾取道具 -> 溢出属性自动流入墓地
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  // 设置玩家 0 泡泡数量为满属性 (bombsMax)
  sim.bombsCap[0] = sim.bombsMax;
  assert.strictEqual(sim.bombsCap[0], sim.bombsMax);

  // 玩家 0 拾取一个泡泡道具 (type=0)
  sim._grow(0, false, 0);

  assert.strictEqual(sim.bombsCap[0], sim.bombsMax, '满属性不应再增加数值');
  assert.strictEqual(sim.graveyard.length, 1, '溢出的属性应当进入墓地');
  assert.strictEqual(sim.graveyard[0].type, 0, '溢出道具应为泡泡(0)');
  assert.strictEqual(sim.graveyard[0].isSuper, false);
  console.log('  ✓ [测试 2] 验证通过：满属性玩家吃道具，溢出属性成功流入墓地！');
}

// --------------------------------------------------------------------------
// 测试 3：超级道具档位提升至 +5 档（选项 A）
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  sim.bombsCap[0] = 2; // 起步 2 颗泡
  // 拾取超级泡泡道具
  sim._grow(0, true, 0);
  // 2 + 5 = 7
  assert.strictEqual(sim.bombsCap[0], 7, '超级道具应当提供 +5 档提升');
  console.log('  ✓ [测试 3] 验证通过：超级道具提升档位成功由 +4 升级至 +5 档！');
}

// --------------------------------------------------------------------------
// 测试 4：墓地道具空投落地写入地图数据
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  const targetCell = 5 * W + 5;
  assert.strictEqual(sim.crate[targetCell], 0);

  const ok = sim.spawnGraveyardDrop(targetCell, 1, true); // 超级威力
  assert.strictEqual(ok, true);
  assert.strictEqual(sim.crate[targetCell], 1);
  assert.strictEqual(sim.crateType[targetCell], 1);
  assert.strictEqual(sim.superCrate[targetCell], 1);
  console.log('  ✓ [测试 4] 验证通过：空投落地接口成功向地图数据原子写入道具！');
}

// --------------------------------------------------------------------------
// 测试 5：无 UI 仿真模式下，第 300 tick（30s）墓地道具自动空投刷向地面
// --------------------------------------------------------------------------
{
  const sim = createCleanSim();
  sim.graveyard.push({ type: 0, isSuper: false });
  sim.graveyard.push({ type: 1, isSuper: true });
  sim.t = 299;

  // 下一步到达 tick 300 (30s)
  sim.step([[MOVE_IDLE, 0], [MOVE_IDLE, 0]]);
  assert.strictEqual(sim.t, 300);
  assert.strictEqual(sim.graveyard.length, 0, '第 300 tick 墓地道具应当全部空投刷向地面');

  let crateCount = 0;
  for (let i = 0; i < H * W; i++) {
    if (sim.crate[i]) crateCount++;
  }
  assert.strictEqual(crateCount, 2, '地面应当成功刷新 2 个空投道具');
  console.log('  ✓ [测试 5] 验证通过：30s 周期无 UI 自驱空投刷新机制运行完美！');
}

console.log('==============================================================');
console.log('🎉 全部 5 项墓地、飞鸟与超级道具单元测试 100% 验收通过！');
console.log('==============================================================');
