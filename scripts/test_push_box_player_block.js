// test_push_box_player_block.js
// 验证推箱子目标格被角色占领时无法推动，角色移开后可正常推动。

const QQT = require('../web/sim.js');
const { Sim, CFG, DIRS, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_IDLE } = QQT;

const H = 13, W = 15;
let passes = 0, fails = 0;

function assert(condition, msg) {
  if (condition) {
    console.log('  [PASS]', msg);
    passes++;
  } else {
    console.error('  [FAIL]', msg);
    fails++;
  }
}

console.log('=== 开始推箱子角色占位阻挡测试 ===');

// 1. 测试 _cellOccupiedByPlayer 几何重叠与排除
console.log('\n-- 测试 1: _cellOccupiedByPlayer 判定准确度 --');
{
  const sim = new Sim(42);
  sim.pos[0] = 5.5; sim.pos[1] = 5.5; // P0 在 (5, 5)
  sim.pos[2] = 6.5; sim.pos[3] = 5.5; // P1 在 (6, 5)
  sim.alive = [true, true];

  const targetCell = 6 * W + 5;
  assert(sim._cellOccupiedByPlayer(targetCell, 0) === true, 'P1 位于目标格时 _cellOccupiedByPlayer(target, P0) 为 true');
  assert(sim._cellOccupiedByPlayer(targetCell, 1) === false, '排除 P1 自身时为 false');

  // P1 走到相邻格 (6, 6)
  sim.pos[2] = 6.5; sim.pos[3] = 6.5;
  assert(sim._cellOccupiedByPlayer(targetCell, 0) === false, 'P1 离开目标格到 (6, 6) 后为 false');

  // P1 死亡
  sim.pos[2] = 6.5; sim.pos[3] = 5.5;
  sim.alive[1] = false;
  assert(sim._cellOccupiedByPlayer(targetCell, 0) === false, 'P1 死亡后即使留在该格也不阻挡');
}

// 2. 测试真实推箱物理流程（P1 挡在箱子前方）
console.log('\n-- 测试 2: 真实对局中目标格有角色无法推动 --');
{
  const brickArr = new Array(H * W).fill(0);
  brickArr[5 * W + 5] = 1; // 可推箱本身作为障碍物也是 brick

  const level = {
    id: 999,
    wall: new Array(H * W).fill(0),
    brick: brickArr,
    spawns: [[4, 5], [6, 5]],
    push_boxes: [[5, 5, 1, 1]], // (5, 5) 处有一个 1x1 可推箱
    initial_stats: { bombs: 1, blast: 1, speed: 1.0 }
  };
  const sim = new Sim(101, { playerModes: ['new', 'new'] });
  sim._loadLevel(level);

  // 摆放位置：
  // P0 在 (4.5, 5.5)，顶着箱子 (5, 5) 向下推 (MOVE_DOWN)
  // P1 在 (6.5, 5.5)，正正站在目标格 (6, 5)
  sim.pos[0] = 4.5; sim.pos[1] = 5.5;
  sim.pos[2] = 6.5; sim.pos[3] = 5.5;
  sim.alive = [true, true];

  const boxOrigin = 5 * W + 5;
  const targetCell = 6 * W + 5;

  assert(sim.pushBoxAt[boxOrigin] >= 0, '箱子初始在 (5, 5)');
  assert(sim.brick[boxOrigin] === 1, '箱子初始占用 brick[5, 5]');
  assert(sim.brick[targetCell] === 0, '目标格 (6, 5) 初始无砖');

  // P0 连续向下推 10 个 tick (1.0 秒，远超 PUSH_TIME 0.3s)
  for (let t = 0; t < 10; t++) {
    sim.step([[MOVE_DOWN, 0], [MOVE_IDLE, 0]]);
  }

  assert(sim.pushBoxAt[boxOrigin] >= 0, '由于 P1 占位阻挡，箱子仍在原位 (5, 5)');
  assert(sim.pushBoxAt[targetCell] === -1, '目标格 (6, 5) 没有被箱子推入');
  assert(sim.brick[targetCell] === 0, '目标格 (6, 5) 未产生砖块碰撞');
  assert(sim.pushT[boxOrigin] === 0, '受阻时推动计时器被清零');

  // 3. P1 移开后，箱子恢复可推
  console.log('\n-- 测试 3: 角色移开后，箱子恢复正常推动 --');
  // P1 走到右侧 (6.5, 8.5)
  sim.pos[2] = 6.5; sim.pos[3] = 8.5;

  // P0 连续向下推 3 个 tick (0.3s = PUSH_TIME)
  for (let t = 0; t < 3; t++) {
    sim.step([[MOVE_DOWN, 0], [MOVE_IDLE, 0]]);
  }

  assert(sim.pushBoxAt[boxOrigin] === -1, 'P1 离开后推满 0.3s，原位 (5, 5) 箱子已离开');
  assert(sim.pushBoxAt[targetCell] >= 0, '箱子成功推入目标格 (6, 5)');
  assert(sim.brick[boxOrigin] === 0, '原位 (5, 5) brick 已清除');
  assert(sim.brick[targetCell] === 1, '目标格 (6, 5) brick 已占领');
}

// 4. 测试横向推动阻挡
console.log('\n-- 测试 4: 横向推动（向右推）被角色阻挡 --');
{
  const brickArr = new Array(H * W).fill(0);
  brickArr[5 * W + 5] = 1;

  const level = {
    id: 998,
    wall: new Array(H * W).fill(0),
    brick: brickArr,
    spawns: [[5, 4], [5, 6]],
    push_boxes: [[5, 5, 1, 1]],
    initial_stats: { bombs: 1, blast: 1, speed: 1.0 }
  };
  const sim = new Sim(102, { playerModes: ['new', 'new'] });
  sim._loadLevel(level);

  // P0 在 (5.5, 4.5)，向右推
  // P1 在 (5.5, 6.5)，占领目标格 (5, 6)
  sim.pos[0] = 5.5; sim.pos[1] = 4.5;
  sim.pos[2] = 5.5; sim.pos[3] = 6.5;
  sim.alive = [true, true];

  const boxOrigin = 5 * W + 5;
  const targetCell = 5 * W + 6;

  for (let t = 0; t < 5; t++) {
    sim.step([[MOVE_RIGHT, 0], [MOVE_IDLE, 0]]);
  }

  assert(sim.pushBoxAt[boxOrigin] >= 0, '横向推动受 P1 占位阻挡，箱子仍在原位 (5, 5)');
  assert(sim.pushBoxAt[targetCell] === -1, '目标格 (5, 6) 未被推入');

  // P1 走开到 (7.5, 6.5)
  sim.pos[2] = 7.5; sim.pos[3] = 6.5;

  for (let t = 0; t < 3; t++) {
    sim.step([[MOVE_RIGHT, 0], [MOVE_IDLE, 0]]);
  }

  assert(sim.pushBoxAt[targetCell] >= 0, 'P1 离开后，箱子成功推入 (5, 6)');
}

console.log(`\n测试汇总: PASS=${passes}, FAIL=${fails}`);
if (fails > 0) {
  process.exit(1);
} else {
  console.log('全部推箱子角色占位测试通过！✔');
  process.exit(0);
}
