// test_lowfreq_150ms.js
// 验证「降低 AI 反应 50%」推理降频机制（150ms 推理间隔）

const fs = require('fs');
const path = require('path');
const QQT = require('../web/sim.js');
const { Sim, MLPModel, TransformerModel } = QQT;

let passes = 0, fails = 0;
function assert(cond, msg) {
  if (cond) {
    console.log('  [PASS]', msg);
    passes++;
  } else {
    console.error('  [FAIL]', msg);
    fails++;
  }
}

console.log('=== 开始测试 150ms 推理降频机制 ===');

// 1. 测试 HTML 文案
console.log('\n-- 测试 1: index.html 界面文案与控件属性 --');
const htmlContent = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
assert(htmlContent.includes('降低 AI 反应 50%'), 'HTML 包含「降低 AI 反应 50%」文案');
assert(!htmlContent.includes('AI 模型推理降频（省 CPU）'), '旧文案「AI 模型推理降频（省 CPU）」已替换');

// 2. 测试 main.js 中降频系数
console.log('\n-- 测试 2: main.js 逻辑绑定 1.5 (150ms) --');
const mainContent = fs.readFileSync(path.join(__dirname, '../web/main.js'), 'utf8');
assert(mainContent.includes('elModelLowfreq.checked ? 1.5 : 1'), 'main.js 使用 1.5 (150ms) 降频系数');

// 3. 测试 TransformerModel 在 inferEvery = 1 (100ms) 下的行为
console.log('\n-- 测试 3: inferEvery = 1 (默认 100ms) 每 tick 推理 --');
{
  const sim = new Sim(1);
  const mockModel = new TransformerModel({
    meta: { arch: 'transformer', obs_shape: [14, 13, 15] },
    tensors: {}
  }, true);

  let decideCount = 0;
  mockModel._decide = () => {
    decideCount++;
    return [0, 0];
  };
  mockModel.inferEvery = 1;

  for (let t = 0; t < 6; t++) {
    sim.t = t;
    mockModel.act(sim, 1, () => 0.5);
  }
  assert(decideCount === 6, `100ms 模式下 6 tick 触发 6 次推理 (实际=${decideCount})`);
}

// 4. 测试 TransformerModel 在 inferEvery = 1.5 (150ms) 下的节拍序列与平均耗时
console.log('\n-- 测试 4: inferEvery = 1.5 (150ms) 节拍序列与平均耗时 --');
{
  const sim = new Sim(1);
  const mockModel = new TransformerModel({
    meta: { arch: 'transformer', obs_shape: [14, 13, 15] },
    tensors: {}
  }, true);

  const inferTicks = [];
  mockModel._decide = () => {
    inferTicks.push(sim.t);
    return [0, 0];
  };
  mockModel.inferEvery = 1.5;

  // 运行 12 个 tick (1200ms)
  for (let t = 0; t < 12; t++) {
    sim.t = t;
    mockModel.act(sim, 1, () => 0.5);
  }

  // 预期推理发生的 tick: 0, 2, 3, 5, 6, 8, 9, 11
  const expectedTicks = [0, 2, 3, 5, 6, 8, 9, 11];
  const match = JSON.stringify(inferTicks) === JSON.stringify(expectedTicks);
  assert(match, `12 个 tick 内推理触发序列符合预期 [${expectedTicks.join(',')}]，实际=[${inferTicks.join(',')}]`);

  const avgIntervalMs = (12 * 100) / inferTicks.length;
  assert(avgIntervalMs === 150, `1200ms 内触发 8 次推理，平均推理间隔为 150ms (实际=${avgIntervalMs}ms)`);
}

// 5. 测试 bothAct 双模型批处理在 1.5 下的表现
console.log('\n-- 测试 5: bothAct 双模型批处理在 1.5 下的节拍 --');
{
  const sim = new Sim(1);
  const mockModel = new TransformerModel({
    meta: { arch: 'transformer', obs_shape: [14, 13, 15] },
    tensors: {}
  }, true);

  let bothCount = 0;
  mockModel.forward2 = () => {
    bothCount++;
    return [{ value: 0, move: new Float32Array(5), bomb: new Float32Array(2) },
            { value: 0, move: new Float32Array(5), bomb: new Float32Array(2) }];
  };
  mockModel.inferEvery = 1.5;

  for (let t = 0; t < 12; t++) {
    sim.t = t;
    mockModel.bothAct(sim, () => 0.5);
  }
  assert(bothCount === 8, `bothAct 在 1200ms 内同样恰好推理 8 次 (平均 150ms，实际=${bothCount}次)`);
}

// 6. 测试 MLPModel 基类在 1.5 下的表现
console.log('\n-- 测试 6: MLPModel 基类在 1.5 下的节拍 --');
{
  const sim = new Sim(1);
  const mockModel = new MLPModel({
    meta: { arch: 'mlp', obs_shape: [14, 13, 15] },
    tensors: {}
  }, true);

  let forwardCount = 0;
  mockModel.forward = () => {
    forwardCount++;
    return { move: new Float32Array(5), bomb: new Float32Array(2) };
  };
  mockModel.inferEvery = 1.5;

  for (let t = 0; t < 12; t++) {
    sim.t = t;
    mockModel.act(sim, 1, () => 0.5);
  }
  assert(forwardCount === 8, `MLPModel 基类在 1200ms 内同样触发 8 次推理 (实际=${forwardCount})`);
}

console.log(`\n测试汇总: PASS=${passes}, FAIL=${fails}`);
if (fails > 0) {
  process.exit(1);
} else {
  console.log('150ms 降频验证全部通过！✔');
  process.exit(0);
}
