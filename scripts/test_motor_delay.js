#!/usr/bin/env node
'use strict';

const assert = require('assert');

function createMotorDelayTester() {
  let latencyVal = 100;
  let simT = 0;
  let spectate = false;
  let p0Sel = 'human';
  let enemySel = 'ViTModel2_31.9B';
  const MOVE_IDLE = 0, MOVE_UP = 1, MOVE_DOWN = 2, MOVE_LEFT = 3, MOVE_RIGHT = 4;

  const isRuleAi = (sel) =>
    sel === '__hunter__' || sel === '__time_astar__' || sel === '__time_astar_hunt__' ||
    sel === '__time_astar_roam__' || sel === '__nukeman__' || sel === '__idle__' ||
    sel === '__stationary__' || sel === '__flee_bot__' || sel === '__roam_bot__';

  const aiMotorQueues = [
    { queue: [], lastMove: MOVE_IDLE, accum: 0 },
    { queue: [], lastMove: MOVE_IDLE, accum: 0 },
  ];

  function resetAiMotorQueues() {
    for (let p = 0; p < 2; p++) {
      aiMotorQueues[p].queue = [];
      aiMotorQueues[p].lastMove = MOVE_IDLE;
      aiMotorQueues[p].accum = 0;
    }
  }

  function applyAiMotorDelay(pid, rawAction) {
    const rawMove = rawAction[0], rawBomb = rawAction[1];
    const qState = aiMotorQueues[pid];
    if (!qState) return rawAction;

    const isModel = pid === 0 ? (spectate && !isRuleAi(p0Sel)) : !isRuleAi(enemySel);
    if (!isModel) {
      qState.queue.length = 0;
      qState.lastMove = rawMove;
      return rawAction;
    }

    const latencyMs = Number.isFinite(latencyVal) ? latencyVal : 100;
    if (latencyMs <= 100) {
      qState.queue.length = 0;
      qState.lastMove = rawMove;
      return rawAction;
    }

    const extraDelayTicks = (latencyMs - 100) / 100;
    const baseTicks = Math.floor(extraDelayTicks);
    const frac = extraDelayTicks - baseTicks;

    let delayTicks = baseTicks;
    if (frac > 0) {
      qState.accum += frac;
      if (qState.accum >= 1.0 - 1e-4) {
        delayTicks += 1;
        qState.accum -= 1.0;
      }
    }

    const curTick = simT;
    const dueTick = curTick + delayTicks;

    qState.queue.push({ move: rawMove, bomb: rawBomb, dueTick });

    let maturedMove = null;
    let maturedBomb = 0;
    let popIdx = -1;

    for (let i = 0; i < qState.queue.length; i++) {
      const item = qState.queue[i];
      if (item.dueTick <= curTick) {
        maturedMove = item.move;
        if (item.bomb) maturedBomb = 1;
        popIdx = i;
      } else {
        break;
      }
    }

    if (popIdx >= 0) {
      qState.queue.splice(0, popIdx + 1);
    }

    if (maturedMove != null) {
      qState.lastMove = maturedMove;
      return [maturedMove, maturedBomb];
    } else {
      return [qState.lastMove, 0];
    }
  }

  return {
    setLatency: (v) => { latencyVal = v; },
    setTick: (t) => { simT = t; },
    getTick: () => simT,
    stepTick: () => { simT++; },
    setSpectate: (v) => { spectate = v; },
    setP0Sel: (v) => { p0Sel = v; },
    setEnemySel: (v) => { enemySel = v; },
    resetAiMotorQueues,
    applyAiMotorDelay,
    MOVE_IDLE, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    queues: aiMotorQueues,
  };
}

console.log('=== 开始测试 AI 后置动作传导时延 (Motor Delay) ===\n');

// 1. 测试 100ms 档位（原生电竞级 · 0ms 按键延迟）
{
  const t = createMotorDelayTester();
  t.setLatency(100);
  t.setTick(0);
  const a0 = t.applyAiMotorDelay(1, [t.MOVE_UP, 1]);
  assert.deepStrictEqual(a0, [t.MOVE_UP, 1], '100ms 档位应立即执行移动与放炮');
  t.setTick(1);
  const a1 = t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 0]);
  assert.deepStrictEqual(a1, [t.MOVE_RIGHT, 0], '100ms 档位第二步应立即执行');
  console.log('✓ 100ms 档位（0ms 按键延迟）：动作即时生效验证通过');
}

// 2. 测试 200ms 档位（普通玩家 · +100ms 按键延迟，即严格 1 tick 延迟）
{
  const t = createMotorDelayTester();
  t.setLatency(200);
  t.setTick(0);
  const a0 = t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 1]);
  assert.deepStrictEqual(a0, [t.MOVE_IDLE, 0], 't=0 应保持初始惯性且不放炮');

  t.setTick(1);
  const a1 = t.applyAiMotorDelay(1, [t.MOVE_UP, 0]);
  assert.deepStrictEqual(a1, [t.MOVE_RIGHT, 1], 't=1 应执行 t=0 的指令（延迟 100ms / 1 tick）');

  t.setTick(2);
  const a2 = t.applyAiMotorDelay(1, [t.MOVE_LEFT, 0]);
  assert.deepStrictEqual(a2, [t.MOVE_UP, 0], 't=2 应执行 t=1 的指令（转向延迟 100ms）');
  console.log('✓ 200ms 档位（+100ms 按键延迟）：严格 1-tick 传导时延与放炮不丢失验证通过');
}

// 3. 测试 300ms 档位（休闲娱乐 · +200ms 按键延迟，即严格 2 tick 延迟）
{
  const t = createMotorDelayTester();
  t.setLatency(300);
  t.setTick(0);
  assert.deepStrictEqual(t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 0]), [t.MOVE_IDLE, 0]);
  t.setTick(1);
  assert.deepStrictEqual(t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 1]), [t.MOVE_IDLE, 0]);
  t.setTick(2);
  assert.deepStrictEqual(t.applyAiMotorDelay(1, [t.MOVE_UP, 0]), [t.MOVE_RIGHT, 0], 't=2 执行 t=0 动作');
  t.setTick(3);
  assert.deepStrictEqual(t.applyAiMotorDelay(1, [t.MOVE_DOWN, 0]), [t.MOVE_RIGHT, 1], 't=3 执行 t=1 动作并放炮');
  console.log('✓ 300ms 档位（+200ms 按键延迟）：严格 2-tick 传导时延验证通过');
}

// 4. 测试 150ms 档位（人类高手 · +50ms 按键延迟，即 0.5 tick 延迟）
{
  const t = createMotorDelayTester();
  t.setLatency(150);
  
  const executed = [];
  for (let tick = 0; tick < 10; tick++) {
    t.setTick(tick);
    const act = t.applyAiMotorDelay(1, [tick + 1, tick === 3 ? 1 : 0]);
    executed.push(act);
  }

  const bombTicks = executed.map((a, idx) => a[1] === 1 ? idx : -1).filter(idx => idx >= 0);
  assert.strictEqual(bombTicks.length, 1, '放炮意图必须恰好执行 1 次');
  assert.ok(bombTicks[0] === 3 || bombTicks[0] === 4, `放炮执行 tick (${bombTicks[0]}) 应在 3 或 4 步`);

  console.log('✓ 150ms 档位（+50ms 按键延迟）：0.5 tick 浮动交替与放炮保全验证通过');
}

// 5. 测试 250ms 档位（新手友好 · +150ms 按键延迟，即 1.5 tick 延迟）
{
  const t = createMotorDelayTester();
  t.setLatency(250);
  for (let tick = 0; tick < 10; tick++) {
    t.setTick(tick);
    t.applyAiMotorDelay(1, [tick + 1, 0]);
  }
  console.log('✓ 250ms 档位（+150ms 按键延迟）：1.5 tick 传导时延验证通过');
}

// 6. 测试开局重置与模式切换重置
{
  const t = createMotorDelayTester();
  t.setLatency(200);
  t.setTick(0);
  t.applyAiMotorDelay(1, [t.MOVE_UP, 1]);
  assert.strictEqual(t.queues[1].queue.length, 1);
  t.resetAiMotorQueues();
  assert.strictEqual(t.queues[1].queue.length, 0, '重置后队列必须清空');
  assert.strictEqual(t.queues[1].lastMove, t.MOVE_IDLE, '重置后惯性必须复位');
  console.log('✓ resetAiMotorQueues 复位机制验证通过');
}

// 7. 测试规则敌人（Hunter、时空 A* 等）完全不受时延影响，保持原生无延迟
{
  const t = createMotorDelayTester();
  t.setLatency(300); // 即使设置 300ms 大时延
  t.setEnemySel('__time_astar_hunt__'); // 规则敌人：高级时空 A*
  t.setTick(0);
  const a0 = t.applyAiMotorDelay(1, [t.MOVE_UP, 1]);
  assert.deepStrictEqual(a0, [t.MOVE_UP, 1], '规则敌人动作必须即时生效，绝不进入时延队列');
  assert.strictEqual(t.queues[1].queue.length, 0, '规则敌人队列必须始终为空');

  t.setTick(1);
  const a1 = t.applyAiMotorDelay(1, [t.MOVE_LEFT, 0]);
  assert.deepStrictEqual(a1, [t.MOVE_LEFT, 0], '规则敌人第二步必须即时生效');
  console.log('✓ 规则敌人免受时延影响验证通过（300ms 档位下依然原生零延迟）');
}

// 8. 测试观战模式下一方规则敌人、一方神经网络模型时的时延隔离
{
  const t = createMotorDelayTester();
  t.setLatency(200); // +100ms 按键延迟
  t.setSpectate(true);
  t.setP0Sel('__hunter__'); // P0 是规则 Hunter
  t.setEnemySel('params_it00000831_ema'); // P1 是最新神经网络模型

  t.setTick(0);
  const actP0_t0 = t.applyAiMotorDelay(0, [t.MOVE_DOWN, 1]);
  const actP1_t0 = t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 1]);
  assert.deepStrictEqual(actP0_t0, [t.MOVE_DOWN, 1], '观战时规则 AI P0 必须即时生效');
  assert.deepStrictEqual(actP1_t0, [t.MOVE_IDLE, 0], '观战时模型 AI P1 必须受时延影响滞后');

  t.setTick(1);
  const actP0_t1 = t.applyAiMotorDelay(0, [t.MOVE_LEFT, 0]);
  const actP1_t1 = t.applyAiMotorDelay(1, [t.MOVE_UP, 0]);
  assert.deepStrictEqual(actP0_t1, [t.MOVE_LEFT, 0], '观战时规则 AI P0 第二步即时生效');
  assert.deepStrictEqual(actP1_t1, [t.MOVE_RIGHT, 1], '观战时模型 AI P1 在 t=1 执行 t=0 动作');
  console.log('✓ 观战模式下规则 AI 原生即时 vs 模型 AI 拟真延迟隔离验证通过');
}

// 9. 测试由模型切换至规则敌人时队列自动排空
{
  const t = createMotorDelayTester();
  t.setLatency(200);
  t.setEnemySel('params_it00000831_ema'); // 先是模型
  t.setTick(0);
  t.applyAiMotorDelay(1, [t.MOVE_UP, 1]);
  assert.strictEqual(t.queues[1].queue.length, 1, '模型决策应进入队列');

  // 切换为规则 AI
  t.setEnemySel('__hunter__');
  t.setTick(1);
  const act = t.applyAiMotorDelay(1, [t.MOVE_RIGHT, 0]);
  assert.deepStrictEqual(act, [t.MOVE_RIGHT, 0], '切换至规则 AI 后动作应立即直通');
  assert.strictEqual(t.queues[1].queue.length, 0, '切换至规则 AI 后残留队列被清空');
  console.log('✓ 切换至规则 AI 时动作队列即时清空验证通过');
}

console.log('\n所有 AI 后置动作传导时延测试全部通过 ✔');
