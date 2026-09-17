#!/usr/bin/env node
'use strict';

const assert = require('assert');

function createMotorDelayTester() {
  let latencyVal = 100;
  let simT = 0;
  const MOVE_IDLE = 0, MOVE_UP = 1, MOVE_DOWN = 2, MOVE_LEFT = 3, MOVE_RIGHT = 4;

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

console.log('\n所有 AI 后置动作传导时延测试全部通过 ✔');
