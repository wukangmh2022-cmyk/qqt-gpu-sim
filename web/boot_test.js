#!/usr/bin/env node
// 无浏览器冒烟：用最小 DOM/Canvas mock 加载 main.js，验证页面启动全流程
// （素材加载 → 模型加载 → 开局 → 逻辑节拍推进）无运行时错误。
//
// 用法： node web/boot_test.js

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

// ------------------------------------------------------------ DOM mock
function makeEl(id) {
  const el = {
    id,
    _html: '',
    _val: undefined,
    checked: false,
    textContent: '',
    children: [],
    listeners: {},
    _classes: new Set(),
    classList: {
      add: (c) => { els[id]._classes.add(c); },
      remove: (c) => { els[id]._classes.delete(c); },
      contains: (c) => els[id]._classes.has(c),
    },
    style: {},
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    appendChild(c) { this.children.push(c); },
    // value：显式赋值优先，否则取第一个 option（模拟默认选中）
    get value() { return this._val !== undefined ? this._val
      : (this.children[0] && this.children[0].value) || ''; },
    set value(v) { this._val = v; },
    set innerHTML(v) { this._html = v; this.children = []; },
    get innerHTML() { return this._html; },
    set width(v) { this._w = v; },
    get width() { return this._w || 780; },
    set height(v) { this._h = v; },
    get height() { return this._h || 876; },
    getContext() { return ctxMock; },
  };
  return el;
}
const els = {};
function byId(id) { return els[id] || (els[id] = makeEl(id)); }

const ctxMock = new Proxy({}, {
  get(t, prop) {
    if (prop === 'getImageData') return () => ({ data: new Uint8ClampedArray(4) });
    if (prop === 'putImageData') return () => {};
    if (prop === 'measureText') return () => ({ width: 0 });
    if (typeof prop === 'string' && prop !== 'then') return () => {};
    return undefined;
  },
  set() { return true; },
});

global.window = global;
const winListeners = {};
global.addEventListener = (ev, fn) => {
  (winListeners[ev] = winListeners[ev] || []).push(fn);
};
global.dispatch = (ev, e) => {
  (winListeners[ev] || []).forEach((fn) => fn({ code: e, preventDefault() {} }));
};
global.removeEventListener = () => {};
global.document = {
  getElementById: byId,
  createElement(tag) {
    if (tag === 'canvas') {
      return { width: 0, height: 0, getContext: () => ctxMock,
               getContext2d: ctxMock };
    }
    return makeEl(tag);
  },
};
global.Image = class {
  set src(_) {
    // 模拟加载成功（异步触发 onload）
    setTimeout(() => this.onload && this.onload(), 0);
  }
};
global.AudioContext = class {
  constructor() { this.destination = {}; }
  createBufferSource() {
    return { buffer: null, connect() { return this; },
             start() { global._acStart = (global._acStart || 0) + 1; },
             stop() { global._acStop = (global._acStop || 0) + 1; } };
  }
  createGain() { return { gain: { value: 0 }, connect() { return this; } }; }
  async decodeAudioData() { return {}; }
};
global.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now() + 16), 16);
global.performance = global.performance || require('perf_hooks').performance;

// fetch mock：模型 JSON 从磁盘读
global.fetch = async (url) => {
  const p = url.replace(/^https?:\/\/[^/]*\//, '').replace(/^\//, '');
  const file = path.join(ROOT, 'web', p);
  if (!fs.existsSync(file)) return { ok: false, status: 404, json: async () => ({}) };
  const buf = fs.readFileSync(file);
  const text = buf.toString('utf8');
  return {
    ok: true,
    status: 200,
    json: async () => JSON.parse(text),
    text: async () => text,
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
};

// ------------------------------------------------------------ 运行
console.log('加载 main.js（DOM mock）…');
// 模拟浏览器里 <script src="sim.js"> 先加载并挂到 window.QQT
const QQT = require('./sim.js');
global.window.QQT = QQT;
require('./main.js');

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  await wait(800);                     // boot()：素材 + 模型加载
  const qqt = global.window.__QQT__;
  if (!qqt || !qqt.model) {
    console.error('FAIL: window.__QQT__.model 未就绪');
    console.error('status:', els['status'] && els['status']._html);
    process.exit(1);
  }

  // 加载完成：loading 隐藏 + 欢迎窗口（操作说明）显示，未开局（sim 未创建）
  if (!els['loading'].classList.contains('hidden')) {
    console.error('FAIL: loading 层未隐藏');
    process.exit(1);
  }
  if (els['banner'].classList.contains('hidden')) {
    console.error('FAIL: 欢迎窗口未显示（banner 应为可见）');
    process.exit(1);
  }
  const wlHtml = els['banner']._html || '';
  if (!wlHtml.includes('开始游戏')) {
    console.error('FAIL: 欢迎窗口缺少操作提示（按空格开始）');
    process.exit(1);
  }
  console.log('loading 已隐藏 + 欢迎窗口显示（含操作说明）✔');

  // 按空格 → 开始第一局
  global.dispatch('keydown', 'Space');
  await wait(300);
  if (!qqt.sim) {
    console.error('FAIL: 按空格后未开局（sim 未创建）');
    process.exit(1);
  }
  console.log('按空格开局成功:', '模型:', qqt.model.meta.name, ' 地图:', qqt.sim.mode);

  // 当前模型信息已展示在 HUD（cur-model 元素有内容、且与加载的模型名一致）
  const curText = els['cur-model'].textContent;
  if (!curText || !curText.includes(qqt.model.meta.name)) {
    console.error(`FAIL: cur-model 未显示当前模型（"${curText}"）`);
    process.exit(1);
  }
  console.log(`当前模型显示: ${curText} ✔`);

  // 敌人 AI 下拉已有选项（模型列表 + 规则 Hunter）
  const aiCount = els['enemy-ai'].children.length;
  const idx = JSON.parse(fs.readFileSync(path.join(ROOT, 'web', 'models', 'index.json'), 'utf8'));
  const expectAi = (idx.models ? idx.models.length : idx.length) + 1;   // 模型数 + 规则 Hunter
  if (aiCount !== expectAi) {
    console.error(`FAIL: 敌人 AI 下拉应 ${expectAi} 个候选（${expectAi - 1} 模型 + 规则 Hunter），实际 ${aiCount}`);
    process.exit(1);
  }
  console.log(`敌人 AI 下拉: ${aiCount} 个候选（${expectAi - 1} 模型 + 规则 Hunter）✔`);

  // 点击「应用」重载当前选中敌人 AI（默认 = 最强模型），不炸
  const click = (id) => (els[id].listeners['click'] || []).forEach((fn) => fn());
  click('apply-model');
  await wait(400);
  if (!qqt.model.meta.name) {
    console.error('FAIL: apply-model 后模型丢失');
    process.exit(1);
  }
  console.log('apply-model 重载正常 ✔');

  // 敌人 AI 切到规则 Hunter → 应用 → P1 由 hunter 决策
  els['enemy-ai'].value = '__hunter__';
  (els['enemy-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(500);
  if (qqt.enemySel !== '__hunter__') {
    console.error('FAIL: 敌人切规则 Hunter 后 enemySel 未更新');
    process.exit(1);
  }
  console.log('敌人 AI 切到规则 Hunter 后正常 ✔');
  // 切回默认最强模型（后续观战测试用）
  const defaultModel = els['enemy-ai'].children[1].value;   // 第 0 项是规则 Hunter
  els['enemy-ai'].value = defaultModel;
  (els['enemy-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(400);

  // BGM 开关：勾选 → 开启（不炸）；取消勾选 → 停播
  els['bgm'].checked = true;
  (els['bgm'].listeners['change'] || []).forEach((fn) => fn());
  await wait(300);
  els['bgm'].checked = false;
  (els['bgm'].listeners['change'] || []).forEach((fn) => fn());
  console.log('BGM 开关切换正常（on/off）✔');

  // 观战勾选 → 「我方：」下拉显示
  els['spectate'].checked = true;
  (els['spectate'].listeners['change'] || []).forEach((fn) => fn());
  if (els['p0-ai-wrap'].style.display !== '') {
    console.error(`FAIL: 观战勾选后 p0-ai-wrap 未显示（display=${els['p0-ai-wrap'].style.display}）`);
    process.exit(1);
  }
  // 观战「我方：」默认策略已初始化（修 p0Sel=null → aiOf 返回 IDLE 站着不动）
  if (qqt.p0Sel === null || qqt.p0Sel !== els['p0-ai'].value) {
    console.error(`FAIL: 观战我方默认策略未初始化（p0Sel=${qqt.p0Sel}）`);
    process.exit(1);
  }
  // 默认模型观战：P0 应能动（有位置变化），不是站着 —— 等几 tick 看 pos
  {
    const y0 = qqt.sim.pos[0], x0 = qqt.sim.pos[1];
    let moved = false;
    for (let k = 0; k < 25; k++) {
      await wait(120);
      const d = Math.abs(qqt.sim.pos[0] - y0) + Math.abs(qqt.sim.pos[1] - x0);
      if (d > 1e-6) { moved = true; break; }
    }
    if (!moved) {
      console.error('FAIL: 观战默认模型 P0 站着不动（策略未生效）');
      process.exit(1);
    }
  }
  console.log('观战：我方默认模型可动（策略已生效）✔');
  // 观战 + 我方也换规则 Hunter（双规则 AI 对局）
  els['p0-ai'].value = '__hunter__';
  (els['p0-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(600);
  if (qqt.p0Sel !== '__hunter__') {
    console.error('FAIL: 观战我方切规则 Hunter 后 p0Sel 未更新');
    process.exit(1);
  }
  console.log('观战：我方替换为规则 Hunter 后正常 ✔');
  // 恢复：我方 = 默认最强模型
  els['p0-ai'].value = defaultModel;
  (els['p0-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(300);

  // CNN 模型（duel_cnn）出现在下拉 + 按 arch 用 CNNModel 加载 + 能决策对局
  const cnnOpt = [...els['enemy-ai'].children].find((o) => o.value === 'duel_cnn');
  if (!cnnOpt) {
    console.error('FAIL: 敌人 AI 下拉缺少 CNN 模型 duel_cnn');
    process.exit(1);
  }
  els['enemy-ai'].value = 'duel_cnn';
  (els['enemy-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(1200);                    // 1.4MB 权重加载 + base64 解码
  const cnnModel = qqt.modelCache.get('duel_cnn');
  if (!cnnModel) {
    console.error('FAIL: duel_cnn 未加载进 modelCache');
    process.exit(1);
  }
  if (!(cnnModel instanceof QQT.CNNModel)) {
    console.error(`FAIL: duel_cnn 应按 CNNModel 加载（实际 ${cnnModel.constructor.name}）`);
    process.exit(1);
  }
  // 无头对局：CNN 当 P0 vs 规则 Hunter（P1），验证 encodeObs→forward→采样全链路
  {
    const sim = new QQT.Sim(7);
    sim.reset('open');
    const hunter = new QQT.HunterAI();
    const rng = QQT.mulberry32(1);
    let steps = 0;
    while (!sim.done && steps < 600) {
      const a0 = cnnModel.act(sim, 0, rng);
      const a1 = hunter.act(sim, 1);
      if (a0[0] < 0 || a0[0] > 4 || a0[1] < 0 || a0[1] > 1) {
        console.error(`FAIL: CNN 输出非法动作 ${a0}`);
        process.exit(1);
      }
      sim.step([a0, a1]);
      steps++;
    }
    console.log(`CNN 模型加载 + 对局完成（${steps}tick，winner=${sim.winner}）✔`);
  }
  // 恢复默认敌人
  els['enemy-ai'].value = defaultModel;
  (els['enemy-ai'].listeners['change'] || []).forEach((fn) => fn());
  await wait(400);

  // 人物皮肤下拉：3 种候选（海王子/小虾米/火影），切换后重开不炸
  if (els['skin'].children.length !== 3) {
    console.error(`FAIL: 皮肤下拉应 3 个候选（实际 ${els['skin'].children.length}）`);
    process.exit(1);
  }
  els['skin'].value = '小虾米';
  (els['skin'].listeners['change'] || []).forEach((fn) => fn());
  await wait(300);
  if (qqt.sim && qqt.res && qqt.res.players !== qqt.res.skins['小虾米']) {
    console.error('FAIL: 切小虾米后 res.players 未更新');
    process.exit(1);
  }
  els['skin'].value = '海王子';
  (els['skin'].listeners['change'] || []).forEach((fn) => fn());
  await wait(300);
  console.log('皮肤下拉 3 候选 + 切换重开正常 ✔');

  // 逻辑节拍推进检查（setInterval 100ms）
  const t0 = qqt.sim.t;
  await wait(1200);
  const t1 = qqt.sim.t;
  await wait(1200);
  const t2 = qqt.sim.t;
  console.log(`tick 推进: ${t0} → ${t1} → ${t2}（应单调增加）`);
  if (!(t1 > t0 && t2 > t1)) {
    console.error('FAIL: 逻辑节拍未推进');
    process.exit(1);
  }

  // 移动端虚拟摇杆：pointer 事件 → P0 按摇杆方向帧级移动
  // 先取消观战（前面观战测试一直勾着，观战模式不采样人类输入）
  els['spectate'].checked = false;
  (els['spectate'].listeners['change'] || []).forEach((fn) => fn());
  await wait(400);
  // mock 里 joystick 无 getBoundingClientRect → main.js 兜底 {0,0,128,128}，
  // 中心 = (64,64)。向右拖 40px（> 死区 14px）→ joyMove = MOVE_RIGHT。
  {
    const joy = els['joystick'];
    const fire = (ev, e) => (joy.listeners[ev] || []).forEach((fn) => fn(e));
    const yBefore = qqt.sim.pos[0], xBefore = qqt.sim.pos[1];
    fire('pointerdown', { pointerId: 1, clientX: 64, clientY: 64, preventDefault() {} });
    fire('pointermove', { pointerId: 1, clientX: 64 + 40, clientY: 64, preventDefault() {} });
    await wait(700);                     // 帧循环采样摇杆方向持续移动
    fire('pointerup', { pointerId: 1 });
    if (!(qqt.sim.pos[1] > xBefore + 0.05)) {
      console.error(`FAIL: 摇杆右推后 P0 未向右移动（x ${xBefore} → ${qqt.sim.pos[1]}）`);
      process.exit(1);
    }
    console.log(`移动端摇杆：右推后 P0 向右移动（${xBefore.toFixed(2)} → ${qqt.sim.pos[1].toFixed(2)}）✔`);
  }

  // 渲染帧无异常：手动多跑几帧（rAF mock 每 16ms 一帧）
  await wait(1000);
  console.log(`渲染帧正常（tick=${qqt.sim.t}，done=${qqt.sim.done}）`);
  if (!Number.isFinite(qqt.sim.pos[0])) {
    console.error('FAIL: 位置非法');
    process.exit(1);
  }

  // 切观战模式，等 AI 放泡爆炸，确认 explosion 掩码被设置（十字爆炸渲染的输入）
  els['spectate'].checked = true;
  (els['spectate'].listeners['change'] || []).forEach((fn) => fn());
  let explosionSeen = false;
  for (let i = 0; i < 60; i++) {
    await wait(300);
    const ex = qqt.explosion;
    if (ex && ex.some((v) => v > 0)) { explosionSeen = true; break; }
  }
  if (!explosionSeen) {
    console.error('FAIL: 18s 内没观察到爆炸（explosion 掩码未设置）');
    process.exit(1);
  }
  console.log(`爆炸特效触发正常（tick=${qqt.sim.t}，covered 格=${qqt.explosion.filter(v => v).length}）✔`);

  // 换场景/换模式/观战 各触发一次 startGame，确认不炸
  const fire = (id) => (els[id].listeners['change'] || []).forEach((fn) => fn());
  // BGM：先确保在播（重新勾选），再切场景 → 旧曲必须被 stop
  els['bgm'].checked = true;
  fire('bgm');
  await wait(400);
  const stopsBefore = global._acStop || 0;
  els['scene'].value = '矿洞';
  fire('scene');
  await wait(500);
  const stopsAfter = global._acStop || 0;
  if (stopsAfter <= stopsBefore) {
    console.error(`FAIL: 切场景后旧 BGM 未停（stop 次数 ${stopsBefore} → ${stopsAfter}）`);
    process.exit(1);
  }
  console.log(`切场景旧 BGM 已停（stop ${stopsBefore} → ${stopsAfter}）✔`);
  els['mode'].value = 'corridor';
  fire('mode');
  await wait(600);
  console.log(`切场景/切模式后正常（tick=${qqt.sim.t}，mode=${qqt.sim.mode}）`);

  // 结算行为：跑完一局后（done），空格不重开、R 重开
  // 用观战 AI 对打快速结束一局
  els['spectate'].checked = true;
  fire('spectate');
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    await wait(1000);
    if (qqt.sim.done) break;
  }
  const doneBefore = qqt.sim.done;
  const seedBefore = qqt.sim.t;
  if (doneBefore) {
    // 结算界面：空格不应重开（running=false、sim 非 null）
    global.dispatch('keydown', 'Space');
    await wait(300);
    if (qqt.sim.t !== seedBefore) {
      console.error('FAIL: 结算界面按空格不应重开');
      process.exit(1);
    }
    console.log('结算界面按空格不重开 ✔');
    // R 重开
    global.dispatch('keydown', 'KeyR');
    await wait(300);
    if (qqt.sim.done) {
      console.error('FAIL: 结算界面按 R 未重开');
      process.exit(1);
    }
    console.log('结算界面按 R 重开 ✔');
  } else {
    console.log('（观战 45s 未结束，跳过结算断言）');
  }

  console.log('\n页面启动全流程无运行时错误 ✔');
  process.exit(0);
})().catch((e) => {
  console.error('FAIL:', e.stack || e);
  process.exit(1);
});
