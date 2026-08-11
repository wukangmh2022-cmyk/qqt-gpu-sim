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
global.addEventListener = () => {};
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
    return { buffer: null, connect() { return this; }, start() {}, gain: { value: 0 } };
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
  if (!qqt || !qqt.sim) {
    console.error('FAIL: window.__QQT__.sim 未就绪');
    console.error('status:', els['status'] && els['status']._html);
    process.exit(1);
  }
  console.log('模型:', qqt.model.meta.name, ' 地图:', qqt.sim.mode);

  // 当前模型信息已展示在 HUD（cur-model 元素有内容、且与加载的模型名一致）
  const curText = els['cur-model'].textContent;
  if (!curText || !curText.includes(qqt.model.meta.name)) {
    console.error(`FAIL: cur-model 未显示当前模型（"${curText}"）`);
    process.exit(1);
  }
  console.log(`当前模型显示: ${curText} ✔`);

  // 模型下拉已有选项（模型列表加载成功）
  if (els['model'].children.length === 0) {
    console.error('FAIL: 模型下拉列表为空');
    process.exit(1);
  }
  console.log(`模型下拉: ${els['model'].children.length} 个候选 ✔`);

  // 点击「应用此模型」重载当前选中模型，不炸
  const click = (id) => (els[id].listeners['click'] || []).forEach((fn) => fn());
  click('apply-model');
  await wait(400);
  if (!qqt.model.meta.name) {
    console.error('FAIL: apply-model 后模型丢失');
    process.exit(1);
  }
  console.log('apply-model 重载正常 ✔');

  // BGM 开关：勾选 → 开启（不炸）；取消勾选 → 停播
  els['bgm'].checked = true;
  (els['bgm'].listeners['change'] || []).forEach((fn) => fn());
  await wait(300);
  els['bgm'].checked = false;
  (els['bgm'].listeners['change'] || []).forEach((fn) => fn());
  console.log('BGM 开关切换正常（on/off）✔');

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

  // 渲染帧无异常：手动多跑几帧（rAF mock 每 16ms 一帧）
  await wait(1000);
  console.log(`渲染帧正常（tick=${qqt.sim.t}，done=${qqt.sim.done}）`);
  if (!Number.isFinite(qqt.sim.pos[0])) {
    console.error('FAIL: 位置非法');
    process.exit(1);
  }

  // loading 层在 boot 完成后隐藏
  if (!els['loading'].classList.contains('hidden')) {
    console.error('FAIL: loading 层未隐藏');
    process.exit(1);
  }
  console.log('loading 层已隐藏 ✔');

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
  els['scene'].value = '矿洞';
  fire('scene');
  els['mode'].value = 'corridor';
  fire('mode');
  await wait(600);
  console.log(`切场景/切模式后正常（tick=${qqt.sim.t}，mode=${qqt.sim.mode}）`);

  console.log('\n页面启动全流程无运行时错误 ✔');
  process.exit(0);
})().catch((e) => {
  console.error('FAIL:', e.stack || e);
  process.exit(1);
});
