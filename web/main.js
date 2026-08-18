// main.js —— 浏览器端：原版素材渲染 + 游戏主循环 + 输入 + 模型加载。
//
// 渲染移植自 play/duel.py::draw_grid + play/res.py：同一套 res/ 素材（角色
// 4×4 精灵图、炸弹呼吸、爆炸臂切片、场景砖块/背景、道具图、无敌罩），
// 同一套画家算法（z = 所在行，远→近绘制）与底边对齐锚点。
// 玩法与 play/duel.py 对齐：人类 60Hz 帧级移动（自动转向 + AABB 滑动碰撞），
// AI 决策与模拟推进走 10Hz；AI 用导出权重（已折 pid=0 视角）+ 合法动作掩码采样。
//
// 逻辑节拍用 setInterval(100ms) 驱动（rAF 在标签页后台会被浏览器节流停发，
// 用定时器保证对局不受影响）；rAF 只做输入采样 + 渲染。

'use strict';

(() => {
  const Q = window.QQT;
  const { Sim, MLPModel, CNNModel, CFG, DIRS, MOVE_IDLE, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP } = Q;

  const H = Q.H, W = Q.W, N = Q.N;
  const CELL = 60;                 // 与 play/duel.py 一致：素材原生 40px/格 × 1.5
  const SCALE = CELL / 40.0;
  const BOARD_PX = CELL * W;       // 780
  const HUD_PX = 96;
  const TICK = 1.0 / CFG.tickHz;   // 0.1s
  const DY = [-1, 1, 0, 0], DX = [0, 0, -1, 1];
  const TURN_EPS = 0.4;
  // 动作编码 → 精灵行（行序：下/左/右/上，与 res.py MOVE_TO_SPRITE_ROW 一致）
  const MOVE_TO_SPRITE_ROW = { [MOVE_DOWN]: 0, [MOVE_LEFT]: 1, [MOVE_RIGHT]: 2, [MOVE_UP]: 3 };

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const $ = (id) => document.getElementById(id);
  const elMode = $('mode'), elScene = $('scene'), elSkin = $('skin'),
        elSpectate = $('spectate'), elDanger = $('danger'), elSound = $('sound'),
        elBgm = $('bgm'), elApplyModel = $('apply-model'), elCurModel = $('cur-model'),
        elEnemyAi = $('enemy-ai'), elP0Ai = $('p0-ai'), elP0AiWrap = $('p0-ai-wrap'),
        elRestart = $('restart'), elStatus = $('status'), elBanner = $('banner'),
        elLoading = $('loading'), elLoadingText = $('loading-text'),
        elSaveReplay = $('save-replay'), elSaveWebp = $('save-webp'),
        elSaveVideo = $('save-video'), elRecMsg = $('rec-msg');

  // ------------------------------------------------------------ 状态
  let sim = null, modelList = [], res = null;
  let rng = null;
  let showDanger = true;            // 与启动器一致：危险图红色渐变默认常显
  let soundOn = true;
  let bgmOn = true;
  let running = false;
  let resultShown = false;
  let prevPos = new Float64Array(4), curPos = new Float64Array(4);
  let explosion = null, explosionTrig = null, explosionT = 0;
  let dangerCache = null;           // tick 级危险图缓存（logicTick 每 step 重建）
  let lastTickT = 0;
  let gameSeed = 1;
  // 录像：JSON 结构化重放（seed + 每 tick 动作 + 周期快照）+ WebP 动图（滚动窗口）。
  // 动图管线：滚动窗口按 20fps 存像素级精确的缩采样帧（ImageData）→ 保存时
  // 交给 libwebp WebPAnimEncoder 做帧间差分合成（vendor/webp-anim/）。
  // 差分 vs 全关键帧：贴图细节场景完整关键帧 ~44KB/帧（12s≈10MB），差分帧
  // 只编变化区域 ~1/3 体积。QuickLook/Preview 的 WebP 动画播放慢是 macOS
  // 系统播放器问题（Chrome/Safari 播放 1:1 正常），与帧结构无关。
  const CLIP_WINDOW_MS = 12000, CLIP_FRAME_MS = 1000 / 20, CLIP_SCALE = 0.6;
  let clipFrames = [];          // 滚动帧缓冲 [{ t, img: ImageData }]
  let lastClipCap = 0;
  let gameEndT = 0;             // 终局时刻（performance.now）：终局后冻结画面不进录像
  let clipC = null, clipCtx = null;   // 复用缩采样 canvas（避免每次采集新建的开销）
  // 视频录制：canvas.captureStream + MediaRecorder（VP9/VP8/H.264，浏览器内置
  // 视频编码器）。WebP 动图对高动态画面效率低（等效码率 ~6.9Mbps），视频编码
  // 同样内容 ~1MB/12s。环形缓冲只留最近 ~13s，点保存时合并导出。
  let mediaRec = null, mediaMime = '', mediaChunks = [];
  function startVideoRecorder() {
    try {
      if (!canvas.captureStream || !window.MediaRecorder) return;
      const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8',
        'video/webm', 'video/mp4'].find((t) => MediaRecorder.isTypeSupported(t));
      if (!mime) return;
      const stream = canvas.captureStream(20);
      mediaRec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 800000 });
      mediaMime = mime;
      mediaChunks = [];
      mediaRec.ondataavailable = (e) => {
        if (e.data && e.data.size) mediaChunks.push({ t: performance.now(), blob: e.data });
        const nowT = performance.now();
        while (mediaChunks.length > 2 && nowT - mediaChunks[0].t > CLIP_WINDOW_MS + 1500) {
          mediaChunks.shift();
        }
      };
      mediaRec.start(250);
    } catch (e) {
      mediaRec = null;
      console.warn('视频录制不可用:', e);
    }
  }
  let replay = null;          // { meta, actions: [[m0,b0,m1,b1], ...], snapshots: [...] }
  const face = [MOVE_DOWN, MOVE_DOWN];
  const human = { dirStack: [], latch: new Set(), move: MOVE_IDLE, pendingBomb: false };
  const hunter = new Q.HunterAI();   // 规则 AI（纯进攻寻路），可当敌/我方
  const HUNTER_VAL = '__hunter__';   // 下拉里规则 AI 的 value 哨兵

  // 敌/我方 AI 选择：'__hunter__'（规则）或模型名。模型按需懒加载到缓存。
  // 敌人默认 = 列表第一个（ELO 最高）；观战我方默认 = 同样的最强模型。
  let enemySel = null, p0Sel = null;
  const modelCache = new Map();      // name → MLPModel/CNNModel（懒加载缓存）
  async function ensureModel(name) {
    let m = modelCache.get(name);
    if (m) return m;
    const resp = await fetch(`models/${name}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const doc = await resp.json();
    m = doc.meta.arch === 'cnn' ? new CNNModel(doc) : new MLPModel(doc);
    modelCache.set(name, m);
    return m;
  }

  // 玩家决策来源：'human' | '__hunter__' | 模型名（观战/规则 AI 时用）
  function aiOf(pid) {
    if (pid === 0 && !elSpectate.checked) return [MOVE_IDLE, human.pendingBomb ? 1 : 0];
    const sel = pid === 0 ? p0Sel : enemySel;
    if (sel === HUNTER_VAL) return hunter.act(sim, pid);
    const m = sel ? modelCache.get(sel) : null;
    if (m) return m.act(sim, pid, rng);
    return [MOVE_IDLE, 0];          // 模型还没加载好：先站着
  }

  // ------------------------------------------------------------ 素材加载
  // 进度 = 图片数 + 模型数统一折算百分比，rAF 里缓动（loading 后半段不卡顿）。
  // 模型 JSON 只有 1.8MB（345K 参数 float32 打包），不是 ckpt 的 20MB ——
  // 下载飞快，但旧版进度条只统计图片、模型下载时冻结在 57% 附近，观感像卡住。
  const imgCache = new Map();
  let imgLoaded = 0, imgTotal = 0;
  let modelLoaded = false;
  let progShown = 0;                    // 显示的平滑进度（0~100）
  function updateProgress() {
    const imgPct = imgTotal ? (imgLoaded / imgTotal) * 100 : 0;
    const target = modelLoaded ? 100 : imgPct * 0.9;   // 模型占最后 10%
    progShown += (target - progShown) * 0.25;          // 一阶缓动，避免跳变
    if (Math.abs(target - progShown) < 0.5) progShown = target;  // 收敛停止
    elLoadingText.textContent =
      `正在加载… ${Math.min(99, Math.round(progShown))}%`;
    if (progShown < target) requestAnimationFrame(updateProgress);
  }
  function loadImage(src) {
    if (imgCache.has(src)) return imgCache.get(src);
    imgTotal++;
    const p = new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        imgLoaded++;
        requestAnimationFrame(updateProgress);
        resolve(img);
      };
      img.onerror = () => {
        console.warn('素材缺失（降级占位）:', src);
        imgLoaded++;
        requestAnimationFrame(updateProgress);
        const c = document.createElement('canvas');   // 透明占位，不阻塞启动
        c.width = 40; c.height = 40;
        resolve(c);
      };
      img.src = src;
    });
    imgCache.set(src, p);
    return p;
  }

  function toCanvas(img) {
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    c.getContext('2d').drawImage(img, 0, 0);
    return c;
  }

  function scaleCanvas(src, w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w));
    c.height = Math.max(1, Math.round(h));
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    return c;
  }

  // 无敌罩预乘（res.py::_load_wudi）：透明区清黑 + rgb *= alpha/255，
  // 之后用 'lighter' 加法混合才不会有蓝底/边缘生硬
  function premulAlpha(src) {
    const c = toCanvas(src);
    const g = c.getContext('2d');
    const d = g.getImageData(0, 0, c.width, c.height);
    for (let i = 0; i < d.data.length; i += 4) {
      const a = d.data[i + 3];
      if (a === 0) { d.data[i] = d.data[i + 1] = d.data[i + 2] = 0; }
      else {
        d.data[i] = (d.data[i] * a / 255) | 0;
        d.data[i + 1] = (d.data[i + 1] * a / 255) | 0;
        d.data[i + 2] = (d.data[i + 2] * a / 255) | 0;
      }
    }
    g.putImageData(d, 0, 0);
    return c;
  }

  // 加载全部素材（失败降级：缺图用色块，保证可玩）
  async function loadAssets() {
    const scenes = await (await fetch('assets/scenes.json')).json();
    const sceneNames = Object.keys(scenes);
    elScene.innerHTML = '';
    for (const s of sceneNames) {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      elScene.appendChild(opt);
    }
    elScene.value = '比武';

    // 预加载全部场景素材（bg + 砖块 + 墙），渲染全程同步；并行加载
    const sceneAssets = {};
    await Promise.all(sceneNames.map(async (name) => {
      const sc = scenes[name];
      const bgImg = await loadImage('assets/' + sc.bg);
      // 与 build_static 一致：整体缩放（比例 = CELL/40）后左上角铺一张
      const bg = scaleCanvas(bgImg, Math.round(bgImg.width * SCALE),
                             Math.round(bgImg.height * SCALE));
      const brick = await Promise.all(sc.brick.map(async (rel) => {
        const img = await loadImage('assets/' + rel);
        // 与 res.py::load_one 一致：等比缩放（宽 = cell，高按比例）
        const h1 = Math.max(1, Math.round(img.height * CELL / img.width));
        return scaleCanvas(img, CELL, h1);
      }));
      let wall = null;
      if (sc.wall) {
        // wall 可配单张或多张元件图（scenes.json 约定：除 z*（可炸砖）和底图
        // 外都是不可炸墙元件），渲染时每格确定性随机选一张（同 brick 机制）。
        const rels = Array.isArray(sc.wall) ? sc.wall : [sc.wall];
        wall = await Promise.all(rels.map(async (rel) => {
          const img = await loadImage('assets/' + rel);
          const h1 = Math.max(1, Math.round(img.height * CELL / img.width));
          return scaleCanvas(img, CELL, h1);
        }));
      }
      sceneAssets[name] = { bg, brick, wall };
    }));

    // 角色皮肤：3 种可选行走图 + 敌人固定角色c（res.py::_load_player 移植）
    //   海王子 角色4×4精灵图.png（85px 帧）  小虾米 角色b4×4.png（100px）
    //   火影   角色火影4x4.png（80px）       敌人   角色c4×4.png（100px）
    const SKINS = [
      { key: '海王子', file: '角色4×4精灵图.png', scale: 1.0 },
      { key: '小虾米', file: '角色b4×4.png', scale: 85 / 100 },
      { key: '火影', file: '角色火影4x4.png', scale: 85 / 80 },
    ];
    const ENEMY_SKIN = { file: '角色c4×4.png', scale: 85 / 100 };
    async function loadSkinRows(file, scale) {
      const sheet = await loadImage('assets/' + file);
      const fw = sheet.width / 4, fh = sheet.height / 4;
      const target = Math.max(1, Math.round(fw * SCALE * scale));
      const rows = [];                           // 4 行方向 × 4 帧
      for (let r = 0; r < 4; r++) {
        rows.push([]);
        for (let c = 0; c < 4; c++) {
          const fr = document.createElement('canvas');
          fr.width = target; fr.height = target;
          fr.getContext('2d').drawImage(sheet, c * fw, r * fh, fw, fh, 0, 0, target, target);
          // 视觉主体顶部：帧顶常有透明留白（人物视觉头顶不在帧顶边）。
          // 头顶指示箭头要对齐它而不是帧顶 —— 每帧检测第一个非透明像素行。
          fr._top = 0;
          try {
            const g = fr.getContext('2d');
            const d = g.getImageData(0, 0, target, target).data;
            for (let y = 0; y < target; y++) {
              let hit = false;
              for (let x = 0; x < target; x++) {
                if (d[(y * target + x) * 4 + 3] > 8) { hit = true; break; }
              }
              if (hit) { fr._top = y; break; }
            }
          } catch (e) { fr._top = 0; }
          rows[r].push(fr);
        }
      }
      return rows;
    }
    const skinRows = {};
    for (const sk of SKINS) skinRows[sk.key] = await loadSkinRows(sk.file, sk.scale);
    const enemyRows = await loadSkinRows(ENEMY_SKIN.file, ENEMY_SKIN.scale);
    elSkin.innerHTML = '';
    for (const sk of SKINS) {
      const opt = document.createElement('option');
      opt.value = sk.key; opt.textContent = sk.key;
      elSkin.appendChild(opt);
    }
    elSkin.value = '海王子';
    const wudi = premulAlpha(await loadImage('assets/无敌.PNG'));
    const boomImg = await loadImage('assets/bomb1.png');
    const props = [];
    for (const name of ['威力道具.png', '泡泡数量道具.png', '鞋子道具.png']) {
      props.push(scaleCanvas(await loadImage('assets/' + name), CELL, CELL));
    }
    const exploArms = {};
    for (const [key, f] of [['up', '向上爆炸.png'], ['down', '向下爆炸.png'],
                            ['left', '向左爆炸.png'], ['right', '向右爆炸.png']]) {
      exploArms[key] = await loadImage('assets/' + f);
    }
    res = {
      scenes, sceneAssets,
      skins: skinRows,             // 3 种玩家皮肤
      players: skinRows[elSkin.value],   // 当前玩家皮肤（切换后重绑）
      enemyRows,                   // 敌人固定角色c（不再染红）
      playerAi: enemyRows,
      wudi: scaleCanvas(wudi, Math.round(85 * SCALE), Math.round(85 * SCALE)),
      bomb: scaleCanvas(boomImg, CELL, CELL),
      props,
      point: scaleCanvas(await loadImage('assets/point.png'),
                         Math.round(40 * SCALE * 0.5), Math.round(40 * SCALE * 0.5)),
      exploCenter: scaleCanvas(await loadImage('assets/爆炸中心.png'), CELL, CELL),
      exploArms,
    };
    // 音效（Web Audio；失败静默）
    try {
      res.audio = new AudioContext();
      res.snd = {};
      const names = { place: '放炮.wav', boom: '爆炸.wav', pickup: '吃道具音效.wav',
                      hurt: '生命损失音效.wav', die: '角色消失音效.wav' };
      for (const [k, f] of Object.entries(names)) {
        const buf = await (await fetch('assets/snd/' + f)).arrayBuffer();
        res.snd[k] = await res.audio.decodeAudioData(buf);
      }
    } catch (e) { res.snd = {}; }
  }

  function sceneOf() {
    return res.sceneAssets[elScene.value] || res.sceneAssets['比武'];
  }

  // ------------------------------------------------------------ 音效
  function playSnd(name, vol) {
    if (!soundOn || !res || !res.snd || !res.snd[name] || !res.audio) return;
    try {
      const src = res.audio.createBufferSource();
      src.buffer = res.snd[name];
      const g = res.audio.createGain();
      g.gain.value = vol == null ? 0.6 : vol;
      src.connect(g).connect(res.audio.destination);
      src.start();
    } catch (e) { /* 忽略 */ }
  }

  // ------------------------------------------------------------ 背景音乐
  // 场景 BGM（ogg）懒加载 + 循环播放；随场景切换换曲；浏览器自动播放策略
  // 要求用户先有交互，首次按键/点击时 resume AudioContext 并开始播放。
  // bgmGen 是"代际号"：每次启动/停止递增，异步加载完成后校验自己是否还是
  // 最新一代 —— 快速连切场景时旧请求作废，绝不出现"旧曲加载完又把新场景
  // 的 BGM 顶掉/两首叠播"的竞态。
  const bgmBuffers = new Map();
  let bgmSrc = null, bgmGain = null, bgmUrl = null, bgmGen = 0;

  async function startBgm() {
    if (!res || !res.audio || !bgmOn) return;
    const sc = res.scenes[elScene.value];
    if (!sc || !sc.bgm) { stopBgm(); return; }   // 新场景无 BGM：旧曲必须停
    const url = 'assets/' + sc.bgm;
    if (url === bgmUrl && bgmSrc) return;        // 同一首已在播
    const gen = ++bgmGen;
    if (bgmSrc) { try { bgmSrc.stop(); } catch (e) { /* */ } }
    bgmSrc = null; bgmUrl = null;                // 先停旧曲，再加载新曲
    try {
      if (res.audio.state === 'suspended') await res.audio.resume();
      let buf = bgmBuffers.get(url);
      if (!buf) {
        const ab = await (await fetch(url)).arrayBuffer();
        if (gen !== bgmGen) return;              // 已被更新的切换作废
        buf = await res.audio.decodeAudioData(ab);
        if (gen !== bgmGen) return;
        bgmBuffers.set(url, buf);
      }
      if (gen !== bgmGen) return;
      const src = res.audio.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      if (!bgmGain) {
        bgmGain = res.audio.createGain();
        bgmGain.gain.value = 0.22;             // 背景音量压低，不盖音效
        bgmGain.connect(res.audio.destination);
      }
      src.connect(bgmGain);
      src.start();
      bgmSrc = src;
      bgmUrl = url;
    } catch (e) { /* 解码/网络失败静默 */ }
  }

  function stopBgm() {
    bgmGen++;                                   // 作废任何进行中的加载
    if (bgmSrc) { try { bgmSrc.stop(); } catch (e) { /* */ } }
    bgmSrc = null;
    bgmUrl = null;
  }

  // 首次用户交互：解锁 AudioContext + 开 BGM（自动播放策略）
  let audioUnlocked = false;
  function unlockAudio() {
    if (audioUnlocked) return;
    audioUnlocked = true;
    if (res && res.audio && res.audio.state === 'suspended') res.audio.resume().catch(() => {});
    startBgm();
  }
  window.addEventListener('pointerdown', unlockAudio);
  window.addEventListener('keydown', unlockAudio);

  // ------------------------------------------------------------ 输入
  const KEY_TO_MV = {
    ArrowUp: 0, KeyW: 0, ArrowDown: 1, KeyS: 1,
    ArrowLeft: 2, KeyA: 2, ArrowRight: 3, KeyD: 3,
  };
  const MV_KEYS = [[], [], [], []];
  for (const k of Object.keys(KEY_TO_MV)) MV_KEYS[KEY_TO_MV[k]].push(k);
  const held = new Set();

  // ---- 移动端触屏：虚拟摇杆 + 放泡键（joyMove 优先于键盘） ----
  // 摇杆位移 → 主方向（4 向 + 死区）；释放回 IDLE。放泡键按下即 pendingBomb。
  let joyMove = null;
  const elJoy = document.getElementById('joystick');
  const elKnob = document.getElementById('joystick-knob');
  const elBombBtn = document.getElementById('bomb-btn');
  if (elJoy && elKnob) {
    const JOY_R = 54;    // 摇杆可拖动半径（≈ 底座 128px 的一半）
    const joyRect = () => (elJoy.getBoundingClientRect
      ? elJoy.getBoundingClientRect()
      : { left: 0, top: 0, width: 128, height: 128 });   // mock 兜底
    let joyActive = false;
    const moveKnob = (dx, dy) => {
      const d = Math.hypot(dx, dy) || 1;
      const c = Math.min(d, JOY_R);
      elKnob.style.transform = `translate(${dx / d * c}px, ${dy / d * c}px)`;
    };
    const dirFrom = (dx, dy) => {
      if (Math.hypot(dx, dy) < 14) return null;          // 死区
      return Math.abs(dx) > Math.abs(dy)
        ? (dx > 0 ? MOVE_RIGHT : MOVE_LEFT)
        : (dy > 0 ? MOVE_DOWN : MOVE_UP);
    };
    const joyMoveEv = (e) => {
      if (!joyActive) return;
      const r = joyRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      moveKnob(dx, dy);
      joyMove = dirFrom(dx, dy);
    };
    const joyDown = (e) => {
      joyActive = true;
      if (elJoy.setPointerCapture) elJoy.setPointerCapture(e.pointerId);
      joyMove = null;
      joyMoveEv(e);
    };
    const joyUp = () => {
      joyActive = false;
      joyMove = null;
      elKnob.style.transform = 'translate(0px, 0px)';
    };
    elJoy.addEventListener('pointerdown', joyDown);
    elJoy.addEventListener('pointermove', joyMoveEv);
    elJoy.addEventListener('pointerup', joyUp);
    elJoy.addEventListener('pointercancel', joyUp);
  }
  if (elBombBtn) {
    elBombBtn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      human.pendingBomb = true;
    });
  }

  window.addEventListener('keydown', (e) => {
    held.add(e.code);
    if (e.code === 'Space') {
      e.preventDefault();
      // 仅欢迎窗口（还没开局）用空格开始；结算界面用 R 重开，空格不触发
      if (!running && sim === null) { startGame(); return; }
      human.pendingBomb = true;
    }
    if (e.code === 'KeyR') { startGame(); }
    if (e.code === 'KeyD') { showDanger = !showDanger; elDanger.checked = showDanger; }
    if (e.code === 'KeyM') { soundOn = !soundOn; elSound.checked = soundOn; }
    const mv = KEY_TO_MV[e.code];
    if (mv !== undefined) {
      human.latch.add(mv);
      if (!human.dirStack.includes(mv)) human.dirStack.push(mv);
      e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => {
    held.delete(e.code);
    const mv = KEY_TO_MV[e.code];
    if (mv !== undefined) {
      const i = human.dirStack.indexOf(mv);
      if (i >= 0) human.dirStack.splice(i, 1);
    }
  });

  function sampleHumanMove() {
    if (joyMove !== null) return joyMove;   // 移动端摇杆优先（覆盖键盘方向）
    const active = new Set(human.latch);
    for (const mv of human.dirStack) {
      if (MV_KEYS[mv].some((k) => held.has(k))) active.add(mv);
    }
    human.dirStack = human.dirStack.filter((mv) => active.has(mv));
    for (const mv of active) if (!human.dirStack.includes(mv)) human.dirStack.push(mv);
    human.latch.clear();
    return human.dirStack.length ? human.dirStack[human.dirStack.length - 1] : MOVE_IDLE;
  }

  // ------------------------------------------------------------ 帧级移动（duel.py 同款）
  function blockedGrid() {
    const b = new Uint8Array(N);
    for (let i = 0; i < N; i++) b[i] = sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 ? 1 : 0;
    return b;
  }

  function probeMove(pid, mv) {
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const blocked = blockedGrid();
    const dist = CFG.stepLen;
    const [dy, dx] = DIRS[mv];
    if (dy !== 0) {
      const ny = Q.resolveAxis(y + dy * dist, dy * dist, x, y, x, blocked, CFG.radius, H, W, true);
      return Math.abs(ny - y) > Q.EPS * 2;
    }
    const nx = Q.resolveAxis(x + dx * dist, dx * dist, y, y, x, blocked, CFG.radius, H, W, false);
    return Math.abs(nx - x) > Q.EPS * 2;
  }

  function autoTurn(pid, move) {
    if (move >= 4 || probeMove(pid, move)) return move;
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const fx = x - Math.floor(x), fy = y - Math.floor(y);
    let alt = null;
    if (move === 0 || move === 1) alt = fx <= TURN_EPS ? 2 : (fx >= 1 - TURN_EPS ? 3 : null);
    else alt = fy <= TURN_EPS ? 0 : (fy >= 1 - TURN_EPS ? 1 : null);
    if (alt === null) return move;
    const r = Math.floor(y), c = Math.floor(x);
    const nr = r + DY[move] + DY[alt], nc = c + DX[move] + DX[alt];
    if (nr >= 0 && nr < H && nc >= 0 && nc < W &&
        !sim.wall[nr * W + nc] && !sim.brick[nr * W + nc] && sim.fuse[nr * W + nc] <= 0) {
      return alt;
    }
    return move;
  }

  function frameMove(pid, mv, dt) {
    if (mv === MOVE_IDLE || !sim.alive[pid]) return;
    const dist = CFG.speed * sim.spdG[pid] * Math.min(dt, 0.1);
    if (dist <= 0) return;
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const blocked = blockedGrid();
    const [dy, dx] = DIRS[mv];
    if (dy !== 0) {
      sim.pos[pid * 2] = Q.resolveAxis(y + dy * dist, dy * dist, x, y, x,
                                       blocked, CFG.radius, H, W, true);
    }
    if (dx !== 0) {
      sim.pos[pid * 2 + 1] = Q.resolveAxis(x + dx * dist, dx * dist, y, y, x,
                                           blocked, CFG.radius, H, W, false);
    }
    sim.pos[pid * 2] = Math.min(Math.max(sim.pos[pid * 2], CFG.radius), H - CFG.radius);
    sim.pos[pid * 2 + 1] = Math.min(Math.max(sim.pos[pid * 2 + 1], CFG.radius), W - CFG.radius);
  }

  // ------------------------------------------------------------ 开局
  function startGame() {
    if (!res) return;                // 模型未就绪由 logicTick 兜底等待
    gameSeed = (Math.random() * 0xFFFFFFFF) >>> 0;
    sim = new Sim(gameSeed);
    sim.reset(elMode.value === 'corridor' ? 'corridor' : 'open');
    rng = Q.mulberry32(gameSeed ^ 0x13579BDF);
    human.dirStack = []; human.latch.clear(); human.move = MOVE_IDLE; human.pendingBomb = false;
    joyMove = null;                    // 摇杆归位（移动端）
    explosion = null; explosionTrig = null; resultShown = false;
    dangerCache = null;             // 开局清掉旧危险图缓存
    replay = {
      meta: {
        mode: elMode.value === 'corridor' ? 'corridor' : 'open',
        seed: gameSeed,
        scene: elScene.value,
        skin: elSkin.value,
        spectate: elSpectate.checked,
        p0: elSpectate.checked ? p0Sel : 'human',
        p1: enemySel,
        cfg: Object.assign({}, CFG),   // CFG 快照：重放/分析时不依赖当前版本常量
      },
      actions: [],
      snapshots: [],
    };
    clipFrames = [];
    lastClipCap = 0;
    gameEndT = 0;
    prevPos.set(sim.pos); curPos.set(sim.pos);
    face[0] = MOVE_DOWN; face[1] = MOVE_DOWN;
    lastTickT = performance.now();
    running = true;
    elBanner.classList.add('hidden');
    startBgm();               // 换场景后切 BGM（同曲则跳过）
  }

  // ------------------------------------------------------------ 模型加载
  function fmtStep(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M';
    return String(n);
  }

  function fillAiSelect(sel, includeHunter) {
    sel.innerHTML = '';
    if (includeHunter) {
      const h = document.createElement('option');
      h.value = HUNTER_VAL;
      h.textContent = '规则 Hunter（纯进攻寻路）';
      sel.appendChild(h);
    }
    for (const m of modelList) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name}  · ${fmtStep(m.global_step)}步 · elo ${m.elo} · 导出于 ${(m.generated_at || '').slice(0, 10)}`;
      sel.appendChild(opt);
    }
  }

  async function loadModelList() {
    const resp = await fetch('models/index.json');
    modelList = (await resp.json()).models;
    // 敌人 AI 下拉 + 观战「我方：」下拉：都列全部模型 + 规则 Hunter
    fillAiSelect(elEnemyAi, true);
    fillAiSelect(elP0Ai, true);
    elEnemyAi.value = modelList[0].name;     // 默认：ELO 最高的模型
    elP0Ai.value = modelList[0].name;
    // 观战「我方：」默认策略**同步初始化**（之前只在下拉 change 时赋值，
    // 直接勾观战 → p0Sel=null → aiOf(0) 返回 IDLE → 我方站着不动）
    p0Sel = elP0Ai.value;
    await applyModel();            // 预加载默认敌人模型（我方默认同款，已入缓存）
  }

  // 应用选中的 AI（敌人）：模型名 → 加载权重；规则 Hunter → 无需权重
  async function applyModel() {
    const sel = elEnemyAi.value;
    if (!sel) return;
    if (sel === HUNTER_VAL) {
      enemySel = HUNTER_VAL;
      elCurModel.textContent = '规则 Hunter（纯进攻寻路）';
      elStatus.innerHTML = '敌人：<b>规则 Hunter</b>（纯进攻寻路 AI，无需模型权重）';
      return;
    }
    elStatus.innerHTML = `正在加载模型 <b>${sel}</b>…`;
    try {
      const m = await ensureModel(sel);
      enemySel = sel;
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent =
        `${m.meta.name}（${fmtStep(m.meta.global_step)}步 · elo ${m.meta.elo} · 导出于 ${(m.meta.generated_at || '').slice(0, 10)}）`;
      elStatus.innerHTML =
        `当前模型：<b>${m.meta.name}</b><br>` +
        `训练步数 ${fmtStep(m.meta.global_step)} · elo ${m.meta.elo}<br>` +
        `观测 ${m.meta.obs_shape.join('×')} · 参数 ${Object.values(m.tensors)
          .reduce((s, [, n]) => s + n, 0).toLocaleString()}`;
    } catch (e) {
      elStatus.innerHTML = `模型加载失败：${e.message}`;
    }
  }

  elRestart.addEventListener('click', startGame);
  elBanner.addEventListener('click', () => { if (!running) startGame(); });  // 欢迎窗口点击开始
  elMode.addEventListener('change', startGame);
  elSkin.addEventListener('change', () => {
    if (res && res.skins) res.players = res.skins[elSkin.value];   // 换皮肤
    startGame();
  });
  elApplyModel.addEventListener('click', applyModel);
  elSpectate.addEventListener('change', () => {
    // 勾选观战时显示「我方：」下拉（模型 / 规则）
    elP0AiWrap.style.display = elSpectate.checked ? '' : 'none';
    startGame();
  });
  elScene.addEventListener('change', () => { stopBgm(); startGame(); });  // 先停旧曲，startGame 内再播新曲
  elEnemyAi.addEventListener('change', applyModel);   // 换敌人 AI → 应用并重开
  elP0Ai.addEventListener('change', async () => {
    // 观战「我方：」：规则 → hunter；模型 → 懒加载进缓存（不阻塞开局）
    p0Sel = elP0Ai.value;
    if (p0Sel !== HUNTER_VAL) {
      try { await ensureModel(p0Sel); } catch (e) { elStatus.innerHTML = `我方模型加载失败：${e.message}`; }
    }
    startGame();
  });
  elDanger.addEventListener('change', () => { showDanger = elDanger.checked; });
  elSound.addEventListener('change', () => { soundOn = elSound.checked; });
  elBgm.addEventListener('change', () => {
    bgmOn = elBgm.checked;
    if (bgmOn) startBgm(); else stopBgm();
  });

  // ------------------------------------------------------------ 录像保存
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function timeStamp() {
    const d = new Date(), p = (x) => String(x).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }
  function recMsg(html) {
    elRecMsg.innerHTML = html;
    clearTimeout(recMsg._t);
    recMsg._t = setTimeout(() => { elRecMsg.innerHTML = ''; }, 10000);
  }

  elSaveReplay.addEventListener('click', () => {
    if (!replay || !replay.actions.length) { recMsg('还没有本局录像 —— 先开始一局再保存'); return; }
    const doc = {
      format: 'qqt-replay',
      version: 1,
      meta: Object.assign({}, replay.meta, {
        savedAt: new Date().toISOString(),
        done: sim ? sim.done : false,
        result: sim && sim.done ? sim.winner : null,   // 保存时刻的终局（0/1/null）
        finalT: sim ? sim.t : 0,
      }),
      ticks: replay.actions.length,
      actions: replay.actions,
      snapshots: replay.snapshots,
    };
    const blob = new Blob([JSON.stringify(doc, null, 1)], { type: 'application/json' });
    downloadBlob(blob, `replay_${doc.meta.mode}_s${doc.meta.seed}_${timeStamp()}.json`);
    recMsg(`录像已保存：${doc.ticks} tick，${(blob.size / 1024).toFixed(0)}KB`);
  });

  elSaveWebp.addEventListener('click', () => {
    if (!replay || !replay.actions.length) { recMsg('还没有可录的画面 —— 先开始一局再保存'); return; }
    const now = performance.now();
    // 终局后保存：只保留游戏结束前的帧（冻结结果画面不进屋，避免"结尾静止=慢放"）
    const frames = clipFrames.filter((f) =>
        now - f.t <= CLIP_WINDOW_MS + 250 && (!gameEndT || f.t <= gameEndT))
      .sort((a, b) => a.t - b.t);
    if (frames.length < 2) { recMsg('画面不足，等几秒再点'); return; }
    recMsg(`WebP 编码中…（${frames.length} 帧，帧间差分压缩）`);
    setTimeout(async () => {
      try {
        const anim = window.WebPAnim;
        if (!anim) throw new Error('动画编码器未加载（vendor/webp-anim/）');
        await Promise.race([
          anim.ready,
          new Promise((_, rej) => setTimeout(() => rej(new Error('WebP wasm 加载超时')), 15000)),
        ]);
        // 帧 → libwebp WebPAnimEncoder 帧间差分合成：静止背景增量帧极小。
        // 全关键帧会因贴图细节每帧 ~44KB（12s≈10MB），差分只有 ~1/3。
        // QuickLook/Preview 的 WebP 动画播放是系统播放器问题（Chrome/Safari
        // 播放正常），与帧结构无关，差分方案不影响浏览器播放。
        // 每帧时长用真实采集间隔（保证播放 = 直播 1:1），首帧用 20fps 标称值。
        const f0 = frames[0].img;
        let prevT = null;
        const rgbaFrames = frames.map((f) => {
          const d = prevT === null ? Math.round(CLIP_FRAME_MS)
            : Math.max(20, Math.min(500, Math.round(f.t - prevT)));
          prevT = f.t;
          return { data: f.img.data, duration: d, config: { quality: 85, lossless: 0 } };
        });
        const webp = await anim.encodeAnimation(f0.width, f0.height, true, rgbaFrames);
        if (!webp) throw new Error('编码失败（wasm 返回 null）');
        const blob = new Blob([webp], { type: 'image/webp' });
        downloadBlob(blob, `clip_${replay.meta.mode}_s${replay.meta.seed}_${timeStamp()}.webp`);
        recMsg(`WebP 已保存：${frames.length} 帧，${(blob.size / 1024).toFixed(0)}KB（差分压缩）`);
      } catch (e) {
        recMsg(`WebP 编码失败：${e.message}`);
        console.error(e);
      }
    }, 30);
  });

  elSaveVideo.addEventListener('click', async () => {
    if (!mediaRec || mediaRec.state !== 'recording') {
      recMsg('视频录制不可用（浏览器不支持 MediaRecorder）');
      return;
    }
    if (!replay || !replay.actions.length) { recMsg('还没有可录的画面 —— 先开始一局再保存'); return; }
    recMsg('视频导出中…');
    try {
      mediaRec.requestData();                      // 冲刷未落盘的数据
      await new Promise((r) => setTimeout(r, 80));
      const now = performance.now();
      // 与 WebP 一致：只取最近 12s + 终局前（冻结结算画面不入片）
      const chunks = mediaChunks.filter((c) =>
        now - c.t <= CLIP_WINDOW_MS + 250 && (!gameEndT || c.t <= gameEndT));
      const bytes = chunks.reduce((s, c) => s + c.blob.size, 0);
      if (!chunks.length || bytes < 8192) { recMsg('视频数据不足，等几秒再点'); return; }
      const blob = new Blob(chunks.map((c) => c.blob), { type: mediaMime });
      const ext = mediaMime.indexOf('mp4') >= 0 ? 'mp4' : 'webm';
      downloadBlob(blob, `clip_${replay.meta.mode}_s${replay.meta.seed}_${timeStamp()}.${ext}`);
      recMsg(`视频已保存：${(blob.size / 1024).toFixed(0)}KB（${ext}，最近 ${(CLIP_WINDOW_MS / 1000).toFixed(0)}s）`);
    } catch (e) {
      recMsg(`视频导出失败：${e.message}`);
      console.error(e);
    }
  });

  // ------------------------------------------------------------ 10Hz 逻辑节拍
  function logicTick() {
    if (!running || !sim || sim.done) return;
    // 模型未就绪（正在加载/加载失败）先不推进：敌人 + 观战时的我方；
    // 规则 AI 随时可用。缺这一步观战 P0 模型没进缓存 → aiOf 返回 IDLE 站着。
    const spectate = elSpectate.checked;
    if (enemySel !== HUNTER_VAL && !modelCache.has(enemySel)) return;
    if (spectate && p0Sel !== HUNTER_VAL && !modelCache.has(p0Sel)) return;
    const a0 = aiOf(0);
    if (!spectate) human.pendingBomb = false;
    const a1 = aiOf(1);
    // 拾取判定：人类玩家脚下 step 前有宝箱 → step 后没有 = 吃到
    const hc = Math.floor(sim.pos[1]), hr = Math.floor(sim.pos[0]);
    const hadCrate = !spectate && sim.alive[0] && sim.crate[hr * W + hc] === 1;
    // 录像：记录本 tick 实际喂给 step 的动作 + 每 20 tick 一个状态快照
    if (replay) {
      replay.actions.push([a0[0], a0[1], a1[0], a1[1]]);
      if (replay.actions.length % 20 === 1) {
        const bombs = [];
        for (let i = 0; i < N; i++) {
          if (sim.fuse[i] > 0) bombs.push([i, sim.fuse[i], sim.owner[i], sim.bombBlast[i]]);
        }
        replay.snapshots.push({ t: sim.t, pos: Array.from(sim.pos), hp: sim.hp.slice(),
                                alive: sim.alive.slice(), bombs });
      }
    }
    prevPos.set(sim.pos);
    const info = sim.step([a0, a1]);
    curPos.set(sim.pos);
    lastTickT = performance.now();
    // danger 缓存：tick 级重建（10Hz），渲染帧直接复用 —— 60fps 每帧重算
    // dangerMap（不动点传播 O(炸弹×blast)）是 AI 对打帧率低的主因
    dangerCache = sim.dangerMap();
    // 朝向：人类玩家（非观战）的朝向由 60Hz 帧级移动维护，10Hz tick 不覆盖
    //（否则每 tick 把 face[0] 重置成 IDLE → 渲染回退朝下，按左/右后总朝下）
    if (spectate) face[0] = a0[0];
    face[1] = a1[0];
    // 音效（以人类玩家为监听者，只播人类相关事件）
    if (info.placed[0]) playSnd('place');
    if (hadCrate && !sim.crate[hr * W + hc]) playSnd('pickup');
    // 只有真的有火焰（任一格被覆盖）才播爆炸音效/显示爆炸特效
    const hasBlast = info.covered && info.covered.some((v) => v > 0);
    if (hasBlast) {
      explosion = info.covered;
      explosionTrig = info.triggered;      // 引爆源格（step 内已清场，必须用返回掩码）
      explosionT = performance.now();
      playSnd('boom');
    }
    if (info.died[0]) playSnd('die');
    if (sim.done && !resultShown) {
      resultShown = true;
      gameEndT = performance.now();   // 冻结画面从此刻起不进动图窗口
      const w = sim.winner;
      const msg = w === null ? '平局' : (w === 0 ? '🎉 你赢了！' : '🤖 敌人赢了');
      elBanner.innerHTML = `${msg}<span class="tip">按 R 或点「重新开局」再来一局</span>`;
      elBanner.classList.remove('hidden');
      running = false;
    }
  }
  setInterval(logicTick, TICK * 1000);

  // ------------------------------------------------------------ 渲染（draw_grid 移植）
  function drawBoard(bg) {
    // 背景层（build_static 的 JS 版）：缩放后从左上角铺一张
    if (bg) ctx.drawImage(bg, 0, 0);
    else { ctx.fillStyle = '#2d2a32'; ctx.fillRect(0, 0, BOARD_PX, BOARD_PX); }
  }

  function drawDangerOverlay() {
    if (!showDanger) return;
    const d = dangerCache;          // 10Hz 缓存；开局首帧尚未生成时按需算一次
    if (!d) return;
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const v = d[r * W + c];
        if (v <= 0.04) continue;
        const a = Math.min(255, Math.round(20 + 235 * v));
        ctx.fillStyle = `rgba(255,30,60,${(a / 255).toFixed(3)})`;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
      }
    }
  }

  // 渲染一帧：背景 → 危险区 → 画家算法精灵（墙砖/爆炸/泡/宝箱/角色）→ 无敌罩 → 血条 → HUD
  function render(now) {
    if (!sim || !res) return;
    const sc = sceneOf();
    const alpha = Math.min(1, (now - lastTickT) / (TICK * 1000));
    drawBoard(sc.bg);
    drawDangerOverlay();

    const items = [];   // (z, drawFn)

    // 墙/砖 tile：底边对齐格底（向上延伸超一格，靠 z 排序盖住上方角色脚部）
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const i = r * W + c;
        if (sim.wall[i] && sc.wall && sc.wall.length) {
          // 多张墙元件图：按格确定性随机（同 brick 的 (r*7+c*13) 哈希）
          const t = sc.wall[(r * 7 + c * 13) % sc.wall.length];
          const tw = t.width, th = t.height;
          items.push([r, () => ctx.drawImage(
            t, c * CELL + (CELL - tw) / 2, r * CELL + CELL - th)]);
        } else if (sim.brick[i] && sc.brick.length) {
          const t = sc.brick[(r * 7 + c * 13) % sc.brick.length];
          const tw = t.width, th = t.height;
          items.push([r, () => ctx.drawImage(
            t, c * CELL + (CELL - tw) / 2, r * CELL + CELL - th)]);
        }
      }
    }

    const nowS = now / 1000;
    const bob = Math.round(Math.sin(nowS * 2 * Math.PI) * 3);
    const bombW = res.bomb.width, bombH = res.bomb.height;

    // 爆炸：中心格用中心图；臂图按实际爆炸格数从炸弹边缘端切片（duel.py 同款算法）
    if (explosion) {
      const age = (now - explosionT) / 1000;
      if (age <= 0.6 && explosionTrig) {
        const blast = explosion;
        const maxBlast = 7;   // 成长上限，与 duel.py 的 res_blast 一致（含 open 关）
        // 引爆源格画中心图
        for (let i = 0; i < N; i++) {
          if (!explosionTrig[i]) continue;
          const r = (i / W) | 0, c = i % W;
          items.push([r, () => ctx.drawImage(res.exploCenter, c * CELL, r * CELL)]);
        }
        // 臂：从引爆源向 4 方向按实际长度画（尊重挡火规则）
        for (let i = 0; i < N; i++) {
          if (!explosionTrig[i]) continue;
          const sr = (i / W) | 0, sc = i % W;
          for (let d = 0; d < 4; d++) {
            const [dr, dc] = DIRS[d];
            let n = 0;
            for (let k = 1; k <= maxBlast; k++) {
              const r = sr + dr * k, c = sc + dc * k;
              if (r < 0 || r >= H || c < 0 || c >= W) break;
              if (!blast[r * W + c]) break;
              n++;
            }
            const arm = res.exploArms[['up', 'down', 'left', 'right'][d]];
            const len = maxBlast * 40;
            for (let k = 1; k <= n; k++) {
              const r = sr + dr * k, c = sc + dc * k;
              let sx, sy;
              if (dc !== 0) {
                sx = dc > 0 ? (arm.width - len) + (k - 1) * 40 : (n - k) * 40;
                sy = 0;
              } else {
                sx = 0;
                sy = dr > 0 ? (arm.height - len) + (k - 1) * 40 : (n - k) * 40;
              }
              items.push([r, () => ctx.drawImage(arm, sx, sy, 40, 40,
                                                 c * CELL, r * CELL, CELL, CELL)]);
            }
          }
        }
      } else {
        explosion = null;
        explosionTrig = null;
      }
    }

    // 泡泡：底部贴格底线 + 垂直呼吸
    for (let i = 0; i < N; i++) {
      if (sim.fuse[i] <= 0) continue;
      const r = (i / W) | 0, c = i % W;
      const bx = c * CELL + (CELL - bombW) / 2;
      const by = (r + 1) * CELL - bombH + bob;
      items.push([r, () => ctx.drawImage(res.bomb, bx, by)]);
    }

    // 宝箱：三张道具图轮流展示 + 呼吸（底部贴格底线）
    const propIdx = Math.floor(nowS * 2) % res.props.length;
    const prop = res.props[propIdx];
    for (let i = 0; i < N; i++) {
      if (!sim.crate[i]) continue;
      const r = (i / W) | 0, c = i % W;
      const px = c * CELL + (CELL - prop.width) / 2;
      const py = (r + 1) * CELL - prop.height + bob;
      items.push([r, () => ctx.drawImage(prop, px, py)]);
    }

    // 角色：z = 脚所在行；帧底边 = 中心格底边；底线不越地图底
    const chars = [];   // {z, x, y, surf, wudi, wx, wy, hpv, mx}
    for (let pid = 0; pid < 2; pid++) {
      if (!sim.alive[pid]) continue;
      const rows = pid === 0 ? res.players : res.playerAi;
      let gy, gx;
      if (elSpectate.checked || pid === 0) {
        gy = sim.pos[pid * 2]; gx = sim.pos[pid * 2 + 1];
      } else {
        gy = prevPos[2] + (curPos[2] - prevPos[2]) * alpha;
        gx = prevPos[3] + (curPos[3] - prevPos[3]) * alpha;
      }
      const cx = gx * CELL, cy = gy * CELL;
      const row = MOVE_TO_SPRITE_ROW[face[pid]] != null ? MOVE_TO_SPRITE_ROW[face[pid]] : 0;
      const frame = (humanMoveState(pid) ? Math.floor(nowS * 8) % 4 : 0);
      const s = rows[row][frame];
      const blitX = Math.round(cx - s.width / 2);
      const blitY = Math.min(Math.round(cy + CELL / 2 - s.height), H * CELL - s.height);
      let wudi = null, wx = blitX, wy = blitY;
      if (sim.invuln[pid] > 0) {
        wudi = res.wudi;
        // 无敌光晕居中于角色帧中心（近似 res.py 的 body_center）
        wx = blitX + s.width / 2 - wudi.width / 2;
        wy = blitY + s.height / 2 - wudi.height / 2;
      }
      const z = Math.floor(gy);
      items.push([z, () => ctx.drawImage(s, blitX, blitY)]);
      chars.push({ pid, z, blitX, blitY, s, wudi, wx, wy, hpv: sim.hp[pid], mx: CFG.maxHp });
    }

    // 画家算法：z 升序（远→近）绘制
    items.sort((a, b) => a[0] - b[0]);
    for (const [, fn] of items) fn();

    // 无敌罩（加法混合，UI 层最后画）
    ctx.globalCompositeOperation = 'lighter';
    for (const ch of chars) {
      if (ch.wudi) ctx.drawImage(ch.wudi, ch.wx, ch.wy);
    }
    ctx.globalCompositeOperation = 'source-over';

    // 血条（段式，最后画不被墙挡）
    // 水平：右移一格宽（+CELL）后再回移 12px（纯右移一格子偏过头，视觉主体
    // 实际偏 ~半格多）—— 最终偏移 +48px。
    // 垂直：贴角色头顶上方 8px（与旧版一致，**不随箭头移动**）。
    for (const ch of chars) {
      const segW = 5, segH = 4, gap = 1;
      const color = ch.hpv > ch.mx / 3 ? '#50dc5a' : '#f04646';
      const barX = ch.blitX + CELL - 12;
      for (let i = 0; i < ch.mx; i++) {
        ctx.fillStyle = i < ch.hpv ? color : '#3c3c42';
        ctx.fillRect(barX + i * (segW + gap), ch.blitY - 8, segW, segH);
      }
    }

    // 我方控制指示箭头（res/point.png 向下箭头，已缩到 50%）：非观战时放在
    // **血条下方**、紧贴角色**视觉头顶**（帧内第一个非透明像素行，不是帧顶
    // 的透明留白）—— 箭头尖离视觉头顶 2px 指着头；顶行角色不越画布顶。
    if (!elSpectate.checked) {
      const me = chars.find((ch) => ch.pid === 0);
      if (me) {
        const aw = res.point.width, ah = res.point.height;
        const top = me.s._top || 0;
        const ax = me.blitX + me.s.width / 2 - aw / 2;
        const ay = Math.max(2, me.blitY + top - ah - 2);
        ctx.drawImage(res.point, Math.round(ax), Math.round(ay));
      }
    }
    drawHUD();
  }

  // 角色是否在行走（动画帧推进用）
  function humanMoveState(pid) {
    if (pid === 0) {
      return !elSpectate.checked && human.move !== MOVE_IDLE && sim.alive[0];
    }
    return face[1] !== MOVE_IDLE && sim.alive[1];
  }

  function drawHUD() {
    const y0 = BOARD_PX;
    ctx.fillStyle = '#10131a';
    ctx.fillRect(0, y0, BOARD_PX, HUD_PX);
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.strokeRect(0, y0, BOARD_PX, HUD_PX);
    const aiName = (sel) => sel === HUNTER_VAL ? '规则 Hunter' : (sel || '模型');
    const p0Kind = elSpectate.checked ? aiName(p0Sel) : '你';
    const p1Kind = aiName(enemySel);
    // 第 1 行：双方状态各自合并成一行（名字 + HP/属性同排），右侧倒计时（无 tick）
    const colors = ['#ff6b6b', '#5aa7ff'];
    const leftHalf = BOARD_PX * 0.46;
    for (let p = 0; p < 2; p++) {
      const name = p === 0 ? p0Kind : p1Kind;
      const tag = sim.alive[p] ? `P${p}` : `P${p}·阵亡`;
      const bx = 18 + p * leftHalf;
      const nameStr = `${name}（${tag}）`;
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = colors[p];
      ctx.font = 'bold 13px sans-serif';
      const nameW = ctx.measureText(nameStr).width;
      ctx.fillText(nameStr, bx, y0 + 10);
      ctx.fillStyle = '#e8e6df';
      ctx.font = '12px sans-serif';
      ctx.fillText(`HP ${sim.hp[p]}/${CFG.maxHp} · 泡 ${sim.bombsCap[p]} · 威 ${sim.blastCap[p]} · 速 ${sim.spdG[p].toFixed(2)}`,
                   bx + nameW + 8, y0 + 12);
    }
    // 倒计时（剩余秒，倒着走）
    const remain = Math.max(0, Math.ceil(CFG.maxSteps / CFG.tickHz - sim.t / CFG.tickHz));
    ctx.fillStyle = '#f5a623';
    ctx.font = 'bold 15px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`⏱ ${remain}s`, BOARD_PX - 18, y0 + 10);
    ctx.fillStyle = '#8b93a5';
    ctx.font = '12px sans-serif';
    ctx.fillText(`地图：${sim.mode === 'open' ? '空场' : '走廊'} · ${elScene.value} · 对局 #${gameSeed % 100000}`,
                 BOARD_PX - 18, y0 + 32);
    ctx.fillStyle = '#5a6275';
    ctx.font = '11px sans-serif';
    const em = enemySel && enemySel !== HUNTER_VAL ? modelCache.get(enemySel) : null;
    ctx.fillText(`敌人：${em ? em.meta.name + '（' + fmtStep(em.meta.global_step) + '步）' : p1Kind}`,
                 18, y0 + 78);
  }

  // ------------------------------------------------------------ 主循环
  let prevFrame = 0;
  function loop(now) {
    // 人类输入 60Hz 采样 + 帧级移动
    if (running && !elSpectate.checked) {
      const dt = Math.min((now - prevFrame) / 1000 || 0, 0.25);
      human.move = sampleHumanMove();
      if (human.move !== MOVE_IDLE && sim.alive[0]) {
        const eff = autoTurn(0, human.move);
        frameMove(0, eff, dt);
        human.move = eff;
        face[0] = eff;
      }
    }
    prevFrame = now;
    render(now);
    // 动图滚动窗口：按 20fps 把画面缩采样进环形缓冲，保存时取最近 12 秒。
    // 直接存原始像素（getImageData）：中间任何有损编码都会让静止背景
    // 每帧重量化出不同噪声 → 帧间差分失效（体积暴涨回关键帧量级）。
    // 像素级一致 → AnimEncoder 增量帧极小。
    // 终局后（running=false / sim.done）不再采集：冻结的结果画面会占满
    // 保存窗口后半段，看起来像慢放。
    // 缩采样 canvas 全局复用（每次 createElement+getContext 有 ~ms 级开销，
    // 在 30fps 主循环里会拖慢 rAF → 采集帧率掉到 20fps）。
    if (now - lastClipCap >= CLIP_FRAME_MS) {
      lastClipCap = now;
      if (running && sim && !sim.done) {
        if (!clipC) {
          clipC = document.createElement('canvas');
          clipC.width = Math.max(1, Math.round(canvas.width * CLIP_SCALE));
          clipC.height = Math.max(1, Math.round(canvas.height * CLIP_SCALE));
          clipCtx = clipC.getContext('2d');
        }
        clipCtx.drawImage(canvas, 0, 0, clipC.width, clipC.height);
        clipFrames.push({ t: now, img: clipCtx.getImageData(0, 0, clipC.width, clipC.height) });
      }
      while (clipFrames.length > 2 && now - clipFrames[0].t > CLIP_WINDOW_MS) clipFrames.shift();
    }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------ 启动
  // 触屏检测：移动端竖屏显示摇杆布局 + 欢迎文案
  const isTouch = () => !!(window.matchMedia &&
    window.matchMedia('(pointer: coarse)').matches);
  // 加载完成后显示欢迎窗口（操作说明），按空格或点击开始第一局
  function showWelcome() {
    elBanner.innerHTML =
      `<div class="wl-title">💣 QQT 格斗</div>` +
      (isTouch()
        ? `<span class="tip">左摇杆移动 · 💣 键放泡</span>`
        : `<span class="tip">方向键 / WASD 移动 · 空格 放泡</span>`) +
      `<span class="tip">R 重开</span>` +
      `<span class="tip act">点击 开始游戏</span>`;
    elBanner.classList.remove('hidden');
  }

  async function boot() {
    elLoadingText.textContent = '正在加载… 0%';
    // 模型（开局）与素材（渲染）互不依赖，并行加载
    await Promise.all([loadModelList(), loadAssets()]);
    await new Promise((r) => setTimeout(r, 150));   // 进度条缓动走完最后一段再切
    startVideoRecorder();     // 视频环形缓冲（VP9/VP8，常驻录制最近 12s）
    showWelcome();            // 先出欢迎窗口，等玩家按空格开局
    elLoading.classList.add('hidden');
    requestAnimationFrame(loop);
  }
  boot().catch((e) => {
    elStatus.innerHTML = `启动失败：${e.message}`;
    console.error(e);
  });

  // 调试钩子（只读）：无头验证用
  window.__QQT__ = {
    get sim() { return sim; },
    get res() { return res; },
    get model() {
      const m = enemySel && enemySel !== HUNTER_VAL ? modelCache.get(enemySel) : null;
      return m || (enemySel === HUNTER_VAL ? null : modelCache.get(modelList[0] && modelList[0].name) || null);
    },
    get enemySel() { return enemySel; },
    get p0Sel() { return p0Sel; },
    get modelCache() { return modelCache; },
    get running() { return running; },
    get explosion() { return explosion; },
    get explosionTrig() { return explosionTrig; },
  };
})();
