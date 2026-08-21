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
  const { Sim, MLPModel, CNNModel, TransformerModel, ORTTransformerModel, CFG, DIRS, MOVE_IDLE, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP } = Q;

  const H = Q.H, W = Q.W, N = Q.N;
  const CELL = 60;                 // 与 play/duel.py 一致：素材原生 40px/格 × 1.5
  const SCALE = CELL / 40.0;
  const BOARD_PX = CELL * W;       // 宽: 15 列 × 60px = 900
  const BOARD_H = CELL * H;        // 高: 13 行 × 60px = 780
  const BOARD_OFFSET = 20 * SCALE; // 顶部留半格(20原生px×1.5=30): 首行元件有上溢出
  const HUD_PX = 96;
  const TICK = 1.0 / CFG.tickHz;   // 0.1s
  const DY = [-1, 1, 0, 0], DX = [0, 0, -1, 1];
  // 动作编码 → 精灵行（行序：下/左/右/上，与 res.py MOVE_TO_SPRITE_ROW 一致）
  const MOVE_TO_SPRITE_ROW = { [MOVE_DOWN]: 0, [MOVE_LEFT]: 1, [MOVE_RIGHT]: 2, [MOVE_UP]: 3 };

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const $ = (id) => document.getElementById(id);
  const elSkin = $('skin'),
        elSpectate = $('spectate'), elDanger = $('danger'), elSound = $('sound'),
        elBgm = $('bgm'), elApplyModel = $('apply-model'), elCurModel = $('cur-model'),
        elEnemyAi = $('enemy-ai'), elP0Ai = $('p0-ai'), elP0AiWrap = $('p0-ai-wrap'),
        elRestart = $('restart'), elStatus = $('status'), elBanner = $('banner'),
        elLoading = $('loading'), elLoadingText = $('loading-text'),
        elSaveReplay = $('save-replay'), elSaveGif = $('save-gif'), elRecClip = $('rec-clip'),
        elSaveVideo = $('save-video'), elRecMsg = $('rec-msg'),
        elModelLowfreq = $('model-lowfreq');

  // ------------------------------------------------------------ 状态
  let sim = null, modelList = [], res = null;
  let rng = null;
  // 新地图系统: 241 张原版关卡 (levels.json) + 元素属性表 (elements.json)
  let levels = [], levelById = new Map(), elements = {};
  let selectedLevel = null;         // 黑屏菜单选中的关卡对象
  let elemImgCache = new Map();     // eid → Image (按需懒加载, 防启动加载 268 张)
  let showDanger = true;            // 与启动器一致：危险图红色渐变默认常显
  let showBox = false;              // B 键：绘制角色碰撞包围盒(调试)
  let soundOn = true;
  let bgmOn = true;
  let running = false;
  let mapMenuOpen = false;        // 黑屏选图菜单打开时冻结渲染（不再重绘游戏画面）
  let resultShown = false;
  let prevPos = new Float64Array(4), curPos = new Float64Array(4);
  let explosion = null, explosionTrig = null, explosionT = 0;
  // 砖被炸毁的中间态(_die 帧)特效: cell -> {eid, until}; 显示约 0.35s
  let dieFx = new Map();
  // 掉血回收宝箱飞行动画: {x0,y0,x1,y1,cell,t0} —— 从掉血玩家抛物线飞向落点(100ms)
  let flyFx = [];
  let dangerCache = null;           // tick 级危险图缓存（logicTick 每 step 重建）
  let lastTickT = 0;
  let gameSeed = 1;
  // 录像：JSON 结构化重放（seed + 每 tick 动作 + 周期快照）+ WebP 动图（滚动窗口）。
  // 动图管线：滚动窗口按 20fps 存像素级精确的缩采样帧（ImageData）→ 保存时
  // 交给 libwebp WebPAnimEncoder 做帧间差分合成（vendor/webp-anim/）。
  // 差分 vs 全关键帧：贴图细节场景完整关键帧 ~44KB/帧（12s≈10MB），差分帧
  // 只编变化区域 ~1/3 体积。QuickLook/Preview 的 WebP 动画播放慢是 macOS
  // 系统播放器问题（Chrome/Safari 播放 1:1 正常），与帧结构无关。
  // 采样降频(20→10fps)+窗口缩短(12→8s): 环形缓冲常驻从 ~280MB 降到 ~90MB,
  // 消除 getImageData 引发的周期性 Major GC 停顿(突变帧 267ms 的元凶)
  const CLIP_WINDOW_MS = 8000, CLIP_FRAME_MS = 1000 / 10, CLIP_SCALE = 0.6;
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
  let joyBombDown = false;   // 摇杆放泡按钮按住状态(tick 判断锁存清除用)
  let spaceDownSince = 0, joyDownSince = 0;   // 按下时刻: 长按>180ms 才连放, 点按=1颗
  const hunter = new Q.HunterAI();   // 规则 AI（纯进攻寻路），可当敌/我方
  const HUNTER_VAL = '__hunter__';   // 下拉里规则 AI 的 value 哨兵
  const IDLE_VAL = '__idle__';      // 静止敌人(不动不炸)哨兵
  const LATEST_VIT = 'ViTModel_500';         // 最新 ViT 模型(默认敌人)

  // 敌/我方 AI 选择：'__hunter__'（规则）或模型名。模型按需懒加载到缓存。
  // 敌人默认 = 列表第一个（ELO 最高）；观战我方默认 = 同样的最强模型。
  let enemySel = null, p0Sel = null;
  const modelCache = new Map();      // name → MLPModel/CNNModel/TransformerModel/ORT…（懒加载缓存）

  // transformer 模型优先用 onnxruntime（WebGPU→WASM），失败回退纯 JS
  async function makeOrtModel(name, doc) {
    const ort = window.ort;
    if (!ort || !ort.InferenceSession) return null;
    ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;  // 动态 import 需要绝对 URL
    // COOP/COEP 头存在(crossOriginIsolated) → WASM 可用多线程, 按核数设;
    // 否则禁线程(WebGPU 不受影响)。单线程跑 7.5M 参数 transformer ~10ms/次,
    // 多线程可到 ~1ms。
    ort.env.wasm.numThreads = (typeof crossOriginIsolated === 'boolean' && crossOriginIsolated)
      ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
    const providers = (typeof navigator !== 'undefined' && navigator.gpu)
      ? ['webgpu', 'wasm'] : ['wasm'];
    loadPhase = `正在加载 ONNX 推理引擎（${name}）`;
    requestAnimationFrame(updateProgress);
    const session = await ort.InferenceSession.create(`models/${name}.onnx`, {
      executionProviders: providers,
    });
    loadPhase = '';
    return new ORTTransformerModel(doc, session);
  }

  async function ensureModel(name) {
    let m = modelCache.get(name);
    if (m) return m;
    loadPhase = `正在加载模型 ${name}`;
    requestAnimationFrame(updateProgress);
    const resp = await fetch(`models/${name}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const doc = await resp.json();
    if (doc.meta.arch === 'transformer') {
      try { m = await makeOrtModel(name, doc); }
      catch (e) {
        console.warn('[ort] 会话创建失败，回退纯 JS 前向：', e);
        m = new TransformerModel(doc);
        m._ortError = String(e && e.message ? e.message : e);
      }
      if (!m) m = new TransformerModel(doc);
    } else {
      m = doc.meta.arch === 'cnn' ? new CNNModel(doc) : new MLPModel(doc);
    }
    m.inferEvery = elModelLowfreq.checked ? 2 : 1;   // 降频开关即时生效
    modelCache.set(name, m);
    return m;
  }

  // 玩家决策来源：'human' | '__hunter__' | 模型名（观战/规则 AI 时用）。
  // async：ORT 模型的 act 是异步 session.run；纯 JS 模型同步值 await 后不变。
  async function aiOf(pid) {
    if (pid === 0 && !elSpectate.checked) return [MOVE_IDLE, human.pendingBomb ? 1 : 0];
    const sel = pid === 0 ? p0Sel : enemySel;
    if (sel === IDLE_VAL) return [MOVE_IDLE, 0];   // 静止：不动不炸
    if (sel === HUNTER_VAL) return hunter.act(sim, pid);
    const m = sel ? modelCache.get(sel) : null;
    if (m) {
      try { return await m.act(sim, pid, rng); }
      catch (e) { return [MOVE_IDLE, 0]; }   // 13x13 旧模型在 15x13 上不适用：先站着
    }
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
  let loadPhase = '';                   // 当前加载阶段提示(素材/模型JSON/ONNX…)
  function updateProgress() {
    const imgPct = imgTotal ? (imgLoaded / imgTotal) * 100 : 0;
    const target = modelLoaded ? 100 : imgPct * 0.9;   // 模型占最后 10%
    progShown += (target - progShown) * 0.25;          // 一阶缓动，避免跳变
    if (Math.abs(target - progShown) < 0.5) progShown = target;  // 收敛停止
    const pct = Math.min(99, Math.round(progShown));
    // 阶段提示优先: 模型加载中显示"在干什么", 素材阶段显示百分比
    elLoadingText.textContent = loadPhase
      ? `${loadPhase}… ${pct}%`
      : `正在加载… ${pct}%`;
    if (progShown < target) requestAnimationFrame(updateProgress);
  }
  function loadImage(src) {
    if (imgCache.has(src)) {
      const v = imgCache.get(src);
      return v instanceof Promise ? v : Promise.resolve(v);
    }
    imgTotal++;
    const p = new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        imgLoaded++;
        requestAnimationFrame(updateProgress);
        imgCache.set(src, img);        // 缓存**已加载的 Image**（同步读得到）
        resolve(img);
      };
      img.onerror = () => {
        console.warn('素材缺失（降级占位）:', src);
        imgLoaded++;
        requestAnimationFrame(updateProgress);
        const c = document.createElement('canvas');   // 透明占位，不阻塞启动
        c.width = 40; c.height = 40;
        imgCache.set(src, c);
        resolve(c);
      };
      img.src = src;
    });
    imgCache.set(src, p);              // 加载完成前先占位 Promise（await 可用）
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
    loadPhase = '正在加载素材';
    // ---- 新地图系统：241 张原版关卡 + 元素属性表（旧 scenes.json 场景砖块废除）----
    const [levelsDoc, elementsDoc] = await Promise.all([
      (await fetch('assets/maps/levels.json?v=' + Date.now())).json(),
      (await fetch('assets/maps/elements.json?v=' + Date.now())).json(),
    ]);
    levels = levelsDoc;
    elements = elementsDoc;
    levelById = new Map(levels.map((l) => [l.id, l]));
    // 默认选中第一个普通竞技地图（黑屏菜单可改）
    selectedLevel = levels.find((l) => l.category === '普通竞技') || levels[0];

    // 主题背景（9 张，按关卡 background 引用；缩放同 build_static：比例 = CELL/40）
    const themes = [...new Set(levels.map((l) => l.theme).filter(Boolean))];
    const bgImages = {};
    await Promise.all(themes.map(async (t) => {
      const img = await loadImage('assets/bg/' + t + '.png');
      bgImages[t] = scaleCanvas(img, Math.round(img.width * SCALE),
                                Math.round(img.height * SCALE));
    }));

    // 顶部底色带资源：从水面背景**复制并裁切一行**(40原生px×SCALE 高, 全宽)，
    // 独立一份，绘制时自然高度显示、不把整图压进 30px 带
    let baseBand = null;
    if (bgImages['水面']) {
      baseBand = document.createElement('canvas');
      baseBand.width = bgImages['水面'].width;
      baseBand.height = Math.round(40 * SCALE);
      baseBand.getContext('2d').drawImage(
        bgImages['水面'], 0, 0, baseBand.width, baseBand.height,
        0, 0, baseBand.width, baseBand.height);
    }

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
    // 宝箱单图标：威力/泡泡数量/鞋子（炸开时种类已定，不再轮播）
    // 按原图像素 × 场景缩放系数(SCALE) 放大，绘制时格内居中（与整个场景一致）
    const iconScale = (img) => scaleCanvas(img, Math.round(img.width * SCALE), Math.round(img.height * SCALE));
    // 图标数组顺序必须与 crateType 一致: 0=泡 / 1=威 / 2=速
    const propIcons = [];
    for (const name of ['泡泡数量道具.png', '威力道具.png', '鞋子道具.png']) {
      propIcons.push(iconScale(await loadImage('assets/' + name)));
    }
    const superIcons = [];
    for (const name of ['超级泡泡.png', '超级威力.png', '超级速度.png']) {
      superIcons.push(iconScale(await loadImage('assets/' + name)));
    }
    // 随机宝箱（带问号的箱子，原版宝箱道具.GIF）：空场景十字箱 / 掉血回收箱
    const boxQ = iconScale(await loadImage('assets/宝箱道具.png'));
    const exploArms = {};
    for (const [key, f] of [['up', '向上爆炸.png'], ['down', '向下爆炸.png'],
                            ['left', '向左爆炸.png'], ['right', '向右爆炸.png']]) {
      exploArms[key] = await loadImage('assets/' + f);
    }
    res = {
      levels, levelById, elements, bgImages,
      skins: skinRows,             // 3 种玩家皮肤
      players: skinRows[elSkin.value],   // 当前玩家皮肤（切换后重绑）
      enemyRows,                   // 敌人固定角色c（不再染红）
      playerAi: enemyRows,
      wudi: scaleCanvas(wudi, Math.round(85 * SCALE), Math.round(85 * SCALE)),
      bomb: scaleCanvas(boomImg, CELL, CELL),
      propIcons, superIcons, boxQ, baseBand,
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
    // 关卡自带的主题背景；无则空
    const lv = sim && sim.level || selectedLevel;
    return (lv && res.bgImages && res.bgImages[lv.theme]) || null;
  }

  // 元件贴图懒加载缓存（eid → 缩放好的 canvas）
  // 未加载完的**不缓存**（返回 null，下帧重试），避免永久占位色块。
  function elemImage(eid) {
    const el = elements[eid];
    if (!el || !res) return null;
    const v = imgCache.get(el.file);
    if (!v || v instanceof Promise) return null;              // 还在加载：下帧重试
    const ready = (v instanceof HTMLCanvasElement) ||
      (v instanceof HTMLImageElement && v.complete && v.naturalWidth > 0);
    if (!ready) return null;
    if (elemImgCache.has(eid)) return elemImgCache.get(eid);
    // 缩放：40px/格 × SCALE (1.5) → 60px；多格元件整体按比例放大
    const s = CELL / 40.0;
    const out = scaleCanvas(v, Math.round(v.width * s), Math.round(v.height * s));
    elemImgCache.set(eid, out);
    return out;
  }
  // _die 炸毁帧加载（懒加载，同 elemImage 语义）
  function dieImage(eid) {
    const el = elements[eid];
    if (!el || !el.die || !res) return null;
    const v = imgCache.get(el.die);
    if (!v || v instanceof Promise) return null;
    const ready = (v instanceof HTMLCanvasElement) ||
      (v instanceof HTMLImageElement && v.complete && v.naturalWidth > 0);
    if (!ready) return null;
    const key = 'die:' + eid;
    if (elemImgCache.has(key)) return elemImgCache.get(key);
    const s = CELL / 40.0;
    const out = scaleCanvas(v, Math.round(v.width * s), Math.round(v.height * s));
    elemImgCache.set(key, out);
    return out;
  }
  // 预取某关卡用到的全部元件贴图（切图后渲染前调用，缺图降级色块）
  function preloadLevelImages(lv) {
    // 旧模型兼容: 右两列填墙的贴图也预载
    if (sim && sim.oldMode && elements[PAD_WALL_EID]) {
      const f = elements[PAD_WALL_EID].file;
      if (!imgCache.has(f)) loadImage(f).catch(() => {});
    }
    for (const layer of lv.layers) {
      for (const v of layer) {
        const eid = Math.abs(v);
        if (eid && elements[eid]) {
          if (!imgCache.has(elements[eid].file)) {
            loadImage(elements[eid].file).catch(() => {});
          }
          // 炸毁中间态 _die 帧也要预取，否则 dieFx 渲染时永远查不到图
          if (elements[eid].die && !imgCache.has(elements[eid].die)) {
            loadImage(elements[eid].die).catch(() => {});
          }
        }
      }
    }
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
    const lv = sim && sim.level || selectedLevel;
    if (!lv || !lv.music) { stopBgm(); return; }   // 本图无 BGM：旧曲必须停
    const url = lv.music;
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

  // 首次用户交互：只解锁 AudioContext（自动播放策略）。
  // BGM 只在真正开局时由 startGame → startBgm 播放，选地图阶段不响。
  let audioUnlocked = false;
  function unlockAudio() {
    if (audioUnlocked) return;
    audioUnlocked = true;
    if (res && res.audio && res.audio.state === 'suspended') res.audio.resume().catch(() => {});
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
    const bombDown = (e) => { e.preventDefault(); human.pendingBomb = true; joyBombDown = true; joyDownSince = performance.now(); };
    const bombUp = () => { joyBombDown = false; };   // 状态改, pendingBomb 由 tick 清
    elBombBtn.addEventListener('pointerdown', bombDown);
    elBombBtn.addEventListener('pointerup', bombUp);
    elBombBtn.addEventListener('pointercancel', bombUp);
  }

  window.addEventListener('keydown', (e) => {
    held.add(e.code);
    if (e.code === 'Space') {
      e.preventDefault();
      // 仅欢迎窗口（还没开局）用空格开始；结算界面用 R 重开，空格不触发
      if (!running && sim === null) { startGame(); return; }
      human.pendingBomb = true;
      spaceDownSince = performance.now();
    }
    if (e.code === 'KeyR') { startGame(); }
    // ESC → 回首页（黑屏选图菜单）；局内/结算画面都生效
    if (e.code === 'Escape') { openMapMenu(); }
    if (e.code === 'KeyD') { showDanger = !showDanger; elDanger.checked = showDanger; }
    if (e.code === 'KeyB') { showBox = !showBox; }   // 碰撞包围盒调试
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
    // 松手不清 pendingBomb: 由 tick 读取后按需清(见 logicTick),
    // 快速点按(按下-松开<100ms)也不会丢
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

  // 探测实际能移动多少并返回移动量(格)。三态判定:
  //   <5% 步长 = 贴墙被挡; 5%~95% = 部分可走(还能往墙滑); ≥95% = 完全可走。
  // 不能用 EPS 判断“可走”: 贴墙时 resolveAxis 的 stopPos 含 +EPS, 会误报能走。
  function probeMoveDist(pid, mv) {
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const blocked = blockedGrid();
    const dist = CFG.stepLen;
    const [dy, dx] = DIRS[mv];
    if (dy !== 0) {
      const ny = Q.resolveAxis(y + dy * dist, dy * dist, x, y, x, blocked, CFG.radius, H, W, true);
      return Math.abs(ny - y);
    }
    const nx = Q.resolveAxis(x + dx * dist, dx * dist, y, y, x, blocked, CFG.radius, H, W, false);
    return Math.abs(nx - x);
  }

  // 自动转向（只针对玩家）：直接移动被挡时，若碰撞盒**侧向**越过当前格边界
  // 且盒子横跨的某一侧有缺口，则转向滑入缺口。
  //   - 最小阈值 MIN_PEN：盒子刚擦边(近中心)不触发 —— 避免无谓偏移把玩家带偏；
  //   - 无上阈值：滑动必须能**完成**（滑到缺口列对齐后走出），
  //     不能中途截断 —— 否则盒子横跨两列、上方被墙挡时卡在角落动不了。
  // 唯一可调阈值 MIN_OFF: 触发条件 = bbox边线相对格子边线的偏移 < MIN_OFF。
  //   偏移越接近0越该转(盒子刚擦边), 越接近0.5越在正中间不转。
  //   调大=更容易触发(擦边一点就转), 调小=要更贴边才转; 0=永不转。
  const MIN_OFF = 0.399;
  let turnInput = -1, turnSlide = -1;    // 状态: 上次输入方向 + 承诺滑动方向(-1=无; 0=上合法!)
  function autoTurn(pid, move) {
    const stepLen = CFG.stepLen;
    const moved = move >= 4 ? stepLen : probeMoveDist(pid, move);
    // 完全可走(盒子能走满一步) → 正常移动, 取消滑动
    if (move >= 4 || moved >= stepLen * 0.95) { turnSlide = -1; return move; }
    if (move !== turnInput) turnSlide = -1;  // 输入方向变了: 取消旧滑动
    turnInput = move;
    const R = CFG.radius;
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const r = Math.floor(y), c = Math.floor(x);
    const open = (cr, cc) => cr >= 0 && cr < H && cc >= 0 && cc < W &&
      !sim.wall[cr * W + cc] && !sim.brick[cr * W + cc] && sim.fuse[cr * W + cc] <= 0;
    // 缺口方向: 垂直移动看水平跨度, 水平移动看垂直跨度
    // 注意: MOVE_UP=0, 不能用 `if(dir)` 判断(0 是合法方向) → 用 -1 作哨兵
    let dir = -1;
    if (move === MOVE_UP || move === MOVE_DOWN) {
      const cl = Math.floor(x - R), cr = Math.floor(x + R);
      const dy = move === MOVE_UP ? -1 : 1;
      if (cl !== cr) {
        if (open(r + dy, cl) && open(r, cl)) dir = MOVE_LEFT;
        else if (open(r + dy, cr) && open(r, cr)) dir = MOVE_RIGHT;
      }
    } else {
      const rl = Math.floor(y - R), rr2 = Math.floor(y + R);
      const dx = move === MOVE_LEFT ? -1 : 1;
      if (rl !== rr2) {
        if (open(rl, c + dx) && open(rl, c)) dir = MOVE_UP;
        else if (open(rr2, c + dx) && open(rr2, c)) dir = MOVE_DOWN;
      }
    }
    if (turnSlide !== -1) {
      // 已承诺滑动: 只要缺口方向仍可行就**继续滑到底**(不被部分移动/pen 截断),
      // 直到盒子对齐(直接移动完全可走)为止 —— 中途截断会卡在角落
      if (dir !== -1) return turnSlide;
      turnSlide = -1;
      return move;
    }
    // 初次触发: 贴墙被挡(moved <5% 步长)才触发 —— 还能往墙滑(部分可走)不触发
    if (moved > stepLen * 0.05) return move;
    // 简单版: 盒子横跨格边界时, 用 bbox 边线相对格子边线的偏移决定滑动方向。
    //   左滑判定 = bbox右缘 − 其所在格左线; 右滑判定 = 其所在格右线 − bbox左缘;
    //   上下同理。取偏移小的那一侧为滑动方向; 偏移越接近0越该转(0=100%转,
    //   0.5=在正中间不转) → 初次触发条件: 偏移 < MIN_OFF。
    let dir2 = -1, off = 99, okSlide = false;
    if (move === MOVE_UP || move === MOVE_DOWN) {
      const cl = Math.floor(x - R), cr = Math.floor(x + R);
      if (cl !== cr) {
        const lo = (x + R) - cr, ro = (cl + 1) - (x - R);
        const dy = move === MOVE_UP ? -1 : 1;
        // 侧移格(r,cl/r,cr)若是**玩家自己所在格** → 恒可走(脚下放泡能走出),
        // 不能因为自己格放了泡就判不可走, 否则自动转向永远不触发
        if (lo <= ro) { dir2 = MOVE_LEFT; off = lo; okSlide = open(r + dy, cl) && (cl === c || open(r, cl)); }
        else { dir2 = MOVE_RIGHT; off = ro; okSlide = open(r + dy, cr) && (cr === c || open(r, cr)); }
      }
    } else {
      const rl = Math.floor(y - R), rr2 = Math.floor(y + R);
      if (rl !== rr2) {
        const uo = (y + R) - rr2, do2 = (rl + 1) - (y - R);
        const dx = move === MOVE_LEFT ? -1 : 1;
        if (uo <= do2) { dir2 = MOVE_UP; off = uo; okSlide = open(rl, c + dx) && (rl === r || open(rl, c)); }
        else { dir2 = MOVE_DOWN; off = do2; okSlide = open(rr2, c + dx) && (rr2 === r || open(rr2, c)); }
      }
    }
    if (dir2 === -1 || !okSlide) { turnSlide = -1; return move; }   // 不横跨/滑不动: 不转
    if (turnSlide !== -1) return turnSlide;                          // 已承诺: 继续滑到底
    if (off >= MIN_OFF) return move;                                 // 偏移不够近0: 不触发
    turnSlide = dir2;
    return dir2;
  }

  function frameMove(pid, mv, dt) {
    if (mv === MOVE_IDLE || !sim.alive[pid]) return;
    const dist = CFG.speed * sim.spdG[pid] * Math.min(dt, 0.1);
    if (dist <= 0) return;
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const blocked = blockedGrid();
    const [dy, dx] = DIRS[mv];
    // 中心路径硬约束（对齐 sim.js Sim.step / JAX _move_player）：中心扫过的
    // 每一格必须可通行（起点格脚下豁免）。resolveAxis 的盒覆盖豁免允许盒压着
    // 泡格擦边，但中心不能进入泡/墙/砖格 —— 防穿炮（放泡后能离开泡格，
    // 但不能踩回泡格中心 / 穿过泡格继续走）。
    const startR = Math.max(0, Math.min(H - 1, Math.floor(y)));
    const startC = Math.max(0, Math.min(W - 1, Math.floor(x)));
    let ny = y, nx = x;
    if (dy !== 0) {
      ny = Q.resolveAxis(y + dy * dist, dy * dist, x, y, x,
                         blocked, CFG.radius, H, W, true);
      const yLo = Math.max(0, Math.min(H - 1, Math.floor(Math.min(y, ny))));
      const yHi = Math.max(0, Math.min(H - 1, Math.floor(Math.max(y, ny))));
      for (let r = yLo; r <= yHi; r++) {
        if (r === startR) continue;
        if (blocked[r * W + startC]) { ny = y; break; }
      }
    }
    if (dx !== 0) {
      nx = Q.resolveAxis(x + dx * dist, dx * dist, y, ny, x,
                         blocked, CFG.radius, H, W, false);
      const xLo = Math.max(0, Math.min(W - 1, Math.floor(Math.min(x, nx))));
      const xHi = Math.max(0, Math.min(W - 1, Math.floor(Math.max(x, nx))));
      const cy0 = Math.max(0, Math.min(H - 1, Math.floor(ny)));
      for (let c = xLo; c <= xHi; c++) {
        if (c === startC && cy0 === startR) continue;
        if (blocked[cy0 * W + c]) { nx = x; break; }
      }
    }
    sim.pos[pid * 2] = ny;
    sim.pos[pid * 2 + 1] = nx;
    sim.pos[pid * 2] = Math.min(Math.max(sim.pos[pid * 2], CFG.radius), H - CFG.radius);
    sim.pos[pid * 2 + 1] = Math.min(Math.max(sim.pos[pid * 2 + 1], CFG.radius), W - CFG.radius);
  }

  // ------------------------------------------------------------ 开局
  // 旧 13x13 模型(MLP/CNN, obs_shape=[14,13,13])兼容: 当前地图 15 宽, 旧模型
  // 期望 13 宽。选中旧模型 → 地图第 13/14 列填不可通行墙 + encodeObs 按 13 宽输出。
  // 注意: 旧模型**只在空场景训练过**, 泛化性差 → 仅空场景做此适配, 其他地图不适用。
  function isOldModelName(name) {
    const m = name ? modelCache.get(name) : null;
    const shp = m && m.meta && m.meta.obs_shape;
    return !!shp && shp.length === 3 && shp[2] === 13;   // 旧模型观测宽度=13
  }
  function oldModeActive() {
    const oldSel = isOldModelName(elSpectate.checked ? p0Sel : enemySel) ||
                   (elSpectate.checked && isOldModelName(enemySel));
    return oldSel && selectedLevel && selectedLevel.source === 'empty_scene';
  }

  function startGame() {
    if (!res || !selectedLevel) return;      // 素材/地图未就绪由 logicTick 兜底等待
    gameSeed = (Math.random() * 0xFFFFFFFF) >>> 0;
    sim = new Sim(gameSeed);
    window.__sim = sim;                        // 调试钩子：读 sim 状态/帧率用
    sim.reset(selectedLevel, { oldMode: oldModeActive() });  // 旧模型: 13/14列填墙+13宽观测
    preloadLevelImages(selectedLevel);       // 预取本图元件贴图
    prevCovered = new Set();                 // 清空结构覆盖/进入动画状态
    structAnim.clear();
    rng = Q.mulberry32(gameSeed ^ 0x13579BDF);
    human.dirStack = []; human.latch.clear(); human.move = MOVE_IDLE; human.pendingBomb = false;
    tickDebt = 0;                  // 新开局清空节流补偿欠账
    joyMove = null;                    // 摇杆归位（移动端）
    explosion = null; explosionTrig = null; resultShown = false;
    dangerCache = null;             // 开局清掉旧危险图缓存
    replay = {
      meta: {
        map: selectedLevel.source,
        mapName: selectedLevel.name,
        category: selectedLevel.category,
        seed: gameSeed,
        initial: selectedLevel.initial_stats,
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
    mapMenuOpen = false;                      // 开局恢复渲染
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
    const idle = document.createElement('option');
    idle.value = IDLE_VAL;
    idle.textContent = '静止（不动不炸）';
    sel.appendChild(idle);
    if (includeHunter) {
      const h = document.createElement('option');
      h.value = HUNTER_VAL;
      h.textContent = '规则 Hunter（纯进攻寻路）';
      sel.appendChild(h);
    }
    for (const m of modelList) {
      const opt = document.createElement('option');
      opt.value = m.name;
      const gstep = m.global_step != null ? m.global_step : (m.it || 0);
      const elo = m.elo != null ? ` · elo ${m.elo}` : '';
      opt.textContent = `${m.name}  · ${fmtStep(gstep)}步${elo} · 导出于 ${(m.generated_at || '').slice(0, 10)}`;
      sel.appendChild(opt);
    }
  }

  async function loadModelList() {
    const resp = await fetch('models/index.json');
    modelList = (await resp.json()).models;
    // 下拉按时间倒序(最新在前): generated_at 缺失的排最后
    modelList.sort((a, b) => String(b.generated_at || '').localeCompare(String(a.generated_at || '')));
    // 敌人 AI 下拉 + 观战「我方：」下拉：都列全部模型 + 规则 Hunter
    fillAiSelect(elEnemyAi, true);
    fillAiSelect(elP0Ai, true);
    elEnemyAi.value = LATEST_VIT;             // 默认敌人 = 最新 ViT 模型
    elP0Ai.value = HUNTER_VAL;                // 观战「我方：」默认规则 Hunter
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
    if (sel === IDLE_VAL) {
      enemySel = IDLE_VAL;
      elCurModel.textContent = '静止（不动不炸）';
      elStatus.innerHTML = '敌人：<b>静止</b>（不动不炸）';
      return;
    }
    elStatus.innerHTML = `正在加载模型 <b>${sel}</b>…`;
    try {
      const m = await ensureModel(sel);
      enemySel = sel;
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent =
        `${m.meta.name}（${fmtStep(m.meta.global_step ?? m.meta.it ?? 0)}步 · 导出于 ${(m.meta.generated_at || '').slice(0, 10)}）`;
      elStatus.innerHTML =
        `当前模型：<b>${m.meta.name}</b><br>` +
        `训练步数 ${fmtStep(m.meta.global_step ?? m.meta.it ?? 0)}<br>` +
        `观测 ${m.meta.obs_shape.join('×')} · 参数 ${Object.values(m.tensors)
          .reduce((s, [, n]) => s + n, 0).toLocaleString()}<br>` +
        `推理后端：${m.constructor.name === 'ORTTransformerModel'
          ? (navigator.gpu ? 'WebGPU' : 'WASM') : '纯 JS'}` +
        (m._ortError ? `<br><span class="dim">ORT 失败：${m._ortError.slice(0, 120)}</span>` : '');
    } catch (e) {
      elStatus.innerHTML = `模型加载失败：${e.message}`;
    }
  }

  elRestart.addEventListener('click', startGame);
  const elMapBtn = $('map-btn');
  if (elMapBtn) elMapBtn.addEventListener('click', openMapMenu);
  elBanner.addEventListener('click', () => { if (!running) startGame(); });  // 欢迎窗口点击开始
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
  // AI 模型推理降频：对已缓存模型即时生效（重新开局时新加载的模型在
  // ensureModel 里同样读取该开关）
  elModelLowfreq.addEventListener('change', () => {
    const every = elModelLowfreq.checked ? 2 : 1;
    for (const m of modelCache.values()) if (m.inferEvery) m.inferEvery = every;
  });
  // 「录制动图」开关：勾选启动动图采样+视频录制, 取消停止(零开销)
  elRecClip.addEventListener('change', () => {
    if (elRecClip.checked) {
      if (!mediaRec) startVideoRecorder();
      clipFrames = [];
      recMsg('录制动图：已开启（保存 GIF/视频需此开关）');
    } else {
      if (mediaRec && mediaRec.state === 'recording') { try { mediaRec.stop(); } catch (e) {} }
      mediaRec = null; mediaChunks = [];
      clipFrames = [];
      recMsg('录制动图：已关闭（零开销）');
    }
  });
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
    elRecMsg.classList.add('active');
    clearTimeout(recMsg._t);
    recMsg._t = setTimeout(() => {
      elRecMsg.innerHTML = '';
      elRecMsg.classList.remove('active');
    }, 10000);
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

  // GIF 导出参数：GIF 对贴图噪点画面天生低效（原尺寸 256 色 12s ≈ 17MB），
  // README 内嵌用降采样 + 10fps + 128 色全局调色板 ≈ 3MB。
  // 全局调色板贯穿全片，避免逐帧调色板导致的颜色闪烁。
  const GIF_SUB_FRAME = 2;   // 20fps 采集 → 10fps 导出
  const GIF_SCALE = 0.6;     // 相对采集画布的额外降采样（0.6×0.6 ≈ 0.36 游戏画面）
  const GIF_COLORS = 128;    // 128 色 PSNR ~30dB，256 色只多 ~17% 体积
  const gifScratch = {};

  elSaveGif.addEventListener('click', () => {
    if (!clipFrames.length) {
      recMsg('未录制画面：请先勾选「录制动图」再开局');
      return;
    }
    if (!replay || !replay.actions.length) { recMsg('还没有可录的画面 —— 先开始一局再保存'); return; }
    const now = performance.now();
    // 终局后保存：只保留游戏结束前的帧（冻结结果画面不进屋，避免"结尾静止=慢放"）
    const frames = clipFrames.filter((f) =>
        now - f.t <= CLIP_WINDOW_MS + 250 && (!gameEndT || f.t <= gameEndT))
      .sort((a, b) => a.t - b.t);
    if (frames.length < 2) { recMsg('画面不足，等几秒再点'); return; }
    const gif = window.Gifenc;
    if (!gif) { recMsg('GIF 编码器未加载（vendor/gifenc.global.js）'); return; }
    const sub = frames.filter((_, i) => i % GIF_SUB_FRAME === 0);
    recMsg(`GIF 编码中…（${frames.length} 帧 → 10fps ${GIF_COLORS} 色全局调色板，约几秒请稍候）`);
    setTimeout(() => {
      const t0 = performance.now();
      try {
        const f0 = sub[0].img;
        const gW = Math.max(1, Math.round(f0.width * GIF_SCALE));
        const gH = Math.max(1, Math.round(f0.height * GIF_SCALE));
        if (!gifScratch.src) {   // 1:1 中转画布（putImageData 不缩放）
          gifScratch.src = document.createElement('canvas');
          gifScratch.src.width = f0.width; gifScratch.src.height = f0.height;
          gifScratch.srcCtx = gifScratch.src.getContext('2d');
        }
        if (!gifScratch.dst || gifScratch.dst.width !== gW) {  // 缩放目标画布
          gifScratch.dst = document.createElement('canvas');
          gifScratch.dst.width = gW; gifScratch.dst.height = gH;
          gifScratch.dstCtx = gifScratch.dst.getContext('2d', { willReadFrequently: true });
        }
        // 1) 逐帧降采样（双线性，drawImage 缩放 + getImageData 读回）
        const scaled = sub.map((f) => {
          gifScratch.srcCtx.putImageData(f.img, 0, 0);
          gifScratch.dstCtx.drawImage(gifScratch.src, 0, 0, gW, gH);
          return gifScratch.dstCtx.getImageData(0, 0, gW, gH).data;
        });
        // 2) 全局调色板：隔像素隔帧采样 → 量化一版贯穿全片
        const stepPx = 8, stepF = 2;
        const sample = new Uint8Array(Math.ceil(scaled.length / stepF) * Math.ceil((gW * gH) / stepPx) * 4);
        let si = 0;
        for (let fi = 0; fi < scaled.length; fi += stepF) {
          const d = scaled[fi];
          for (let i = 0; i < d.length; i += stepPx * 4) {
            sample[si++] = d[i]; sample[si++] = d[i + 1]; sample[si++] = d[i + 2]; sample[si++] = d[i + 3];
          }
        }
        const palette = gif.quantize(sample.subarray(0, si), GIF_COLORS);
        // 3) 逐帧映射 + 写入（delay 毫秒 → gifenc 内部转厘秒；10fps ≈ 100ms/帧）
        const enc = gif.GIFEncoder();
        let prevT = null;
        for (let i = 0; i < scaled.length; i++) {
          const idx = gif.applyPalette(scaled[i], palette);
          const durMs = prevT === null ? Math.round(CLIP_FRAME_MS * GIF_SUB_FRAME)
            : Math.max(20, Math.min(1000, Math.round(sub[i].t - prevT)));
          prevT = sub[i].t;
          enc.writeFrame(idx, gW, gH, { palette, delay: durMs });
        }
        enc.finish();
        const blob = new Blob([enc.bytes()], { type: 'image/gif' });
        downloadBlob(blob, `clip_${replay.meta.mode}_s${replay.meta.seed}_${timeStamp()}.gif`);
        recMsg(`GIF 已保存：${scaled.length} 帧，${(blob.size / 1024).toFixed(0)}KB，耗时 ${((performance.now() - t0) / 1000).toFixed(1)}s（${GIF_COLORS} 色全局调色板，可内嵌 README）`);
      } catch (e) {
        recMsg(`GIF 编码失败：${e.message}`);
        console.error(e);
      }
    }, 30);
  });

  elSaveVideo.addEventListener('click', async () => {
    if (!mediaRec || mediaRec.state !== 'recording') {
      recMsg('未录制视频：请先勾选「录制动图」再开局');
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
  let tickBusy = false;              // async tick 重入保护（await 期间 setInterval 再触发时跳过）
  let tickDebt = 0;                  // 后台节流补偿欠账(ms): 标签页隐藏时 setInterval 被
                                     // 节流到 ~1Hz(10倍慢) → 每次触发补跑缺失的 tick 保持实时
  async function logicTick() {
    if (!running || !sim || sim.done || tickBusy) return;
    tickBusy = true;
    try {
      tickDebt += TICK * 1000;       // 本次触发期望推进 100ms
      let guard = 0;
      while (tickDebt >= TICK * 1000 && guard < 12 && running && sim && !sim.done) {
        await logicTickInner();
        tickDebt -= TICK * 1000;
        guard++;
      }
      if (tickDebt > TICK * 1000 * 30) tickDebt = TICK * 1000 * 30;   // 欠账封顶30 tick
    } finally {
      tickBusy = false;
    }
  }
  async function logicTickInner() {
    const tk0 = performance.now();
    // 模型未就绪（正在加载/加载失败）先不推进：敌人 + 观战时的我方；
    // 规则 AI 随时可用。缺这一步观战 P0 模型没进缓存 → aiOf 返回 IDLE 站着。
    const spectate = elSpectate.checked;
    if (enemySel !== HUNTER_VAL && !modelCache.has(enemySel)) return;
    if (spectate && p0Sel !== HUNTER_VAL && !modelCache.has(p0Sel)) return;
    // 观战 + 双方同一模型 → 一次批处理前向出双玩家动作（ORT 为 batch=2 一次 run）
    let a0, a1;
    const pairM = (spectate && enemySel === p0Sel) ? modelCache.get(enemySel) : null;
    if (pairM && pairM.bothAct) {
      const pair = await pairM.bothAct(sim, rng);
      a0 = pair[0]; a1 = pair[1];
    } else {
      a0 = await aiOf(0);
      // 点按/长按区分: 每次 tick **消费** pendingBomb(只放一次), 仅当按键
      // 仍按住且超过长按阈值(180ms)才**重新武装** → 点按=恰好1颗, 长按=连放
      if (!spectate) {
        human.pendingBomb = false;
        const heldNow = held.has('Space') || joyBombDown;
        const downSince = held.has('Space') ? spaceDownSince : joyDownSince;
        if (heldNow && performance.now() - downSince > 180) human.pendingBomb = true;
      }
      a1 = await aiOf(1);
    }
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
    const prevBrick = sim.brick.slice();      // 砖快照（找被炸毁的格）
    const prevBush = sim.bush ? sim.bush.slice() : null;
    const hpBefore = sim.hp.slice();          // 血量快照（找掉血玩家）
    const prevCrate = sim.crate.slice();      // 宝箱快照（找新回收箱）
    const info = sim.step([a0, a1]);
    curPos.set(sim.pos);
    // 掉血回收：新出现的 recycle 宝箱 → 从掉血玩家身上抛物线飞向落点（100ms）
    const dmgP = hpBefore[0] > sim.hp[0] ? 0 : (hpBefore[1] > sim.hp[1] ? 1 : -1);
    if (dmgP >= 0) {
      const fNow = performance.now();
      for (let i = 0; i < N; i++) {
        if (sim.crate[i] === 1 && !prevCrate[i] && sim.recycle[i] === 1) {
          flyFx.push({
            x0: sim.pos[dmgP * 2 + 1], y0: sim.pos[dmgP * 2],
            x1: (i % W) + 0.5, y1: ((i / W) | 0) + 0.5, cell: i, t0: fNow,
          });
        }
      }
    }
    // 被炸毁的砖/灌木 → 记录 _die 中间态特效（元素 ID 从本图 L1/L0 层取）
    if (sim.level && sim.level.layers) {
      const l1 = sim.level.layers[1];
      const l0 = sim.level.layers[0];
      for (let i = 0; i < N; i++) {
        if (prevBrick[i] === 1 && sim.brick[i] === 0 && l1[i]) {
          dieFx.set(i, { eid: Math.abs(l1[i]), until: performance.now() + 350 });
        }
        if (prevBush && prevBush[i] === 1 && sim.bush[i] === 0 && l0[i]) {
          dieFx.set(i, { eid: Math.abs(l0[i]), until: performance.now() + 350 });
        }
      }
    }
    lastTickT = performance.now();
    prof.tickLast = performance.now() - tk0;   // tick 耗时(ms)
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
    // 调试用: ?auto=1 → 双 AI 自动开局(?ai=模型名 → 双方用该模型; 默认双 Hunter)
    if (location.search.includes('auto=1')) {
      const ai = new URLSearchParams(location.search).get('ai');
      if (ai) {
        elEnemyAi.value = ai; elP0Ai.value = ai;
        p0Sel = ai; enemySel = ai;
        try { await ensureModel(ai); } catch (e) { console.warn('[auto] 模型加载失败', e); }
      } else {
        elEnemyAi.value = HUNTER_VAL;
        elP0Ai.value = HUNTER_VAL;
        p0Sel = HUNTER_VAL; enemySel = HUNTER_VAL;
      }
      selectedLevel = levels.find((l) => l.source === 'empty_scene') || levels[0];
      startGame();
      console.log('[auto] 自动开局: 双方 ' + (ai || '规则 Hunter') + ', empty_scene');
    }
    // (B键坐标点击复制已移除)
      running = false;
    }
  }
  setInterval(logicTick, TICK * 1000);

  // ------------------------------------------------------------ 渲染（draw_grid 移植）
  // 新图块体系：按关卡 layers_raw 逐格渲染**原版元件贴图**（elements.json），
  // 多格元件只在原点格画一次（负值延续格跳过，精灵整体覆盖 w×h 格）。
  //   Z 排序（画家算法，z 升序先画）：
  //   纵向：越靠下（r 越大）z 越高，后画 —— 高元件（>40px，甚至 80px+）靠
  //         z 排序盖住上方行的角色脚部；
  //   横向：元件投影往**左**，所以同一行内**越靠右 z 越低**（先画垫底），
  //         让左边的图块盖住右边图块的投影（shadow 踩在自己图块之下）；
  //   可进入结构（拱门/房子）：角色在结构**下方/前面**时按行 z 正常排序
  //         （角色盖住结构）；角色中心**完全进入足迹**时才把结构 z 抬到角色
  //         之上（挡在前面），并在进入瞬间播放果冻扭动动画（横/纵不同相位
  //         的缩放，像钻进小房子那样扭一下）。
  const Z_ROW_STRIDE = W + 1;              // 纵向主序步长
  // 行内 z 范围 0..14；泡泡=15、角色=16 —— 同一行时实体恒在墙/元件前面，
  // 只有实体在墙体上一行(更小 r)时才被墙盖住。
  function tileZ(r, c) {
    return r * Z_ROW_STRIDE + (W - 1 - c);
  }
  function drawBoard(bg) {
    // 背景层（build_static 的 JS 版）：缩放后从左上角铺一张
    if (bg) ctx.drawImage(bg, 0, 0);
    else { ctx.fillStyle = '#2d2a32'; ctx.fillRect(0, 0, BOARD_PX, BOARD_H); }
    // 地板层（L2）：每格地面贴图铺满格（40×40 或带偏移）
    const lv = sim && sim.level;
    if (!lv || !lv.layers || !lv.layers[2]) return;
    const g = lv.layers[2];
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const v = g[r * W + c];
        if (!v) continue;
        const el = elements[v];
        const img = el && elemImage(v);
        if (!el) continue;
        const x = c * CELL - el.xo * SCALE, y = r * CELL - el.yo * SCALE;
        if (img) ctx.drawImage(img, x, y);
        else { ctx.fillStyle = '#454a52'; ctx.fillRect(c * CELL, r * CELL, CELL, CELL); }
      }
    }
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
  // ---- 结构（L1 墙/砖 + L0 头顶装饰）足迹与进入动画 ----
  // 结构清单缓存（每关一次）：多格元件只在原点格画一次，足迹 = 原点 + w×h。
  const structCache = new Map();
  function levelStructures(lv) {
    if (structCache.has(lv.id)) return structCache.get(lv.id);
    const out = [];
    for (let li = 0; li < 2; li++) {
      for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
          const v = lv.layers[li][r * W + c];
          if (!v || v < 0) continue;
          const el = elements[v];
          if (!el) continue;
          out.push({ key: li + ':' + r + ':' + c, layer: li, r, c, eid: v, w: el.w, h: el.h,
                    isBush: !!(lv.bush && lv.bush[r * W + c]) });
        }
      }
    }
    structCache.set(lv.id, out);
    return out;
  }
  // 旧模型兼容模式(空场景+旧模型): 第 13/14 列填的墙在 level.layers 里没有
  // 对应元素, 渲染时补上虚拟墙结构(中国城墙体 elem 3007), 让填墙可见。
  const PAD_WALL_EID = 3007;   // 中国城墙体
  const PAD_WALLS = [];
  for (let r = 0; r < H; r++) {
    for (let c = W - 2; c < W; c++) {
      PAD_WALLS.push({ key: 'pad:' + r + ':' + c, layer: 1, r, c, eid: PAD_WALL_EID,
                       w: 1, h: 1, isBush: false, pad: true });
    }
  }

  // 结构进入动画：key -> 进入时刻 (performance.now)；果冻扭动 = 横/纵不同
  // 相位、随时间衰减的正弦缩放（锚点在结构底部中心，像钻进小房子扭一下）。
  const structAnim = new Map();
  let prevCovered = new Set();
  function drawStructImg(st, img, x, y, nowMs) {
    const age = structAnim.has(st.key) ? (nowMs - structAnim.get(st.key)) / 1000 : 1;
    if (age < 0.6) {
      const dec = 1 - age / 0.6;
      const sx = 1 + 0.12 * Math.sin(2 * Math.PI * 2.5 * age) * dec;
      const sy = 1 - 0.10 * Math.sin(2 * Math.PI * 2.5 * age + Math.PI / 2) * dec;
      const cx = x + img.width / 2, cy = y + img.height;   // 底部中心为锚
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(sx, sy);
      ctx.drawImage(img, -img.width / 2, -img.height);
      ctx.restore();
    } else {
      ctx.drawImage(img, x, y);
    }
  }

  // 渲染一帧：背景+地板 → 危险区 → 画家算法精灵（墙砖/爆炸/泡/宝箱/角色）→ 无敌罩 → 血条 → HUD
  function render(now) {
    if (mapMenuOpen) return;          // 换地图黑屏菜单：保持已清空的画布，不重绘
    if (!sim || !res) return;
    const alpha = Math.min(1, (now - lastTickT) / (TICK * 1000));
    // 顶部半格底色带(最低优先级): 用裁好的水面一行(自然高度, 不压缩),
    // 只露出带内的部分(居中裁取), 横向铺满
    const base = res.baseBand;
    if (base) {
      const sy = (base.height - BOARD_OFFSET) / 2;
      ctx.drawImage(base, 0, sy, base.width, BOARD_OFFSET,
                    0, 0, canvas.width, BOARD_OFFSET);
    } else { ctx.fillStyle = '#1f2b33'; ctx.fillRect(0, 0, canvas.width, BOARD_OFFSET); }
    // 游戏画面整体下移 20 原生像素×SCALE(首行元件顶部溢出不再被截断)
    ctx.save();
    ctx.translate(0, BOARD_OFFSET);
    drawBoard(sceneOf());
    drawDangerOverlay();

    const items = [];   // (z, drawFn)
    const hideCells = new Set();   // 角色进入果冻结构 → 足迹内角色隐藏（render 作用域）

    // 结构精灵（L1 墙/砖 + L0 头顶装饰）：负值延续格跳过，多格元件整体在原点
    // 画一次；L1 由 sim 状态驱动（砖炸掉即消失）；角色中心进入结构足迹 →
    // 结构抬到角色之上（挡在前面）+ 进入果冻扭动动画。Z 排序见 tileZ。
    if (sim.level && sim.level.layers) {
      const nowMs = performance.now();
      const charCells = [];
      for (let p = 0; p < 2; p++) {
        charCells.push(sim.alive[p]
          ? [Math.floor(sim.pos[p * 2]), Math.floor(sim.pos[p * 2 + 1])] : null);
      }
      const coveredNow = new Set();
      const structList = sim.oldMode
        ? levelStructures(sim.level).concat(PAD_WALLS) : levelStructures(sim.level);
      for (const st of structList) {
        const v = st.pad ? st.eid : sim.level.layers[st.layer][st.r * W + st.c];
        if (!v || v < 0) continue;
        if (st.layer === 1) {
          const i = st.r * W + st.c;
          // 墙/砖/房子/灌木 都在才画；砖与灌木被炸毁后消失
          if (!sim.wall[i] && !sim.brick[i] && !sim.cover[i] && !sim.bush[i]) continue;
        } else if (st.isBush && !sim.bush[st.r * W + st.c]) {
          continue;                            // 顶层灌木被炸毁 → 不画
        }
        const el = elements[st.eid];
        if (!el) continue;
        const img = el && elemImage(st.eid);
        // 结构足迹格：**所有存活结构**的格子都记入 hideCells —— 角色/泡泡
        // 只要在这些格上（进房子/灌木/拱门等，不管可炸不可炸）就 visible=false
        for (let rr = st.r; rr < Math.min(H, st.r + st.h); rr++) {
          for (let cc = st.c; cc < Math.min(W, st.c + st.w); cc++) {
            hideCells.add(rr * W + cc);
          }
        }
        // 角色中心是否在结构足迹内（完全进入）→ 结构挡在前面 + 果冻动画
        let covered = false;
        const r1 = Math.min(H, st.r + st.h), c1 = Math.min(W, st.c + st.w);
        for (const cc of charCells) {
          if (cc && cc[0] >= st.r && cc[0] < r1 && cc[1] >= st.c && cc[1] < c1) {
            covered = true; break;
          }
        }
        if (covered && !prevCovered.has(st.key)) structAnim.set(st.key, nowMs);
        if (!covered && structAnim.has(st.key) &&
            nowMs - structAnim.get(st.key) > 600) structAnim.delete(st.key);
        if (covered) coveredNow.add(st.key);
        // 结构 Z 恒用行内 tileZ：不因角色进入而提升（提升会盖住下方图块向上
        // 溢出的部分）；角色/泡泡靠 hideCells 隐藏即可
        const z = tileZ(st.r, st.c);
        const x = st.c * CELL - el.xo * SCALE, y = st.r * CELL - el.yo * SCALE;
        items.push([z, () => {
          if (img) drawStructImg(st, img, x, y, nowMs);
          else { ctx.fillStyle = '#5a5f68'; ctx.fillRect(st.c * CELL, st.r * CELL, CELL, CELL); }
        }]);
      }
      prevCovered = coveredNow;
    }

    // 砖被炸毁的中间态 (_die 帧)：短暂显示碎墙 ~0.35s，尾段淡出
    if (dieFx.size) {
      const dieNow = performance.now();
      for (const [i, fx] of dieFx) {
        if (dieNow > fx.until) { dieFx.delete(i); continue; }
        const img = dieImage(fx.eid);
        const el = elements[fx.eid];
        if (!img || !el) continue;
        const r = (i / W) | 0, c = i % W;
        const x = c * CELL - el.xo * SCALE, y = r * CELL - el.yo * SCALE;
        const fade = Math.min(1, (fx.until - dieNow) / 150);
        items.push([tileZ(r, c) + 1, () => {
          ctx.globalAlpha = fade;
          ctx.drawImage(img, x, y);
          ctx.globalAlpha = 1;
        }]);
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
          items.push([r * Z_ROW_STRIDE + 2, res.exploCenter, c * CELL, r * CELL]);
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
              items.push([r * Z_ROW_STRIDE + 2, arm, sx, sy, 40, 40,
                           c * CELL, r * CELL, CELL, CELL]);
            }
          }
        }
      } else {
        explosion = null;
        explosionTrig = null;
      }
    }

    // 泡泡：底部贴格底线 + 垂直呼吸（原样，无半透明/闪烁效果）
    for (let i = 0; i < N; i++) {
      if (sim.fuse[i] <= 0) continue;
      // 泡泡在任何果冻遮挡结构(房子/灌木/拱门等)上 → 隐藏（visible=false）
      if (hideCells.has(i)) continue;
      const r = (i / W) | 0, c = i % W;
      const bx = c * CELL + (CELL - bombW) / 2;
      const by = (r + 1) * CELL - bombH + bob;
      items.push([r * Z_ROW_STRIDE + (Z_ROW_STRIDE - 1), res.bomb, bx, by]);
    }

    // 宝箱：三张道具图轮流展示 + 呼吸（底部贴格底线）
    // 宝箱图标：随机(带?箱子) / 普通按炸开时定的种类 / 超级按种类+超级图标
    const boxQ = res.boxQ;
    // 正在飞行的宝箱目标格：飞行期间不画落点宝箱（避免双影）
    const flyNow = performance.now();
    const flyTargets = new Set();
    for (const f of flyFx) {
      if ((flyNow - f.t0) / 1000 < 0.1) flyTargets.add(f.cell);
    }
    for (let i = 0; i < N; i++) {
      if (!sim.crate[i] || flyTargets.has(i)) continue;
      const r = (i / W) | 0, c = i % W;
      // 随机宝箱(带?箱子) / 普通(种类定好) / 超级(种类+超级图标)
      let p;
      if (sim.crateType[i] < 0) p = boxQ;
      else if (sim.superCrate[i]) p = res.superIcons[sim.crateType[i]];
      else p = res.propIcons[sim.crateType[i]];
      // 原图尺寸，格内居中（不拉伸）
      const px = c * CELL + (CELL - p.width) / 2;
      const py = r * CELL + (CELL - p.height) / 2 + bob * 0.5;
      items.push([r * Z_ROW_STRIDE + 4, p, Math.round(px), Math.round(py)]);
    }

    // 掉血回收宝箱飞行：从玩家身上抛物线(上拱)飞向落点，100ms 完成
    for (let k = flyFx.length - 1; k >= 0; k--) {
      const f = flyFx[k];
      const age = (flyNow - f.t0) / 1000;
      if (age >= 0.1) { flyFx.splice(k, 1); continue; }
      const tt = age / 0.1;
      const px = (f.x0 + (f.x1 - f.x0) * tt) * CELL;
      // 开口向下的抛物线：弧顶在起点与终点中间上方 48px
      const py = (f.y0 + (f.y1 - f.y0) * tt) * CELL - 48 * 4 * tt * (1 - tt);
      // 掉血回收 = 随机宝箱(带?箱子), 原图尺寸居中
      items.push([50000, res.boxQ,
        Math.round(px - res.boxQ.width / 2), Math.round(py - res.boxQ.height / 2)]);
    }

    // 角色：z = 脚所在行；帧底边 = 中心格底边；底线不越地图底
    const chars = [];   // {z, x, y, surf, wudi, wx, wy, hpv, mx}
    let p0Arrow = null;   // 我方箭头位置：进果冻遮挡隐藏后仍要显示
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
      // 走进果冻结构（房子/灌木/拱门等，可炸或不可炸）内 → 角色隐藏，
      // 但控制箭头仍显示（记录位置供箭头绘制）
      const cx = gx * CELL, cy = gy * CELL;
      const row = MOVE_TO_SPRITE_ROW[face[pid]] != null ? MOVE_TO_SPRITE_ROW[face[pid]] : 0;
      const frame = (humanMoveState(pid) ? Math.floor(nowS * 8) % 4 : 0);
      const s = rows[row][frame];
      const blitX = Math.round(cx - s.width / 2);
      // 精灵脚底对齐**碰撞盒底** (gy + radius)，不再是格底 (gy+0.5)：
      // 穿越一格通道时视觉与碰撞一致，不会差 12px
      const blitY = Math.min(Math.round(cy + CFG.radius * CELL - s.height), H * CELL - s.height);
      // 我方箭头位置：**先记录**，进果冻遮挡隐藏后箭头仍显示
      if (pid === 0) p0Arrow = { blitX, blitY, s };
      // 走进果冻结构（房子/灌木/拱门等，可炸或不可炸）内 → 角色隐藏
      if (hideCells.has(Math.floor(gy) * W + Math.floor(gx))) continue;
      let wudi = null, wx = blitX, wy = blitY;
      if (sim.invuln[pid] > 0) {
        wudi = res.wudi;
        // 无敌光晕居中于角色帧中心，向下 20 原生px 再**上移 15** → 净 +5 原生px
        wx = blitX + s.width / 2 - wudi.width / 2;
        wy = blitY + s.height / 2 - wudi.height / 2 + 5 * SCALE;
      }
      // 角色 Z = 行×16+15：同行墙(≤+14)在角色后画(角色在前)；
      // 下一行最后一列墙(+16)在角色后画(墙溢出盖住角色脚部) —— 不产生平级
      const z = Math.floor(gy) * Z_ROW_STRIDE + (Z_ROW_STRIDE - 1);
      items.push([z, s, blitX, blitY]);
      chars.push({ pid, z, blitX, blitY, s, wudi, wx, wy, hpv: sim.hp[pid], mx: CFG.maxHp });
    }

    // 画家算法：z 升序（远→近）绘制
    // 元组形式 [z,img,x,y] / [z,img,sx,sy,sw,sh,dx,dy,dw,dh] 或闭包(结构/die 特效)
    items.sort((a, b) => a[0] - b[0]);
    for (const it of items) {
      if (typeof it[1] === 'function') it[1]();
      else if (it.length >= 6) ctx.drawImage(it[1], it[2], it[3], it[4], it[5], it[6], it[7], it[8], it[9]);
      else ctx.drawImage(it[1], it[2], it[3]);
    }

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
    if (!elSpectate.checked && p0Arrow) {
      // 箭头**恒在血条下方**（血条 = blitY-8 起 4px 高 → 下方即 blitY-4）：
      // 动画帧"视觉头顶"变化不再让箭头跳到血条上方；进果冻遮挡也不消失。
      const aw = res.point.width, ah = res.point.height;
      const top = p0Arrow.s._top || 0;
      const ax = p0Arrow.blitX + p0Arrow.s.width / 2 - aw / 2;
      const ay = Math.max(p0Arrow.blitY - 4, Math.max(2, p0Arrow.blitY + top - ah - 2));
      ctx.drawImage(res.point, Math.round(ax), Math.round(ay));
    }
    // 碰撞包围盒调试（B 键）：红框=0.6格AABB, 黄点=碰撞中心, 青线=脚底(碰撞盒底)
    if (showBox && sim) {
      ctx.lineWidth = 2;
      for (let p = 0; p < 2; p++) {
        if (!sim.alive[p]) continue;
        const gxp = sim.pos[p * 2 + 1], gyp = sim.pos[p * 2];
        const hx = CFG.radius * CELL;
        ctx.strokeStyle = 'rgba(255,70,70,0.95)';
        ctx.strokeRect((gxp - CFG.radius) * CELL, (gyp - CFG.radius) * CELL, hx * 2, hx * 2);
        ctx.fillStyle = 'rgba(255,255,0,0.95)';
        ctx.fillRect(gxp * CELL - 2, gyp * CELL - 2, 4, 4);
        ctx.fillStyle = 'rgba(0,255,255,0.85)';
        ctx.fillRect(gxp * CELL - 7, (gyp + CFG.radius) * CELL - 1, 14, 2);
      }
    }
    ctx.restore();                 // 结束整体下移(translate), HUD 用绝对坐标
    drawHUD();
    // 帧数显示：右上角 11px 黄色小字
    ctx.font = '13px sans-serif';
    ctx.fillStyle = '#ffd700';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(`${fpsNow} fps`, canvas.width - 8, 8);
    // (坐标/偏移调试显示已移除, 只保留 FPS; 数据走 console 日志)
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  // 角色是否在行走（动画帧推进用）
  function humanMoveState(pid) {
    if (pid === 0) {
      return !elSpectate.checked && human.move !== MOVE_IDLE && sim.alive[0];
    }
    return face[1] !== MOVE_IDLE && sim.alive[1];
  }

  function drawHUD() {
    const y0 = BOARD_H + BOARD_OFFSET;
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
    const lvN = sim && sim.level ? `${sim.level.name}${sim.level.mode.includes('空场景') ? '(空场景)' : ''}` : '-';
    ctx.fillText(`地图：${lvN} · 对局 #${gameSeed % 100000}`,
                 BOARD_PX - 18, y0 + 32);
    ctx.fillStyle = '#5a6275';
    ctx.font = '11px sans-serif';
    const em = enemySel && enemySel !== HUNTER_VAL ? modelCache.get(enemySel) : null;
    ctx.fillText(`敌人：${em ? em.meta.name + '（' + fmtStep(em.meta.global_step) + '步）' : p1Kind}`,
                 18, y0 + 78);
  }

  // ------------------------------------------------------------ 主循环
  let prevFrame = 0;
  let fpsFrames = 0, fpsT0 = 0, fpsNow = 0;   // 帧数统计(右上角显示)
  // ---- profiling: 真实帧间隔/渲染耗时/tick耗时/最大帧 ----
  let prof = { frames: 0, t0: 0, sumDt: 0, maxDt: 0, renderMs: 0, tickLast: 0, clipLast: 0 };
  let profAvg = null;
  let lastRenderMs = 0;   // 最近一帧 render 耗时(突变帧日志用)
  function loop(now) {
    // 人类输入 60Hz 采样 + 帧级移动
    if (running && !elSpectate.checked) {
      const dt = Math.min((now - prevFrame) / 1000 || 0, 0.25);
      human.move = sampleHumanMove();
      if (human.move !== MOVE_IDLE && sim.alive[0]) {
        const eff = autoTurn(0, human.move);
        const keepFace = human.move;      // 原始输入方向(自动转向不改行走图朝向)
        frameMove(0, eff, dt);            // 坐标用转向方向滑移
        human.move = eff;                 // 动画播放状态按实际移动
        face[0] = keepFace;               // 朝向保持玩家按的方向
      }
    }
    const frameDt = now - prevFrame;          // 真实帧间隔(ms, rAF 时间戳)
    prevFrame = now;
    fpsFrames++;
    if (human.move === MOVE_IDLE || !sim.alive[0]) turnSlide = -1;  // 松手/死亡: 取消滑动
    if (now - fpsT0 >= 500) { fpsNow = Math.round(fpsFrames * 1000 / (now - fpsT0)); fpsFrames = 0; fpsT0 = now; }
    // profiling 累计
    prof.frames++;
    prof.sumDt += frameDt;
    prof.maxDt = Math.max(prof.maxDt, frameDt);
    const rs = performance.now();
    render(now);
    lastRenderMs = performance.now() - rs;
    prof.renderMs += lastRenderMs;
    // 突变帧(>25ms)即时告警到 console(带最近一帧渲染耗时)
    if (frameDt > 25) {
      console.warn(`[prof] 突变帧 ${frameDt.toFixed(1)}ms @t=${(now/1000).toFixed(1)}s (渲染${lastRenderMs.toFixed(1)}ms 采样${(prof.clipLast||0).toFixed(1)}ms)`);
    }
    if (now - prof.t0 >= 1000) {
      const win = now - prof.t0;
      // 模型推理耗时（每秒累计，读完清零；被 tick 缓存命中的 tick 不产生推理）
      let inferMs = 0;
      for (const m of modelCache.values()) {
        if (m._inferMs) { inferMs += m._inferMs; m._inferMs = 0; }
      }
      profAvg = {
        fps: Math.round(prof.frames * 1000 / win),
        avgDt: prof.sumDt / prof.frames,
        maxDt: prof.maxDt,
        render: prof.renderMs / prof.frames,
        tick: prof.tickLast,
      };
      const profLine = `[prof] ${profAvg.fps}fps 帧均${profAvg.avgDt.toFixed(1)}ms 渲染${profAvg.render.toFixed(1)}ms tick${profAvg.tick.toFixed(1)}ms 采样${(prof.clipLast||0).toFixed(1)}ms 推理${inferMs.toFixed(1)}ms 最大${profAvg.maxDt.toFixed(0)}ms${profAvg.maxDt>25?' ⚠含突变':''}`;
      console.log(profLine);
      // 调试：?profdom=1 时把 [prof] 行写进 document.title（不开 console 也能读帧率/推理）
      if (location.search.includes('profdom=1')) document.title = profLine;
      prof.frames = 0; prof.t0 = now; prof.sumDt = 0; prof.maxDt = 0; prof.renderMs = 0;
    }
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
      if (elRecClip.checked && running && sim && !sim.done) {
        if (!clipC) {
          clipC = document.createElement('canvas');
          clipC.width = Math.max(1, Math.round(canvas.width * CLIP_SCALE));
          clipC.height = Math.max(1, Math.round(canvas.height * CLIP_SCALE));
          clipCtx = clipC.getContext('2d');
        }
        const ct0 = performance.now();
        clipCtx.drawImage(canvas, 0, 0, clipC.width, clipC.height);
        clipFrames.push({ t: now, img: clipCtx.getImageData(0, 0, clipC.width, clipC.height) });
        prof.clipLast = performance.now() - ct0;   // 采样耗时(ms)
      }
      while (clipFrames.length > 2 && now - clipFrames[0].t > CLIP_WINDOW_MS) clipFrames.shift();
    }
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------ 启动
  // 触屏检测：移动端竖屏显示摇杆布局 + 欢迎文案
  const isTouch = () => !!(window.matchMedia &&
    window.matchMedia('(pointer: coarse)').matches);
  // 二级地图菜单（黑屏欢迎界面内）：第一级=模式分类，第二级=具体地图。
  // 不用 HTML label 下拉：开局在黑屏里选两次（分类 → 地图），选中即开。
  // ---- 树状地图菜单：一级=分类节点(可展开)，二级=地图项，同一控件；
  //      左右分栏：左=树，右=缩略图预览（两级共用同一预览位）。
  function showMapList(cat) { /* 已被树状菜单取代 */ }
  function showWelcome() {
    const CAT_ORDER = ['空场景', '比武', '功夫', '中国城', '沙漠', '雪地',
                       '矿洞', '水面', '野外', '夺宝', '推箱子'];
    const catOf = (l) => l.category === '普通竞技' ? (l.theme || '普通') : l.category;
    const groups = new Map();                  // cat -> [levels]
    for (const cat of CAT_ORDER) {
      const maps = cat === '空场景'
        ? levels.filter((l) => l.category === '空场景')
        : levels.filter((l) => catOf(l) === cat);
      if (maps.length) groups.set(cat, maps);
    }
    elBanner.innerHTML =
      `<div class="wl-title">💣 QQT 格斗 — 选择地图</div>` +
      `<div class="mm-tree-wrap">
         <div class="mm-tree-left">
           <button id="mm-random-btn">🎲 随机地图</button>
           <div class="mm-tree" id="mm-tree"></div>
         </div>
         <div class="mm-prev-panel">
           <div class="mm-prev-img"><img id="mm-prev-img" alt=""></div>
           <div class="mm-prev-name" id="mm-prev-name"></div>
           <div class="mm-prev-meta" id="mm-prev-meta"></div>
         </div>
       </div>` +
      (isTouch()
        ? `<span class="tip">左摇杆移动 · 💣 键放泡</span>`
        : `<span class="tip">方向键 / WASD 移动 · 空格 放泡</span>`) +
      `<span class="tip">点击分类展开 → hover 地图看缩略图 → 点击开局；R 重开</span>` +
      `<span class="tip act">点击地图开始</span>`;
    const tree = document.getElementById('mm-tree');
    const prevImg = document.getElementById('mm-prev-img');
    const prevName = document.getElementById('mm-prev-name');
    const prevMeta = document.getElementById('mm-prev-meta');
    const showPrev = (l, extraName, extraMeta, imgSrc) => {
      if (!l && !extraName && !imgSrc) return;
      if (prevImg) prevImg.src = imgSrc || (l && l.thumb) || '';
      if (prevName) prevName.textContent = extraName || (l ? (l.name || l.source) : '');
      const st = l ? (l.initial_stats || {}) : {};
      if (prevMeta) prevMeta.textContent = extraMeta ||
        (l ? `${l.mode} · ${st.bombs}泡/${st.blast}威/${st.speed}速 · ${l.source}` : '');
    };
    // 随机地图：官方 rand 缩略图；hover 展示、点击随机开局
    const rndBtn = document.getElementById('mm-random-btn');
    if (rndBtn) {
      const RAND_IMG = 'assets/maps/thumb/rand.png';
      rndBtn.addEventListener('mouseenter', () =>
        showPrev(null, '随机地图', '全 241 张地图随机抽取', RAND_IMG));
      rndBtn.addEventListener('mouseleave', () =>
        showPrev(null, '随机地图', '全 241 张地图随机抽取', RAND_IMG));
      rndBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        selectedLevel = levels[Math.floor(Math.random() * levels.length)];
        startGame();
      });
      // 官方随机缩略图（rand.png）常驻预览
      if (prevImg) prevImg.src = 'assets/maps/thumb/rand.png';
      if (prevName) prevName.textContent = '随机地图';
      if (prevMeta) prevMeta.textContent = '全 241 张地图随机抽取';
    }
    // 树：分类节点（▸ 可展开）→ 地图项（hover 缩略图 / 点击开局）
    for (const [cat, maps] of groups) {
      const node = document.createElement('div');
      node.className = 'mm-node';
      const head = document.createElement('button');
      head.className = 'mm-cat';
      if (cat === '空场景') {
        // 空场景：单节点，直接展示（不折叠），hover 缩略图，点击开局
        head.textContent = '空场景';
        const emptyLv = maps[0];
        head.addEventListener('mouseenter', () => showPrev(emptyLv));
        head.addEventListener('click', (ev) => {
          ev.stopPropagation();
          selectedLevel = emptyLv;
          startGame();
        });
        node.appendChild(head);
        tree.appendChild(node);
        continue;
      }
      head.textContent = `▸ ${cat}（${maps.length}）`;
      const children = document.createElement('div');
      children.className = 'mm-children';
      children.style.display = 'none';
      head.addEventListener('mouseenter', () => showPrev(maps[0]));
      head.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const open = children.style.display !== 'none';
        children.style.display = open ? 'none' : 'block';
        head.textContent = `${open ? '▸' : '▾'} ${cat}（${maps.length}）`;
      });
      for (const l of maps) {
        const item = document.createElement('button');
        item.className = 'mm-map';
        const st = l.initial_stats || {};
        item.innerHTML =
          `<span class="mm-name">${l.name || l.source}</span>` +
          `<span class="mm-meta">${st.bombs}泡 / ${st.blast}威 / ${st.speed}速</span>`;
        item.addEventListener('mouseenter', () => showPrev(l));
        item.addEventListener('click', (ev) => {
          ev.stopPropagation();
          selectedLevel = l;
          startGame();
        });
        children.appendChild(item);
      }
      node.appendChild(head);
      node.appendChild(children);
      tree.appendChild(node);
    }
    elBanner.classList.remove('hidden');
  }
  function openMapMenu() {
    running = false;
    mapMenuOpen = true;                       // 冻结渲染
    stopBgm();                                // 关音乐
    ctx.fillStyle = '#0c0e13';                // 清掉游戏画面
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    showWelcome();                            // 回黑屏首页
  }

  async function boot() {
    canvas.width = BOARD_PX;                    // 15 列 × 60px = 900
    canvas.height = BOARD_H + BOARD_OFFSET + HUD_PX;
    elLoadingText.textContent = '正在加载… 0%';
    // 模型（开局）与素材（渲染）互不依赖，并行加载
    await Promise.all([loadModelList(), loadAssets()]);
    await new Promise((r) => setTimeout(r, 150));   // 进度条缓动走完最后一段再切
    // 视频录制不再常驻：由「录制动图」开关控制（默认关 = 零开销）
    loadPhase = '';
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
