// main.js —— 浏览器端：原版素材渲染 + 游戏主循环 + 输入 + 模型加载。
//
// 渲染移植自 play/duel.py::draw_grid + play/res.py：同一套 res/ 素材（角色
// 4×4 精灵图、炸弹序列帧、爆炸臂切片、场景砖块/背景、道具图、无敌罩），
// 同一套画家算法（z = 所在行，远→近绘制）与底边对齐锚点。
// 玩法与 play/duel.py 对齐：人类 60Hz 帧级移动（自动转向 + AABB 滑动碰撞），
// AI 决策与模拟推进走 10Hz；AI 用导出权重（已折 pid=0 视角）+ 合法动作掩码采样。
//
// 逻辑节拍用 setInterval(100ms) 驱动（rAF 在标签页后台会被浏览器节流停发，
// 用定时器保证对局不受影响）；rAF 只做输入采样 + 渲染。

'use strict';

(() => {
  const Q = window.QQT;
  const { Sim, MLPModel, CNNModel, TransformerModel, ORTTransformerModel, CFG, DIRS, EPS, MOVE_IDLE, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP } = Q;

  const H = Q.H, W = Q.W, N = Q.N;
  const CELL = 60;                 // 与 play/duel.py 一致：素材原生 40px/格 × 1.5
  const RADIUS_MIN = 0.20, RADIUS_MAX = 0.49;
  function setRadiusLabel() {
    const r = Number(elRadius.value);
    elRadiusValue.textContent = `${r.toFixed(2)} 格（${Math.round(r * CELL * 2)}px）`;
  }
  function applyRadius() {
    const r = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, Number(elRadius.value)));
    CFG.radius = Number.isFinite(r) ? r : 0.42;
    elRadius.value = CFG.radius.toFixed(2);
    setRadiusLabel();
  }
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
        elMousePath = $('mouse-path'),
        elRadius = $('radius'), elRadiusValue = $('radius-value'),
        elRestart = $('restart'), elStatus = $('status'), elBanner = $('banner'),
        elLoading = $('loading'), elLoadingText = $('loading-text'),
        elSaveReplay = $('save-replay'), elSaveGif = $('save-gif'), elRecClip = $('rec-clip'),
        elSaveVideo = $('save-video'), elRecMsg = $('rec-msg'),
        elModelLowfreq = $('model-lowfreq'),
        elP0WinFill = $('p0-win-fill'), elP1WinFill = $('p1-win-fill'),
        elP0WinPct = $('p0-win-pct'), elP1WinPct = $('p1-win-pct'),
        elP0WinName = $('p0-win-name'), elP1WinName = $('p1-win-name');

  function mouseGridCell(e) {
    const rect = canvas.getBoundingClientRect();
    // 映射回 canvas 原生像素坐标（clientLeft/Top = CSS 边框宽，rect 含边框而绘图区不含）
    const mx = (e.clientX - rect.left - canvas.clientLeft) * (canvas.width / (rect.width - 2 * canvas.clientLeft));
    const my = (e.clientY - rect.top - canvas.clientTop) * (canvas.height / (rect.height - 2 * canvas.clientTop)) - BOARD_OFFSET;
    const gc = Math.floor(mx / CELL), gr = Math.floor(my / CELL);
    return gr >= 0 && gr < H && gc >= 0 && gc < W ? { r: gr, c: gc } : null;
  }

  function cardinalDestinationLegalAt(y, x, dir) {
    const pr = Math.floor(y), pc = Math.floor(x);
    const tr = pr + DY[dir], tc = pc + DX[dir];
    if (tr < 0 || tr >= H || tc < 0 || tc >= W) return false;
    const bi = tr * W + tc;
    return !(sim.wall[bi] || (sim.brick[bi] && !sim.pushable[bi]) || sim.fuse[bi] > 0);
  }

  function mouseDestination(cell, search = true) {
    if (!cell || !sim || !sim.alive[0]) return -1;
    const pr = Math.floor(sim.pos[0]), pc = Math.floor(sim.pos[1]);
    const dr = cell.r - pr, dc = cell.c - pc;
    if (Math.abs(dr) > 1 || Math.abs(dc) > 1) return -1;
    // 直接点击四邻可推箱时，必须返回朝箱子的动作；不能用下面的距离试算，
    // 因为顶箱前两 tick 坐标不变，却正在有效累计 0.3s 推动时间。
    if (Math.abs(dr) + Math.abs(dc) === 1) {
      const direct = dr < 0 ? MOVE_UP : dr > 0 ? MOVE_DOWN : dc < 0 ? MOVE_LEFT : MOVE_RIGHT;
      if (sim.pushable[cell.r * W + cell.c]) return direct;
      if (cardinalDestinationLegalAt(sim.pos[0], sim.pos[1], direct)) return direct;
    }
    // hover 只负责显示九宫格，不需要预测真实下一步；保持 O(1)，避免鼠标
    // 移动时反复创建 blocked 数组和调用碰撞逻辑。
    if (!search) return MOVE_IDLE;
    const y = sim.pos[0], x = sim.pos[1];
    const ty = cell.r + 0.5, tx = cell.c + 0.5;
    const blocked = new Uint8Array(N);
    for (let i = 0; i < N; i++) blocked[i] = sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 ? 1 : 0;
    const dist = CFG.stepLen * sim.spdG[0];
    const score = (py, px) => (py - ty) ** 2 + (px - tx) ** 2;
    const before = score(y, x);
    let bestDir = MOVE_IDLE, bestScore = before;

    // 一步贪心会在大型碰撞体边缘形成局部死点：正确动作可能要先横向/纵向
    // 对齐，第一步距离不降，第二步才进入黄色格。用真实 _steer 做 4 tick
    // 小范围试算（最多 4^4=256 个节点），选择最终最近路径的第一步。
    // 每次点击仍只执行第一步，不把网页控制器变成自动寻路。
    let frontier = [{ y, x, first: MOVE_IDLE }];
    for (let depth = 0; depth < 4; depth++) {
      const next = [];
      for (const node of frontier) {
        for (let dir = 0; dir < 4; dir++) {
          if (!cardinalDestinationLegalAt(node.y, node.x, dir)) continue;
          const [ny, nx] = sim._steer(node.y, node.x, dir, blocked, dist);
          if (Math.abs(ny - node.y) + Math.abs(nx - node.x) <= 2 * EPS) continue;
          const first = node.first === MOVE_IDLE ? dir : node.first;
          const after = score(ny, nx);
          if (after < bestScore - 1e-8) {
            bestScore = after;
            bestDir = first;
          }
          next.push({ y: ny, x: nx, first });
        }
      }
      frontier = next;
      if (!frontier.length) break;
    }
    return bestDir;
  }

  // 鼠标寻路：hover 玩家周围 3×3（含中心），click 映射到上下左右/停留
  canvas.addEventListener('mousemove', (e) => {
    if (!elMousePath.checked) { hoverCell = null; hoverDir = -1; return; }
    if (!sim || !running) { hoverCell = null; hoverDir = -1; return; }
    const cell = mouseGridCell(e);
    hoverDir = mouseDestination(cell, false);
    hoverCell = hoverDir >= 0 ? cell : null;
  });
  canvas.addEventListener('mouseleave', () => { hoverCell = null; hoverDir = -1; });
  if (elMousePath) elMousePath.addEventListener('change', () => {
    if (elMousePath.checked) return;
    hoverCell = null;
    hoverDir = -1;
    mousePush = null;
  });
  canvas.addEventListener('click', (e) => {
    if (!elMousePath.checked) return;
    if (!sim || !running || elSpectate.checked) return;
    // 角色可能已在上次 mousemove 后到达高亮格；点击时必须按当前位置重算。
    // 点当前格会继续向该格中心归位；已经居中或没有改善动作才映射为 IDLE。
    const cell = mouseGridCell(e);
    const searchT0 = performance.now();
    const dir = mouseDestination(cell);
    prof.mouseSearchLast = performance.now() - searchT0;
    if (dir < 0 || !sim.alive[0]) {
      hoverCell = null;
      hoverDir = -1;
      return;
    }
    if (dir === MOVE_IDLE) return;
    const pr = Math.floor(sim.pos[0]), pc = Math.floor(sim.pos[1]);
    const tr = pr + DY[dir], tc = pc + DX[dir];
    if (tr >= 0 && tr < H && tc >= 0 && tc < W && sim.pushable[tr * W + tc]) {
      // 鼠标点击本来只执行一个 tick；推箱需要同方向持续 ≥0.3s。为人工验证
      // 自动保持稍长于阈值的一小段时间，实际推动仍走 frameMove 的原版逻辑。
      mousePush = { dir, until: performance.now() + 420 };
      return;
    }
    const y = sim.pos[0], x = sim.pos[1];
    const blocked = new Uint8Array(N);
    for (let i = 0; i < N; i++) blocked[i] = sim.wall[i] || sim.brick[i] || sim.fuse[i] > 0 ? 1 : 0;
    const dist = CFG.stepLen * sim.spdG[0];
    const [ny, nx] = sim._steer(y, x, dir, blocked, dist);
    face[0] = dir;
    sim.pos[0] = Math.min(Math.max(ny, CFG.radius), H - CFG.radius);
    sim.pos[1] = Math.min(Math.max(nx, CFG.radius), W - CFG.radius);
  });

  // ------------------------------------------------------------ 状态
  let sim = null, modelList = [], res = null;
  let replayExporting = false;
  let replayWinProb = null;
  let currentWinProb = 0.5;
  let replayAnim = null;   // 视频导出中非 null：{moving:[pid0,pid1]} —— 行走动画按导出帧间位移驱动
  let rng = null;
  // 鼠标寻路：hover 周围九宫格，click 映射到四方向 Destination / IDLE
  let hoverCell = null;       // {r, c} 或 null
  let hoverDir = -1;          // 0-4 对应上下左右/停留，-1=无
  let mousePush = null;       // 点击可推箱后的短时持续方向输入
  // 新地图系统: 241 张原版关卡 (levels.json) + 元素属性表 (elements.json)
  let levels = [], levelById = new Map(), elements = {};
  let selectedLevel = null;         // 黑屏菜单选中的关卡对象
  let customStats = null;           // 选图页覆盖属性：{bombs,blast,speed,bombsMax,blastMax,speedMax}
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
  // 飞鸟道具空投抛物线动画: [{sx, sy, tx, ty, cell, item, t0, dur}]
  let birdDropFx = [];
  let birdLastCycle = -1;
  let birdCruiseQueue = [];
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
  let mediaRec = null, mediaMime = '', mediaChunks = [], mediaStopPromise = null;
  function stopVideoRecorder() {
    const rec = mediaRec;
    if (!rec || rec.state === 'inactive') return Promise.resolve();
    if (mediaStopPromise) return mediaStopPromise;
    mediaStopPromise = new Promise((resolve) => {
      rec.addEventListener('stop', resolve, { once: true });
      try { rec.stop(); } catch (e) { resolve(); }
    }).finally(() => { mediaStopPromise = null; });
    return mediaStopPromise;
  }
  function flushVideoRecorder() {
    const rec = mediaRec;
    if (!rec || rec.state !== 'recording') return Promise.resolve();
    return new Promise((resolve, reject) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        rec.removeEventListener('dataavailable', finish);
        resolve();
      };
      const timer = setTimeout(finish, 1500);
      rec.addEventListener('dataavailable', finish, { once: true });
      try { rec.requestData(); } catch (e) {
        clearTimeout(timer);
        rec.removeEventListener('dataavailable', finish);
        reject(e);
      }
    });
  }
  function startVideoRecorder() {
    try {
      if (!canvas.captureStream || !window.MediaRecorder) return;
      const mime = [
        'video/mp4;codecs=avc1.42E01E', 'video/mp4',
        'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm',
      ].find((t) => MediaRecorder.isTypeSupported(t));
      if (!mime) return;
      const stream = canvas.captureStream(20);
      mediaRec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 800000 });
      mediaMime = mime;
      mediaChunks = [];
      mediaStopPromise = null;
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
      mediaMime = '';
      console.warn('视频录制不可用:', e);
    }
  }
  let replay = null;          // { meta, actions: [[m0,b0,m1,b1], ...], snapshots: [...] }
  const face = [MOVE_DOWN, MOVE_DOWN];
  let lastAiMove = [MOVE_IDLE, MOVE_IDLE];
  const human = { dirStack: [], latch: new Set(), move: MOVE_IDLE, pendingBomb: false };
  let joyBombDown = false;   // 摇杆放泡按钮按住状态(tick 判断锁存清除用)
  let spaceDownSince = 0, joyDownSince = 0;   // 按下时刻: 长按>180ms 才连放, 点按=1颗
  const hunter = new Q.HunterAI();   // 规则 AI（纯进攻寻路），可当敌/我方
  const HUNTER_VAL = '__hunter__';   // 下拉里规则 AI 的 value 哨兵
  const IDLE_VAL = '__idle__';      // 静止敌人(不动不炸)哨兵
  const LATEST_VIT = 'ViTModel2_31.9B';       // 最新 ViT 模型(默认敌人)

  // 敌/我方 AI 选择：'__hunter__'（规则）或模型名。模型按需懒加载到缓存。
  // 敌人默认 = 列表第一个（ELO 最高）；观战我方默认 = 同样的最强模型。
  let enemySel = null, p0Sel = null;
  const modelCache = new Map();      // name → MLPModel/CNNModel/TransformerModel/ORT…（懒加载缓存）

  // 流式下载辅助函数：支持 ReadableStream 字节进度汇报与超时熔断保护
  async function fetchWithProgress(url, onProgress, timeoutMs = 60000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error(`网络请求超时 (${timeoutMs / 1000}s)`)), timeoutMs);
    try {
      const resp = await fetch(url, { signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status} (${resp.statusText})`);
      const contentLength = resp.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;
      if (!resp.body || typeof resp.body.getReader !== 'function') {
        const buf = await resp.arrayBuffer();
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return buf;
      }
      const reader = resp.body.getReader();
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (onProgress) onProgress(loaded, total);
      }
      const all = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) {
        all.set(c, off);
        off += c.length;
      }
      return all.buffer;
    } finally {
      clearTimeout(timer);
    }
  }

  // 会话创建超时保护
  async function createOrtSessionWithTimeout(buffer, providers, timeoutMs = 10000) {
    let timer;
    const createP = ort.InferenceSession.create(buffer, { executionProviders: providers });
    const timeoutP = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`ORT 会话初始化超时 (${timeoutMs}ms)`)), timeoutMs);
    });
    try {
      return await Promise.race([createP, timeoutP]);
    } finally {
      clearTimeout(timer);
    }
  }

  // transformer 模型优先用 onnxruntime（WebGPU→WASM），失败回退纯 JS
  async function makeOrtModel(name, meta) {
    const ort = window.ort;
    if (!ort || !ort.InferenceSession) return null;
    ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;  // 动态 import 需要绝对 URL
    // COOP/COEP 头存在(crossOriginIsolated) → WASM 可用多线程, 按核数设
    ort.env.wasm.numThreads = (typeof crossOriginIsolated === 'boolean' && crossOriginIsolated)
      ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

    // 1. 流式下载 ONNX 权重并展示字节级进度
    const buffer = await fetchWithProgress(`models/${name}.onnx`, (loaded, total) => {
      const pct = total ? Math.round((loaded / total) * 100) : 0;
      const mbLoaded = (loaded / (1024 * 1024)).toFixed(1);
      const mbTotal = total ? (total / (1024 * 1024)).toFixed(1) : '?';
      const msg = `下载模型 ${mbLoaded}/${mbTotal}MB (${pct}%)`;
      if (elCurModel) elCurModel.textContent = `⏳ ${msg}`;
      if (elStatus) elStatus.innerHTML = `正在下载模型权重：<b>${msg}</b>`;
      loadPhase = msg;
      requestAnimationFrame(updateProgress);
    });

    // 2. 初始化推理引擎（优先 WebGPU，超时 8s 自动切 WASM CPU，杜绝黑屏死等）
    loadPhase = `正在初始化推理引擎（${name}）`;
    if (elCurModel) elCurModel.textContent = '⚙️ 正在初始化推理引擎...';
    requestAnimationFrame(updateProgress);

    const hasGpu = typeof navigator !== 'undefined' && !!navigator.gpu;
    let session = null;
    if (hasGpu) {
      try {
        session = await createOrtSessionWithTimeout(buffer, ['webgpu', 'wasm'], 8000);
      } catch (err) {
        console.warn('[ort] WebGPU 初始化超时或失败，平滑降级至 WASM：', err);
      }
    }
    if (!session) {
      session = await createOrtSessionWithTimeout(buffer, ['wasm'], 15000);
    }

    loadPhase = '';
    return new ORTTransformerModel({ meta }, session);
  }

  async function ensureModel(name) {
    let m = modelCache.get(name);
    if (m) return m;

    // 获取元数据（优先从已有的 modelList 取，避免为读元数据下载数十兆 JSON）
    const meta = modelList.find(item => item.name === name) || { name, arch: 'transformer' };

    // 优先路径：若是 transformer 且支持 ORT，直接流式下载 ONNX，不下载与解码 38MB 的纯 JS JSON
    if (meta.arch === 'transformer' && typeof window.ort !== 'undefined') {
      try {
        m = await makeOrtModel(name, meta);
      } catch (e) {
        console.warn('[ort] ONNX 路径失败，将尝试纯 JS JSON 回退：', e);
      }
    }

    // 回退路径：若 ORT 失败或属于 CNN/MLP 模型，下载 JSON 权重
    if (!m) {
      loadPhase = `正在下载模型 JSON（${name}）`;
      if (elCurModel) elCurModel.textContent = `⏳ 正在下载 JSON…`;
      requestAnimationFrame(updateProgress);
      const resp = await fetch(`models/${name}.json`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const doc = await resp.json();
      if (doc.meta.arch === 'transformer') {
        m = new TransformerModel(doc);
      } else {
        m = doc.meta.arch === 'cnn' ? new CNNModel(doc) : new MLPModel(doc);
      }
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
      catch (e) {
        // 推理失败不能伪装成正常 IDLE；记录首个错误供状态栏和调试钩子定位。
        if (!m._lastInferError) m._lastInferError = String(e && e.message ? e.message : e);
        console.error('[ai] 推理失败', sel, e);
        return [MOVE_IDLE, 0];
      }
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
    // 炸弹序列帧：每组 4 帧，1s 均匀播放（每帧 0.25s），3s 引信循环三组。
    // 敌方固定 default；我方在每局开始时从红/蓝 custom 随机选一组并固定。
    async function loadBombFrames(dir, names) {
      return Promise.all(names.map(async (name) => {
        const src = await loadImage(`assets/${dir}/${name}`);
        // 保留原始帧尺寸 × SCALE（40px 原图 → 60px 格子）：42px 高帧
        // 应渲染为 63px，允许像人物一样从格子顶部上溢，底边仍贴格底。
        // 先裁掉透明边距，避免蓝色冰泡泡因原图留白看起来偏小。
        let sx = 0, sy = 0, sw = src.width, sh = src.height;
        try {
          const probe = document.createElement('canvas');
          probe.width = src.width; probe.height = src.height;
          const pg = probe.getContext('2d');
          pg.drawImage(src, 0, 0);
          const d = pg.getImageData(0, 0, src.width, src.height).data;
          let x0 = src.width, y0 = src.height, x1 = -1, y1 = -1;
          for (let y = 0; y < src.height; y++) for (let x = 0; x < src.width; x++) {
            if (d[(y * src.width + x) * 4 + 3] > 8) {
              if (x < x0) x0 = x; if (x > x1) x1 = x;
              if (y < y0) y0 = y; if (y > y1) y1 = y;
            }
          }
          if (x1 >= x0 && y1 >= y0) {
            sx = x0; sy = y0; sw = x1 - x0 + 1; sh = y1 - y0 + 1;
          }
        } catch (e) { /* 透明边界探测失败时使用原图 */ }
        const w = Math.max(1, Math.round(sw * SCALE));
        const h = Math.max(1, Math.round(sh * SCALE));
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        c.getContext('2d').drawImage(src, sx, sy, sw, sh, 0, 0, w, h);
        return c;
      }));
    }
    const bombNames = ['bomb1_stand_0_0.png', 'bomb1_stand_0_1.png',
                       'bomb1_stand_0_2.png', 'bomb1_stand_0_3.png'];
    const customNames = [
      ['red_01.png', 'red_02.png', 'red_03.png', 'red_04.png'],
      ['blue_01.png', 'blue_02.png', 'blue_03.png', 'blue_04.png'],
    ];
    const bombDefaultFrames = await loadBombFrames('bomb-default', bombNames);
    const bombCustomFrames = await Promise.all(
      customNames.map((names) => loadBombFrames('bomb-custom', names)));
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
    // 爆炸水泡序列帧素材 (res/flame):
    // C: 中心格 1..2 (0 为透明占位)
    // U, D, L, R: 四方向格 1..6 (1=炸开花边缘, 2-4=中间连续动画帧, 5=伸展边缘, 6=收尾边缘)
    // 统一预生成 CELL × CELL (60×60) 贴图，按物理朝向边缘对齐
    const flameFrames = { C: [], U: [], D: [], L: [], R: [] };
    for (const f of [1, 2]) {
      const img = await loadImage(`assets/flame/flame_C_${f}.png`);
      const c = document.createElement('canvas');
      c.width = CELL; c.height = CELL;
      c.getContext('2d').drawImage(img, 0, 0, CELL, CELL);
      flameFrames.C[f] = c;
    }
    for (const d of ['U', 'D', 'L', 'R']) {
      flameFrames[d] = [];
      for (let f = 1; f <= 6; f++) {
        const img = await loadImage(`assets/flame/flame_${d}_${f}.png`);
        const sw = Math.round(img.width * SCALE);
        const sh = Math.round(img.height * SCALE);
        const c = document.createElement('canvas');
        c.width = CELL; c.height = CELL;
        const g = c.getContext('2d');
        let ox = 0, oy = 0;
        if (d === 'U') oy = CELL - sh;
        else if (d === 'D') oy = 0;
        else if (d === 'L') ox = CELL - sw;
        else if (d === 'R') ox = 0;
        g.drawImage(img, 0, 0, img.width, img.height, ox, oy, sw, sh);
        flameFrames[d][f] = c;
      }
    }
    const rawBird1 = await loadImage('assets/bird1.png');
    const rawBird2 = await loadImage('assets/bird2.png');
    // 飞鸟按 1.5 倍放大展示（128×96 -> 192×144）
    const birdFrames = [
      scaleCanvas(rawBird1, Math.round(rawBird1.width * 1.5), Math.round(rawBird1.height * 1.5)),
      scaleCanvas(rawBird2, Math.round(rawBird2.width * 1.5), Math.round(rawBird2.height * 1.5)),
    ];
    res = {
      levels, levelById, elements, bgImages,
      skins: skinRows,             // 3 种玩家皮肤
      players: skinRows[elSkin.value],   // 当前玩家皮肤（切换后重绑）
      enemyRows,                   // 敌人固定角色c（不再染红）
      playerAi: enemyRows,
      wudi: scaleCanvas(wudi, Math.round(85 * SCALE), Math.round(85 * SCALE)),
      bombFrames: { default: bombDefaultFrames, custom: bombCustomFrames },
      propIcons, superIcons, boxQ, baseBand,
      point: scaleCanvas(await loadImage('assets/point.png'),
                         Math.round(40 * SCALE * 0.5), Math.round(40 * SCALE * 0.5)),
      flames: flameFrames,
      birdFrames,
    };
    // 音效（Web Audio；失败静默）
    try {
      res.audio = new AudioContext();
      res.snd = {};
      const names = { place: '放炮.wav', boom: '爆炸.wav', pickup: '吃道具音效.wav',
                      hurt: '生命损失音效.wav', die: '角色消失音效.wav',
                    pushbox: '推箱.wav' };
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
  // 不能用 EPS 判断"可走": 贴墙时 resolveAxis 的 stopPos 含 +EPS, 会误报能走。
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
  let turnSlideTarget = null;            // {axis:'x'|'y', value}: 缺口格中心线，防高速越线反向
  function clearTurnSlide() {
    turnSlide = -1;
    turnSlideTarget = null;
  }
  function autoTurn(pid, move) {
    const stepLen = CFG.stepLen;
    const moved = move >= 4 ? stepLen : probeMoveDist(pid, move);
    // 完全可走(盒子能走满一步) → 正常移动, 取消滑动
    if (move >= 4 || moved >= stepLen * 0.95) { clearTurnSlide(); return move; }
    if (move !== turnInput) clearTurnSlide();  // 输入方向变了: 取消旧滑动
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
      // 已承诺滑动：不再根据瞬时缺口判定反向。高速/大半径可能一帧跨过
      // 判定边界；方向重算会左→右→左循环。只朝启动时记录的中心线移动，
      // frameMove 后钳制到该线；侧向本身被完全挡住才放弃。
      if (probeMoveDist(pid, turnSlide) > stepLen * 0.05) return turnSlide;
      clearTurnSlide();
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
    if (dir2 === -1 || !okSlide) { clearTurnSlide(); return move; } // 不横跨/滑不动: 不转
    if (turnSlide !== -1) return turnSlide;                          // 已承诺: 继续滑到底
    if (off >= MIN_OFF) return move;                                 // 偏移不够近0: 不触发
    turnSlide = dir2;
    if (move === MOVE_UP || move === MOVE_DOWN) {
      const targetCol = dir2 === MOVE_LEFT ? Math.floor(x - R) : Math.floor(x + R);
      turnSlideTarget = { axis: 'x', value: targetCol + 0.5 };
    } else {
      const targetRow = dir2 === MOVE_UP ? Math.floor(y - R) : Math.floor(y + R);
      turnSlideTarget = { axis: 'y', value: targetRow + 0.5 };
    }
    return dir2;
  }

  function frameMove(pid, mv, dt) {
    if (mv === MOVE_IDLE || !sim.alive[pid]) return;
    const dist = CFG.speed * sim.spdG[pid] * Math.min(dt, 0.1);
    if (dist <= 0) return;
    const y = sim.pos[pid * 2], x = sim.pos[pid * 2 + 1];
    const [dy, dx] = DIRS[mv];
    // 推箱子(人类60Hz): 前缘顶着可推箱 → 累计推动时间, ≥0.3s 后箱子移一格
    // (与 sim.js step 同逻辑; 先于 blockedGrid 执行, 移走后本帧即可前进)
    if (sim.pushBoxAt && (dy !== 0 || dx !== 0)) {
      const R = CFG.radius;
      const pr = dy !== 0 ? (dy > 0 ? Math.floor(y + R + EPS * 8) : Math.floor(y - R - EPS * 8)) : Math.floor(y);
      const pc = dx !== 0 ? (dx > 0 ? Math.floor(x + R + EPS * 8) : Math.floor(x - R - EPS * 8)) : Math.floor(x);
      const pi = pr * W + pc;
      const bi = pi >= 0 && pi < N ? sim.pushBoxAt[pi] : -1;
      if (bi >= 0) {
        const box = sim.pushBoxes[bi];
        let ok = true;
        const targetCells = [];
        for (const cell of box.cells) {
          const rr = (cell / W) | 0, cc = cell % W;
          const tr = rr + dy, tc = cc + dx;
          if (tr < 0 || tr >= H || tc < 0 || tc >= W) { ok = false; break; }
          const ti = tr * W + tc;
          if (sim.wall[ti] || sim.brick[ti] || sim.fuse[ti] > 0 || sim.crate[ti] || sim.pushable[ti]) { ok = false; break; }
          targetCells.push(ti);
        }
        if (ok) {
          sim.pushT[box.o] += dt;
          if (sim.pushT[box.o] >= 0.3) {
            for (let k = 0; k < box.cells.length; k++) {
              const ci = box.cells[k], ti = targetCells[k];
              sim.brick[ci] = 0; sim.brick[ti] = 1;
              sim.pushable[ci] = 0; sim.pushable[ti] = 1;
              sim.pushBoxAt[ci] = -1; sim.pushBoxAt[ti] = bi;
              sim.pushSprite[ti] = sim.pushSprite[ci]; sim.pushSprite[ci] = -1;
            }
            box.cells = targetCells;
            box.o = targetCells[0];
            sim.pushT[box.o] = 0;
            playSnd('pushbox', 0.6);   // 推箱音效
          }
        } else {
          sim.pushT[box.o] = 0;   // 推不动 → 重置
        }
      }
    }
    const blocked = blockedGrid();
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
    if (replayExporting) return;   // 视频导出中：换全局 sim 会打断逐帧渲染，忽略 R/重新开局
    if (!res || !selectedLevel) return;      // 素材/地图未就绪由 logicTick 兜底等待
    gameSeed = (Math.random() * 0xFFFFFFFF) >>> 0;
    sim = new Sim(gameSeed);
    sim._manualBird = true;                    // 前端接管飞鸟与空投抛物线动画
    birdDropFx = [];
    birdCruiseQueue = [];
    birdLastCycle = -1;
    window.__sim = sim;                        // 调试钩子：读 sim 状态/帧率用
    sim.reset(selectedLevel, { oldMode: oldModeActive() });  // 旧模型: 13/14列填墙+13宽观测
    if (customStats) {
      const bombsMax = Math.max(1, customStats.bombsMax | 0);
      const blastMax = Math.max(1, customStats.blastMax | 0);
      const speedMax = Math.max(0.1, Number(customStats.speedMax));
      const bombs = Math.min(bombsMax, Math.max(1, customStats.bombs | 0));
      const blast = Math.min(blastMax, Math.max(1, customStats.blast | 0));
      const speed = Math.min(speedMax, Math.max(0.1, Number(customStats.speed)));
      sim.bombsMax = bombsMax; sim.blastMax = blastMax; sim.speedMax = speedMax;
      for (let p = 0; p < 2; p++) {
        sim.bombsCap[p] = bombs; sim.blastCap[p] = blast; sim.spdG[p] = speed;
        sim.loBombs[p] = bombs; sim.loBlast[p] = blast; sim.loSpeed[p] = speed;
      }
    }
    preloadLevelImages(selectedLevel);       // 预取本图元件贴图
    prevCovered = new Set();                 // 清空结构覆盖/进入动画状态
    structAnim.clear();
    rng = Q.mulberry32(gameSeed ^ 0x13579BDF);
    human.dirStack = []; human.latch.clear(); human.move = MOVE_IDLE; human.pendingBomb = false;
    mousePush = null;
    turnInput = -1;
    clearTurnSlide();
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
        initial: customStats ? Object.assign({}, customStats) : selectedLevel.initial_stats,
        skin: elSkin.value,
        spectate: elSpectate.checked,
        p0: elSpectate.checked ? p0Sel : 'human',
        p1: enemySel,
        cfg: Object.assign({}, CFG),   // CFG 快照：重放/分析时不依赖当前版本常量
        levelId: selectedLevel.id,
        oldMode: !!sim.oldMode,
        tickHz: CFG.tickHz,
        replayStateVersion: 2,
      },
      actions: [],
      snapshots: [],
      // 状态帧恒录（实测 ~7.6KB/帧 JSON、56µs/tick 含序列化，开销可忽略）：
      // 「保存视频」永远有米，不再依赖剪片开关。
      frames: [sim.snapshotReplay()],
      // 60Hz 帧级位置轨迹 [t, p0y, p0x, p1y, p1x] —— 人类的移动发生在渲染帧
      // （frameMove 直接改 sim.pos），10Hz step 动作记录不到，只体现在这里。
      framePos: [],
    };
    clipFrames = [];
    lastClipCap = 0;
    gameEndT = 0;
    prevPos.set(sim.pos); curPos.set(sim.pos);
    face[0] = MOVE_DOWN; face[1] = MOVE_DOWN;
    lastAiMove = [MOVE_IDLE, MOVE_IDLE];
    lastTickT = performance.now();
    fpsFrames = 0; fpsT0 = 0; fpsNow = 0;
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

  function modelDisplayName(meta) {
    return meta.display_name || meta.name;
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
      opt.textContent = `${modelDisplayName(m)}  · ${fmtStep(gstep)}步${elo} · 导出于 ${(m.generated_at || '').slice(0, 10)}`;
      sel.appendChild(opt);
    }
  }

  async function loadModelList() {
    const resp = await fetch('models/index.json');
    modelList = (await resp.json()).models || [];
    // 按时间倒序排列（最新导出的模型排在最前）
    modelList.sort((a, b) => {
      const parseTime = (str) => {
        if (!str) return 0;
        const norm = str.replace(' UTC', 'Z').replace(' ', 'T');
        const t = Date.parse(norm);
        return isNaN(t) ? 0 : t;
      };
      const ta = parseTime(a.generated_at);
      const tb = parseTime(b.generated_at);
      if (tb !== ta && ta > 0 && tb > 0) return tb - ta;
      const sa = a.global_step != null ? a.global_step : (a.it || 0);
      const sb = b.global_step != null ? b.global_step : (b.it || 0);
      return sb - sa;
    });
    // 敌人 AI 下拉 + 观战「我方：」下拉：都列全部模型 + 规则 Hunter
    fillAiSelect(elEnemyAi, true);
    fillAiSelect(elP0Ai, true);
    elEnemyAi.value = modelList[0] ? modelList[0].name : LATEST_VIT;
    elP0Ai.value = HUNTER_VAL;                // 观战「我方：」默认规则 Hunter
    p0Sel = elP0Ai.value;
    try {
      await applyModel();            // 预加载默认敌人模型（我方默认同款，已入缓存）
    } catch (e) {
      console.warn('默认模型加载异常，降级至规则 Hunter：', e);
      elEnemyAi.value = HUNTER_VAL;
      await applyModel();
    }
  }

  let isApplyingModel = false;
  // 应用选中的 AI（敌人）：模型名 → 加载权重；规则 Hunter → 无需权重
  async function applyModel() {
    if (isApplyingModel) return;
    const sel = elEnemyAi.value;
    if (!sel) return;
    if (sel === HUNTER_VAL) {
      enemySel = HUNTER_VAL;
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent = '规则 Hunter（纯进攻寻路）';
      elStatus.innerHTML = '敌人：<b>规则 Hunter</b>（纯进攻寻路 AI，无需模型权重）';
      return;
    }
    if (sel === IDLE_VAL) {
      enemySel = IDLE_VAL;
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent = '静止（不动不炸）';
      elStatus.innerHTML = '敌人：<b>静止</b>（不动不炸）';
      return;
    }

    isApplyingModel = true;
    if (elApplyModel) elApplyModel.disabled = true;
    if (elCurModel) elCurModel.textContent = `⏳ 正在连接下载 ${sel}…`;
    elStatus.innerHTML = `正在加载模型 <b>${sel}</b>…`;

    try {
      const m = await ensureModel(sel);
      enemySel = sel;
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent =
        `${modelDisplayName(m.meta)}（${fmtStep(m.meta.global_step ?? m.meta.it ?? 0)}步 · 导出于 ${(m.meta.generated_at || '').slice(0, 10)}）`;
      const numParams = m.tensors && Object.keys(m.tensors).length > 0
        ? Object.values(m.tensors).reduce((s, [, n]) => s + n, 0) : 7500000;
      elStatus.innerHTML =
        `当前模型：<b>${modelDisplayName(m.meta)}</b><br>` +
        `训练步数 ${fmtStep(m.meta.global_step ?? m.meta.it ?? 0)}<br>` +
        `观测 ${m.meta && m.meta.obs_shape ? m.meta.obs_shape.join('×') : '14×13×15'} · 参数约 ${numParams.toLocaleString()}<br>` +
        `推理后端：${m.constructor.name === 'ORTTransformerModel'
          ? (navigator.gpu ? 'WebGPU' : 'WASM') : '纯 JS'}` +
        (m._ortError ? `<br><span class="dim">ORT 失败：${m._ortError.slice(0, 120)}</span>` : '') +
        (m._lastInferError ? `<br><span class="dim">推理失败：${m._lastInferError.slice(0, 160)}</span>` : '');
    } catch (e) {
      modelLoaded = true;
      requestAnimationFrame(updateProgress);
      elCurModel.textContent = '❌ 加载失败（点击「应用」重试）';
      elStatus.innerHTML = `模型加载失败：${e.message}。请检查网络后点击「应用」重试。`;
      console.error('[model] 加载失败:', e);
    } finally {
      isApplyingModel = false;
      if (elApplyModel) elApplyModel.disabled = false;
    }
  }

  elRestart.addEventListener('click', startGame);
  // 黑屏横幅不再"点击任意位置开局"：选图菜单就挂在横幅里，点分类/滑块/空白
  // 都会冒泡到这里误开局 —— 开局只认选图菜单的「点击进入」按钮（mm-enter-btn）。
  // if (elBanner) {
  //   elBanner.addEventListener('click', () => {
  //     if (!elBanner.classList.contains('hidden') && !replayExporting) {
  //       startGame();
  //     }
  //   });
  // }
  const elMapBtn = $('map-btn');
  if (elMapBtn) elMapBtn.addEventListener('click', openMapMenu);
  // 选图页由独立“点击进入”按钮确认，避免调滑块/展开分类时误开局。
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
  // 「录制剪片」开关：只管 GIF/剪片采样（canvas 20fps 环形缓冲 + MediaRecorder）。
  // 状态帧与 60Hz 轨迹已恒录，保存视频无需此开关。
  elRecClip.addEventListener('change', () => {
    if (elRecClip.checked) {
      if (!mediaRec) startVideoRecorder();
      clipFrames = [];
      recMsg('录制剪片：已开启（保存 GIF 用；保存视频无需此开关）');
    } else {
      stopVideoRecorder().finally(() => {
        mediaRec = null; mediaMime = ''; mediaChunks = [];
      });
      clipFrames = [];
      recMsg('录制剪片：已关闭（零开销）');
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
  elRadius.addEventListener('input', setRadiusLabel);
  elRadius.addEventListener('change', () => { applyRadius(); startGame(); });
  applyRadius();  // 初始化同步半径显示与 CFG.radius (0.42)
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

  function buildReplayDoc() {
    if (!replay || !replay.actions.length) return null;
    return {
      format: 'qqt-replay',
      version: 2,
      meta: Object.assign({}, replay.meta, {
        savedAt: new Date().toISOString(),
        done: sim ? sim.done : false,
        result: sim && sim.done ? sim.winner : null,
        finalT: sim ? sim.t : 0,
      }),
      ticks: replay.actions.length,
      actions: replay.actions.map((a) => a.slice()),
      snapshots: replay.snapshots.map((s) => Object.assign({}, s)),
      frames: replay.frames.map((f) => f),
      framePos: replay.framePos.map((f) => f.slice()),
    };
  }

  function downloadReplayDoc(doc) {
    const text = JSON.stringify(doc, null, 1);
    const blob = new Blob([text], { type: 'application/json' });
    const mapName = doc.meta.mapName || doc.meta.map || 'map';
    downloadBlob(blob, `replay_${mapName}_s${doc.meta.seed}_${timeStamp()}.json`);
    return { text, blob };
  }

  elSaveReplay.addEventListener('click', () => {
    const doc = buildReplayDoc();
    if (!doc) { recMsg('还没有本局录像 —— 先开始一局再保存'); return; }
    const { blob } = downloadReplayDoc(doc);
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
      recMsg('未录制画面：请先勾选「录制剪片/GIF 采样」再开局');
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

  async function exportReplayVideo(doc) {
    if (!canvas.captureStream || !window.MediaRecorder) {
      throw new Error('当前浏览器不支持 Canvas 视频录制；JSON 已保存');
    }
    if (!doc.frames || doc.frames.length < 2) {
      throw new Error('录像没有完整状态帧；请重新开局后再保存');
    }

    const oldRunning = running;
    const oldSim = sim;
    const oldLevel = selectedLevel;
    const oldDanger = dangerCache;
    const oldExplosion = explosion;
    const oldExplosionTrig = explosionTrig;
    const oldExplosionT = explosionT;
    const replayLevel = levels.find((l) => l.id === doc.meta.levelId) ||
      levels.find((l) => l.source === doc.meta.map);
    if (!replayLevel) throw new Error(`找不到录像地图 ${doc.meta.map}`);

    const exportSim = new Sim(doc.meta.seed);
    exportSim.reset(replayLevel, { oldMode: !!doc.meta.oldMode });
    // 60Hz 帧级位置轨迹：新版录像恒录。有轨迹 → 60fps 平滑重放；
    // 旧录像（无 framePos）退回每 tick 一帧。
    const tickHz = Number(doc.meta.tickHz || CFG.tickHz);
    const framePos = Array.isArray(doc.framePos) ? doc.framePos : [];
    // 音频录制：重放期间把 BGM + 音效按事件重新触发，接进 MediaStream
    // 目的地并并进录制轨道（对局音效由 tick 差分触发，见下方主循环）。
    const ac = res && res.audio ? res.audio : null;
    const audioDst = ac ? ac.createMediaStreamDestination() : null;
    if (audioDst && ac.state === 'suspended') ac.resume().catch(() => {});
    let bgmRecNode = null, bgmRecGain = null;
    const mime = (audioDst ? [
      'video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus', 'video/webm',
    ] : []).concat([
      'video/mp4;codecs=avc1.42E01E', 'video/mp4',
      'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm',
    ]).find((t) => MediaRecorder.isTypeSupported(t));
    if (!mime) throw new Error('当前浏览器没有可用的视频编码器；JSON 已保存');
    const withAudio = !!audioDst && /mp4a|opus/.test(mime);
    const sndRec = (name, vol) => {
      if (!withAudio || !res.snd || !res.snd[name]) return;
      try {
        const src = ac.createBufferSource();
        src.buffer = res.snd[name];
        const g = ac.createGain();
        g.gain.value = vol == null ? 0.6 : vol;
        src.connect(g).connect(audioDst);
        src.start();
      } catch (e) { /* 忽略 */ }
    };
    // BGM：停现场曲，改用录像关卡自己的曲子（同时接外放与录制轨道）
    stopBgm();
    if (withAudio && replayLevel.music) {
      try {
        let buf = bgmBuffers.get(replayLevel.music);
        if (!buf) {
          buf = await ac.decodeAudioData(await (await fetch(replayLevel.music)).arrayBuffer());
          bgmBuffers.set(replayLevel.music, buf);
        }
        bgmRecNode = ac.createBufferSource();
        bgmRecNode.buffer = buf;
        bgmRecNode.loop = true;
        bgmRecGain = ac.createGain();
        bgmRecGain.gain.value = 0.22;          // 与对局 BGM 同音量
        bgmRecNode.connect(bgmRecGain);
        bgmRecGain.connect(audioDst);
        bgmRecGain.connect(ac.destination);
      } catch (e) { bgmRecNode = null; }
    }
    const stream = canvas.captureStream(framePos.length ? 60 : tickHz);
    if (withAudio) {
      for (const t of audioDst.stream.getAudioTracks()) stream.addTrack(t);
    }
    const chunks = [];
    const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 1200000 });
    const stopped = new Promise((resolve, reject) => {
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      recorder.onerror = (e) => reject(e.error || new Error('视频编码失败'));
      recorder.onstop = resolve;
    });
    const frameDelay = 1000 / tickHz;    // 每 tick 真实时长（在其 n 个子帧内均分）
    replayExporting = true;
    running = false;
    sim = exportSim;
    selectedLevel = replayLevel;
    const savedRadius = CFG.radius;
    if (doc.meta.cfg && Number.isFinite(Number(doc.meta.cfg.radius))) CFG.radius = Number(doc.meta.cfg.radius);
    const oldBanner = elBanner.innerHTML;
    const oldFace = face.slice();
    const lastPos = Float64Array.from(exportSim.pos);
    const animMoving = [false, false];
    dangerCache = null;                // 危险图默认不录入：清掉现场缓存, 重放也不重建
    try {
      elBanner.innerHTML = '⏳ 正在重放录制中…<span class="tip">按完整状态帧逐帧渲染，请勿切换标签页</span>';
      recorder.start();
      if (bgmRecNode) bgmRecNode.start();
      let subPtr = 0;
      let prevFrame = null;
      let prevP0 = [exportSim.pos[0], exportSim.pos[1]];
      let prevP1 = [exportSim.pos[2], exportSim.pos[3]];
      for (const frame of doc.frames) {
        exportSim.restoreReplay(frame);
        // 该 tick 的 60Hz 子帧（人类帧级移动只记录在 framePos；旧录像没有）
        const subs = [];
        while (subPtr < framePos.length && framePos[subPtr][0] <= frame.t) {
          if (framePos[subPtr][0] === frame.t) subs.push(framePos[subPtr]);
          subPtr++;
        }
        const n = Math.min(Math.max(subs.length, 1), 15);
        const p0HasPath = subs.length > 1 &&
          subs.some((s) => s[1] !== subs[0][1] || s[2] !== subs[0][2]);
        // 重建对局里由 tick 前后差分产生的事件特效/音效（快照含全部字段）：
        // 砖/灌木炸毁的 _die 中间态帧（炸开散落图 0.35s）、掉血回收宝箱抛物线、
        // 推箱完成、放泡/拾取/死亡音效
        if (prevFrame) {
          const l1 = replayLevel.layers && replayLevel.layers[1];
          const l0 = replayLevel.layers && replayLevel.layers[0];
          const nowMs = performance.now();
          let pushDone = false;
          for (let i = 0; i < N; i++) {
            // 新语义下砖在余威期间仍保留为碰撞体，因此用本 tick 的 covered
            // 事件识别炸毁；兼容旧录像仍保留 brick 由 1→0 的判断。
            const newlyBurned = frame.brickLinger != null
              ? (frame.covered && frame.covered[i] &&
                 (!prevFrame.brickLinger || prevFrame.brickLinger[i] === 0))
              // 旧录像没有 brickLinger/covered 语义，只能用砖状态差分兼容。
              : frame.brick[i] === 0;
            if (prevFrame.brick[i] === 1 && newlyBurned && l1 && l1[i]) {
              dieFx.set(i, { eid: Math.abs(l1[i]), until: nowMs + 200 });
            }
            if (prevFrame.bush && prevFrame.bush[i] === 1 && frame.bush[i] === 0 && l0 && l0[i]) {
              dieFx.set(i, { eid: Math.abs(l0[i]), until: nowMs + 200 });
            }
            // 新出现的回收宝箱 = 掉血散落（对局 _scatterRecycle → flyFx）
            if (prevFrame.crate[i] === 0 && frame.crate[i] === 1 && frame.recycle[i] === 1) {
              const dmgP = prevFrame.hp[0] > frame.hp[0] ? 0 : (prevFrame.hp[1] > frame.hp[1] ? 1 : -1);
              if (dmgP >= 0) {
                flyFx.push({
                  x0: exportSim.pos[dmgP * 2 + 1], y0: exportSim.pos[dmgP * 2],
                  x1: (i % W) + 0.5, y1: ((i / W) | 0) + 0.5, cell: i, t0: nowMs,
                });
              }
            }
            // 箱子从格上消失 = 一次推动完成（对局只在人推时响，这里一并收录）
            if (!pushDone && prevFrame.pushBoxAt[i] >= 0 && frame.pushBoxAt[i] === -1) {
              pushDone = true;
              sndRec('pushbox', 0.6);
            }
          }
          if (frame.placed && frame.placed[0]) sndRec('place');
          if (frame.died && frame.died[0]) sndRec('die');
          // 拾取：pid0 前一帧中心格有箱、本帧没了（对局 hadCrate 判定）
          const pcell = Math.floor(prevFrame.pos[0]) * W + Math.floor(prevFrame.pos[1]);
          if (prevFrame.crate[pcell] === 1 && frame.crate[pcell] === 0) sndRec('pickup');
        }
        // 火光与对局同语义：只在新爆炸时覆盖 —— 无爆炸保留旧值按 0.4s 自然
        // 到期（否则下一 tick 被置空 → 火焰/糖浆渲染时长比真实游戏短）。
        // 危险图默认不录入视频：危险叠层是调试/可视化用途，导出重放不重建
        // dangerCache（置空 → drawDangerOverlay 直接跳过）。
        if (frame.covered && frame.covered.some((v) => v > 0)) {
          explosion = new Uint8Array(frame.covered);
          explosionTrig = frame.triggered ? new Uint8Array(frame.triggered) : null;
          explosionT = performance.now();
          sndRec('boom');
        }
        // 每 tick 的真实时长在 n 个子帧内均分 → 视频时长与对局一致
        for (let i = 0; i < n; i++) {
          const a01 = (i + 1) / n;
          // P0：有 60Hz 轨迹用记录值（帧级转向/滑移的真实路径）；
          // 静止或旧录像按 tick 间线性插值。P1 恒按 tick 间插值。
          if (p0HasPath) {
            exportSim.pos[0] = subs[i][1]; exportSim.pos[1] = subs[i][2];
          } else {
            exportSim.pos[0] = prevP0[0] + (frame.pos[0] - prevP0[0]) * a01;
            exportSim.pos[1] = prevP0[1] + (frame.pos[1] - prevP0[1]) * a01;
          }
          exportSim.pos[2] = prevP1[0] + (frame.pos[2] - prevP1[0]) * a01;
          exportSim.pos[3] = prevP1[1] + (frame.pos[3] - prevP1[1]) * a01;
          // 恢复 60Hz 胜率轨迹（跟随画面与操作实时变化）
          if (subs[i] && subs[i][5] !== undefined) {
            replayWinProb = subs[i][5];
          } else if (frame.winProb !== undefined) {
            replayWinProb = frame.winProb;
          } else {
            const hpDiff = (exportSim.hp[0] - exportSim.hp[1]) / CFG.maxHp;
            replayWinProb = Math.max(0.05, Math.min(0.95, 0.5 + hpDiff * 0.45));
          }

          // 快照不含 sprite 状态：朝向/行走动画按本渲染帧与上一帧的位移推断，
          // 否则导出的视频角色锁朝向、腿部静止。
          for (let pid = 0; pid < 2; pid++) {
            const dy = exportSim.pos[pid * 2] - lastPos[pid * 2];
            const dx = exportSim.pos[pid * 2 + 1] - lastPos[pid * 2 + 1];
            animMoving[pid] = Math.abs(dy) > 1e-6 || Math.abs(dx) > 1e-6;
            if (animMoving[pid]) {
              if (Math.abs(dy) >= Math.abs(dx)) face[pid] = dy < 0 ? MOVE_UP : MOVE_DOWN;
              else face[pid] = dx < 0 ? MOVE_LEFT : MOVE_RIGHT;
            }
          }
          lastPos.set(exportSim.pos);
          replayAnim = { moving: animMoving };
          render(performance.now());
          await new Promise((resolve) => setTimeout(resolve, Math.max(0, frameDelay / n)));
        }
        prevP0 = [frame.pos[0], frame.pos[1]];
        prevP1 = [frame.pos[2], frame.pos[3]];
        prevFrame = frame;
      }
      recorder.stop();
      await stopped;
      const blob = new Blob(chunks, { type: mime });
      if (!blob.size) throw new Error('视频编码器没有输出数据；JSON 已保存');
      const isMp4 = mime.includes('mp4');
      const ext = isMp4 ? 'mp4' : 'webm';
      downloadBlob(blob, `video_${doc.meta.mapName || doc.meta.map}_s${doc.meta.seed}_${timeStamp()}.${ext}`);
      return { blob, mime, ext };
    } finally {
      if (bgmRecNode) { try { bgmRecNode.stop(); } catch (e) { /* */ } }
      bgmRecNode = null; bgmRecGain = null;
      CFG.radius = savedRadius;
      replayExporting = false;
      replayWinProb = null;
      replayAnim = null;
      face[0] = oldFace[0]; face[1] = oldFace[1];
      elBanner.innerHTML = oldBanner;
      sim = oldSim;
      selectedLevel = oldLevel;
      dangerCache = oldDanger;
      explosion = oldExplosion;
      explosionTrig = oldExplosionTrig;
      explosionT = oldExplosionT;
      running = oldRunning;
      startBgm();                              // 恢复现场 BGM（走缓存，无网络开销）
    }
  }

  elSaveVideo.addEventListener('click', async () => {
    if (replayExporting) return;
    const doc = buildReplayDoc();
    if (!doc) { recMsg('还没有本局录像 —— 先开始一局再保存'); return; }
    const { text } = downloadReplayDoc(doc);
    recMsg('JSON 已保存，正在按完整状态帧渲染视频…');
    try {
      const out = await exportReplayVideo(JSON.parse(text));
      recMsg(`视频已保存：${(out.blob.size / 1024).toFixed(0)}KB（${out.ext}${out.ext === 'webm' ? '；当前浏览器不支持 MP4，已保留真实 WebM' : ''}）`);
    } catch (e) {
      recMsg(`视频导出失败：${e.message}`);
      console.error(e);
    }
  });

  // ------------------------------------------------------------ 10Hz 逻辑节拍
  let tickBusy = false;              // async tick 重入保护（await 期间 setInterval 再触发时跳过）
  let tickDebt = 0;                  // 后台节流补偿欠账(ms): 标签页隐藏时 setInterval 被
                                     // 节流到 ~1Hz(10倍慢) → 每次触发补跑缺失的 tick 保持实时
  const tickTimeline = [];
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
    const actionT0 = performance.now();
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
    const actionMs = performance.now() - actionT0;
    // 拾取判定：人类玩家脚下 step 前有宝箱 → step 后没有 = 吃到
    const hc = Math.floor(sim.pos[1]), hr = Math.floor(sim.pos[0]);
    const hadCrate = !spectate && sim.alive[0] && sim.crate[hr * W + hc] === 1;
    // 录像：记录本 tick 实际喂给 step 的动作 + 每 20 tick 一个状态快照
    const snapshotT0 = performance.now();
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
    const prevBrickLinger = sim.brickLinger.slice();
    const prevBush = sim.bush ? sim.bush.slice() : null;
    const hpBefore = sim.hp.slice();          // 血量快照（找掉血玩家）
    const prevCrate = sim.crate.slice();      // 宝箱快照（找新回收箱）
    const snapshotMs = performance.now() - snapshotT0;
    const stepT0 = performance.now();
    const info = sim.step([a0, a1]);
    if (replay) {
      const snap = sim.snapshotReplay(info);
      snap.winProb = currentWinProb;
      replay.frames.push(snap);
    }
    curPos.set(sim.pos);
    const stepMs = performance.now() - stepT0;
    const eventT0 = performance.now();
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
        // 砖的碰撞体要等余威结束才清除，所以炸毁特效只以本 tick 的
        // lastCovered 事件为准，不能把残威结束时的 brick 1→0 当成第二次爆炸。
        const newlyBurned = sim.lastCovered && sim.lastCovered[i] &&
          prevBrickLinger[i] === 0;
        if (prevBrick[i] === 1 && newlyBurned && l1[i]) {
          dieFx.set(i, { eid: Math.abs(l1[i]), until: performance.now() + 200 });
        }
        if (prevBush && prevBush[i] === 1 && sim.bush[i] === 0 && l0[i]) {
          dieFx.set(i, { eid: Math.abs(l0[i]), until: performance.now() + 200 });
        }
      }
    }
    const eventMs = performance.now() - eventT0;
    // danger 缓存：tick 级重建（10Hz），渲染帧直接复用 —— 60fps 每帧重算
    // dangerMap（不动点传播 O(炸弹×blast)）是 AI 对打帧率低的主因
    const dangerT0 = performance.now();
    dangerCache = sim.dangerMap();
    const dangerMs = performance.now() - dangerT0;
    const postT0 = performance.now();
    // 朝向：人类玩家（非观战）的朝向由 60Hz 帧级移动维护，10Hz tick 不覆盖；
    // AI 做出移动决策时才更新朝向；IDLE 静止时保持上一次朝向，绝不闪回朝下。
    if (spectate && a0[0] !== MOVE_IDLE) face[0] = a0[0];
    if (a1[0] !== MOVE_IDLE) face[1] = a1[0];
    lastAiMove = [a0[0], a1[0]];
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
    const tickEnd = performance.now();
    tickTimeline.push({ start: tk0, end: tickEnd, duration: tickEnd - tk0 });
    while (tickTimeline.length > 200) tickTimeline.shift();
    lastTickT = tickEnd;
    prof.tickLast = tickEnd - tk0;
    prof.tickParts = {
      at: tickEnd,
      action: actionMs,
      snapshot: snapshotMs,
      step: stepMs,
      events: eventMs,
      danger: dangerMs,
      post: tickEnd - postT0,
    };
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
      await stopVideoRecorder();
      running = false;
    }
  }
  setInterval(() => {
    // 将 10Hz 逻辑工作放进浏览器空闲窗口，避免固定 timer 与下一次
    // vsync 同相位触发而错过一帧。隐藏页仍走原路径，让 tickDebt 补偿继续工作。
    if (!document.hidden && window.requestIdleCallback) {
      requestIdleCallback(() => logicTick(), { timeout: TICK * 500 });
    } else {
      logicTick();
    }
  }, TICK * 1000);

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
  const Z_ROW_STRIDE = 24;                 // 纵向主序步长 (各行严格按 24 隔离，行内：墙体 0..14、道具 15、泡泡 17、人物 18、水泡/火焰 19)
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
                    isBush: li === 0 && Math.abs(v) === 6003 ||
                      !!(lv.bush && lv.bush[r * W + c]) });
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
    // 画一次；L1 由 sim 状态驱动（砖在余威结束后消失）；角色中心进入结构足迹 →
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
        if (sim.pushable && sim.pushable[st.r * W + st.c]) continue;   // 可推箱由运行时画
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

    // 可推箱(运行时位置): 被推走后精灵跟随新格(静态 layers 已被跳过)
    if (sim.pushBoxes) {
      for (const box of sim.pushBoxes) {
        if (!box || box.dead || !sim.brick[box.o]) continue;
        const el = elements[box.eid];
        if (!el) continue;
        const img = elemImage(box.eid);
        if (!img) continue;
        const r = (box.o / W) | 0, c = box.o % W;
        items.push([tileZ(r, c), img, Math.round(c * CELL - el.xo * SCALE),
                    Math.round(r * CELL - el.yo * SCALE)]);
      }
    }

    // 砖/灌木被炸毁的中间态 (_die 帧)：固定 0.2s 碎墙动画，不遮挡水泡且不提前淡出
    if (dieFx.size) {
      const dieNow = performance.now();
      for (const [i, fx] of dieFx) {
        if (dieNow > fx.until) { dieFx.delete(i); continue; }
        const img = dieImage(fx.eid);
        const el = elements[fx.eid];
        if (!img || !el) continue;
        const r = (i / W) | 0, c = i % W;
        const x = c * CELL - el.xo * SCALE, y = r * CELL - el.yo * SCALE;
        items.push([tileZ(r, c) + 1, () => {
          ctx.drawImage(img, x, y);
        }]);
      }
    }

    const nowS = now / 1000;
    const bob = Math.round(Math.sin(nowS * 2 * Math.PI) * 3);

    // 爆炸：中心格用中心图；臂图按实际爆炸格数从炸弹边缘端切片（duel.py 同款算法）
    // ------------------------------------------------------------ 爆炸水泡逐格序列帧 (res/flame)
    // 爆炸持续动画一共 0.45s：
    //   A 阶段 [0.00s ~ 0.06s]：动画波及威力=1 的范围初始帧动画（视觉先导，使用帧 1 炸开花边缘）
    //   B 阶段 [0.06s ~ 0.14s]：动画波及到最远处（0.06s-0.30s 完整格子造成伤害，中间格使用帧 2-4 轮播，边缘使用帧 5）
    //   C 阶段 [0.14s ~ 0.45s]：动画收汁消散：
    //     - 边缘格经历 1 -> 5 -> 1 -> 6 脉动消散序列（0.30s 起伤害自然截止）
    //     - 0.39s ~ 0.45s（后 0.06s）回到 A 阶段范围（收缩回威力=1 的范围，边缘使用帧 6）
    if (explosion) {
      const age = (now - explosionT) / 1000;
      if (age <= 0.45 && explosionTrig) {
        const blast = explosion;
        const maxBlast = 8;   // 成长上限，与 sim/jax_env.py / 地图数据一致
        if (res && res.flames) {
          const DIR_KEYS = ['U', 'D', 'L', 'R'];
          // 中心格动画帧：帧 1, 2 交替循环
          const centerFrame = (Math.floor(age * 12) % 2 === 0) ? 1 : 2;
          const centerImg = res.flames.C[centerFrame] || res.flames.C[1];

          // 1. 引爆源格画中心图 (Z = r * Z_ROW_STRIDE + 19)
          for (let i = 0; i < N; i++) {
            if (!explosionTrig[i]) continue;
            if (dieFx.has(i)) continue;
            const r = (i / W) | 0, c = i % W;
            items.push([r * Z_ROW_STRIDE + 19, centerImg, c * CELL, r * CELL]);
          }

          // 2. 臂：从引爆源向 4 方向按实际长度画
          // 中间格子：完整按照 2 -> 3 -> 4 -> 3 序列过渡
          //   0.06s ~ 0.15s 初生波涌使用帧 2
          //   0.15s ~ 0.24s 展开过渡使用帧 3
          //   0.24s ~ 0.33s 鼎盛饱满使用帧 4
          //   0.33s ~ 0.45s 阶段 C 边缘过渡到帧 6 时，中间格同步切入帧 3 收敛
          let bodyFrame;
          if (age < 0.15) {
            bodyFrame = 2;  // 0.06s ~ 0.15s: 帧 2
          } else if (age < 0.24) {
            bodyFrame = 3;  // 0.15s ~ 0.24s: 帧 3
          } else if (age < 0.33) {
            bodyFrame = 4;  // 0.24s ~ 0.33s: 帧 4
          } else {
            bodyFrame = 3;  // 0.33s ~ 0.45s: 帧 3 (边缘过渡到帧 6 时同步收尾)
          }

          for (let i = 0; i < N; i++) {
            if (!explosionTrig[i]) continue;
            const sr = (i / W) | 0, sc = i % W;
            for (let d = 0; d < 4; d++) {
              const [dr, dc] = DIRS[d];
              const dirKey = DIR_KEYS[d];
              let n = 0;
              for (let k = 1; k <= maxBlast; k++) {
                const r = sr + dr * k, c = sc + dc * k;
                if (r < 0 || r >= H || c < 0 || c >= W) break;
                if (!blast[r * W + c]) break;
                n++;
              }
              if (n === 0) continue;

              // 根据扩散时间线确定当前有效展示长度 activeLen 与边缘格的帧:
              let activeLen = n;
              let tipFrame = 1;

              if (age < 0.06) {
                // A 阶段 [0.00 ~ 0.06s]: 仅波及威力=1 范围，使用帧 1（炸开花初始边缘形态）
                activeLen = Math.min(n, 1);
                tipFrame = 1;
              } else if (age < 0.14) {
                // B 阶段 [0.06 ~ 0.14s]: 扩散到最远处，边缘使用帧 1（与阶段 C 接续）
                activeLen = n;
                tipFrame = 1;
              } else {
                // C 阶段 [0.14 ~ 0.45s]: 动画消散收汁，边缘按 1 -> 5 -> 1 -> 6 连贯消散
                // 后 0.06s (0.39s ~ 0.45s) 收缩回威力=1 范围收尾
                activeLen = age < 0.39 ? n : Math.min(n, 1);
                if (age < 0.20) {
                  tipFrame = 1;       // 0.14 ~ 0.20s: 帧 1
                } else if (age < 0.26) {
                  tipFrame = 5;       // 0.20 ~ 0.26s: 帧 5
                } else if (age < 0.33) {
                  tipFrame = 1;       // 0.26 ~ 0.33s: 帧 1
                } else {
                  tipFrame = 6;       // 0.33 ~ 0.45s: 帧 6
                }
              }

              for (let k = 1; k <= activeLen; k++) {
                const r = sr + dr * k, c = sc + dc * k;
                const cellIdx = r * W + c;
                // 如果该格本身是另一个引爆中心格，优先保留中心格显示
                if (explosionTrig[cellIdx]) continue;
                // 当炸弹威力覆盖到可炸毁墙体：在 0.2s 碎墙动画期间不显示水泡，播放完毕后立刻显现水泡
                if (dieFx.has(cellIdx)) continue;
                const img = (k === activeLen)
                  ? res.flames[dirKey][tipFrame]
                  : res.flames[dirKey][bodyFrame];
                if (img) {
                  items.push([r * Z_ROW_STRIDE + 19, img, c * CELL, r * CELL]);
                }
              }
            }
          }
        }
      } else {
        explosion = null;
        explosionTrig = null;
      }
    }

    // 泡泡：去掉伪呼吸位移，按引信年龄播放 4 帧序列。
    // 每 1s 完整播放一组（每帧 0.25s），3s 引信循环三组；按原图 × SCALE
    // 绘制，底边贴格底线，超过一格的高度允许向上溢出。
    for (let i = 0; i < N; i++) {
      if (sim.fuse[i] <= 0) continue;
      // 泡泡在任何果冻遮挡结构(房子/灌木/拱门等)上 → 隐藏（visible=false）
      if (hideCells.has(i)) continue;
      const r = (i / W) | 0, c = i % W;
      const age = Math.max(0, CFG.fuse - sim.fuse[i]);
      const frame = Math.floor((age / CFG.tickHz) * 4) % 4;
      const owner = sim.owner[i];
      const custom = owner === 0
        ? (sim.playerBombStyle != null ? (sim.playerBombStyle & 1)
                                       : (sim.bombStyle[i] & 1))
        : 0;
      const frames = owner === 0 ? res.bombFrames.custom[custom] : res.bombFrames.default;
      const img = frames[frame];
      const bx = c * CELL + (CELL - img.width) / 2;
      const by = (r + 1) * CELL - img.height;
      items.push([r * Z_ROW_STRIDE + 17, img, bx, by]);
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
    for (const f of birdDropFx) {
      flyTargets.add(f.cell);
    }
    for (let i = 0; i < N; i++) {
      if (!sim.crate[i] || flyTargets.has(i)) continue;
      // 砖还在碎墙动画 (dieFx) 期间，不提前显示该格的道具/宝箱
      if (dieFx.has(i)) continue;
      const r = (i / W) | 0, c = i % W;
      // 随机宝箱(带?箱子) / 普通(种类定好) / 超级(种类+超级图标)
      let p;
      if (sim.crateType[i] < 0) p = boxQ;
      else if (sim.superCrate[i]) p = res.superIcons[sim.crateType[i]];
      else p = res.propIcons[sim.crateType[i]];
      // 原图尺寸，格内居中（不拉伸）
      const px = c * CELL + (CELL - p.width) / 2;
      const py = r * CELL + (CELL - p.height) / 2 + bob * 0.5;
      // 道具 Z 设为 15，严格低于同一行的水泡 (19)，水泡拥有更高优先级
      items.push([r * Z_ROW_STRIDE + 15, p, Math.round(px), Math.round(py)]);
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

    // 飞鸟空投抛物线飞行动画（高空飞行物，Z = 55000）
    for (let k = birdDropFx.length - 1; k >= 0; k--) {
      const drop = birdDropFx[k];
      const ageMs = now - drop.t0;
      const progress = ageMs / drop.dur;
      if (progress >= 1.0) {
        // 落地：正式写入地图数据
        sim.spawnGraveyardDrop(drop.cell, drop.item.type, drop.item.isSuper);
        birdDropFx.splice(k, 1);
        continue;
      }
      const curX = drop.sx + (drop.tx - drop.sx) * progress;
      // 抛物线：向上拱起 90px
      const arc = -90 * 4 * progress * (1 - progress);
      const curY = drop.sy + (drop.ty - drop.sy) * progress + arc;
      let icon = drop.item.isSuper ? res.superIcons[drop.item.type] : res.propIcons[drop.item.type];
      if (!icon) icon = res.boxQ;
      items.push([55000, icon, Math.round(curX - icon.width / 2), Math.round(curY - icon.height / 2)]);
    }

    // 飞鸟巡航控制器：30s 一个循环（前 25s 冷却，后 5s 飞行）
    // 严格与倒计时时钟（sim.t / 10）绑定对齐：
    // - 倒计时 180s（t=0）开局；
    // - 倒计时 155s（t=25s，即 sim.t=250）第一只鸟准时从右侧场外开始向左飞入；
    // - 倒计时 153s（t=27s，即 sim.t=270）飞鸟头部进入画面右边界；
    // - 倒计时 153s ~ 150s（flightTime 2.0s ~ 4.7s）：
    //   飞鸟横跨场内飞行期间，沿途向所经之处上下方均匀洒落墓地道具；
    // - 倒计时 150s（t=30s，即 sim.t=300）飞鸟完全飞离画面左边界，完成本轮周期。
    if (res.birdFrames && running && !sim.done) {
      const subTick = Math.min(1.0, Math.max(0.0, (now - lastTickT) / (TICK * 1000)));
      const matchElapsedS = (sim.t + subTick) / CFG.tickHz;
      const cycleIndex = Math.floor(matchElapsedS / 30);
      const cycleTime = matchElapsedS % 30; // 0 ~ 30s

      // 新周期开始：重置本轮空投规划队列
      if (birdLastCycle !== cycleIndex) {
        birdLastCycle = cycleIndex;
        birdCruiseQueue = [];
      }

      if (cycleTime >= 25.0) {
        const flightTime = cycleTime - 25.0; // 0 ~ 5s
        // 帧动画：1s 播放一轮（0.5s 一帧）
        const frameIdx = Math.floor(flightTime * 2) % 2;
        const birdImg = res.birdFrames[frameIdx];

        // 坐标计算：
        // 0s ~ 2s：从 x = 22 移动到 x = 15 (进入画面右边界)
        // 2s ~ 5s：从 x = 15 移动到 x = -3.5 (完全飞离画面左边界，1.5x 宽为 3.2 格)
        let bx;
        if (flightTime < 2.0) {
          bx = (22 - 3.5 * flightTime) * CELL;
        } else {
          bx = (15 - (18.5 / 3.0) * (flightTime - 2.0)) * CELL;
        }
        // y 坐标固定为从上往下第 3.5 个格子 (oy = 3.5 * CELL)
        const by = 3.5 * CELL;

        // 飞鸟处于最高空 (Z = 60000)
        items.push([60000, birdImg, Math.round(bx), Math.round(by)]);

        // 场内飞行阶段（flightTime 2.0s ~ 4.7s）：
        // 若墓地有积攒道具，按剩余飞行时间均匀规划空投时刻
        if (flightTime >= 2.0 && flightTime <= 4.7 && sim.graveyard && sim.graveyard.length > 0) {
          const count = sim.graveyard.length;
          const toDrop = sim.graveyard.splice(0, count);
          const tStart = Math.max(flightTime + 0.05, 2.1);
          const tEnd = 4.7;
          for (let k = 0; k < toDrop.length; k++) {
            const dropT = tStart + ((k + 0.5) / count) * (tEnd - tStart);
            birdCruiseQueue.push({ flightTime: dropT, item: toDrop[k] });
          }
          birdCruiseQueue.sort((a, b) => a.flightTime - b.flightTime);
        }

        // 沿途准时抛出空投道具：飞鸟在其所在 X 处向上下方安全格洒落
        if (flightTime >= 2.0 && flightTime <= 4.85 && birdCruiseQueue.length > 0) {
          while (birdCruiseQueue.length > 0 && birdCruiseQueue[0].flightTime <= flightTime) {
            const nextDrop = birdCruiseQueue.shift();
            const birdCenterX = bx + birdImg.width / 2;
            const birdCenterY = by + birdImg.height / 2;
            const birdCol = Math.round(bx / CELL);

            // 寻找落点：优先在飞鸟当前经过的列（及其相邻列）的上下方空闲格
            const inFlight = new Set();
            for (const f of birdDropFx) inFlight.add(f.cell);
            const candidates = [];
            for (let i = 0; i < N; i++) {
              if (!sim.wall[i] && !sim.brick[i] && !sim.crate[i] && !sim.pushable[i] && sim.fuse[i] <= 0 && !inFlight.has(i)) {
                candidates.push(i);
              }
            }
            if (candidates.length > 0) {
              // 计算离飞鸟当前列的水平距离，优先同列或就近列
              let minColDist = 999;
              for (const c of candidates) {
                const dist = Math.abs((c % W) - birdCol);
                if (dist < minColDist) minColDist = dist;
              }
              const bestCols = candidates.filter((c) => Math.abs((c % W) - birdCol) === minColDist);
              // 在最优列集合中，随机挑选一个空闲格（自然分散在上下方）
              const tc = bestCols[Math.floor(Math.random() * bestCols.length)];
              const targetX = (tc % W) * CELL + CELL / 2;
              const targetY = (((tc / W) | 0) + 0.5) * CELL;
              birdDropFx.push({
                sx: birdCenterX,
                sy: birdCenterY,
                tx: targetX,
                ty: targetY,
                cell: tc,
                item: nextDrop.item,
                t0: now,
                dur: 600, // 0.6s 抛物线落地
              });
            }
          }
        }
      }
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
      const moving = replayAnim ? replayAnim.moving[pid] : humanMoveState(pid);
      const frame = (moving ? Math.floor(nowS * 8) % 4 : 0);
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
      // 角色 Z = 行×24+18：
      // 同行墙(≤+15)在角色后画(角色在前)；同行水泡(+19)在角色前画(盖住角色右/下半身)；
      // 上一行水泡((r-1)*24+19 = r*24-5 < r*24+18)在角色后画(角色头部帽子盖住上一行水泡)
      const z = Math.floor(gy) * Z_ROW_STRIDE + 18;
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
    // 鼠标寻路：hover 周围九宫格高亮（在 translate 内画，与游戏板同一坐标系）
    if (hoverCell && !elSpectate.checked && sim && running) {
      const r = hoverCell.r, c = hoverCell.c;
      if (r >= 0 && r < H && c >= 0 && c < W) {
        ctx.fillStyle = 'rgba(255,255,100,0.25)';
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        ctx.strokeStyle = 'rgba(255,255,100,0.6)';
        ctx.lineWidth = 2;
        ctx.strokeRect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
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
    if (pid === 0 && !elSpectate.checked) {
      return human.move !== MOVE_IDLE && sim.alive[0];
    }
    const m = pid === 0 ? lastAiMove[0] : lastAiMove[1];
    return m !== MOVE_IDLE && sim.alive[pid];
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
    // 辅助文本截断函数，保证模型名过长时属性不被推出视野
    function truncateText(text, maxWidth) {
      if (!text || ctx.measureText(text).width <= maxWidth) return text;
      let low = 0, high = text.length;
      let res = text;
      while (low <= high) {
        const mid = (low + high) >> 1;
        const candidate = text.slice(0, mid) + '…';
        if (ctx.measureText(candidate).width <= maxWidth) {
          res = candidate;
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }
      return res;
    }

    // 第 1 行：双方状态各自合并成一行（名字截断 + HP/属性同排），右侧倒计时（无 tick）
    const colors = ['#ff6b6b', '#5aa7ff'];
    const pWidth = 350; // 单阵营占用上限，右侧保留 180px 供倒计时与地图信息
    for (let p = 0; p < 2; p++) {
      const name = p === 0 ? p0Kind : p1Kind;
      const tag = sim.alive[p] ? `P${p}` : `P${p}·阵亡`;
      const bx = 18 + p * pWidth;
      const tagStr = `（${tag}）`;
      const attrStr = `HP ${sim.hp[p]}/${CFG.maxHp} · 泡 ${sim.bombsCap[p]} · 威 ${sim.blastCap[p]} · 速 ${sim.spdG[p].toFixed(2)}`;

      ctx.font = '12px sans-serif';
      const attrW = ctx.measureText(attrStr).width;

      ctx.font = 'bold 13px sans-serif';
      const tagW = ctx.measureText(tagStr).width;

      // 动态截断模型名字：给属性、标签留足空间（至少保留 40px 名字宽度）
      const maxNameW = Math.max(40, pWidth - attrW - tagW - 14);
      const truncatedName = truncateText(name, maxNameW);
      const nameStr = `${truncatedName}${tagStr}`;
      const nameW = ctx.measureText(nameStr).width;

      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = colors[p];
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(nameStr, bx, y0 + 10);

      ctx.fillStyle = '#e8e6df';
      ctx.font = '12px sans-serif';
      ctx.fillText(attrStr, bx + nameW + 6, y0 + 12);
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
                 BOARD_PX - 18, y0 + 30);

    // ---- 实时 AI 胜率评估 (仅同步更新网页顶部 Header 胜率条，去除底部冗余胜率条) ----
    const em = enemySel && enemySel !== HUNTER_VAL ? modelCache.get(enemySel) : null;
    let p0WinProb = 0.5;
    if (replayExporting && replayWinProb !== null) {
      p0WinProb = replayWinProb;
    } else {
      const p0m = elSpectate.checked && p0Sel && p0Sel !== HUNTER_VAL ? modelCache.get(p0Sel) : null;
      if (p0m && p0m._lastVal && p0m._lastVal[0] !== undefined) {
        p0WinProb = Math.max(0.02, Math.min(0.98, (p0m._lastVal[0] + 1.0) / 2.0));
      } else if (em && em._lastVal) {
        const v = (em._lastVal[1] !== undefined && !elSpectate.checked) ? em._lastVal[1] : (em._lastVal[0] !== undefined ? em._lastVal[0] : 0.0);
        p0WinProb = Math.max(0.02, Math.min(0.98, 1.0 - (v + 1.0) / 2.0));
      } else if (sim) {
        const hpDiff = (sim.hp[0] - sim.hp[1]) / CFG.maxHp;
        p0WinProb = Math.max(0.05, Math.min(0.95, 0.5 + hpDiff * 0.45));
      }
      currentWinProb = p0WinProb;
    }

    // 同步更新网页顶部 Header 胜率条 (Header Win Probability Gauge)
    if (elP0WinFill) {
      const p0Pct = Math.round(p0WinProb * 100);
      const p1Pct = 100 - p0Pct;
      elP0WinFill.style.width = `${p0Pct}%`;
      elP1WinFill.style.width = `${p1Pct}%`;
      elP0WinPct.textContent = `${p0Pct}%`;
      elP1WinPct.textContent = `${p1Pct}%`;
      if (elP0WinName) elP0WinName.textContent = `我方：${p0Kind}`;
      if (elP1WinName) elP1WinName.textContent = `敌人：${p1Kind}`;
    }

    // 第 2 行：完整对阵模型明细（辅助文本，替代底部重复胜率条）
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    const p0Full = elSpectate.checked && p0Sel && p0Sel !== HUNTER_VAL && modelCache.get(p0Sel)
      ? `${modelDisplayName(modelCache.get(p0Sel).meta)}（${fmtStep(modelCache.get(p0Sel).meta.global_step)}步）`
      : p0Kind;
    const p1Full = em
      ? `${modelDisplayName(em.meta)}（${fmtStep(em.meta.global_step)}步）`
      : p1Kind;
    const vsInfo = `红方(P0)：${p0Full}   ⚔️   蓝方(P1)：${p1Full}`;
    ctx.fillText(truncateText(vsInfo, BOARD_PX - 36), 18, y0 + 46);
  }

  // ------------------------------------------------------------ 主循环
  let prevFrame = 0;
  let fpsFrames = 0, fpsT0 = 0, fpsNow = 0;   // 帧数统计(右上角显示)
  // ---- profiling: 真实帧间隔/渲染耗时/tick耗时/最大帧 ----
  let prof = { frames: 0, t0: 0, sumDt: 0, maxDt: 0, renderMs: 0, tickLast: 0,
               inputLast: 0, callbackLast: 0, clipLast: 0, mouseSearchLast: 0 };
  let profAvg = null;
  let lastRenderMs = 0;   // 最近一帧 render 耗时(突变帧日志用)
  const spikeHistory = [];          // 完整保留，不依赖 DevTools 控制台 UI
  window.__qqtProfSpikes = spikeHistory;
  const longTasks = [];
  const longFrames = [];
  if (window.PerformanceObserver) {
    try {
      const po = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          longTasks.push({
            start: e.startTime,
            end: e.startTime + e.duration,
            duration: e.duration,
            name: e.name || 'self',
            attribution: (e.attribution || []).map((a) => ({
              name: a.name || '',
              entryType: a.entryType || '',
              containerType: a.containerType || '',
              containerSrc: a.containerSrc || '',
              containerName: a.containerName || '',
            })),
          });
        }
        while (longTasks.length > 20) longTasks.shift();
      });
      po.observe({ entryTypes: ['longtask'] });
    } catch (_) {}
    try {
      const loaf = new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          const scripts = (e.scripts || []).map((s) => ({
            duration: s.duration || 0,
            invoker: s.invoker || s.sourceFunctionName || s.sourceURL || 'script',
          })).sort((a, b) => b.duration - a.duration);
          longFrames.push({
            start: e.startTime,
            end: e.startTime + e.duration,
            duration: e.duration,
            blocking: e.blockingDuration || 0,
            script: scripts[0] || null,
          });
        }
        while (longFrames.length > 20) longFrames.shift();
      });
      loaf.observe({ type: 'long-animation-frame', buffered: true });
    } catch (_) {}
  }
  const PROF_SPIKE_MS = 25;
  function loop(now) {
    const callbackT0 = performance.now();
    // 人类输入 60Hz 采样 + 帧级移动
    const inputT0 = callbackT0;
    if (running && !elSpectate.checked) {
      const dt = Math.min((now - prevFrame) / 1000 || 0, 0.25);
      const keyMove = sampleHumanMove();
      if (keyMove !== MOVE_IDLE) mousePush = null;  // 键盘/摇杆立即接管
      const mousePushing = mousePush && now < mousePush.until && sim.alive[0];
      if (mousePush && !mousePushing) mousePush = null;
      human.move = keyMove !== MOVE_IDLE ? keyMove : (mousePushing ? mousePush.dir : MOVE_IDLE);
      if (human.move !== MOVE_IDLE && sim.alive[0]) {
        // 只要有人类按键意图，立即更新朝向（即使撞墙被阻挡，也必须正确面朝输入目标方向）
        face[0] = human.move;
        // 顶箱期间禁止玩家自动转向；必须保持同一方向累计推动时间。
        const eff = mousePushing ? human.move : autoTurn(0, human.move);
        frameMove(0, eff, dt);            // 坐标用转向方向滑移
        if (!mousePushing && turnSlideTarget && eff === turnSlide) {
          // 只截断本帧的侧滑轴，不改另一轴碰撞结果。到达中心线后下一帧
          // autoTurn 会发现原方向可完整移动并恢复玩家的意图方向。
          if (turnSlideTarget.axis === 'x') {
            sim.pos[1] = eff === MOVE_LEFT
              ? Math.max(sim.pos[1], turnSlideTarget.value)
              : Math.min(sim.pos[1], turnSlideTarget.value);
          } else {
            sim.pos[0] = eff === MOVE_UP
              ? Math.max(sim.pos[0], turnSlideTarget.value)
              : Math.min(sim.pos[0], turnSlideTarget.value);
          }
        }
        human.move = eff;                 // 动画播放状态按实际移动
      }
    }
    prof.inputLast = performance.now() - inputT0;
    // 60Hz 帧级位置轨迹（不管对局是谁）：人类/观战下角色在渲染帧间的真实
    // 位置只存在于这里，导出视频据此做 60fps 平滑重放。
    if (replay && running && sim && !sim.done && !replayExporting) {
      replay.framePos.push([sim.t, sim.pos[0], sim.pos[1], sim.pos[2], sim.pos[3], currentWinProb]);
    }
    const frameDt = now - prevFrame;          // 真实帧间隔(ms, rAF 时间戳)
    prevFrame = now;
    if (!fpsT0) fpsT0 = now;
    else fpsFrames++;                 // 统计帧间隔数，不把窗口首尾两帧都算进去
    if (human.move === MOVE_IDLE || !sim.alive[0]) clearTurnSlide(); // 松手/死亡: 取消滑动
    if (now - fpsT0 >= 1000) { fpsNow = Math.round(fpsFrames * 1000 / (now - fpsT0)); fpsFrames = 0; fpsT0 = now; }
    // profiling 累计
    prof.frames++;
    prof.sumDt += frameDt;
    prof.maxDt = Math.max(prof.maxDt, frameDt);
    const rs = performance.now();
    render(now);
    lastRenderMs = performance.now() - rs;
    prof.callbackLast = performance.now() - callbackT0;
    prof.renderMs += lastRenderMs;
    // 只在实际对局中告警。选图/欢迎页仍使用同一 rAF 做 UI 动画，
    // 但那里的调度抖动不属于游戏帧卡顿，不能混入诊断日志。
    if (running && !mapMenuOpen && frameDt > PROF_SPIKE_MS) {
      const p = prof.tickParts;
      const age = p ? (now - p.at).toFixed(0) : '-';
      const parts = p ? ` tick分段动作${p.action.toFixed(1)} 快照${p.snapshot.toFixed(1)} step${p.step.toFixed(1)} 事件${p.events.toFixed(1)} danger${p.danger.toFixed(1)} post${p.post.toFixed(1)}ms（距今${age}ms）` : '';
      const frameStart = now - frameDt;
      const renderMs = lastRenderMs;
      const callbackMs = prof.callbackLast;
      const inputMs = prof.inputLast;
      // rAF 时间戳之间的间隔减去当前回调自身耗时，近似表示浏览器/系统
      // 没有调度页面 JS 的时间；它不是页面函数耗时。
      const scheduleGapMs = Math.max(0, frameDt - callbackMs);
      const overlappingTicks = tickTimeline.filter((t) => t.start < now && t.end > frameStart)
        .map((t) => ({
          start: t.start,
          end: t.end,
          duration: t.duration,
          overlap: Math.max(0, Math.min(t.end, now) - Math.max(t.start, frameStart)),
        }));
      const spike = { at: now, frameDt, callbackMs, inputMs, renderMs,
                      scheduleGapMs, overlappingTicks, tickParts: p ? { ...p } : null };
      spikeHistory.push(spike);
      while (spikeHistory.length > 200) spikeHistory.shift();
      canvas.dataset.profSpikes = String(spikeHistory.length);
      canvas.dataset.profLastDt = frameDt.toFixed(1);
      // PerformanceObserver 通常在当前 rAF 回调之后投递，延迟一个 task 再做
      // 时间重叠匹配，避免把历史 Long Task 错标到当前突变帧。
      if (!location.search.includes('profquiet=1')) setTimeout(() => {
        if (!running || mapMenuOpen) return;
        const lt = [...longTasks].reverse().find((e) => e.start < now && e.end > frameStart);
        const lf = [...longFrames].reverse().find((e) => e.start < now && e.end > frameStart);
        const ltSrc = lt && lt.attribution && lt.attribution.length
          ? (lt.attribution[0].containerSrc || lt.attribution[0].name || '') : '';
        const longInfo = lt
          ? ` longtask${lt.duration.toFixed(1)}ms[${lt.start.toFixed(1)}-${lt.end.toFixed(1)}${ltSrc ? ` ${ltSrc}` : ''}]`
          : ' longtask无重叠';
        const frameInfo = lf
          ? ` LoAF${lf.duration.toFixed(1)}ms 阻塞${lf.blocking.toFixed(1)}ms${lf.script ? ` 最重脚本${lf.script.invoker}:${lf.script.duration.toFixed(1)}ms` : ''}`
          : '';
        console.warn(`[prof] 突变帧 ${frameDt.toFixed(1)}ms @t=${(now/1000).toFixed(1)}s (回调${callbackMs.toFixed(1)}ms 输入${inputMs.toFixed(1)}ms 渲染${renderMs.toFixed(1)}ms 调度空档${scheduleGapMs.toFixed(1)}ms 采样${(prof.clipLast||0).toFixed(1)}ms${parts}${longInfo}${frameInfo})`);
      }, 0);
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
      const profLine = `[prof] ${profAvg.fps}fps 帧均${profAvg.avgDt.toFixed(1)}ms 渲染${profAvg.render.toFixed(1)}ms tick${profAvg.tick.toFixed(1)}ms 采样${(prof.clipLast||0).toFixed(1)}ms 鼠标寻路${(prof.mouseSearchLast||0).toFixed(1)}ms 推理${inferMs.toFixed(1)}ms 最大${profAvg.maxDt.toFixed(0)}ms${profAvg.maxDt>25?' ⚠含突变':''}`;
      // DevTools 打开时，每秒 console.log 的格式化/界面重绘会抢占下一次
      // vsync，制造 profiler 自己报告的 26~35ms 突变。默认不写控制台；
      // 显式加 ?profconsole=1 时才输出每秒摘要。
      if (running && !mapMenuOpen && location.search.includes('profconsole=1')) console.log(profLine);
      // 调试：?profdom=1 时把 [prof] 行写进 document.title（不开 console 也能读帧率/推理）
      if (location.search.includes('profdom=1')) document.title = profLine;
      prof.frames = 0; prof.t0 = now; prof.sumDt = 0; prof.maxDt = 0; prof.renderMs = 0;
      prof.mouseSearchLast = 0;
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
           <div class="mm-stats" id="mm-stats">
             <div class="mm-stat"><span>初始泡泡 <output id="mm-bombs-v"></output></span><input id="mm-bombs" type="range" min="1" max="10" step="1"></div>
             <div class="mm-stat"><span>最大泡泡 <output id="mm-bombs-max-v"></output></span><input id="mm-bombs-max" type="range" min="1" max="10" step="1"></div>
             <div class="mm-stat"><span>初始威力 <output id="mm-blast-v"></output></span><input id="mm-blast" type="range" min="1" max="8" step="1"></div>
             <div class="mm-stat"><span>最大威力 <output id="mm-blast-max-v"></output></span><input id="mm-blast-max" type="range" min="1" max="8" step="1"></div>
             <div class="mm-stat"><span>初始速度 <output id="mm-speed-v"></output></span><input id="mm-speed" type="range" min="0.5" max="2.3" step="0.05"></div>
             <div class="mm-stat"><span>最大速度 <output id="mm-speed-max-v"></output></span><input id="mm-speed-max" type="range" min="0.5" max="2.3" step="0.05"></div>
           </div>
         </div>
       </div>` +
      `<button id="mm-enter-btn" class="mm-enter" type="button">点击进入</button>` +
      (isTouch()
        ? `<span class="tip">左摇杆移动 · 💣 键放泡</span>`
        : `<span class="tip">方向键 / WASD 移动 · 空格 放泡</span>`) +
      `<span class="tip">点击分类展开 → 选择地图 → 调整属性 → 点击进入；R 重开</span>`;
    const tree = document.getElementById('mm-tree');
    const prevImg = document.getElementById('mm-prev-img');
    const prevName = document.getElementById('mm-prev-name');
    const prevMeta = document.getElementById('mm-prev-meta');
    const enterBtn = document.getElementById('mm-enter-btn');
    const statIds = ['bombs', 'bombs-max', 'blast', 'blast-max', 'speed', 'speed-max'];
    const statEls = Object.fromEntries(statIds.map((id) => [id, document.getElementById(`mm-${id}`)]));
    const statOut = Object.fromEntries(statIds.map((id) => [id, document.getElementById(`mm-${id}-v`)]));
    const setSelected = (l) => {
      selectedLevel = l;
      const st = l.initial_stats || { bombs: 2, blast: 2, speed: 1.3 };
      customStats = {
        bombs: st.bombs, blast: st.blast, speed: st.speed,
        bombsMax: l.bombs_max || CFG.growthBombsMax,
        blastMax: l.blast_max || CFG.growthBlastMax,
        speedMax: l.speed_max || CFG.growthSpeedMax,
      };
      const values = [customStats.bombs, customStats.bombsMax, customStats.blast,
                      customStats.blastMax, customStats.speed, customStats.speedMax];
      statIds.forEach((id, i) => { statEls[id].value = values[i]; statOut[id].textContent = values[i]; });
      for (const el of tree.children) {
        const children = el.children && el.children[1];
        if (!children || !children.children) continue;
        for (const item of children.children) {
          item.classList.remove('mm-cur');
          if (item._levelId === String(l.id)) item.classList.add('mm-cur');
        }
      }
    };
    const syncStats = () => {
      const n = (id) => Number(statEls[id].value);
      let bombsMax = n('bombs-max'), blastMax = n('blast-max'), speedMax = n('speed-max');
      let bombs = Math.min(n('bombs'), bombsMax);
      let blast = Math.min(n('blast'), blastMax);
      let speed = Math.min(n('speed'), speedMax);
      statEls.bombs.value = bombs; statEls.blast.value = blast; statEls.speed.value = speed;
      customStats = { bombs, blast, speed, bombsMax, blastMax, speedMax };
      const vals = [bombs, bombsMax, blast, blastMax, speed.toFixed(2), speedMax.toFixed(2)];
      statIds.forEach((id, i) => { statOut[id].textContent = vals[i]; });
    };
    statIds.forEach((id) => statEls[id].addEventListener('input', (ev) => {
      ev.stopPropagation(); syncStats();
    }));
    if (enterBtn) enterBtn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (selectedLevel) startGame();
    });
    const showPrev = (l, extraName, extraMeta, imgSrc) => {
      if (!l && !extraName && !imgSrc) return;
      if (prevImg) prevImg.src = imgSrc || (l && l.thumb) || '';
      if (prevName) prevName.textContent = extraName || (l ? (l.name || l.source) : '');
      const st = l ? (l.initial_stats || {}) : {};
      if (prevMeta) prevMeta.textContent = extraMeta ||
        (l ? `${l.mode} · ${l.source}` : '');
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
        const l = levels[Math.floor(Math.random() * levels.length)];
        setSelected(l);
        showPrev(l);
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
          setSelected(emptyLv);
          showPrev(emptyLv);
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
        item._levelId = String(l.id);
        item.innerHTML =
          `<span class="mm-name">${l.name || l.source}</span>`;
        item.addEventListener('mouseenter', () => showPrev(l));
        item.addEventListener('click', (ev) => {
          ev.stopPropagation();
          setSelected(l);
          showPrev(l);
        });
        children.appendChild(item);
      }
      node.appendChild(head);
      node.appendChild(children);
      tree.appendChild(node);
    }
    if (selectedLevel) {
      setSelected(selectedLevel);
      showPrev(selectedLevel);
    }
      elBanner.classList.remove('hidden');
    }
  function openMapMenu() {
    running = false;
    mapMenuOpen = true;                       // 冻结渲染
    prof.tickLast = 0;
    prof.tickParts = null;                    // 不把上一局 tick 误标到选图页抖动
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
