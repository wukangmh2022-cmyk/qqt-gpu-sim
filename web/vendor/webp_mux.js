/* 动画 WebP 容器合成器（全关键帧）。
 *
 * macOS QuickLook/Preview（ImageIO）对增量帧动画的解码是 O(N²)：每帧都从
 * 第 0 帧重解整条增量链，帧越靠后越慢（245 帧实测整片解码 100s）。因此这里
 * 只用"每帧一个完整关键帧"的格式 —— 每帧独立解码 ~10ms，任意播放器实时播放。
 *
 * 输入：每帧的静态 WebP 字节（native canvas.toBlob 或 libwebp wasm 编码），
 * 提取 VP8/VP8L 位流后按 ANMF 合成。RIFF 结构：
 *   VP8X（flags：全不透明帧 0x02=animation，透明帧 0x12=alpha+animation；
 *   libwebp 惯例，QuickLook/ImageIO/浏览器通用）→ ANIM（背景色 4B + loop 2B，
 *   loop=0 无限循环）→ ANMF×N（16B 头 + 内层 'VP8 '/'VP8L' + size + 位流）。
 */
(function (root) {
  'use strict';

  function u24(n) { return [(n >> 0) & 255, (n >> 8) & 255, (n >> 16) & 255]; }
  function u32(n) { return [(n >> 0) & 255, (n >> 8) & 255, (n >> 16) & 255, (n >> 24) & 255]; }
  function cat(arrs) {
    let total = 0;
    for (const a of arrs) total += a.length;
    const out = new Uint8Array(total);
    let o = 0;
    for (const a of arrs) { out.set(a, o); o += a.length; }
    return out;
  }
  function chunk(fourCC, payload) {
    return cat([new TextEncoder().encode(fourCC), u32(payload.length), payload]);
  }

  // 从静态 WebP（RIFF）里取出 'VP8 '/'VP8L' 位流
  function extractVp8(still) {
    if (still[0] !== 0x52 || still[1] !== 0x49 || still[2] !== 0x46 || still[3] !== 0x46) {
      throw new Error('不是 RIFF/WebP 数据');
    }
    let p = 12;
    while (p + 8 <= still.length) {
      const tag = String.fromCharCode(still[p], still[p + 1], still[p + 2], still[p + 3]);
      const size = still[p + 4] | (still[p + 5] << 8) | (still[p + 6] << 16) | (still[p + 7] << 24);
      if (tag === 'VP8 ' || tag === 'VP8L') {
        return { tag, data: still.slice(p, p + 8 + size) };
      }
      p += 8 + size + (size & 1);
    }
    throw new Error('静态 WebP 里没有 VP8/VP8L 位流');
  }

  // parts: [{ tag: 'VP8 '/'VP8L', data: 内层块字节(含 4CC+size+位流), durMs }]
  // hasAlpha: 帧像素是否含透明通道。全不透明写 0x02（libwebp 惯例，QuickLook/
  // ImageIO/各浏览器都认）；有透明才写 0x12。注意 0x10（alpha 无动画位）会被
  // macOS 解码器拒收 —— 动画位必须置 1。
  function muxWebpAnimation(parts, canvasW, canvasH, loop, hasAlpha) {
    if (!parts.length) throw new Error('没有帧');
    loop = loop || 0;
    // VP8X：flags(1) + reserved(3) + canvasW-1(3) + canvasH-1(3)
    const vp8x = chunk('VP8X', cat([new Uint8Array([hasAlpha ? 0x12 : 0x02, 0, 0, 0]),
      u24(canvasW - 1), u24(canvasH - 1)]));
    // ANIM：背景色 4B（全透明黑，libwebp 惯例）+ loop 2B（0 = 无限循环）
    const anim = chunk('ANIM', cat([new Uint8Array([0xff, 0xff, 0xff, 0xff]), u24(loop).slice(0, 2)]));
    const frames = parts.map((f) => {
      const header = cat([new Uint8Array([0, 0, 0, 0, 0, 0]),   // x=0, y=0
        u24(canvasW - 1), u24(canvasH - 1),                     // w-1, h-1
        u24(f.durMs), new Uint8Array([0])]);                    // duration, flags
      return chunk('ANMF', cat([header, f.data]));
    });
    return cat([new TextEncoder().encode('RIFF'),
      u32(4 + vp8x.length + anim.length + frames.reduce((s, f) => s + f.length, 0)),
      new TextEncoder().encode('WEBP'), vp8x, anim, ...frames]);
  }

  const api = { extractVp8, muxWebpAnimation };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.QQTWebpMux = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
