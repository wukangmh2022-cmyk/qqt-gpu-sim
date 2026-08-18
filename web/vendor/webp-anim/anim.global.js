/* wasm-webp（libwebp 1.3.2）浏览器胶水：暴露帧间差分编码接口。
 *
 * webp-wasm.js 是 emscripten -sMODULARIZE 产物，裸 <script> 加载时工厂函数
 * 在全局变量 Module 上（见 webp-wasm.js 末尾补丁）。这里把 libwebp 的
 * WebPAnimEncoder（帧间差分压缩）包装成 window.WebPAnim：
 *   - decodeRGBA(bytes)      → { width, height, data } 静态 WebP → RGBA
 *   - encodeAnimation(w,h,hasAlpha,frames) → 动画 WebP 字节（真差分，非全关键帧）
 *     frames: [{ data: Uint8Array(RGBA), duration: ms, config?: {quality, lossless} }]
 */
(function () {
  const factory = window.WebPAnimModule;
  if (!factory) {
    window.WebPAnim = null;
    return;
  }
  let instPromise = null;
  function mod() {
    if (!instPromise) instPromise = factory();
    return instPromise;
  }
  window.WebPAnim = {
    ready: mod(),
    async decodeRGBA(data) {
      return (await mod()).decodeRGBA(data);
    },
    async encodeAnimation(width, height, hasAlpha, frames) {
      const m = await mod();
      const vec = new m.VectorWebPAnimationFrame();
      for (const f of frames) {
        const config = Object.assign({ quality: 85, lossless: 0 }, f.config || {});
        // data 必须立即拷进 C++ 堆（embind push_back 时拷贝），
        // 防止复用 wasm 堆上 decodeRGBA 的 view 被下一次调用覆盖。
        vec.push_back({
          duration: f.duration,
          data: new Uint8Array(f.data),
          config,
          has_config: 1,
        });
      }
      return m.encodeAnimation(width, height, hasAlpha, vec);
    },
  };
})();
