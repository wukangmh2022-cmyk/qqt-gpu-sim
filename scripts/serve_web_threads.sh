#!/bin/bash
# 带 COOP/COEP 头的静态服务：允许 onnxruntime-web WASM 多线程推理
# （原 serve_web.sh 的 http.server 无这些头 → numThreads 被禁 → ViT 单线程慢）
#
# 用法：
#   bash scripts/serve_web_threads.sh [端口]     # 默认 8080
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/web"
PORT="${1:-8080}"
echo "== 开服 http://localhost:${PORT} (COOP/COEP 已启用, WASM 多线程可用) =="
PORT="$PORT" exec "$ROOT/.venv/bin/python" - <<'PY'
import http.server, socketserver, os

PORT = int(os.environ.get('PORT', '8080'))

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # COOP/COEP: 让 onnxruntime-web 的 WASM 可以用多线程(SharedArrayBuffer)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    def log_message(self, fmt, *args):
        pass

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

with S(('127.0.0.1', PORT), H) as httpd:
    httpd.serve_forever()
PY
