#!/usr/bin/env python3
"""
web/server.py - 带有 TypeSafe Jev API 代理的静态文件服务器
托管 web/ 目录的所有静态资源，并提供 /api/typesafe 代理端点，解决浏览器直接调用云端大模型 API 的跨域与密钥保护问题。
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
WEB_DIR = os.path.dirname(os.path.abspath(__file__))

def get_typesafe_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    possible_envs = [
        os.path.join(os.path.dirname(WEB_DIR), ".env"),
        os.path.expanduser("~/Documents/杂项/jev_lab/jev-browser/.env"),
        os.path.expanduser("~/.env"),
    ]
    for env_path in possible_envs:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TYPESAFE_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
    return ""

TYPESAFE_API_KEY = get_typesafe_key()
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"

LOG_FILE = os.path.join(WEB_DIR, "jev_decisions.log")
recent_logs = []

class GameProxyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/jev_logs" or self.path == "/api/jev_logs/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(recent_logs[-50:], ensure_ascii=False).encode("utf-8"))
            return
        super().do_GET()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/typesafe" or self.path == "/api/typesafe/":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            t0 = time.time()
            try:
                # 转发到 TypeSafe API
                req = urllib.request.Request(
                    TYPESAFE_URL,
                    data=post_data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {TYPESAFE_API_KEY}",
                        "User-Agent": "QQT-Jev-Bridge/1.0"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = resp.read()
                    status_code = resp.status
                    latency_ms = int((time.time() - t0) * 1000)

                    # 解析并记录决策日志
                    try:
                        req_json = json.loads(post_data.decode("utf-8"))
                        res_json = json.loads(resp_data.decode("utf-8"))
                        st = req_json.get("state", {})
                        answers = res_json.get("answers", {})
                        
                        prio = answers.get("tactical_priority", {}).get("choice", "N/A")
                        target = answers.get("target_selection", {}).get("choice", "N/A")
                        bomb_noul = answers.get("place_bomb_now", {}).get("noul", None)
                        
                        log_entry = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tick": st.get("step", 0),
                            "priority": prio,
                            "target": target,
                            "place_bomb": bomb_noul,
                            "player_pos": st.get("player", {}).get("pos"),
                            "opp_pos": st.get("opponent", {}).get("pos"),
                            "opp_dist": st.get("opponent", {}).get("distance"),
                            "in_line_of_fire": st.get("tactical_context", {}).get("in_line_of_fire", False),
                            "latency_ms": latency_ms
                        }
                        recent_logs.append(log_entry)
                        
                        log_line = f"[{log_entry['time']}] [Tick {log_entry['tick']}] 优先级: {prio} | 目标: {target} | 放泡置信: {bomb_noul} | 自身: {log_entry['player_pos']} | 对手: {log_entry['opp_pos']} (距 {log_entry['opp_dist']}) | 直瞄火线: {log_entry['in_line_of_fire']} | 耗时: {latency_ms}ms\n"
                        with open(LOG_FILE, "a", encoding="utf-8") as lf:
                            lf.write(log_line)
                        print(log_line.strip())
                    except Exception as le:
                        pass

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp_data)
            except urllib.error.HTTPError as e:
                err_data = e.read()
                print(f"[server] ❌ TypeSafe HTTPError {e.code}: {err_data.decode('utf-8', errors='ignore')}")
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(err_data)
            except Exception as e:
                print(f"[server] ❌ 代理异常: {type(e).__name__}: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return
            
        self.send_error(404, "Endpoint not found")

def run_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, GameProxyHandler)
    print(f"QQT Game Server with TypeSafe Proxy running on http://localhost:{PORT}/ (serving {WEB_DIR})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
