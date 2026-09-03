#!/usr/bin/env python3
"""
monitor/server.py - 训练时监控与人机交互控制面 (Localhost:8088 Web API)
"""
import os
import sys
import json
import glob
import subprocess
import threading
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONITOR_DIR = os.path.join(ROOT, "monitor")
WEB_DIR = os.path.join(MONITOR_DIR, "web")
CONFIGS_DIR = os.path.join(ROOT, "configs")
HISTORY_FILE = os.path.join(MONITOR_DIR, "history.json")
ALERT_FILE = os.path.join(MONITOR_DIR, "ALERT.json")

eval_lock = threading.Lock()
eval_in_progress = False
current_evaluating_model = ""

def get_alert():
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def get_configs():
    files = glob.glob(os.path.join(CONFIGS_DIR, "*.toml"))
    return [os.path.basename(f) for f in sorted(files)]

def get_available_ckpts():
    ckpts = []
    for d in [os.path.join(ROOT, "ckpt"), os.path.join(ROOT, "ckpt_local")]:
        if os.path.exists(d):
            ckpts.extend(glob.glob(os.path.join(d, "params_it*.pkl")))
    ckpts = [os.path.basename(p).replace(".pkl", "") for p in sorted(ckpts, key=os.path.getmtime, reverse=True)]
    return list(dict.fromkeys(ckpts))

class MonitorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            history = get_history()
            latest = history[-1] if history else None
            alert = get_alert()
            ckpts = get_available_ckpts()
            self._send_json({
                "status": "online",
                "evalInProgress": eval_in_progress,
                "currentEvaluating": current_evaluating_model,
                "latestCheckpoint": latest.get("modelName") if latest else None,
                "health": latest.get("health") if latest else {"status": "UNKNOWN"},
                "activeAlert": alert,
                "totalEvaluated": len(history),
                "availableCheckpoints": ckpts[:15],
            })
            return

        if path == "/api/history":
            self._send_json(get_history())
            return

        if path == "/api/configs":
            self._send_json(get_configs())
            return

        if path == "/api/config":
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            target = os.path.join(CONFIGS_DIR, os.path.basename(name))
            if os.path.exists(target):
                with open(target, "r", encoding="utf-8") as f:
                    content = f.read()
                self._send_json({"name": name, "content": content})
            else:
                self._send_json({"error": "Config not found"}, code=404)
            return

        # 前端静态文件路由
        if path == "/" or not os.path.exists(os.path.join(WEB_DIR, path.lstrip("/"))):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        try:
            req_data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            req_data = {}

        if path == "/api/config":
            name = req_data.get("name")
            content = req_data.get("content")
            if not name or not content:
                self._send_json({"error": "Missing name or content"}, code=400)
                return
            target = os.path.join(CONFIGS_DIR, os.path.basename(name))
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            self._send_json({"success": True, "saved": name})
            return

        if path == "/api/eval_now":
            global eval_in_progress, current_evaluating_model
            model_name = req_data.get("model")
            if not model_name:
                self._send_json({"error": "Missing model name"}, code=400)
                return
            if eval_in_progress:
                self._send_json({"error": "Evaluation already in progress", "model": current_evaluating_model}, code=409)
                return

            def worker():
                global eval_in_progress, current_evaluating_model
                eval_in_progress = True
                current_evaluating_model = model_name
                try:
                    # 查找对应 pkl
                    ckpt_path = None
                    for d in [os.path.join(ROOT, "ckpt"), os.path.join(ROOT, "ckpt_local")]:
                        cand = os.path.join(d, f"{model_name}.pkl")
                        if os.path.exists(cand):
                            ckpt_path = cand
                            break
                    if ckpt_path:
                        import monitor.watchdog as wd
                        wd.process_checkpoint(ckpt_path, games=32, workers=4)
                finally:
                    eval_in_progress = False
                    current_evaluating_model = ""

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            self._send_json({"success": True, "started": model_name})
            return

        if path == "/api/action/restart":
            resume_ckpt = req_data.get("resumeCkpt")
            config_file = req_data.get("configFile")
            script_file = req_data.get("scriptFile", "scripts/launch_8node_it68_hlgauss_top25_patch3.sh")

            cmd_preview = f"CONFIG=configs/{os.path.basename(config_file)} RESUME={resume_ckpt} bash {script_file}"
            self._send_json({
                "success": True,
                "command": cmd_preview,
                "note": "建议人工或 Agent 确认超参修正后执行，或点击直接在后台异步启动。"
            })
            return

        self._send_json({"error": "Not Found"}, code=404)

def run_server(port=8088):
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, MonitorHandler)
    print(f"==========================================================================")
    print(f"🌐 QQ堂强化学习训练监控控制面已启动: http://localhost:{port}")
    print(f"   API 接口: /api/status, /api/history, /api/configs")
    print(f"   静态前端: {WEB_DIR}")
    print(f"==========================================================================")
    httpd.serve_forever()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run_server(port)
