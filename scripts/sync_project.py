#!/usr/bin/env python3
import os
import subprocess
import tempfile

PASS = "2TPXRDVQAODQZMK"
HOST = "ssh.zzai.scnet.cn"
PORT = "11345"
USER = "root"
REMOTE_DIR = "/root/qqt-gpu-sim"

def sync():
    # Create remote dir
    print(f"Creating remote dir {REMOTE_DIR}...")
    mkdir_script = f"""
set timeout 30
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST} "mkdir -p {REMOTE_DIR}"
expect {{
    "password:" {{
        send "{PASS}\\r"
        exp_continue
    }}
    eof
}}
"""
    subprocess.run(["/usr/bin/expect", "-c", mkdir_script], check=True)

    # Tar the local workspace with exclusions
    excludes = [
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.DS_Store",
        "--exclude=checkpoints",
        "--exclude=runs",
        "--exclude=node_modules",
        "--exclude=*.log",
    ]
    
    print("Syncing project archive over SSH...")
    pipe_script = f"""
set timeout 300
spawn bash -c "tar -czf - {' '.join(excludes)} -C . . | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST} 'tar -xzf - -C {REMOTE_DIR}'"
expect {{
    "password:" {{
        send "{PASS}\\r"
        exp_continue
    }}
    eof
}}
"""
    p = subprocess.run(["/usr/bin/expect", "-c", pipe_script], capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print("Sync failed with code", p.returncode)
        return p.returncode
    print("Project synced successfully!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(sync())
