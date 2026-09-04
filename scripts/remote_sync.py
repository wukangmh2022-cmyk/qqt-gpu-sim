#!/usr/bin/env python3
import sys
import subprocess
import os

PASS = "2TPXRDVQAODQZMK"
HOST = "ssh.zzai.scnet.cn"
PORT = "11345"
USER = "root"

def upload_dir(local_path, remote_dest):
    # Ensure remote directory exists
    mkdir_script = f"""
set timeout 30
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST} "mkdir -p {remote_dest}"
expect {{
    "password:" {{
        send "{PASS}\\r"
        exp_continue
    }}
    eof
}}
"""
    subprocess.run(["/usr/bin/expect", "-c", mkdir_script], check=True)

    # Tar and pipe over ssh
    base_dir = os.path.dirname(os.path.abspath(local_path))
    target_name = os.path.basename(os.path.abspath(local_path))

    pipe_script = f"""
set timeout 180
spawn bash -c "tar -czf - -C '{base_dir}' '{target_name}' | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST} 'tar -xzf - -C {remote_dest}'"
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
    return p.returncode

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: remote_sync.py <local_path> <remote_dest>")
        sys.exit(1)
    sys.exit(upload_dir(sys.argv[1], sys.argv[2]))
