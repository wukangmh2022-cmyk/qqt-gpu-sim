#!/usr/bin/env python3
import sys
import subprocess

PASS = "2TPXRDVQAODQZMK"
HOST = "ssh.zzai.scnet.cn"
PORT = "11345"
USER = "root"

def run_remote(cmd):
    expect_script = f"""
set timeout 600
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {PORT} {USER}@{HOST} {{bash -l -c {subprocess.list2cmdline([cmd])}}}
expect {{
    "password:" {{
        send "{PASS}\\r"
        exp_continue
    }}
    eof
}}
"""
    p = subprocess.run(["/usr/bin/expect", "-c", expect_script], capture_output=True, text=True)
    out = p.stdout
    lines = out.splitlines()
    clean = []
    skip = True
    for l in lines:
        if "password:" in l:
            skip = False
            continue
        if not skip:
            clean.append(l)
    return "\n".join(clean), p.returncode

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "uname -a"
    out, code = run_remote(cmd)
    print(out)
    sys.exit(code)
