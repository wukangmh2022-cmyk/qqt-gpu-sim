#!/usr/bin/env python3
"""从节点清单生成 /tmp/ndrun 的 SSH 封装（cmd_N / scp_N）。

用法: python3 scripts/gen_ssh_wrappers.py [nodes文件] [真实网关IP]
  nodes文件格式（两行一组）:
    ssh -p 端口 root@主机
    密码
  真实网关IP: 必须填！本机代理是 fake-ip 模式，域名解析全被劫持成
  198.18.x.x 假 IP（连接全死）。用 DoH 查真实 IP:
    https://dns.alidns.com/resolve?name=<域名>&type=A
  当前真实 IP = 42.228.13.144（2026-08-30）。
"""
import os, re, sys

nodes_file = sys.argv[1] if len(sys.argv) > 1 else "deploy_10node/nodes_24x2.txt"
real_ip = sys.argv[2] if len(sys.argv) > 2 else "42.228.13.144"
OUT = "/tmp/ndrun"
lines = [l.strip() for l in open(nodes_file) if l.strip()]
pairs = []
for i in range(0, len(lines), 2):
    m = re.match(r"ssh -p (\d+) root@(\S+)", lines[i])
    if m and i + 1 < len(lines):
        pairs.append((m.group(1), lines[i + 1]))
print(f"{len(pairs)} 台节点, 网关 {real_ip}")
os.makedirs(OUT, exist_ok=True)
for rank, (port, pw) in enumerate(pairs):
    for kind, tmo in (("cmd", 300), ("scp", 600)):
        if kind == "cmd":
            body = (f"set timeout {tmo}\nset cmd [lindex $argv 0]\n"
                    f"spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p {port} root@{real_ip} $cmd\n"
                    'expect {\n  "password:" { send -- "$pw\r"; exp_continue }\n'
                    '  "yes/no" { send "yes\r"; exp_continue }\n  eof {}\n'
                    f'  timeout {{ puts "TIMEOUT_{rank}" }}\n}}')
        else:
            body = (f"set timeout {tmo}\nset src [lindex $argv 0]\nset dst [lindex $argv 1]\n"
                    f"spawn scp -o StrictHostKeyChecking=accept-new -P {port} $src root@{real_ip}:$dst\n"
                    'expect {\n  "password:" { send -- "$pw\r"; exp_continue }\n'
                    '  "yes/no" { send "yes\r"; exp_continue }\n  eof {}\n'
                    f'  timeout {{ puts "SCP_TIMEOUT_{rank}" }}\n}}')
        path = f"{OUT}/{kind}_{rank}"
        open(path, "w").write(f'#!/usr/bin/expect -f\nset pw "{pw}"\n{body}\n')
        os.chmod(path, 0o755)
print(f"封装写入 {OUT}/cmd_0..{len(pairs)-1} 与 scp_0..{len(pairs)-1}")
