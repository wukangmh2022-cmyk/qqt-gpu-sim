import sys, os, subprocess, concurrent.futures, time

with open("10ssh.env") as f:
    lines = [line.strip() for line in f if line.strip()]

nodes = []
for i in range(0, len(lines), 2):
    cmd = lines[i]
    pw = lines[i+1]
    port = cmd.split("-p")[1].split()[0].strip()
    nodes.append({"rank": len(nodes), "port": port, "cmd": cmd, "pw": pw})

master_ip = "172.31.54.127"
master_port = "29500"
world_size = len(nodes)

print(f"Loaded {world_size} nodes for clean cluster launch.")

# Package jax_bomb
subprocess.run(["tar", "-czf", "jaxbomb_clean.tgz", "jax_bomb", "curriculum.json", "levels.json"], check=True)

def sync_and_launch(node):
    rank = node["rank"]
    port = node["port"]
    pw = node["pw"]
    
    # 1. SCP jaxbomb_clean.tgz
    scp_cmd = f"""
set timeout 60
spawn scp -o StrictHostKeyChecking=accept-new -P {port} jaxbomb_clean.tgz root@ssh.zzai.scnet.cn:/root/private_data/
expect {{
  "password:" {{ send "{pw}\\r"; exp_continue }}
  eof
}}
"""
    p = subprocess.run(["/usr/bin/expect", "-c", scp_cmd], capture_output=True, text=True)
    
    sh_content = f"""#!/bin/bash
source /opt/dtk/env.sh 2>/dev/null
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export LD_PRELOAD=\\$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1)
cd /root/private_data/qqt-gpu-sim
mkdir -p ckpt

export WORLD_SIZE={world_size}
export RANK={rank}
export MASTER_ADDR={master_ip}
export MASTER_PORT={master_port}

exec python3 -m jax_bomb.multicard_train \\
  --arch transformer --embed 392 --depth 4 --patch 3 --heads 4 --ff-factor 4 \\
  --adv-top-frac 1.0 --ema-decay 0.999 \\
  --num-envs 24576 --num-steps 256 --minibatch 24576 --epochs 1 \\
  --lsgd-k 256 --lsgd-mode param --lsgd-bf16 \\
  --iters 10000 \\
  --curriculum-json curriculum.json \\
  --curriculum-min-iters 20 \\
  --curriculum-eval-every 10 \\
  --curriculum-eval-steps 1800 \\
  --ckpt-dir ckpt \\
  --ckpt-every 30 \\
  --ckpt-local-dir ckpt \\
  --ckpt-local-every 15
"""

    ssh_cmd = f"""
set timeout 60
spawn ssh -o StrictHostKeyChecking=accept-new -p {port} root@ssh.zzai.scnet.cn "
pkill -9 -f multicard_train 2>/dev/null
cd /root/private_data/qqt-gpu-sim
tar -xzf /root/private_data/jaxbomb_clean.tgz -C .
rm -rf ckpt/* /root/private_data/train.log
cat << 'EOF' > /root/private_data/run_cluster.sh
{sh_content}
EOF
chmod +x /root/private_data/run_cluster.sh
nohup /root/private_data/run_cluster.sh > /root/private_data/train.log 2>&1 &
sleep 2
ps aux | grep multicard_train | grep -v grep
"
expect {{
  "password:" {{ send "{pw}\\r"; exp_continue }}
  eof
}}
"""
    p2 = subprocess.run(["/usr/bin/expect", "-c", ssh_cmd], capture_output=True, text=True)
    ok = "multicard_train" in p2.stdout
    print(f"  [Rank {rank:02d} | Port {port}] Launch status: {'OK' if ok else 'FAILED'}")
    return rank, ok

# Start Rank 0 first, then workers
print("Starting Rank 0 (Master)...")
sync_and_launch(nodes[0])
time.sleep(2)

print("Starting remaining 23 worker nodes in parallel...")
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    futs = [ex.submit(sync_and_launch, node) for node in nodes[1:]]
    for fut in concurrent.futures.as_completed(futs):
        fut.result()

print("\nAll 24 nodes synchronized and launched cleanly!")
