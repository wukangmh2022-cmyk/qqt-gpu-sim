"""并发快速拉起 10 节点训练进程。"""
import subprocess
import threading
import time

MASTER_IP = "172.31.204.214"
MASTER_PORT = 29540

def run_cmd(rank):
    cmd_str = (
        f"cd /root/private_data/qqt-gpu-sim; "
        f"export LC_ALL=C.UTF-8 LANG=C.UTF-8; source /opt/dtk/env.sh 2>/dev/null; "
        f"unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; "
        f"export LD_PRELOAD=$(ls /usr/mpi/gcc/openmpi-*/lib/libmpi.so 2>/dev/null | head -1); "
        f"export WORLD_SIZE=10 RANK={rank} MASTER_ADDR={MASTER_IP} MASTER_PORT={MASTER_PORT}; "
        f"export LSGD_K=32 LSGD_MODE=grad CKPT_DIR=ckpt CKPT_EVERY=30 CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=5; "
        f"nohup python3 -u -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4 "
        f"--num-envs 32768 --num-steps 256 --minibatch 32768 --epochs 2 --iters 15500 --lsgd-k 32 --lsgd-mode grad "
        f"--levels levels.json --level-weights 'empty=0.05,功夫=0.1,比武=0.15' "
        f"--crate-reward-coef 0.5 --crate-reward-anneal-steps 30000000000 "
        f"--explore-reward-coef 0.01 --explore-reward-anneal-steps 30000000000 "
        f"--brick-reward-coef 0.05 --reward-anneal-k 1.2 --fresh > /root/private_data/train_r{rank}.log 2>&1 & "
        f"echo RANK{rank}_LAUNCHED"
    )
    # 先清理
    clean_cmd = f"/tmp/ndrun/cmd_{rank} \"pkill -9 -f python3 2>/dev/null; fuser -k {MASTER_PORT}/tcp 2>/dev/null; rm -rf /root/private_data/train_r*.log /root/private_data/qqt-gpu-sim/ckpt/* /root/private_data/qqt-gpu-sim/ckpt_local/* 2>/dev/null\""
    subprocess.run(clean_cmd, shell=True, capture_output=True, timeout=15)
    time.sleep(1)
    # 启动
    launch_cmd = f"/tmp/ndrun/cmd_{rank} \"{cmd_str}\""
    res = subprocess.run(launch_cmd, shell=True, capture_output=True, text=True, timeout=20)
    print(f"[Rank {rank}] 启动结果: {res.stdout.strip()}", flush=True)

print(f"=== 并发拉起 10 节点 (MASTER={MASTER_IP}:{MASTER_PORT}) ===", flush=True)

# Rank 0 先启动
t0 = threading.Thread(target=run_cmd, args=(0,))
t0.start()
t0.join()
time.sleep(3)

# Rank 1~9 并发启动
threads = []
for r in range(1, 10):
    t = threading.Thread(target=run_cmd, args=(r,))
    threads.append(t)
    t.start()
    time.sleep(0.5)

for t in threads:
    t.join()

print("=== 10 节点全部并发拉起完毕！ ===", flush=True)
