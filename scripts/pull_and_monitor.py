import os, sys, subprocess, time, glob

NODE0_PORT = "11187"
NODE0_PW = "YTD5MAZHDEY5JRE"
LOCAL_CKPT_DIR = os.path.expanduser("~/Documents/llm-train/qqt-gpu-sim/ckpt")
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

def run_cmd(cmd, timeout=30):
    exp = f"""
set timeout {timeout}
spawn ssh -o StrictHostKeyChecking=accept-new -p {NODE0_PORT} root@ssh.zzai.scnet.cn "{cmd}"
expect {{
  "password:" {{ send "{NODE0_PW}\\r"; exp_continue }}
  eof
}}
"""
    res = subprocess.run(["expect", "-c", exp], capture_output=True, text=True)
    return res.stdout

def pull_ckpt():
    # SCP latest pkl from node 0
    scp_exp = f"""
set timeout 60
spawn scp -o StrictHostKeyChecking=accept-new -P {NODE0_PORT} root@ssh.zzai.scnet.cn:/root/private_data/qqt-gpu-sim/ckpt/params_it*.pkl {LOCAL_CKPT_DIR}/
expect {{
  "password:" {{ send "{NODE0_PW}\\r"; exp_continue }}
  eof
}}
"""
    subprocess.run(["expect", "-c", scp_exp], capture_output=True, text=True)

def get_status():
    out = run_cmd("tail -n 25 /root/private_data/qqt-gpu-sim/multicard_result.txt")
    lines = [l.strip() for l in out.splitlines() if l.strip() and not "spawn" in l and not "password" in l]
    return lines

if __name__ == "__main__":
    lines = get_status()
    print("=== LATEST CLUSTER TRAINING LOGS ===")
    for l in lines[-10:]:
        print(l)
    print("\n=== PULLING LATEST SNAPSHOT TO LOCAL ===")
    pull_ckpt()
    local_files = glob.glob(os.path.join(LOCAL_CKPT_DIR, "params_it*.pkl"))
    print(f"Local snapshots available: {len(local_files)}")
