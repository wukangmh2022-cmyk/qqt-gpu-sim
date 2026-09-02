import subprocess, time, sys

print("Starting auto-poll on Rank 10 (11195) and Rank 11 (11196)...", flush=True)

def ping_node(rank):
    try:
        cmd = f"/tmp/ndrun/cmd_{rank} 'echo PING_OK'"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
        out = res.stdout.strip()
        return "PING_OK" in out
    except Exception:
        return False

attempt = 0
while True:
    attempt += 1
    ok10 = ping_node(10)
    ok11 = ping_node(11)
    
    t_str = time.strftime('%H:%M:%S')
    s10 = "ONLINE" if ok10 else "... waiting"
    s11 = "ONLINE" if ok11 else "... waiting"
    print(f"[{t_str}] Attempt {attempt:03d}: Rank 10 = {s10}, Rank 11 = {s11}", flush=True)
    
    if ok10 and ok11:
        print("\n=== BOTH RANK 10 & 11 ARE ONLINE! Launching 24-node training NOW... ===", flush=True)
        proc = subprocess.Popen(["bash", "/tmp/launch5.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()
        proc.wait()
        print("\n=== Launch sequence complete! ===", flush=True)
        break
        
    time.sleep(5)
