#!/usr/bin/env python3
"""固定锚点 Elo 追踪：拉 rank0 最新快照，vs 三个永久锚点各打 ~120 局（换边均衡，
含平局三方计数），追加 monitor/anchor_elo.csv 并重建报告图。SSH 不可达时安全退出。"""
import sys, os, pickle, subprocess, csv
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MON = os.path.join(ROOT, "monitor")

ndrun = "/tmp/ndrun/cmd_0"
if not os.path.exists(ndrun):
    print("SSH 封装不存在（本机重启过），跳过"); sys.exit(1)
# 1) 找 rank0 最新快照名并拉回
r = subprocess.run([ndrun, "ls /root/private_data/qqt-gpu-sim/ckpt_local/params_it*.pkl | grep -v ema | sort | tail -1"],
                   capture_output=True, text=True, timeout=120)
latest = r.stdout.strip().splitlines()[-1].split("/")[-1] if r.stdout.strip() else ""
if not latest:
    print("取数失败（SSH 不可达？）"); sys.exit(1)
local = os.path.join(MON, "snap_" + latest)
r2 = subprocess.run(["/tmp/ndrun/scp_0", f"/root/private_data/qqt-gpu-sim/ckpt_local/{latest}", MON],
                    capture_output=True, text=True, timeout=300)
if not os.path.exists(local):
    print(f"拉取失败: {latest}"); sys.exit(1)
print(f"[pull] {latest} -> {local}", flush=True)

import jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, sample_actions

_levels.set_active(os.path.join(ROOT, "web/assets/maps/levels.json"), weights="")
def load(p):
    with open(p, "rb") as f: return jax.tree.map(jnp.asarray, pickle.load(f))
ANCHORS = [("it00000158", "params_it00000158.pkl"), ("it00000801", "params_it00000801.pkl"),
           ("it00001450", "params_it00001450.pkl")]
N_ENVS, STEPS = 32, 1800
chal = load(local)
out_row = [datetime.now().strftime("%Y-%m-%d %H:%M"), latest.replace("params_", "").replace(".pkl", "")]
for aname, afile in ANCHORS:
    apath = os.path.join(MON, "anchor_" + afile)
    if not os.path.exists(apath): apath = os.path.join(ROOT, "ckpt_local", afile)
    if not os.path.exists(apath):
        print(f"锚点缺失: {aname}"); out_row += [None]; continue
    anchor = load(apath)
    tot = np.zeros(3, np.int64)
    for swap in (0, 1):
        fpa, fpb = (anchor, chal) if swap else (chal, anchor)
        states = init_batch(jrandom.PRNGKey(777 + swap), N_ENVS)
        key = jrandom.PRNGKey(55 + swap)
        n = N_ENVS
        def one_step(carry, _):
            states, key = carry
            key, k0, k1, kstep = jrandom.split(key, 4)
            obs = both_perspectives(states); masks = both_masks(states); gv = both_states(states)
            a0 = sample_actions(fpa, "transformer", obs[:n], (masks[0][:n], masks[1][:n]), k0, state=gv[:n])[0]
            a1 = sample_actions(fpb, "transformer", obs[n:], (masks[0][n:], masks[1][n:]), k1, state=gv[n:])[0]
            env_acts = jnp.stack([a0, a1], axis=1)
            keys = jrandom.split(kstep, n)
            new_states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, return_info=True))(states, env_acts, keys)
            n_alive = info["alive"].sum(-1)
            dd = done & (n_alive == 1)
            p0w = dd & info["alive"][:, 0]
            p0l = dd & ~info["alive"][:, 0]
            aa = done & (n_alive == 2)
            hp = info["hp"].astype(jnp.float32)
            p0w = p0w | (aa & (hp[:, 0] > hp[:, 1]))
            p0l = p0l | (aa & (hp[:, 0] < hp[:, 1]))
            draw = done & ~(p0w | p0l)
            return (new_states, key), (p0w.sum(), p0l.sum(), draw.sum())
        (states, key), (w, l, d) = jax.lax.scan(one_step, (states, key), None, length=STEPS)
        w, l, d = int(w.sum()), int(l.sum()), int(d.sum())
        a_win, a_lose, a_draw = (w, l, d) if swap == 0 else (l, w, d)
        tot += np.array([a_win, a_lose, a_draw])
    T = int(tot.sum())
    print(f"[anchor] vs {aname}: 胜{tot[0]} 负{tot[1]} 平{tot[2]} / {T}局 (含平胜率={tot[0]/max(T,1):.1%})", flush=True)
    out_row += [round(tot[0]/max(T,1), 3), T]

new = not os.path.exists(os.path.join(MON, "anchor_elo.csv"))
with open(os.path.join(MON, "anchor_elo.csv"), "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["time", "snapshot", "wr_vs_it158", "wr_vs_it801", "wr_vs_it1450"])
    w.writerow(out_row)
subprocess.run([sys.executable, os.path.join(MON, "build_report.py")])
print("[DONE]", flush=True)
