"""列出 open 地图 vs ckpt(AI) 的录像，按对手分组各取最长 2 局（供选演示候选）。"""
import ast, glob, numpy as np, os

recs = sorted(glob.glob("recordings/*.npz"))
rows = []
for p in recs:
    d = np.load(p, allow_pickle=True)
    meta = ast.literal_eval(str(d["meta"][0])) if d["meta"].size else {}
    rows.append((d["action"].shape[0], meta.get("map", "?"),
                 str(meta.get("opp", "?")), int(d["pid"]), os.path.basename(p)))

op_ai = [r for r in rows if r[1] == "open" and r[2].endswith(".pt")]
print(f"open vs ckpt(AI): {len(op_ai)} 局")
by_opp = {}
for r in op_ai:
    by_opp.setdefault(r[2], []).append(r)
for opp, lst in sorted(by_opp.items(), key=lambda kv: -len(kv[1])):
    lst.sort(reverse=True)
    print(f"  {opp}: {len(lst)} 局")
    for r in lst[:2]:
        print(f"    {r[0]:5d}tick pid={r[3]} {r[4]}")
