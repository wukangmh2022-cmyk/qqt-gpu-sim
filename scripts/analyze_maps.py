#!/usr/bin/env python3
"""241 张关卡地图的"初始状态"统计 —— Spawn-Distance 课程设计的数据基础。

对每张图计算：
  room_size   = 出生点房间大小：从出生点 BFS，撞到墙/砖前能到达的格数
               （= 开局"选择空间"；Pommerman 式 2x2 小房间 ≈ 4）
  room_r2     = 出生点 2 格半径内的可走格数（更宽松的选择空间口径）
  sp_dist     = 出生点对最短曼哈顿距离
  isolated    = 出生点间是否被墙/砖完全隔开（BFS 不可达）
  brick_pct   = 砖密度（brick / 总格数）
  n_spawns    = 可选出生点数
  open        = 是否空场景（无墙无砖）

用法：python3 scripts/analyze_maps.py [--json out.json]
输出：按难度分组统计 + 每图一行。
"""
import sys
import os
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'web', 'assets', 'maps', 'levels.json')

maps = json.load(open(MAPS))
if isinstance(maps, dict):
    maps = maps.get('levels', maps.get('maps', []))

H, W = 13, 15
D = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def bfs_blocked(wall, brick, start, limit=None, stop_at_block=False):
    """从 start 出发的可达格数；blocked=墙|砖。stop_at_block：撞墙停（房间口径）。"""
    w, h = W, H
    blocked = [1 if wall[i] or brick[i] else 0 for i in range(w * h)]
    from collections import deque
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        if limit is not None and d >= limit:
            continue
        for dr, dc in D:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nk = nr * w + nc
            if nk in seen or blocked[nk]:
                continue
            seen.add(nk)
            q.append(((nr, nc), d + 1))
    return len(seen)


def room_size(wall, brick, sp):
    """房间大小：从出生点 BFS 撞墙停（不含墙砖）——Pommerman 2x2 ≈ 4。"""
    return bfs_blocked(wall, brick, sp)


rows = []
for m in maps:
    wall, brick = m.get('wall', []), m.get('brick', [])
    sps = m.get('spawns') or []
    if len(wall) != H * W or len(sps) < 2:
        continue
    sp = (sps[0][0], sps[0][1])
    rs = room_size(wall, brick, sp)
    r2 = bfs_blocked(wall, brick, sp, limit=2)
    # 出生点对：所有两两组合的最小距离 + 是否全隔离
    dists = []
    iso_all = 0
    for i in range(len(sps)):
        for j in range(i + 1, len(sps)):
            a, b = sps[i], sps[j]
            dists.append(abs(a[0] - b[0]) + abs(a[1] - b[1]))
            # 隔离：b 从 a BFS 不可达
            if bfs_blocked(wall, brick, (a[0], a[1]), limit=None) and False:
                pass
            # 用可达集合判断
            blocked = [1 if wall[k] or brick[k] else 0 for k in range(H * W)]
            from collections import deque
            seen = {(a[0], a[1])}
            q = deque([(a[0], a[1])])
            while q:
                r, c = q.popleft()
                for dr, dc in D:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    nk = nr * W + nc
                    if nk in seen or blocked[nk]:
                        continue
                    seen.add(nk)
                    q.append((nr, nc))
            if (b[0], b[1]) not in seen:
                iso_all += 1
    brick_cnt = sum(brick)
    rows.append({
        'id': m.get('id'), 'name': m.get('name'), 'theme': m.get('theme'),
        'source': m.get('source'),
        'room': rs, 'room2': r2,
        'dist_min': min(dists) if dists else 0,
        'dist_max': max(dists) if dists else 0,
        'iso_pairs': iso_all,
        'n_pairs': len(dists),
        'brick_pct': brick_cnt / (H * W),
        'n_spawns': len(sps),
        'crate_rate': m.get('crate_rate'),
    })

# ---- 分组 ----
def bucket(r):
    if r['source'] == 'empty_scene':
        return 'A空场景(无墙无砖,大空间)'
    if r['room'] <= 4 and r['iso_pairs'] == 0 and r['dist_min'] <= 6:
        return 'B小房间可通(选择少,起点)'
    if r['room'] <= 4 and r['iso_pairs'] == 0:
        return 'C小房间远距(选择少,距离大)'
    if r['iso_pairs'] == r['n_pairs']:
        return 'D全隔离(出生点被隔开)'
    if r['iso_pairs'] > 0:
        return 'E部分隔离'
    return 'F通图(大空间可通)'

for r in rows:
    r['bucket'] = bucket(r)

by = Counter(r['bucket'] for r in rows)
print(f"共 {len(rows)} 张（13×15），按开局形态分组：")
for k, v in sorted(by.items()):
    print(f"  {k}: {v}")
print()
print("每张图（按组）: id 名称 [主题] 房间 room2 dist_min iso对/总对 砖%% 出生点")
for g in sorted(set(r['bucket'] for r in rows)):
    print(f"\n== {g} ==")
    for r in sorted(rows, key=lambda x: (x['bucket'], x['room'], -x['dist_min'])):
        if r['bucket'] != g:
            continue
        print(f"  {r['id']:>3} {r['name']:<12}[{r['theme'] or '?'}] "
              f"room={r['room']:>2} r2={r['room2']:>2} "
              f"dist={r['dist_min']}-{r['dist_max']:>2} "
              f"iso={r['iso_pairs']}/{r['n_pairs']} "
              f"砖{r['brick_pct']*100:>3.0f}% spawns={r['n_spawns']}")

if len(sys.argv) > 1 and sys.argv[1] == '--json':
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/map_stats.json'
    json.dump(rows, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f"\nJSON → {out}")

# ---- Spawn-Distance 课程：按开局形态分 4 阶段（S1 最简单 → S4 全图）----
#   S1 = Pommerman 式起点：出生点 2x2 房间(room=4) + 出生点距离≤4（21 张候选）
#   S2 = room 4-6 + dist≤8
#   S3 = room≤10 + dist≤12（覆盖主流图）
#   S4 = 全部 241
if '--curriculum' in sys.argv:
    s1 = sorted(r['id'] for r in rows if r['room'] == 4 and r['dist_min'] <= 4)
    s2 = sorted(r['id'] for r in rows
                if r['room'] in (4, 5, 6) and r['dist_min'] <= 8 and r['id'] not in s1)
    s3 = sorted(r['id'] for r in rows
                if r['room'] <= 10 and r['dist_min'] <= 12
                and r['id'] not in s1 + s2 and r['source'] != 'empty_scene')
    s4 = sorted(r['id'] for r in rows)   # 全部 241（含空场景，S4 均分）
    cur = {'thresholds': [0.01, 0.04, 0.15],   # 全局步比例：S1<1% → S2<4% → S3<15% → S4
           'stages': [s1, s2, s3, s4]}
    out = sys.argv[sys.argv.index('--curriculum') + 1] \
        if len(sys.argv) > sys.argv.index('--curriculum') + 1 \
        else '/tmp/curriculum.json'
    json.dump(cur, open(out, 'w'))
    print(f"课程 → {out}: S1={len(s1)} S2={len(s2)} S3={len(s3)} S4={len(s4)} 张")
