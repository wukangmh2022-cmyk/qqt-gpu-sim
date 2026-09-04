"""标准化关卡数据加载与采样（纯 jax，无 torch 依赖）。

训练地图改为从 241 张 QQ堂原版关卡随机采样，替代过程式生成：
- 权威格式 levels_qqt/*.pt（qqt_to_levels.py + qqt_level_enrich.py 产出），
  但 DCU 训练机无 torch，统一从 export_web.py 导出的
  web/assets/maps/levels.json 加载 —— 同一数据的 torch-free 标准导出，
  Web 推理侧同源，保证训练与 Web 用同一批关卡。

每关字段（levels.json）：
  wall / brick       (H,W) 布尔（平铺行主序，全部 15×13）
  spawns             [[y,x],...] 出生点（每关 2-9 个；两人出生点必须不同）
  initial_stats      {bombs, blast, speed} 本关初始属性（掉血惩罚 clamp 下限）
  initial_crates     [[y,x],...] 预置宝箱（踩到必升，等价旧 open 十字池）
  crate_rate         本关炸砖变宝箱爆率（替代全局 CRATE_PROB）

权重可配置：weights="240=0.2"（或 LEVEL_WEIGHTS 环境变量）给指定关权重，
其余关均分剩余概率；"empty=0.2" 指名字含"空场景"的那关（默认第 240 关）。
未加载时 jax_env 保持过程式生成（旧行为，_fresh 走 _make_map）。

用法（任何 jit/vmap 之前调用一次）：
    from jax_bomb_v2 import levels
    levels.set_active("levels.json", weights="empty=0.2")
    # 之后 jax_env._fresh 自动走关卡采样（每 env 每次重置独立采样一关）
"""

from __future__ import annotations

import json
from typing import NamedTuple, Optional

import jax.numpy as jnp
import jax.random as jrandom

# 出生点 padding 上限（levels_qqt 实测最多 9 个，配对上限 66）
S_MAX = 12
P_MAX = 66


class LevelSample(NamedTuple):
    level_id: jnp.ndarray    # () int32 关卡 id
    pos: jnp.ndarray         # (2,2) float32 两个出生点（格中心，P0/P1 不同）
    wall: jnp.ndarray        # (H,W) bool
    brick: jnp.ndarray       # (H,W) bool
    bush: jnp.ndarray        # (H,W) bool 灌木：可通行 + 可炸毁（炸后有概率掉宝箱）
    pushable: jnp.ndarray    # (H,W) bool 可推墙（推箱子关；必 ⊆ brick，预留）
    crate: jnp.ndarray       # (H,W) bool 预置宝箱
    rec: jnp.ndarray         # (H,W) bool 预置宝箱标记（踩到必升）
    lo: jnp.ndarray          # (3,) float32 [bombs, blast, speed] 初始属性
    caps: jnp.ndarray        # (3,) float32 [bombs_max, blast_max, speed_max] 成长上限
    rate: jnp.ndarray        # () float32 本关炸砖/炸灌木爆率
    super_f: jnp.ndarray     # () float32 掉落的超级道具占比（0=无超级）
    is_open: jnp.ndarray     # () bool 无砖 = open 语义（信息性）


class LevelSet:
    """241 关的静态 jnp 栈 + 权重 + 出生点距离课程支持。构建后只读；sample 是纯函数，vmap 安全。"""

    def __init__(self, wall, brick, bush, pushable, crate, rec, lo, caps, rate,
                 super_f, spawns, cnt, pairs, pair_dists, n_pairs, logw, is_open,
                 max_spawn_dist=0):
        self.wall = wall          # (L,H,W) bool
        self.brick = brick
        self.bush = bush          # (L,H,W) bool 灌木（野外关；与 brick/wall 零重叠）
        self.pushable = pushable  # (L,H,W) bool 可推墙（⊆ brick，推箱子关）
        self.crate = crate
        self.rec = rec
        self.lo = lo              # (L,3) float32
        self.caps = caps          # (L,3) float32 成长上限 [bombs_max, blast_max, speed_max]
        self.rate = rate          # (L,) float32
        self.super_f = super_f    # (L,) float32 超级道具占比
        self.spawns = spawns      # (L,S_MAX,2) int32
        self.cnt = cnt            # (L,) int32 每关有效出生点数
        self.pairs = pairs        # (L,P_MAX,2) int32 按曼哈顿距离升序排序的出生点对
        self.pair_dists = pair_dists  # (L,P_MAX) int32 每对曼哈顿距离
        self.n_pairs = n_pairs    # (L,) int32 每关有效点对数
        self.logw = logw          # (L,) float32 log 权重
        self.is_open = is_open    # (L,) bool
        self.max_spawn_dist = jnp.asarray(max_spawn_dist, jnp.int32)
        self.L = wall.shape[0]

    def sample(self, key) -> LevelSample:
        """按权重抽一关 + 满足距离课程的出生点对（空间 50/50 随机翻转保证绝对对称）。"""
        k1, k2, k3 = jrandom.split(key, 3)
        i = jrandom.categorical(k1, self.logw)          # () int32 关卡

        # 出生点距离课程：按曼哈顿距离限制采样候选对
        cnt_p = self.n_pairs[i]
        dists_i = self.pair_dists[i]
        valid_pairs = (dists_i <= self.max_spawn_dist) & (jnp.arange(P_MAX) < cnt_p)
        n_eligible = jnp.where(self.max_spawn_dist > 0,
                               jnp.maximum(jnp.sum(valid_pairs), 1),
                               cnt_p)
        pidx = jrandom.randint(k2, (), 0, n_eligible)
        pair = self.pairs[i, pidx]
        s0, s1 = self.spawns[i, pair[0]], self.spawns[i, pair[1]]

        # 空间对称翻转：50% 概率 P0/P1 出生点互换，彻底消除空间几何先手优势
        flip = jrandom.bernoulli(k3)
        pos = jnp.where(flip, jnp.stack([s1, s0]), jnp.stack([s0, s1])).astype(jnp.float32) + 0.5
        return LevelSample(
            i, pos, self.wall[i], self.brick[i], self.bush[i], self.pushable[i],
            self.crate[i], self.rec[i], self.lo[i], self.caps[i], self.rate[i],
            self.super_f[i], self.is_open[i])

    # ---- 统计辅助（训练日志用）----
    def summary(self) -> str:
        n_empty = int(jnp.sum(self.is_open))
        lo = self.lo
        d_str = f"max_dist={int(self.max_spawn_dist)}" if int(self.max_spawn_dist) > 0 else "dist=unlimited"
        return (f"L={self.L} {d_str} 空场景/无砖关={n_empty} "
                f"lo bombs {float(lo[:, 0].min()):.0f}-{float(lo[:, 0].max()):.0f} "
                f"blast {float(lo[:, 1].min()):.0f}-{float(lo[:, 1].max()):.0f} "
                f"speed {float(lo[:, 2].min()):.2f}-{float(lo[:, 2].max()):.2f} "
                f"caps bombs {float(self.caps[:, 0].min()):.0f}-{float(self.caps[:, 0].max()):.0f}")


_ACTIVE: Optional[LevelSet] = None


def active() -> Optional[LevelSet]:
    """当前激活的关卡集；None = 过程式生成。"""
    return _ACTIVE


def clear() -> None:
    global _ACTIVE
    _ACTIVE = None


def _parse_weights(weights: str | dict, names: list[str], themes: list[str]) -> list[float]:
    """解析权重配置。
    weights 支持三种输入形式：
    1. .toml 文件路径：如 'configs/map_pool.toml'（含 [weights] 或直接顶层 key=val）
    2. 字典：如 {'empty': 0.05, '比武': 0.15, ...}
    3. 逗号分隔字符串：如 'empty=0.05,功夫=0.1,比武=0.15'

    token 匹配规则：
      - "240=0.2": 关卡 id 精确匹配
      - "empty=0.2": 名字含"空场景"的关
      - "比武=0.15": 主题精确匹配（若匹配主题，该主题下所有关卡均分该权重）
      - "功夫01=0.02": 主题不匹配时，回退按名字包含匹配（所有匹配关均分该权重）
    未指定的关均分剩余权重（若指定权重总和已达 1.0 则其余为 0）。"""
    if isinstance(weights, str) and weights.strip().endswith(".toml"):
        import tomllib
        with open(weights.strip(), "rb") as f:
            tdata = tomllib.load(f)
        weights = tdata.get("weights", tdata.get("map_pool", tdata.get("level_weights", tdata)))

    n = len(names)
    specs: dict[int, float] = {}

    if isinstance(weights, dict):
        items = weights.items()
    else:
        items = []
        for tok in str(weights).split(","):
            tok = tok.strip()
            if not tok:
                continue
            id_s, _, w_s = tok.partition("=")
            items.append((id_s.strip(), float(w_s)))

    for raw_k, raw_w in items:
        w = float(raw_w)
        id_s = str(raw_k).strip().lower()
        if id_s == "empty":
            hit = [k for k, nm in enumerate(names) if "空" in nm]
            if not hit:
                raise ValueError("权重 'empty' 找不到空场景关")
            specs[hit[0]] = w
        elif id_s.isdigit():
            idx = int(id_s)
            if not 0 <= idx < n:
                raise ValueError(f"关卡 id {idx} 越界 [0,{n})")
            specs[idx] = w
        else:
            hit = [k for k, th in enumerate(themes) if th.lower() == id_s]
            if not hit:
                hit = [k for k, nm in enumerate(names) if id_s in nm.lower()]
            if not hit:
                raise ValueError(f"权重主题/关卡 '{id_s}' 不存在（可用主题: {sorted(set(themes))}）")
            per_level = w / len(hit)
            for k in hit:
                specs[k] = per_level
    total = sum(specs.values())
    if total > 1.0 + 1e-6:
        raise ValueError(f"权重总和 {total} > 1.0")
    if len(specs) == n:                    # 全部图都指定（S4 全图课程）→ 无剩余
        w = [0.0] * n
        for idx, v in specs.items():
            w[idx] = v
        return w
    # total ≥ 1（课程 stage 全图权重分配浮点累加略超 1）时未指定图 = 0 权重；
    # 负权重 → np.log → nan 毒化 categorical，必须 clamp
    w_rest = max(0.0, (1.0 - total) / (n - len(specs)))
    w = [w_rest] * n
    for idx, v in specs.items():
        w[idx] = v
    return w


def set_active(path: str, weights: str | dict = "empty=0.05,功夫=0.1,比武=0.15",
               max_spawn_dist: int = 0) -> LevelSet:
    """从 levels.json 加载并激活（进程内一次；jit/vmap 前调用）。

    默认权重：空场景 5% + 功夫 10% + 比武 15%，其余 70% 均分随机。
    空 weights="" 则全部关卡均分。
    max_spawn_dist > 0 时激活出生点距离限制（例如 4 贴脸、6 近距、10 中距）。"""
    global _ACTIVE
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < 1:
        raise ValueError(f"{path}: 空关卡列表")
    h, w = data[0]["h"], data[0]["w"]
    for lvl in data:
        assert lvl["h"] == h and lvl["w"] == w, \
            f"关卡尺寸不一致: {lvl['id']} ({lvl['w']}x{lvl['h']}) vs {w}x{h}"
    import numpy as np
    L = len(data)
    names = [str(lvl["name"]) for lvl in data]
    themes = [str(lvl.get("theme", "")) for lvl in data]
    w_arr = _parse_weights(weights, names, themes)
    wall = np.zeros((L, h, w), np.bool_)
    brick = np.zeros((L, h, w), np.bool_)
    bush = np.zeros((L, h, w), np.bool_)
    pushable = np.zeros((L, h, w), np.bool_)
    crate = np.zeros((L, h, w), np.bool_)
    lo = np.zeros((L, 3), np.float32)
    caps = np.zeros((L, 3), np.float32)
    rate = np.zeros((L,), np.float32)
    super_f = np.zeros((L,), np.float32)
    spawns = np.full((L, S_MAX, 2), -1, np.int32)
    cnt = np.zeros((L,), np.int32)
    pairs = np.full((L, P_MAX, 2), 0, np.int32)
    pair_dists = np.full((L, P_MAX), 999, np.int32)
    n_pairs = np.zeros((L,), np.int32)

    for lvl in data:
        i = int(lvl["id"])
        wall[i] = np.asarray(lvl["wall"], np.bool_).reshape(h, w)
        brick[i] = np.asarray(lvl["brick"], np.bool_).reshape(h, w)
        bush[i] = np.asarray(lvl.get("bush", []), np.bool_).reshape(h, w)
        assert not (bush[i] & (wall[i] | brick[i])).any(), \
            f"关卡 {i} 灌木与墙/砖重叠"
        pushable[i] = np.asarray(lvl.get("pushable", []), np.bool_).reshape(h, w)
        assert not (pushable[i] & ~brick[i]).any(), \
            f"关卡 {i} 可推墙不在砖上（可推墙必须是障碍物）"
        st = lvl["initial_stats"]
        lo[i] = (st["bombs"], st["blast"], st["speed"])
        caps[i] = (lvl["bombs_max"], lvl["blast_max"], lvl["speed_max"])
        rate[i] = float(lvl["crate_rate"])
        super_f[i] = float(lvl.get("crate_super_fraction", 0) or 0)
        if not (rate[i] > 0):        # JS 钳制：crate_rate <= 0 / 缺失 → 1.0（炸砖必成箱）
            rate[i] = 1.0
        sp = lvl["spawns"]
        c_i = min(len(sp), S_MAX)
        cnt[i] = c_i
        for j, (r, c) in enumerate(sp[:S_MAX]):
            spawns[i, j] = (r, c)
        for (r, c) in lvl.get("initial_crates", []):
            if 0 <= r < h and 0 <= c < w:
                crate[i, r, c] = True

        # 计算所有有效出生点对并按曼哈顿距离升序排序（过滤完全重叠点）
        p_list = []
        for a in range(c_i):
            for b in range(a + 1, c_i):
                if sp[a][0] == sp[b][0] and sp[a][1] == sp[b][1]:
                    continue
                d = abs(sp[a][0] - sp[b][0]) + abs(sp[a][1] - sp[b][1])
                p_list.append((d, a, b))
        if not p_list:
            p_list = [(1, 0, min(1, c_i - 1))]
        p_list.sort()
        n_pairs[i] = min(len(p_list), P_MAX)
        for p_idx, (d, a, b) in enumerate(p_list[:P_MAX]):
            pairs[i, p_idx] = [a, b]
            pair_dists[i, p_idx] = d

    logw = np.log(np.asarray(w_arr, np.float32))
    is_open = brick.sum(axis=(1, 2)) == 0
    ls = LevelSet(
        jnp.asarray(wall), jnp.asarray(brick), jnp.asarray(bush),
        jnp.asarray(pushable), jnp.asarray(crate), jnp.asarray(crate),
        # 预置宝箱全部必升（rec == crate）
        jnp.asarray(lo), jnp.asarray(caps), jnp.asarray(rate),
        jnp.asarray(super_f), jnp.asarray(spawns),
        jnp.asarray(cnt, jnp.int32),
        jnp.asarray(pairs, jnp.int32),
        jnp.asarray(pair_dists, jnp.int32),
        jnp.asarray(n_pairs, jnp.int32),
        jnp.asarray(logw),
        jnp.asarray(is_open),
        max_spawn_dist=max_spawn_dist)
    _ACTIVE = ls
    return ls
