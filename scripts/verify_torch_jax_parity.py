"""torch（collect 侧）↔ jax（训练侧）逐 tick 对拍。

验证蒸馏数据质量的前提：collect_distill.py 的 torch 环境与 jax_bomb 环境
在**同一局面、同一动作序列**下，逐 tick 的状态推进和 obs7 输出是否一致。

方法（同 deploy/parity_ref.py 的固定状态注入法，排除初始差异）：
  1. 手动构造相同初始状态（pos/fuse/owner/bomb_blast/alive/hp/invuln/t）
    灌进 torch BatchedSim 和 jax BombState；
  2. 固定伪随机动作序列（含放泡/移动/引信递减/爆炸/连锁/伤害/死亡），
     两边每 tick 用相同动作 step；
  3. 逐 tick 比较：pos/fuse/owner/bomb_blast/alive/hp/invuln/t + done +
     obs7（collect 侧 obs7_from_sim 的等价 torch 实现 vs jax make_obs）。

用法：
  .venv/bin/python verify_torch_jax_parity.py [--ticks 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

# torch 只在 --side torch/both 时才 import（远端 miniconda 无 torch）
_side = None
if "--side" in sys.argv:
    _side = sys.argv[sys.argv.index("--side") + 1]
if _side in (None, "both", "torch") and "--compare" not in sys.argv:
    import torch
    from sim.config import SimConfig
    from sim.torch_sim import BatchedSim
    from sim.blast import danger_map

# jax 只在 --side jax/both 时才 import（本地 .venv 无 jax；无参数默认 torch 侧）。
# **不要 sys.path.insert(jax_bomb)**：jax_bomb/platform.py 会劫持标准库
# platform（jax 内部 numpy 会 import platform）→ ImportError 循环。用
# namespace package `from jax_bomb.jax_env import ...`（PROJ 已在 sys.path）。
if _side in ("jax", "both") and "--compare" not in sys.argv:
    import jax
    import jax.numpy as jnp
    from jax_bomb import jax_env as _jax_env
    from jax_bomb.jax_env import (BLAST, FUSE, H, W, MAX_STEPS, MAX_BOMBS,
                                  MAX_HP, INVULN, MAX_CHAIN, BombState,
                                  legal_mask as jax_legal, make_obs,
                                  step as jax_step)
    # 与对拍 CFG（growth_crate_prob=0）对齐：拾取概率归零 → 对拍全程无
    # 成长 RNG（corridor 箱即使被踩也不命中，两侧一致）。step 是 eager
    # 调用（非 jit），每次读模块全局 → 覆盖生效。
    _jax_env.CRATE_PROB = 0.0
else:
    BLAST = 7
    FUSE = 30
    H = W = 13
    MAX_STEPS = 1800
    MAX_BOMBS = 10
    MAX_HP = 5
    INVULN = 30
    MAX_CHAIN = 16

# 与 jax 常量一致的对拍配置（growth/open_* 起点 = jax 模块常量；crate 交互
# 关掉：growth_crate_prob=0 + open_crate_cross=False → 拾取 prob=0 恒不命中、
# 掉血 lost_all=0 无回收 —— 对拍全程确定性，覆盖 brick 爆炸/挡火/变箱共享逻辑）
CFG = SimConfig(height=H, width=W, n_players=2, map_mode="corridor",
                pure_open_fraction=0.5, open_fraction=0.25, ring_fraction=0.0,
                open_obstacle_max=5, random_wall_rows=True, wall_density=0.45,
                open_crate_cross=False, growth_crate_prob=0.0,
                growth_bombs_start=2, growth_blast_start=2,
                growth_speed_start=1.0,
                open_growth_bombs=3, open_growth_blast=3,
                open_growth_speed=0.84,
                speed=3.0, blast=BLAST, max_steps=MAX_STEPS,
                invuln_ticks=INVULN, max_hp=MAX_HP, max_bombs=MAX_BOMBS) \
    if _side in (None, "both", "torch") and "--compare" not in sys.argv \
    else None


def build_torch_state(sim: BatchedSim, pos, fuse, owner, bomb_blast, wall,
                      brick, alive, hp, invuln, t, is_open):
    """把固定状态灌进 torch sim（参考 parity_ref.py 的做法）。"""
    n = 1
    sim.wall[:] = torch.tensor(wall, dtype=torch.bool).unsqueeze(0)
    sim.brick[:] = torch.tensor(brick, dtype=torch.bool).unsqueeze(0)
    sim.crate[:] = False
    sim._recycle_crate[:] = False
    sim._is_open[:] = bool(is_open)
    sim.pos[:] = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    sim.fuse[:] = torch.tensor(fuse, dtype=torch.int16).unsqueeze(0)
    sim.owner[:] = torch.tensor(owner, dtype=torch.int8).unsqueeze(0)
    sim.bomb_blast[:] = torch.tensor(bomb_blast, dtype=torch.int16).unsqueeze(0)
    sim.alive[:] = torch.tensor(alive, dtype=torch.bool).unsqueeze(0)
    sim.hp[:] = torch.tensor(hp, dtype=torch.uint8).unsqueeze(0)
    sim.invuln[:] = torch.tensor(invuln, dtype=torch.long).unsqueeze(0)
    sim.t[:] = torch.tensor([t], dtype=torch.long)
    # 成长属性 = 起点（corridor：2/2/1.0；对拍场景无 open 关）→ 掉血 lost=0
    sim.bombs_cap[:] = 2.0
    sim.blast_cap[:] = 2.0
    sim.spd_g[:] = 1.0
    # 掉血惩罚 clamp 用 _lo_*（reset 时按随机地图类型预计算 —— 构造时可能
    # 残留 open 的 0.84/3，不覆盖则掉血时 spd_g 被扣到 0.84 分叉）。
    # 对拍场景 corridor 起点 = 2/2/1.0 → lost_all=0，无回收 RNG。
    sim._lo_bombs[:] = 2.0
    sim._lo_blast[:] = 2.0
    sim._lo_spd[:] = 1.0
    # crate_prob 显式归零：BatchedSim 构造时 reset 会按随机地图类型把部分
    # env 的 crate_prob 设成 1.0（open 关）—— 不覆盖则对拍随机分叉
    # （踩箱成长概率 torch=1.0/0.0 不定，jax=CRATE_PROB=0.5）。
    # 对拍场景无 crate 交互（crate 全 0），概率归零保证两侧都不成长。
    sim.crate_prob[:] = 0.0


def build_jax_state(pos, fuse, owner, bomb_blast, wall, brick, alive, hp,
                    invuln, t, is_open):
    return BombState(
        pos=jnp.array(pos, jnp.float32),
        fuse=jnp.array(fuse, jnp.int32),
        owner=jnp.array(owner, jnp.int32),
        bomb_blast=jnp.array(bomb_blast, jnp.int32),
        wall=jnp.array(wall, jnp.bool_),
        brick=jnp.array(brick, jnp.bool_),
        crate=jnp.zeros((H, W), jnp.bool_),
        rec_crate=jnp.zeros((H, W), jnp.bool_),
        alive=jnp.array(alive, jnp.bool_),
        hp=jnp.array(hp, jnp.int32),
        invuln=jnp.array(invuln, jnp.int32),
        bombs_cap=jnp.full((2,), 2.0, jnp.float32),
        blast_cap=jnp.full((2,), 2.0, jnp.float32),
        spd_g=jnp.full((2,), 1.0, jnp.float32),
        is_open=jnp.array(bool(is_open)),
        t=jnp.array(t, jnp.int32),
    )


def torch_obs7(sim: BatchedSim, pid: int) -> np.ndarray:
    """collect_distill.py obs7_batch 的单 env 版（jax make_obs 的等价）。"""
    fuse = sim.fuse[0].float().cpu().numpy()
    owner = sim.owner[0].cpu().numpy()
    bomb_blast = sim.bomb_blast[0].float().cpu().numpy()
    pos = sim.pos[0].float().cpu().numpy()
    alive = sim.alive[0].cpu().numpy()
    t = float(sim.t[0].item())
    me, opp = pid, 1 - pid
    obs = np.zeros((8, H, W), np.float32)

    def splat(xy, gate):
        # 与 jax _splat / torch obs._splat 一致：格中心 i 对应 fy=i → 先减半格
        fy = min(max(xy[0] - 0.5, 0.0), float(H - 1))
        fx = min(max(xy[1] - 0.5, 0.0), float(W - 1))
        y0 = min(max(int(fy), 0), H - 1)
        x0 = min(max(int(fx), 0), W - 1)
        y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
        wy, wx = min(max(fy - y0, 0.0), 1.0), min(max(fx - x0, 0.0), 1.0)
        g = 1.0 if gate else 0.0
        out = np.zeros((H, W), np.float32)
        out[y0, x0] += (1 - wy) * (1 - wx) * g
        out[y0, x1] += (1 - wy) * wx * g
        out[y1, x0] += wy * (1 - wx) * g
        out[y1, x1] += wy * wx * g
        return out

    obs[0] = splat(pos[me], alive[me])
    obs[2] = splat(pos[opp], alive[opp])
    fuse_norm = fuse / float(FUSE)
    obs[1] = np.where(owner == me, fuse_norm, 0.0).astype(np.float32)
    obs[3] = np.where(owner == opp, fuse_norm, 0.0).astype(np.float32)
    obs[4] = (sim.wall[0] | sim.brick[0]).float().cpu().numpy()
    # ch5 = 危险图（torch danger_map 同款：wall=永久墙挡火、brick 单独传，
    # 与 obs.py / jax make_obs 同语义 —— 之前把 wall|brick 当 wall 传会让
    # brick 双重挡火、危险图偏小）
    fuse_t = torch.tensor(fuse, dtype=torch.int32)
    wall_t = sim.wall[0]
    b_t = torch.tensor(bomb_blast, dtype=torch.int32)
    blast_map = torch.where(b_t > 0, b_t, torch.tensor(BLAST))
    brick_t = sim.brick[0]
    danger = danger_map(fuse_t, wall_t, blast_map, FUSE, brick_t, MAX_CHAIN)
    obs[5] = danger.float().cpu().numpy()
    obs[6] = np.full((H, W), t / float(MAX_STEPS), np.float32)
    obs[7] = sim.crate[0].float().cpu().numpy()
    return obs


def make_initial_state():
    """构造一个内容丰富（含泡/引信/不同威力/受伤/无敌/墙/砖）的初始状态。"""
    fuse = np.zeros((H, W), np.int32)
    owner = np.full((H, W), -1, np.int32)
    bomb_blast = np.zeros((H, W), np.int32)
    # 三颗泡：不同 owner/引信剩余/威力（覆盖连锁与非连锁）
    for (r, c, f, o, b) in ((5, 5, 15, 0, 3), (5, 7, 5, 1, 7), (8, 9, 20, 0, 2)):
        fuse[r, c] = f
        owner[r, c] = o
        bomb_blast[r, c] = b
    pos = np.array([[4.5, 6.5], [8.5, 6.5]], np.float32)
    alive = np.array([True, True])
    hp = np.array([5, 3], np.int32)          # 玩家1 已受伤
    invuln = np.array([0, 10], np.int32)     # 玩家1 无敌期
    # 一堵墙（(4,8) 在玩家0 右侧两格）：验证墙 blocked + 爆炸挡火 + mask
    wall = np.zeros((H, W), np.bool_)
    wall[4, 8] = True
    wall[3, 4] = True
    wall[6, 6] = True
    # 可炸墙（brick）：在泡的爆炸路径上 —— 验证 brick 挡火但被覆盖、炸掉后
    # 变宝箱（crate |= brick & covered）。玩家移动避开不踩箱（crate_prob=0
    # 下即使踩到也不命中，清箱两侧一致）。
    brick = np.zeros((H, W), np.bool_)
    brick[5, 6] = True       # 泡(5,5) 右侧一格，blast=3 覆盖
    brick[5, 8] = True       # 泡(5,7) 右侧一格
    brick[8, 8] = True
    brick[9, 9] = True
    return fuse, owner, bomb_blast, pos, wall, brick, alive, hp, invuln


def compare_traces(a_path: str, b_path: str) -> None:
    """逐 tick 比较 --side torch 与 --side jax 导出的两条 JSONL 轨迹。

    两侧是独立采样序列（相同 seed 下动作序列相同，但终局 tick 可能因数值
    差异错开），因此按**公共前缀**对齐比较；长度不同只提示，不算失败。
    """
    import json as _json
    la = [_json.loads(l) for l in open(a_path)]
    lb = [_json.loads(l) for l in open(b_path)]
    n = min(len(la), len(lb))
    fails = 0
    for tick in range(n):
        ra, rb = la[tick], lb[tick]
        for k in ("pos", "fuse", "owner", "bomb_blast", "brick", "crate",
                  "alive", "hp", "invuln", "t"):
            va, vb = np.array(ra[k]), np.array(rb[k])
            if not (np.array_equal(va, vb) if k != "pos"
                    else np.allclose(va, vb, atol=1e-5)):
                fails += 1
                print(f"[FAIL] tick={tick} {k}:\n  torch={va}\n  jax  ={vb}")
        if ra["done"] != rb["done"]:
            fails += 1
            print(f"[FAIL] tick={tick} done: torch={ra['done']} jax={rb['done']}")
        for pid in (0, 1):
            oa = np.array(ra[f"obs7_{pid}"])
            ob = np.array(rb[f"obs7_{pid}"])
            if not np.allclose(oa, ob, atol=1e-4):
                fails += 1
                md = np.abs(oa - ob).max()
                idx = np.unravel_index(np.abs(oa - ob).argmax(), oa.shape)
                print(f"[FAIL] tick={tick} pid={pid} obs7 maxdiff={md:.6f} "
                      f"@ch{idx[0]}({idx[1]},{idx[2]})")
    if fails == 0:
        print(f"轨迹一致 ✔（公共前缀 {n} tick；len torch={len(la)} jax={len(lb)}）")
    else:
        print(f"{fails} 处不一致 ✘（公共前缀 {n} tick；len torch={len(la)} "
              f"jax={len(lb)}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", choices=["torch", "jax", "both"], default="both",
                    help="torch/jax 单独跑并导出轨迹；both = 本机同时跑（需两环境）")
    ap.add_argument("--out", default="/tmp/parity_trace.jsonl")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="比较两条已导出轨迹（torch 侧 vs jax 侧）")
    args = ap.parse_args()

    if args.compare:
        return compare_traces(*args.compare)
    if args.side == "both":
        return run_both(args)

    fuse, owner, bomb_blast, pos, wall, brick, alive, hp, invuln = \
        make_initial_state()
    rng = np.random.default_rng(args.seed)
    acts = [rng.integers([0, 0], [5, 2], size=(2, 2)) for _ in range(args.ticks)]

    if args.side == "torch":
        sim = BatchedSim(CFG, 1, device="cpu", seed=args.seed)
        build_torch_state(sim, pos, fuse, owner, bomb_blast, wall, brick,
                          alive, hp, invuln, 0, False)
        rows = []
        for tick in range(args.ticks):
            a_t = torch.tensor(acts[tick], dtype=torch.long).unsqueeze(0)
            rew, done_any, _ = sim.step(a_t, auto_reset=False)
            rows.append({
                "pos": sim.pos[0].tolist(), "fuse": sim.fuse[0].tolist(),
                "owner": sim.owner[0].tolist(),
                "bomb_blast": sim.bomb_blast[0].tolist(),
                "brick": sim.brick[0].tolist(),
                "crate": sim.crate[0].tolist(),
                "alive": sim.alive[0].tolist(), "hp": sim.hp[0].tolist(),
                "invuln": sim.invuln[0].tolist(), "t": int(sim.t[0]),
                "done": bool(done_any),
                "obs7_0": torch_obs7(sim, 0).round(6).tolist(),
                "obs7_1": torch_obs7(sim, 1).round(6).tolist(),
            })
            if done_any:
                break
    else:  # jax
        js = build_jax_state(pos, fuse, owner, bomb_blast, wall, brick,
                             alive, hp, invuln, 0, False)
        rows = []
        for tick in range(args.ticks):
            # auto_reset=False + crate_prob=0/lost=0：key 只驱动零命中 RNG
            js, done_j = jax_step(js, jnp.array(acts[tick], jnp.int32),
                                  jax.random.PRNGKey(0), False)
            rows.append({
                "pos": np.asarray(jax.device_get(js.pos)).tolist(),
                "fuse": np.asarray(jax.device_get(js.fuse)).tolist(),
                "owner": np.asarray(jax.device_get(js.owner)).tolist(),
                "bomb_blast": np.asarray(jax.device_get(js.bomb_blast)).tolist(),
                "brick": np.asarray(jax.device_get(js.brick)).tolist(),
                "crate": np.asarray(jax.device_get(js.crate)).tolist(),
                "alive": np.asarray(jax.device_get(js.alive)).tolist(),
                "hp": np.asarray(jax.device_get(js.hp)).tolist(),
                "invuln": np.asarray(jax.device_get(js.invuln)).tolist(),
                "t": int(np.asarray(jax.device_get(js.t))),
                "done": bool(done_j),
                "obs7_0": np.asarray(jax.device_get(make_obs(js, 0))).round(6).tolist(),
                "obs7_1": np.asarray(jax.device_get(make_obs(js, 1))).round(6).tolist(),
            })
            if done_j:
                break
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{args.side}] {len(rows)} tick -> {args.out}", flush=True)


def run_both(args):
    """本机同时跑两侧（需 torch+jax 同环境），逐 tick 比较。"""
    sim = BatchedSim(CFG, 1, device="cpu", seed=args.seed)
    fuse, owner, bomb_blast, pos, wall, brick, alive, hp, invuln = \
        make_initial_state()
    build_torch_state(sim, pos, fuse, owner, bomb_blast, wall, brick,
                      alive, hp, invuln, 0, False)
    js = build_jax_state(pos, fuse, owner, bomb_blast, wall, brick,
                         alive, hp, invuln, 0, False)
    rng = np.random.default_rng(args.seed)
    fails = 0
    for tick in range(args.ticks):
        a = rng.integers([0, 0], [5, 2], size=(2, 2))
        a_t = torch.tensor(a, dtype=torch.long).unsqueeze(0)
        rew, done_any, _ = sim.step(a_t, auto_reset=False)
        done_t = bool(done_any)
        js, done_j = jax_step(js, jnp.array(a, jnp.int32),
                              jax.random.PRNGKey(tick), False)
        checks = [
            ("pos", js.pos, sim.pos[0], 1e-5),
            ("fuse", js.fuse, sim.fuse[0], 0),
            ("owner", js.owner, sim.owner[0], 0),
            ("bomb_blast", js.bomb_blast, sim.bomb_blast[0], 0),
            ("wall", js.wall, sim.wall[0], 0),
            ("brick", js.brick, sim.brick[0], 0),
            ("crate", js.crate, sim.crate[0], 0),
            ("alive", js.alive, sim.alive[0], 0),
            ("hp", js.hp, sim.hp[0], 0),
            ("invuln", js.invuln, sim.invuln[0], 0),
            ("t", js.t, sim.t[0], 0),
        ]
        for name, jv, tv, atol in checks:
            jn = np.asarray(jax.device_get(jv))
            tn = tv.float().cpu().numpy() if tv.dtype == torch.float32 \
                else tv.cpu().numpy()
            ok = np.allclose(jn, tn, atol=atol) if atol else np.array_equal(jn, tn)
            if not ok:
                fails += 1
                print(f"[FAIL] tick={tick} 状态 {name}:\n  jax  ={jn}\n  torch={tn}")
        # legal_mask 对拍：move (P,5) / bomb (P,2)，方向语义与 IDLE/bomb=0 恒合法
        mm_j, bm_j = jax_legal(js)
        mm_t, bm_t = sim.legal_mask()
        for name, jv, tv in (("move_mask", mm_j, mm_t[0]),
                             ("bomb_mask", bm_j, bm_t[0])):
            jn = np.asarray(jax.device_get(jv))
            tn = tv.cpu().numpy()
            if not np.array_equal(jn, tn):
                fails += 1
                diff = np.argwhere(jn != tn)
                print(f"[FAIL] tick={tick} {name} 差异 {len(diff)} 处，"
                      f"首个 {diff[0].tolist()}：jax={jn[diff[0][0]]} "
                      f"torch={tn[diff[0][0]]}")
        if done_t != bool(done_j):
            fails += 1
            print(f"[FAIL] tick={tick} done: torch={done_t} jax={done_j}")
        for pid in (0, 1):
            o_t = torch_obs7(sim, pid)
            o_j = np.asarray(jax.device_get(make_obs(js, pid)))
            if not np.allclose(o_t, o_j, atol=1e-4):
                fails += 1
                md = np.abs(o_t - o_j).max()
                idx = np.unravel_index(np.abs(o_t - o_j).argmax(), o_t.shape)
                print(f"[FAIL] tick={tick} pid={pid} obs7 maxdiff={md:.6f} "
                      f"@ch{idx[0]}({idx[1]},{idx[2]})\n"
                      f"  torch={o_t[idx]:.4f} jax={o_j[idx]:.4f}")
        if done_t:
            print(f"[info] tick={tick} 终局，停止", flush=True)
            break
    if fails == 0:
        print(f"全部一致 ✔ ({args.ticks} tick，含墙/砖/连锁/伤害/死亡/mask/obs7 双视角)")
    else:
        print(f"{fails} 处不一致 ✘")


if __name__ == "__main__":
    main()
