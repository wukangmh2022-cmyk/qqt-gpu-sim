#!/usr/bin/env python3
"""JS(web/sim.js TransformerModel) ↔ JAX(jax_net.transformer_forward) 前向对拍。

背景：JAX transformer（DCU 训练，bf16）导出到 web/models/*.json 后在浏览器
用 sim.js 的 TransformerModel 前向。本脚本在**真实关卡图**（web/assets/maps/
levels.json，非已废弃的 open/corridor 模式）上做三方一致性检查：

  1. encodeObsJAX(pid)   vs jax_env.make_obs      —— 观测编码逐位一致
  2. encodeStateJAX(pid) vs jax_env.global_vec    —— state token 输入一致
  3. JS forward          vs jax_net 的 fp32 手动前向 —— 输出 ~1e-6

依赖：本机有 node（加载 web/sim.js）、JAX 环境、已导出的模型 JSON
（deploy/export_jax_ckpt.py --verify 通过后产物）。用法：
    PYTHONPATH=. .venv/bin/python scripts/quick_check_js_jax_transformer.py
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jax_bomb.jax_env import make_obs, global_vec, BombState
from deploy.export_jax_ckpt import _forward_fp32

ROOT = Path(__file__).resolve().parent.parent
SIM_JS = ROOT / "web" / "sim.js"
LEVELS = ROOT / "web" / "assets" / "maps" / "levels.json"
import sys as _sys
_DEFAULT = _sys.argv[1] if len(_sys.argv) > 1 else "params_it00000102"
CKPT = ROOT / "ckpt" / f"{_DEFAULT}.pkl"
MODEL = ROOT / "web" / "models" / f"{_DEFAULT}.json"
H, W = 13, 15

JS_DRIVER = r"""
const fs = require('fs');
const Q = require(%(sim_js)r);
const doc = JSON.parse(fs.readFileSync(%(model)r));
const model = new Q.TransformerModel(doc);
const levels = JSON.parse(fs.readFileSync(%(levels)r));
const level = levels[%(lvl_idx)d];
const rng = Q.mulberry32(42);
const sim = new Q.Sim(1);
sim.reset(level);
// 固定脚本走 30 tick（放泡+移动），制造有泡阵/危险图的非平凡状态
const acts = [];
for (let i = 0; i < 30; i++) {
  const mv = [0, 2, 1, 3][i %% 4];
  acts.push([[mv, i %% 5 === 0 ? 1 : 0], [2, i %% 7 === 0 ? 1 : 0]]);
}
for (const a of acts) sim.step(a);
const out = {
  t: sim.t, alive: sim.alive, hp: sim.hp,
  bombsCap: sim.bombsCap, blastCap: sim.blastCap, spdG: sim.spdG,
  pos: Array.from(sim.pos), wall: Array.from(sim.wall),
  brick: Array.from(sim.brick), bush: Array.from(sim.bush),
  crate: Array.from(sim.crate), crateType: Array.from(sim.crateType),
  superCrate: Array.from(sim.superCrate), fuse: Array.from(sim.fuse),
  owner: Array.from(sim.owner), bombBlast: Array.from(sim.bombBlast),
  obs: [], gvec: [], fwd: [], pushable: Array.from(sim.pushable),
};
for (const pid of [0, 1]) {
  out.obs.push(Array.from(sim.encodeObsJAX(pid)));       // 默认 14 通道（含 ch13 可推箱）
  out.gvec.push(Array.from(sim.encodeStateJAX(pid)));
  const f = model.forward(sim.encodeObsJAX(pid, model.obsShape[0]),
                          sim.encodeStateJAX(pid));      // 按模型通道数（旧 ckpt=13）
  out.fwd.push({ move: Array.from(f.move), bomb: Array.from(f.bomb), value: f.value });
}
process.stdout.write(JSON.stringify(out));
"""


def run_js(lvl_idx: int) -> dict:
    code = JS_DRIVER % {"sim_js": str(SIM_JS), "model": str(MODEL),
                        "levels": str(LEVELS), "lvl_idx": lvl_idx}
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True,
                       check=True)
    return json.loads(r.stdout)


def to_bombstate(d: dict) -> BombState:
    st = d
    g2 = lambda name, dt: jnp.array(np.array(st[name], dt).reshape(H, W))
    return BombState(
        pos=jnp.array(st["pos"], jnp.float32).reshape(2, 2),
        fuse=g2("fuse", np.int32), owner=g2("owner", np.int32),
        bomb_blast=g2("bombBlast", np.int32),
        wall=g2("wall", np.int32) > 0, brick=g2("brick", np.int32) > 0,
        pushable=g2("pushable", np.int32) > 0,
        push_t=jnp.zeros((H, W), jnp.float32),
        bush=g2("bush", np.int32) > 0,
        crate=g2("crate", np.int32), rec_crate=jnp.zeros((H, W), jnp.bool_),
        alive=jnp.array([bool(x) for x in st["alive"]], jnp.bool_),
        hp=jnp.array(st["hp"], jnp.int32),
        invuln=jnp.zeros((2,), jnp.int32),
        bombs_cap=jnp.array(st["bombsCap"], jnp.float32),
        blast_cap=jnp.array(st["blastCap"], jnp.float32),
        spd_g=jnp.array(st["spdG"], jnp.float32),
        buffs=jnp.zeros((2,), jnp.int8), debuffs=jnp.zeros((2,), jnp.int8),
        items=jnp.zeros((2, 4), jnp.int8), gametype=jnp.zeros((), jnp.int8),
        is_open=jnp.array(False, jnp.bool_),
        t=jnp.array(st["t"], jnp.int32), level_id=jnp.array(0, jnp.int32),
    )


def main() -> int:
    ck = pickle.load(open(CKPT, "rb"))
    params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), ck)
    ok = True
    for lvl_idx in (0, 1, 7):                     # 抽样 3 张（1=比赛02，含可推箱）
        d = run_js(lvl_idx)
        state = to_bombstate(d)
        worst = 0.0
        for pid in (0, 1):
            o_j = np.asarray(make_obs(state, pid))            # (14,13,15)
            o_js = np.array(d["obs"][pid], np.float32).reshape(14, 13, 15)
            d_obs = float(np.abs(o_j - o_js).max())
            g_j = np.asarray(global_vec(state, pid))
            g_js = np.array(d["gvec"][pid])
            d_g = float(np.abs(g_j - g_js).max())
            # 旧 ckpt 是 13 通道：前向对拍用前 13 通道（ch13 可推箱不进旧模型）
            mv_j, bm_j, v_j = _forward_fp32(params_f32, o_js[:13][None],
                                            g_js[None])
            f = d["fwd"][pid]
            d_m = float(np.abs(np.asarray(mv_j)[0] - np.array(f["move"])).max())
            d_b = float(np.abs(np.asarray(bm_j)[0] - np.array(f["bomb"])).max())
            d_v = float(abs(np.asarray(v_j)[0] - f["value"]))
            worst = max(worst, d_obs, d_g, d_m, d_b, d_v)
            if max(d_obs, d_g, d_m, d_b, d_v) > 1e-4:
                print(f"  [FAIL] level#{lvl_idx} pid={pid}: obs={d_obs:.2e} "
                      f"gvec={d_g:.2e} move={d_m:.2e} bomb={d_b:.2e} "
                      f"value={d_v:.2e}")
                ok = False
        print(f"level#{lvl_idx}: worst={worst:.2e} "
              f"({'PASS' if worst <= 1e-4 else 'FAIL'})")
    print("结果:", "全部通过，transformer 前后端一致" if ok else "存在不一致！")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
