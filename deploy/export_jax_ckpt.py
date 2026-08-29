#!/usr/bin/env python3
"""JAX transformer ckpt → web/models/*.json 转换工具（训练链路第二环）。

与 deploy/export_ckpt.py（torch mlp/cnn）并列：把 `jax_bomb` 训练产出的
pickle ckpt（params 是 jax 嵌套 dict，权重为 (w, b) tuple，bf16）转成浏览器
可直接加载的 JSON：
    - 支持 `jax_bomb/jax_net.py::init_transformer` 的 ViT 结构（patch 切块 +
      state token + pre-norm MHA/FFN + patch-token 均值池化 + 三头）；
    - **不做视角置换**：JAX 训练用 `make_obs(state, pid)` 逐玩家视角（每个
      pid 一份 13 通道 obs），Web 端 TransformerModel 用同款 `encodeObsJAX(pid)`
      直接喂网络，不需要共享观测/折 perm；
    - 输出 web/models/<stem>.json（扁平 float32 数组）+ web/models/index.json。

生产路径只依赖 ckpt/ 目录 + jax_bomb.jax_net（读结构不读网络实现）。
`--verify` 自检：临时构造随机 obs + 用 jax_net.transformer_forward 前向，
与 JS 逻辑的 Python 复刻（transformer_forward_js）逐位对比。

用法：
    .venv/bin/python deploy/export_jax_ckpt.py            # 导出全部 transformer ckpt
    .venv/bin/python deploy/export_jax_ckpt.py --verify   # 导出 + 前向自检
    .venv/bin/python deploy/export_jax_ckpt.py ckpt/ckpt_00000100_r0.pkl   # 只导指定档
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import struct
import sys
from datetime import datetime, timezone

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
CKPT_DIR = os.path.join(PROJ, "ckpt")
OUT_DIR = os.path.join(PROJ, "web", "models")


def model_stem(path: str) -> str:
    """Return the stable browser asset key for a JAX checkpoint."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("ViTModel") and "_it" in stem:
        return stem.split("_it", 1)[0]
    return stem.replace("ckpt_", "jax_")


def pack_tensors(tensors: dict[str, np.ndarray]) -> tuple[str, dict]:
    """把 {name: float32 ndarray} 打包成 (base64 扁平流, {name:[off,cnt]})。"""
    buf = bytearray()
    index = {}
    for name, arr in tensors.items():
        a = np.asarray(arr, np.float32)
        index[name] = (len(buf) // 4, a.size)
        buf += a.tobytes()
    return base64.b64encode(bytes(buf)).decode(), index


def extract_transformer(params: dict) -> dict[str, np.ndarray]:
    """把 jax_net 的 transformer params（嵌套 dict，(w,b) tuple）拍平成
    {name: float32} 表。命名约定与 sim.js TransformerModel.T() 对齐。"""
    out = {}

    def flat(name, arr):
        out[name] = np.asarray(arr, np.float32)

    def lin(name, wb):
        flat(name + "_w", wb[0])
        flat(name + "_b", wb[1])

    lin("tok", params["tok"])
    flat("pos", params["pos"][0])                 # (1, T, E) → (T, E)
    # state 是 _linear_init 返回 (w, b) tuple —— 单独处理
    sw, sb = params["state_w"], params["state_b"]
    flat("state_w", sw)
    flat("state_b", sb)
    for i, blk in enumerate(params["blocks"]):
        p = str(i)
        flat(f"b{p}_ln1_g", blk["ln1_g"])
        flat(f"b{p}_ln1_b", blk["ln1_b"])
        lin(f"b{p}_q", blk["q"])
        lin(f"b{p}_k", blk["k"])
        lin(f"b{p}_v", blk["v"])
        lin(f"b{p}_proj", blk["proj"])
        flat(f"b{p}_ln2_g", blk["ln2_g"])
        flat(f"b{p}_ln2_b", blk["ln2_b"])
        lin(f"b{p}_ff1", blk["ff1"])
        lin(f"b{p}_ff2", blk["ff2"])
    lin("head_wm", params["heads"]["wm"])
    lin("head_wb", params["heads"]["wb"])
    lin("head_wv", params["heads"]["wv"])
    return out


def read_ckpt(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: 加载失败（{type(e).__name__}: {e}）")
        return None


# ---------------- JS 前向的 Python 复刻（--verify 用） ----------------

def _forward_fp32(params, obs, state=None):
    """jax_net.transformer_forward 的 fp32 版（--verify 用，避免 bf16 精度
    干扰结构比对）。结构与 jax_net 完全一致：patch 切块 + state token +
    pre-norm MHA/FFN + patch-token 均值池化 + 三头。"""
    import jax
    import jax.numpy as jnp
    n = obs.shape[0]
    c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
    pd = params["tok"][0].shape[0]
    P = int(round((pd / c) ** 0.5))
    gp = -(-h // P)
    n_tok = gp * gp
    x = obs
    if gp * P != h or gp * P != w:
        x = jnp.pad(x, [(0, 0), (0, 0), (0, gp * P - h), (0, gp * P - w)])
    x = x.reshape(n, c, gp, P, gp, P)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(n, gp * gp, c * P * P)
    x = x @ params["tok"][0] + params["tok"][1] + params["pos"][:, :n_tok]
    if state is not None:
        st = state @ params["state_w"] + params["state_b"]
        x = jnp.concatenate([x, st[:, None] + params["pos"][:, -1:]], axis=1)

    def ln(xx, g, b):
        m = xx.mean(-1, keepdims=True)
        v = xx.var(-1, keepdims=True)
        return (xx - m) * jax.lax.rsqrt(v + 1e-6) * g + b

    heads = 4
    for blk in params["blocks"]:
        e = x.shape[-1]
        ln1 = ln(x, blk["ln1_g"], blk["ln1_b"])
        d = e // heads
        q = (ln1 @ blk["q"][0] + blk["q"][1]).reshape(n, x.shape[1], heads, d).transpose(0, 2, 1, 3)
        k = (ln1 @ blk["k"][0] + blk["k"][1]).reshape(n, x.shape[1], heads, d).transpose(0, 2, 1, 3)
        vv = (ln1 @ blk["v"][0] + blk["v"][1]).reshape(n, x.shape[1], heads, d).transpose(0, 2, 1, 3)
        sc = jnp.einsum("...htd,...hTd->...htT", q, k) / jnp.sqrt(d)
        w_ = jax.nn.softmax(sc, axis=-1)
        att = jnp.einsum("...htT,...hTd->...htd", w_, vv).transpose(0, 2, 1, 3).reshape(*x.shape)
        x = x + att @ blk["proj"][0] + blk["proj"][1]
        ln2 = ln(x, blk["ln2_g"], blk["ln2_b"])
        x = x + jax.nn.relu(ln2 @ blk["ff1"][0] + blk["ff1"][1]) @ blk["ff2"][0] + blk["ff2"][1]
    g = x[:, :n_tok].mean(1)
    mv = g @ params["heads"]["wm"][0] + params["heads"]["wm"][1]
    bm = g @ params["heads"]["wb"][0] + params["heads"]["wb"][1]
    v_raw = g @ params["heads"]["wv"][0] + params["heads"]["wv"][1]
    if v_raw.shape[-1] == 128:
        bin_centers = jnp.linspace(-1.0, 1.0, 128)
        v = jnp.sum(jax.nn.softmax(v_raw, axis=-1) * bin_centers, axis=-1)
    else:
        v = v_raw.squeeze(-1)
    return mv, bm, v


def transformer_forward_js(params_flat: dict, obs: np.ndarray,
                           state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与 sim.js TransformerModel.forward 逐位一致（fp32，float64 中间量）。"""
    c, h, w = obs.shape
    pd = params_flat["tok_w"].shape[0]
    P = int(round((pd / c) ** 0.5))
    gp = -(-h // P)
    n_tok = gp * gp
    hp, wp = gp * P, gp * P
    x = obs.astype(np.float64)
    if hp != h or wp != w:
        # (C, H, W) → 只 pad H/W 两维
        x = np.pad(x, [(0, 0), (0, hp - h), (0, wp - w)])
    x = x.reshape(c, gp, P, gp, P).transpose(1, 3, 0, 2, 4).reshape(n_tok, c * P * P)
    E = params_flat["tok_w"].shape[1]
    x = x @ params_flat["tok_w"].astype(np.float64) + params_flat["tok_b"].astype(np.float64)
    x = x + params_flat["pos"][:n_tok].astype(np.float64)
    heads = 4
    st = state.astype(np.float64) @ params_flat["state_w"].astype(np.float64) \
        + params_flat["state_b"].astype(np.float64)
    x = np.concatenate([x, st[None] + params_flat["pos"][-1:]], axis=0)

    n_blk = sum(1 for k in params_flat if k.startswith("b") and k[1].isdigit()
                and k.endswith("_ln1_g"))
    for i in range(n_blk):
        p = str(i)
        ln1 = _ln(x, params_flat[f"b{p}_ln1_g"], params_flat[f"b{p}_ln1_b"])
        d = E // heads
        # 注意：JAX 是 (T,E).reshape(T,heads,d).transpose(0,2,1,3) → (heads,T,d)，
        # 不能直接 reshape(heads,T,d)（会把 head/token 打乱错位）。
        q = (ln1 @ params_flat[f"b{p}_q_w"] + params_flat[f"b{p}_q_b"]).reshape(x.shape[0], heads, d).transpose(1, 0, 2)
        k = (ln1 @ params_flat[f"b{p}_k_w"] + params_flat[f"b{p}_k_b"]).reshape(x.shape[0], heads, d).transpose(1, 0, 2)
        v = (ln1 @ params_flat[f"b{p}_v_w"] + params_flat[f"b{p}_v_b"]).reshape(x.shape[0], heads, d).transpose(1, 0, 2)
        scores = np.einsum("htd,hTd->htT", q, k) / np.sqrt(d)
        w = _softmax(scores.astype(np.float32)).astype(np.float64)
        att = np.einsum("htT,hTd->htd", w, v).transpose(1, 0, 2).reshape(x.shape[0], E)
        x = x + att @ params_flat[f"b{p}_proj_w"] + params_flat[f"b{p}_proj_b"]
        ln2 = _ln(x, params_flat[f"b{p}_ln2_g"], params_flat[f"b{p}_ln2_b"])
        ff = np.maximum(0, ln2 @ params_flat[f"b{p}_ff1_w"] + params_flat[f"b{p}_ff1_b"])
        x = x + ff @ params_flat[f"b{p}_ff2_w"] + params_flat[f"b{p}_ff2_b"]

    g = x[:n_tok].mean(0)
    mv = g @ params_flat["head_wm_w"] + params_flat["head_wm_b"]
    bm = g @ params_flat["head_wb_w"] + params_flat["head_wb_b"]
    v_raw = g @ params_flat["head_wv_w"] + params_flat["head_wv_b"]
    if v_raw.shape[-1] == 128:
        bin_centers = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
        v = np.sum(_softmax(v_raw.astype(np.float32)).astype(np.float64) * bin_centers, axis=-1)
    else:
        v = v_raw.squeeze(-1)
    return mv.astype(np.float32), bm.astype(np.float32), v.astype(np.float32)


def _ln(x, g, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    var = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(var + eps) * g.astype(np.float64) + b.astype(np.float64)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------- 主流程 ----------------

def export_one(path: str, verify: bool, out_dir: str | None = None,
               incremental: bool = False) -> bool:
    out_dir = out_dir or OUT_DIR
    stem = model_stem(path)
    out_path = os.path.join(out_dir, stem + ".json")
    if incremental and os.path.exists(out_path) \
            and os.path.getmtime(out_path) >= os.path.getmtime(path):
        print(f"  [same] {stem}: 已是最新（跳过）")
        return False
    ck = read_ckpt(path)
    if ck is None:
        return False
    # 两种布局都兼容：ckpt 带 "params" 包装键，或顶层直接就是 params dict
    params = ck.get("params") if isinstance(ck, dict) and "params" in ck else ck
    if not isinstance(params, dict) or "tok" not in params:
        print(f"  [skip] {os.path.basename(path)}: 不是 JAX transformer ckpt"
              f"（params 无 tok 键）")
        return False
    weights = extract_transformer(params)

    # obs_shape / 全局步：ckpt 里没存 obs_shape（jax 版不带），从权重推。
    # tok_w: [patch²·C, embed] → 反推 patch 与通道数（兼容 13 通道旧 ViT
    # ckpt 与 14 通道新 ckpt——ch13=可推箱，2026-08 起新训练默认）。
    tot = int(weights["tok_w"].shape[0])
    c, patch = next((tot // (p * p), p) for p in (4, 3, 2, 5, 6)
                    if tot % (p * p) == 0 and 10 <= tot // (p * p) <= 16)
    h, w = 13, 15
    embed = int(weights["tok_w"].shape[1])
    depth = int(sum(1 for k in weights if k.startswith("b") and k[1].isdigit()
                    and k.endswith("_ln1_g")))
    # 训练进度：ckpt 里没存 it（jax 版不带），从文件名解析（params_it00000204
    # → it=204）；解析不到时退回 ck.get("it")。global_step = it × steps/iter
    # （8 卡口径 2×16384×256=8.39M，见 scnet_train_8gpu_v8.sh 注释）。
    import re
    _m = re.search(r"_it(\d+)", os.path.basename(path))
    it = int(_m.group(1)) if _m else (int(ck.get("it") or 0)
                                      if isinstance(ck, dict) else 0)
    steps_per_iter = 2 * 16384 * 256
    meta = {
        "name": stem,
        "display_name": stem,
        "arch": "transformer",
        "obs_shape": [c, h, w],
        "embed": embed,
        "patch": patch,
        "depth": depth,
        "n_players": 2,
        "it": it,
        "global_step": it * steps_per_iter,
        "source": os.path.basename(path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    if verify:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        # jax_net.transformer_forward 内部强制 bf16 cast（DCU 训练精度）。
        # 自检用**fp32 手动前向**：与 JS 复刻（fp64，导出权重）对比纯结构差。
        # bf16 训练 vs fp32 推理的差（~0.3）是精度差异，不是结构错误。
        params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), params)
        key = jrandom.PRNGKey(0)
        obs = jrandom.normal(key, (2, c, h, w))
        state = jrandom.normal(key, (2, 24))
        mv_j, bm_j, v_j = _forward_fp32(params_f32, obs, state)
        mv_j, bm_j, v_j = (np.asarray(x) for x in (mv_j, bm_j, v_j))
        ok = True
        for pid in (0, 1):
            mv_js, bm_js, v_js = transformer_forward_js(
                weights, np.asarray(obs[pid]), np.asarray(state[pid]))
            d_m = float(np.abs(mv_j[pid] - mv_js).max())
            d_b = float(np.abs(bm_j[pid] - bm_js).max())
            d_v = float(np.abs(v_j[pid] - v_js).max())
            ok &= max(d_m, d_b, d_v) < 0.05
            print(f"  [verify] pid={pid} maxdiff move={d_m:.2e} bomb={d_b:.2e} "
                  f"value={d_v:.2e}")
        if not ok:
            print(f"  [FAIL] {stem}: 前向自检不一致，已跳过导出")
            return False

    b64, index = pack_tensors(weights)
    doc = {"meta": meta, "flat": b64, "tensors": index}
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"  [OK] {stem}: {len(b64) * 3 // 4 / 1e6:.2f}MB → {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                    help="指定 ckpt 文件（默认导出 ckpt/ 下全部 JAX transformer）")
    ap.add_argument("--verify", action="store_true", help="导出后做前向自检")
    ap.add_argument("--incremental", action="store_true",
                    help="只导出比 web/models 里 JSON 新的档（已最新跳过）")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 web/models）")
    args = ap.parse_args()
    out_dir = args.out_dir or OUT_DIR

    paths = args.paths
    if not paths:
        if not os.path.isdir(args.ckpt_dir):
            print(f"没有 ckpt 目录 {args.ckpt_dir}")
            return 1
        paths = sorted(
            os.path.join(args.ckpt_dir, f)
            for f in os.listdir(args.ckpt_dir)
                       if (f.startswith("params_") or f.startswith("ViTModel"))
                       and f.endswith(".pkl"))

    if not paths:
        print("没有可导出的 JAX transformer ckpt")
        return 1

    ok = 0
    for p in paths:
        # export_one 内部用模块级 OUT_DIR；这里通过参数传递
        if export_one(p, args.verify, out_dir, incremental=args.incremental):
            ok += 1
    print(f"\n导出完成: {ok}/{len(paths)}")
    # index.json 由目录扫描重建（与 export_ckpt.py 同一逻辑，保证旧模型不丢）
    try:
        from export_ckpt import scan_out_dir
        metas = scan_out_dir(out_dir)
        if metas:
            metas.sort(key=lambda m: m.get("global_step") or m.get("it") or 0,
                       reverse=True)
            index_path = os.path.join(out_dir, "index.json")
            with open(index_path, "w") as f:
                json.dump({"models": metas}, f, ensure_ascii=False, indent=1)
            print(f"index.json 更新: {len(metas)} 个模型")
    except Exception as e:
        print(f"WARN: index.json 更新失败（{e}）")
    return 0  # 全跳过（incremental 已最新）也是成功，非零会让 serve_web.sh 的 set -e 中断


if __name__ == "__main__":
    sys.exit(main())
