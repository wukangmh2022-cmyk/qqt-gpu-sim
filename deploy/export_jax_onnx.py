#!/usr/bin/env python3
"""JAX transformer ckpt → web/models/*.onnx（onnxruntime-web 推理用）。

与 deploy/export_jax_ckpt.py（JSON，纯 JS 前向）并列：同一份扁平权重
（extract_transformer 的 bf16→fp32 表）用 onnx.helper 直接构建 ONNX 图。
transformer 只有 6 种标准算子（MatMul/Add/LayerNorm/Softmax/Reshape/Relu），
不需要 jax2onnx，也不需要 JAX 运行时。

图结构（batch N 为符号维度，1 玩家 / 2 玩家都行）：
  obs[N,13,13,15] → pad(0,0,3,1) → reshape/transpose/reshape → tok 线性 + pos
  state[N,24] → state 线性 + pos → 与 patch tokens concat → [N,17,392]
  ×depth block：LN1 → q/k/v → scores/softmax/att → proj 残差 → LN2 → ReLU FFN
  → Slice 前 16 tokens → ReduceMean → 三头（move[N,5]/bomb[N,2]/value[N]）

用法：
    PYTHONPATH=. .venv/bin/python deploy/export_jax_onnx.py            # 导出 ckpt/ 下全部
    PYTHONPATH=. .venv/bin/python deploy/export_jax_onnx.py ckpt/x.pkl # 指定档
    # 可选 --verify：onnxruntime CPU 前向 vs jax fp32 手动前向逐位对比
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
CKPT_DIR = os.path.join(PROJ, "ckpt")
OUT_DIR = os.path.join(PROJ, "web", "models")

from deploy.export_jax_ckpt import extract_transformer, model_stem  # noqa: E402


def _init(name, arr):
    import onnx
    from onnx import helper, numpy_helper
    a = np.asarray(arr)
    if a.dtype.kind == "f":                    # float 一律 float32；int/bool 保持原类型
        a = a.astype(np.float32)
    return numpy_helper.from_array(a, name=name)


def build_onnx(weights: dict, embed: int, depth: int, patch: int,
               h: int = 13, w: int = 15, heads: int = 4, c: int = 14) -> bytes:
    """由扁平权重构建 ONNX 图，返回序列化 model（bytes）。

    c = obs 通道数（14=新标准含 ch13 可推箱；旧 13 通道 ckpt 传 13）。
    """
    import onnx
    from onnx import TensorProto, helper

    gp = -(-h // patch)
    n_tok = gp * gp
    tot_tok = n_tok + 1
    d = embed // heads
    F = int(weights["b0_ff1_w"].shape[1])

    inits = []
    nodes = []
    outs = []          # 追加中间输出名
    _c = 0

    def C(name, arr):
        inits.append(_init(name, arr))
        return name

    def N(op, inputs, outputs, **kw):
        nodes.append(helper.make_node(op, inputs, outputs, **kw))
        return outputs[0]

    def mid(prefix=""):
        nonlocal _c
        _c += 1
        return f"{prefix}_{_c}"

    # ---- 常量权重 ----
    tok_w = C("tok_w", weights["tok_w"]); tok_b = C("tok_b", weights["tok_b"])
    pos = C("pos", weights["pos"])                       # (tot_tok, E)
    st_w = C("state_w", weights["state_w"]); st_b = C("state_b", weights["state_b"])
    head_wm = C("head_wm_w", weights["head_wm_w"]); head_bm = C("head_wm_b", weights["head_wm_b"])
    head_wb = C("head_wb_w", weights["head_wb_w"]); head_bb = C("head_wb_b", weights["head_wb_b"])
    head_wv = C("head_wv_w", weights["head_wv_w"]); head_bv = C("head_wv_b", weights["head_wv_b"])

    obs = "obs"; state = "state"

    # ---- patch embedding ----
    # obs [N,C,H,W] → pad → [N,C,gp*P,gp*P]
    pads = C("pads", np.array([0, 0, 0, 0, 0, 0, gp * patch - h, gp * patch - w], np.int64))
    x = N("Pad", [obs, pads], [mid("pad")], mode="constant")
    x = N("Reshape", [x, C("s1", np.array([0, c, gp, patch, gp, patch], np.int64))], [mid("rs")])
    x = N("Transpose", [x], [mid("tp")], perm=[0, 2, 4, 1, 3, 5])
    x = N("Reshape", [x, C("s2", np.array([0, n_tok, c * patch * patch], np.int64))], [mid("rs")])
    x = N("MatMul", [x, tok_w], [mid("mm")])
    x = N("Add", [x, tok_b], [mid("add")])
    pos16 = N("Slice", [pos, C("ps1", np.array([0], np.int64)),
                        C("pe1", np.array([n_tok], np.int64)),
                        C("pa1", np.array([0], np.int64))], [mid("pos")])
    x = N("Add", [x, pos16], [mid("add")])

    # ---- state token ----
    st = N("MatMul", [state, st_w], [mid("mm")])
    st = N("Add", [st, st_b], [mid("add")])
    posl = N("Slice", [pos, C("ps2", np.array([n_tok], np.int64)),
                       C("pe2", np.array([n_tok + 1], np.int64)),
                       C("pa2", np.array([0], np.int64))], [mid("pos")])   # (1, E)
    st = N("Add", [st, posl], [mid("add")])
    st = N("Reshape", [st, C("s3", np.array([0, 1, embed], np.int64))], [mid("rs")])
    x = N("Concat", [x, st], [mid("cat")], axis=1)                         # [N,tot_tok,E]

    inv_sqrt = C("inv_sqrt_d", np.array([1.0 / np.sqrt(d)], np.float32))

    # ---- blocks ----
    for i in range(depth):
        p = str(i)
        ln1g = C(f"b{p}_ln1_g", weights[f"b{p}_ln1_g"])
        ln1b = C(f"b{p}_ln1_b", weights[f"b{p}_ln1_b"])
        qw = C(f"b{p}_q_w", weights[f"b{p}_q_w"]); qb = C(f"b{p}_q_b", weights[f"b{p}_q_b"])
        kw = C(f"b{p}_k_w", weights[f"b{p}_k_w"]); kb = C(f"b{p}_k_b", weights[f"b{p}_k_b"])
        vw = C(f"b{p}_v_w", weights[f"b{p}_v_w"]); vb = C(f"b{p}_v_b", weights[f"b{p}_v_b"])
        prw = C(f"b{p}_proj_w", weights[f"b{p}_proj_w"]); prb = C(f"b{p}_proj_b", weights[f"b{p}_proj_b"])
        ln2g = C(f"b{p}_ln2_g", weights[f"b{p}_ln2_g"])
        ln2b = C(f"b{p}_ln2_b", weights[f"b{p}_ln2_b"])
        f1w = C(f"b{p}_ff1_w", weights[f"b{p}_ff1_w"]); f1b = C(f"b{p}_ff1_b", weights[f"b{p}_ff1_b"])
        f2w = C(f"b{p}_ff2_w", weights[f"b{p}_ff2_w"]); f2b = C(f"b{p}_ff2_b", weights[f"b{p}_ff2_b"])

        ln1 = N("LayerNormalization", [x, ln1g, ln1b], [mid("ln")], axis=-1, epsilon=1e-6)
        q = N("MatMul", [ln1, qw], [mid("mm")]); q = N("Add", [q, qb], [mid("add")])
        k = N("MatMul", [ln1, kw], [mid("mm")]); k = N("Add", [k, kb], [mid("add")])
        v = N("MatMul", [ln1, vw], [mid("mm")]); v = N("Add", [v, vb], [mid("add")])
        q = N("Reshape", [q, C(f"sq{i}", np.array([0, tot_tok, heads, d], np.int64))], [mid("rs")])
        k = N("Reshape", [k, C(f"sk{i}", np.array([0, tot_tok, heads, d], np.int64))], [mid("rs")])
        v = N("Reshape", [v, C(f"sv{i}", np.array([0, tot_tok, heads, d], np.int64))], [mid("rs")])
        q = N("Transpose", [q], [mid("tp")], perm=[0, 2, 1, 3])   # [N,4,tot_tok,d]
        kt = N("Transpose", [k], [mid("tp")], perm=[0, 2, 3, 1])  # [N,4,d,tot_tok]
        v = N("Transpose", [v], [mid("tp")], perm=[0, 2, 1, 3])   # [N,4,tot_tok,d]
        sc = N("MatMul", [q, kt], [mid("mm")])
        sc = N("Mul", [sc, inv_sqrt], [mid("mul")])
        sm = N("Softmax", [sc], [mid("sm")], axis=-1)
        att = N("MatMul", [sm, v], [mid("mm")])                    # [N,4,tot_tok,d]
        att = N("Transpose", [att], [mid("tp")], perm=[0, 2, 1, 3])  # [N,tot_tok,4,d]
        att = N("Reshape", [att, C(f"sa{i}", np.array([0, tot_tok, embed], np.int64))], [mid("rs")])
        proj = N("MatMul", [att, prw], [mid("mm")]); proj = N("Add", [proj, prb], [mid("add")])
        x = N("Add", [x, proj], [mid("add")])
        ln2 = N("LayerNormalization", [x, ln2g, ln2b], [mid("ln")], axis=-1, epsilon=1e-6)
        ff = N("MatMul", [ln2, f1w], [mid("mm")]); ff = N("Add", [ff, f1b], [mid("add")])
        ff = N("Relu", [ff], [mid("relu")])
        ff = N("MatMul", [ff, f2w], [mid("mm")]); ff = N("Add", [ff, f2b], [mid("add")])
        x = N("Add", [x, ff], [mid("add")])

    # ---- 池化 + 三头 ----
    g = N("Slice", [x, C("ps3", np.array([0, 0, 0], np.int64)),
                    C("pe3", np.array([1 << 30, n_tok, embed], np.int64)),
                    C("pa3", np.array([0, 1, 2], np.int64))], [mid("slice")])
    g = N("ReduceMean", [g, C("rm_axes", np.array([1], np.int64))], [mid("mean")], keepdims=0)  # [N,E]
    mv = N("MatMul", [g, head_wm], [mid("mm")]); mv = N("Add", [mv, head_bm], [mid("add")])
    bm = N("MatMul", [g, head_wb], [mid("mm")]); bm = N("Add", [bm, head_bb], [mid("add")])
    va = N("MatMul", [g, head_wv], [mid("mm")]); va = N("Add", [va, head_bv], [mid("add")])
    if int(weights["head_wv_w"].shape[1]) == 128:
        sm_v = N("Softmax", [va], [mid("sm")], axis=-1)
        bc = C("bin_centers", np.linspace(-1.0, 1.0, 128, dtype=np.float32))
        va_mul = N("Mul", [sm_v, bc], [mid("mul")])
        va = N("ReduceSum", [va_mul, C("rs_axes", np.array([1], np.int64))], [mid("sum")], keepdims=0)
    else:
        va = N("Squeeze", [va, C("sqz", np.array([1], np.int64))], [mid("add")])
    mv = N("Identity", [mv], ["move"])
    bm = N("Identity", [bm], ["bomb"])
    va = N("Identity", [va], ["value"])

    graph = helper.make_graph(
        nodes,
        "qqt_transformer",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", c, 13, 15]),
         helper.make_tensor_value_info("state", TensorProto.FLOAT, ["N", 24])],
        [helper.make_tensor_value_info("move", TensorProto.FLOAT, ["N", 5]),
         helper.make_tensor_value_info("bomb", TensorProto.FLOAT, ["N", 2]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["N"])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def export_one(path: str, verify: bool, out_dir: str, incremental: bool = False) -> bool:
    stem = model_stem(path)
    out_path = os.path.join(out_dir, stem + ".onnx")
    if incremental and os.path.exists(out_path) \
            and os.path.getmtime(out_path) >= os.path.getmtime(path):
        print(f"  [same] {stem}.onnx: 已是最新（跳过）")
        return False
    try:
        with open(path, "rb") as f:
            ck = pickle.load(f)
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: 加载失败（{type(e).__name__}: {e}）")
        return False
    params = ck.get("params") if isinstance(ck, dict) and "params" in ck else ck
    if not isinstance(params, dict) or "tok" not in params:
        print(f"  [skip] {os.path.basename(path)}: 不是 JAX transformer ckpt")
        return False
    weights = extract_transformer(params)
    embed = int(weights["tok_w"].shape[1])
    # 通道数从 tok_w 反推（[patch²·C, embed]；兼容 13 通道旧 ckpt / 14 通道新 ckpt）
    tot = int(weights["tok_w"].shape[0])
    c, patch = next((tot // (p * p), p) for p in (4, 3, 2, 5, 6)
                    if tot % (p * p) == 0 and 10 <= tot // (p * p) <= 16)
    depth = int(sum(1 for k in weights if k.startswith("b") and k[1].isdigit()
                    and k.endswith("_ln1_g")))

    blob = build_onnx(weights, embed, depth, patch, c=c)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)

    if verify:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        import onnxruntime as ort
        from deploy.export_jax_ckpt import _forward_fp32
        params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), params)
        key = jrandom.PRNGKey(0)
        obs = jrandom.normal(key, (2, c, 13, 15))
        state = jrandom.normal(key, (2, 24))
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        mv_o, bm_o, v_o = sess.run(None, {
            "obs": np.asarray(obs, np.float32),
            "state": np.asarray(state, np.float32),
        })
        mv_j, bm_j, v_j = _forward_fp32(params_f32, obs, state)
        mv_j, bm_j, v_j = (np.asarray(x) for x in (mv_j, bm_j, v_j))
        d_m = float(np.abs(mv_o - mv_j).max())
        d_b = float(np.abs(bm_o - bm_j).max())
        d_v = float(np.abs(v_o - v_j).max())
        ok = max(d_m, d_b, d_v) < 1e-4
        print(f"  [verify] onnx vs jax-fp32: move={d_m:.2e} bomb={d_b:.2e} "
              f"value={d_v:.2e} ({'PASS' if ok else 'FAIL'})")
        if not ok:
            return False

    print(f"  [OK] {stem}.onnx {os.path.getsize(out_path) / 1e6:.1f}MB → {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--incremental", action="store_true",
                    help="只导出比 web/models 里 ONNX 新的档（已最新跳过）")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or OUT_DIR
    paths = args.paths
    if not paths:
        if not os.path.isdir(args.ckpt_dir):
            print(f"没有 ckpt 目录 {args.ckpt_dir}")
            return 1
        paths = sorted(os.path.join(args.ckpt_dir, f)
                       for f in os.listdir(args.ckpt_dir)
                       if (f.startswith("params_") or f.startswith("ViTModel"))
                       and f.endswith(".pkl"))
    if not paths:
        print("没有可导出的 JAX transformer ckpt")
        return 1
    ok = sum(1 for p in paths
             if export_one(p, args.verify, out_dir, incremental=args.incremental))
    print(f"\n导出完成: {ok}/{len(paths)}")
    return 0  # 全跳过（incremental 已最新）也是成功，非零会让 serve_web.sh 的 set -e 中断


if __name__ == "__main__":
    sys.exit(main())
