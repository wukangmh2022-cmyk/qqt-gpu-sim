"""Policy-value networks (pure-jax, no equinox): mlp | cnn | transformer.

Params are plain dicts of arrays; forward is a pure function so jax.grad /
jit / lax.scan work directly. Designed to be scaled to the 5M-15M parameter
range recommended in architects.md (embed/depth/channels configurable).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import random

from .jax_env import N_BOMB, N_MOVES

# ---------------- HL-Gauss Categorical Value Head Constants ----------------
NUM_VALUE_BINS = 128
V_MIN = -1.0
V_MAX = 1.0
BIN_CENTERS = jnp.linspace(V_MIN, V_MAX, NUM_VALUE_BINS)


# ---------------- MLP ----------------


def _linear_init(key, fan_in, fan_out, scale=1.0):
    w = random.normal(key, (fan_in, fan_out)) * jnp.sqrt(scale / fan_in)
    b = jnp.zeros((fan_out,))
    return w, b


def init_mlp(key, c, h, w, hidden=256):
    k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
    flat = c * h * w
    w1, b1 = _linear_init(k1, flat, hidden)
    w2, b2 = _linear_init(k2, hidden, hidden)
    wm, bm = _linear_init(k3, hidden, N_MOVES, scale=0.01)
    wb, bb = _linear_init(k4, hidden, N_BOMB, scale=0.01)
    wv, bv = _linear_init(k5, hidden, NUM_VALUE_BINS)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2,
            "wm": wm, "bm": bm, "wb": wb, "bb": bb, "wv": wv, "bv": bv}


def mlp_forward(params, obs):
    x = obs.reshape(obs.shape[0], -1)
    x = jax.nn.relu(x @ params["w1"] + params["b1"])
    x = jax.nn.relu(x @ params["w2"] + params["b2"])
    mv = x @ params["wm"] + params["bm"]
    bm = x @ params["wb"] + params["bb"]
    v_logits = x @ params["wv"] + params["bv"]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


def mlp_bf16_forward(params, obs):
    """mlp 的 bf16 计算版（DCU fp32 GEMM 慢，同 mlp4_forward 思路），
    权重 cast bf16 计算、heads 回 fp32，结构同 mlp_forward。
    """
    bf = jnp.bfloat16
    x = obs.astype(bf).reshape(obs.shape[0], -1)
    x = jax.nn.relu(x @ params["w1"].astype(bf) + params["b1"].astype(bf))
    x = jax.nn.relu(x @ params["w2"].astype(bf) + params["b2"].astype(bf))
    x = x.astype(jnp.float32)
    mv = x @ params["wm"] + params["bm"]
    bm = x @ params["wb"] + params["bb"]
    v_logits = x @ params["wv"] + params["bv"]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


# ---------------- MLP-4（正式版结构：4 层 + LayerNorm，hidden=768） ----------------


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + 1e-5) * g + b


def init_mlp4(key, c, h, w, hidden=768):
    """正式版 4 层 MLP：输入投影 + 4 个隐藏层（≈321 万参数 @ hidden=768）。"""
    n_layers = 5                       # 1 输入投影 + 4 隐藏层
    k = random.split(key, n_layers + 3)
    flat = c * h * w
    p = {}
    for i in range(n_layers):
        fan_in = flat if i == 0 else hidden
        w, b = _linear_init(k[i], fan_in, hidden)
        p[f"w{i + 1}"] = w
        p[f"b{i + 1}"] = b
        p[f"ln{i + 1}_g"] = jnp.ones((hidden,))
        p[f"ln{i + 1}_b"] = jnp.zeros((hidden,))
    wm, bm = _linear_init(k[n_layers], hidden, N_MOVES, scale=0.01)
    wb, bb = _linear_init(k[n_layers + 1], hidden, N_BOMB, scale=0.01)
    wv, bv = _linear_init(k[n_layers + 2], hidden, NUM_VALUE_BINS)
    p.update({"wm": wm, "bm": bm, "wb": wb, "bb": bb, "wv": wv, "bv": bv})
    return p


def mlp4_forward(params, obs):
    """bf16 计算（DCU fp32 GEMM ~2× 慢于 bf16，Average Joe 同款），heads 保 fp32。"""
    bf = jnp.bfloat16
    x = obs.astype(bf).reshape(obs.shape[0], -1)
    for i in range(5):
        w = params[f"w{i + 1}"].astype(bf)
        b = params[f"b{i + 1}"].astype(bf)
        g = params[f"ln{i + 1}_g"].astype(bf)
        bb = params[f"ln{i + 1}_b"].astype(bf)
        x = jax.nn.relu(_ln(x @ w + b, g, bb))
    x = x.astype(jnp.float32)
    mv = x @ params["wm"] + params["bm"]
    bm = x @ params["wb"] + params["bb"]
    v_logits = x @ params["wv"] + params["bv"]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


# ---------------- CNN ----------------


def _conv_init(key, cin, cout, kh, kw, scale=2.0):
    w = random.normal(key, (kh, kw, cin, cout)) * jnp.sqrt(scale / (cin * kh * kw))
    b = jnp.zeros((cout,))
    return w, b


def init_cnn(key, c, h, w, ch1=32, ch2=64, hidden=256):
    k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
    w1, b1 = _conv_init(k1, c, ch1, 3, 3)
    w2, b2 = _conv_init(k2, ch1, ch2, 3, 3)
    w3, b3 = _linear_init(k3, ch2, hidden)   # GAP 后接小 MLP（不 flatten）
    wm, bm = _linear_init(k4, hidden, N_MOVES, scale=0.01)
    wb, bb = _linear_init(k5, hidden, N_BOMB, scale=0.01)
    wv, bv = _linear_init(k6, hidden, NUM_VALUE_BINS)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3,
            "wm": wm, "bm": bm, "wb": wb, "bb": bb, "wv": wv, "bv": bv}


def cnn_forward(params, obs):
    """obs (N, C, H, W) → NHWC for jax conv。

    conv 段用 bf16：DCU 上 fp32 conv+bias+relu 的 XLA 融合有缺陷（~100ms，
    实测见 probe_jax_cnn5），bf16 走正常 kernel（~2ms）。GAP 后转回 fp32。
    """
    bf = jnp.bfloat16
    x = obs.astype(bf)
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = jax.nn.relu(jax.lax.conv_general_dilated(
        x, params["w1"].astype(bf), (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params["b1"].astype(bf))
    x = jax.nn.relu(jax.lax.conv_general_dilated(
        x, params["w2"].astype(bf), (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + params["b2"].astype(bf))
    x = x.mean((1, 2)).astype(jnp.float32)        # GAP: (N,9,9,64) → (N,64)
    x = jax.nn.relu(x @ params["w3"] + params["b3"])
    mv = x @ params["wm"] + params["bm"]
    bm = x @ params["wb"] + params["bb"]
    v_logits = x @ params["wv"] + params["bv"]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


# ---------------- Transformer (ViT-ish, patch=3) ----------------


def _attn(q, k, v, mask=None):
    """q,k,v (..., heads, T, d)。mask (..., T, T) 或 None。

    softmax 在 fp32 计算后转回输入 dtype（bf16 下精度更稳，Average Joe 同款）。
    """
    d = q.shape[-1]
    scores = jnp.einsum("...htd,...hTd->...htT", q, k) / jnp.sqrt(d)
    if mask is not None:
        scores = scores + mask  # mask: 0 保留, -inf 屏蔽（causal 时用）
    w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.einsum("...htT,...hTd->...htd", w, v)


def init_transformer(key, c, h, w, embed=392, depth=4, heads=4, ff_factor=4,
                     patch=3, state_dim=24):
    """ViT 风格。patch>1 时按 patch 切块（Average Joe patchify 机制）：
    token 数 = (h//patch)²，attention 计算量 ∝ token²，patch 2/3 对 13×13
    地图把 attention 降 10-100 倍（参数在 ffn，几乎不变）。

    state_dim>0：论文式双序列 —— 全局状态向量（时长/血量/成长属性/存活）
    经 state_w/_b 投影成第 n_tok+1 个 state token，与 patch tokens 一起过
    attention（微操：格内分数坐标在 splat 通道、速度/血量在 state 向量）。
    pos 扩到 n_tok+1（state token 自己的可学习位置编码）。"""
    # 每 block 用 6 个 key 索引：k=keys[i:i+4]（q/k/v/proj）后 i+=4，
    # k2=split(keys[i])（ff1/ff2 seed）后 i+=2；头部再 1 个、state 再 1 个。
    keys = random.split(key, 6 * depth + 3)
    gp = -(-h // patch)                  # ceil(h/patch)：13//2=6 不够，需 7
    n_tok = gp * gp
    patch_dim = c * patch * patch
    p = {}
    p["tok"] = _linear_init(keys[0], patch_dim, embed)   # 每 patch 投影
    p["pos"] = random.normal(keys[1], (1, n_tok + 1, embed)) * 0.02
    i = 1
    p["blocks"] = []
    for d in range(depth):
        k = keys[i:i + 4]; i += 4
        blk = {
            "ln1_g": jnp.ones((embed,)), "ln1_b": jnp.zeros((embed,)),
            "q": _linear_init(k[0], embed, embed),
            "k": _linear_init(k[1], embed, embed),
            "v": _linear_init(k[2], embed, embed),
            "proj": _linear_init(k[3], embed, embed),
            "ln2_g": jnp.ones((embed,)), "ln2_b": jnp.zeros((embed,)),
        }
        # MLP block
        k2 = random.split(keys[i], 2); i += 2
        blk["ff1"] = _linear_init(k2[0], embed, int(ff_factor * embed))
        blk["ff2"] = _linear_init(k2[1], int(ff_factor * embed), embed)
        p["blocks"].append(blk)
    km, kb, kv = random.split(keys[i], 3); i += 1
    p["heads"] = {}
    p["heads"]["wm"] = _linear_init(km, embed, N_MOVES, scale=0.01)
    p["heads"]["wb"] = _linear_init(kb, embed, N_BOMB, scale=0.01)
    p["heads"]["wv"] = _linear_init(kv, embed, NUM_VALUE_BINS)
    # 全局状态向量 → state token（论文式双序列第二路）
    p["state_w"], p["state_b"] = _linear_init(keys[i], state_dim, embed)
    return p


def _tf_block(x, blk, heads):
    """pre-norm + multi-head attention + MLP。权重都是 (w, b) tuple。

    x 是 bf16；LN 参数与投影权重 cast 到 bf16 保持全 bf16 计算
    （DCU fp32 融合缺陷，见 cnn_forward 注释）。
    """
    bf = jnp.bfloat16
    e = x.shape[-1]
    # LayerNorm
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    ln = (x - m) * jax.lax.rsqrt(v + 1e-6)
    ln1 = ln * blk["ln1_g"].astype(bf) + blk["ln1_b"].astype(bf)
    q = ln1 @ blk["q"][0].astype(bf) + blk["q"][1].astype(bf)
    k = ln1 @ blk["k"][0].astype(bf) + blk["k"][1].astype(bf)
    vv = ln1 @ blk["v"][0].astype(bf) + blk["v"][1].astype(bf)
    d = e // heads
    q = q.reshape(*q.shape[:-1], heads, d).transpose(0, -2, 1, -1)
    k = k.reshape(*k.shape[:-1], heads, d).transpose(0, -2, 1, -1)
    vv = vv.reshape(*vv.shape[:-1], heads, d).transpose(0, -2, 1, -1)
    att = _attn(q, k, vv)                                # (N, heads, T, d)
    att = att.transpose(0, 2, 1, 3).reshape(*x.shape[:-1], e)
    x = x + att @ blk["proj"][0].astype(bf) + blk["proj"][1].astype(bf)
    ln2 = (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(x.var(-1, keepdims=True) + 1e-6)
    ln2 = ln2 * blk["ln2_g"].astype(bf) + blk["ln2_b"].astype(bf)
    x = x + jax.nn.relu(ln2 @ blk["ff1"][0].astype(bf) + blk["ff1"][1].astype(bf)) \
        @ blk["ff2"][0].astype(bf) + blk["ff2"][1].astype(bf)
    return x


def transformer_forward(params, obs, state=None):
    """ViT 风格。计算走 bf16（DCU fp32 融合缺陷，同 CNN），softmax/输出保 fp32。

    patch 切块（Average Joe patchify）：obs (N, C, H, W) → 每 patch 展平
    (C*P*P) 经 tok 投影 → (N, n_tok, embed)。P=1 退化为逐格 token。
    13×13 非 P 倍数 → pad 到 ceil(13/P)*P（右/下补零）。pad 量是 Python
    静态量（H/W 是常量），scan 内不依赖 tracer。
    """
    n = obs.shape[0]
    c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
    bf = jnp.bfloat16
    # patch 从 tok 权重 shape 反推（shape 恒静态，避免 jit 参数里 int 被
    # 动态化——DCU 的 JAX 与本地行为不同，dict int 会被 tracer）
    pd = params["tok"][0].shape[0]
    P = int(round((pd / c) ** 0.5))
    gp = -(-h // P)                     # ceil(h/P)：13//2=6 不够，需 7
    n_tok = gp * gp
    hp, wp = gp * P, gp * P
    x = obs.astype(bf)
    if hp != h or wp != w:
        # 静态 pad（Python 层求值：h/w/gp/P 都是编译期常量）
        x = jnp.pad(x, [(0, 0), (0, 0), (0, hp - h), (0, wp - w)])
    # (N, C, gp, P, gp, P) -> (N, gp*gp, C*P*P)
    x = x.reshape(n, c, gp, P, gp, P)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(n, gp * gp, c * P * P)
    tok_w, tok_b = params["tok"]
    pos = params["pos"].astype(bf)                 # (1, n_tok+1, E)
    x = x @ tok_w.astype(bf) + tok_b.astype(bf) + pos[:, :n_tok]
    if state is not None:
        # 论文式双序列：全局状态向量 → state token，与 patch tokens 一起
        # 过 attention（血量/速度/属性与空间信息交叉推理；微操输入）
        st = (state.astype(bf) @ params["state_w"].astype(bf)
              + params["state_b"].astype(bf))      # (N, E)
        x = jnp.concatenate([x, st[:, None] + pos[:, -1:]], axis=1)
    for blk in params["blocks"]:
        x = _tf_block(x, blk, 4)
    g = x[:, :n_tok].mean(1).astype(jnp.float32)   # 池化只用 patch tokens
    mv = g @ params["heads"]["wm"][0] + params["heads"]["wm"][1]
    bm = g @ params["heads"]["wb"][0] + params["heads"]["wb"][1]
    v_logits = g @ params["heads"]["wv"][0] + params["heads"]["wv"][1]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


# ---------------- MLP-Mixer（保留 patch 感受野，无 attention） ----------------

# DCU 上 ViT 慢的根因：attention 是 (T,T) 小矩阵乘 + softmax + 大量小算子
# → launch 开销占比高（54 算子/层 vs GEMM 大矩阵吃不满 bf16 320TFLOPS）。
# MLP-Mixer 用 token/channel 两个 MLP 替代 MHSA：
#   - token-mixing：转置后沿 token 维做 MLP（跨 patch 信息交换 ≈ CNN 感受野）
#   - channel-mixing：沿 channel 维做 MLP（逐 patch 特征变换）
#   核心全是规则大 GEMM（batch=N×C 或 N×T），无 (T,T) 小乘、无 softmax、
#   无位置编码 → launch 开销大幅下降，且保留 patch 结构（归纳偏置）。


def init_mlp_mixer(key, c, h, w, embed=256, depth=8, patch=2, ff_factor=4,
                   token_ratio=2):
    """patch 化 + depth 个 Mixer 层。token_ratio 控制 token-mixing 隐层宽
    （token 数少，太小没意义；默认 2×）。"""
    gp = -(-h // patch)                  # ceil(13/patch)：2→7, 3→5
    n_tok = gp * gp
    patch_dim = c * patch * patch
    keys = random.split(key, 3 + depth * 8)
    p = {}
    p["tok"] = _linear_init(keys[0], patch_dim, embed)
    p["blocks"] = []
    for d in range(depth):
        k = keys[1 + d * 8: 1 + (d + 1) * 8]
        blk = {
            # token-mixing: LN -> (N,C,T) MLP T→T*tr→T -> residual
            "tln_g": jnp.ones((embed,)), "tln_b": jnp.zeros((embed,)),
            "tm1": _linear_init(k[0], n_tok, int(n_tok * token_ratio)),
            "tm2": _linear_init(k[1], int(n_tok * token_ratio), n_tok),
            # channel-mixing: LN -> (N,T,C) MLP C→C*ff→C -> residual
            "cln_g": jnp.ones((embed,)), "cln_b": jnp.zeros((embed,)),
            "cm1": _linear_init(k[2], embed, int(embed * ff_factor)),
            "cm2": _linear_init(k[3], int(embed * ff_factor), embed),
        }
        p["blocks"].append(blk)
    km, kb, kv = random.split(keys[1 + depth * 8], 3)
    p["heads"] = {}
    p["heads"]["wm"] = _linear_init(km, embed, N_MOVES, scale=0.01)
    p["heads"]["wb"] = _linear_init(kb, embed, N_BOMB, scale=0.01)
    p["heads"]["wv"] = _linear_init(kv, embed, NUM_VALUE_BINS)
    return p


def mlp_mixer_forward(params, obs):
    """patch 切块 → Mixer 层 → 全局池化 → heads。全 bf16（DCU GEMM 强项）。"""
    n = obs.shape[0]
    c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
    bf = jnp.bfloat16
    pd = params["tok"][0].shape[0]
    P = int(round((pd / c) ** 0.5))      # shape 反推 patch（恒静态）
    gp = -(-h // P)
    hp, wp = gp * P, gp * P
    x = obs.astype(bf)
    if hp != h or wp != w:
        x = jnp.pad(x, [(0, 0), (0, 0), (0, hp - h), (0, wp - w)])
    x = x.reshape(n, c, gp, P, gp, P)
    x = x.transpose(0, 2, 4, 1, 3, 5).reshape(n, gp * gp, c * P * P)
    tok_w, tok_b = params["tok"]
    x = x @ tok_w.astype(bf) + tok_b.astype(bf)          # (N, T, E)

    for blk in params["blocks"]:
        # token-mixing：(N, T, E) -> (N, E, T) 沿 token 维 MLP
        t = x.mean(-1, keepdims=True)
        tv = x.var(-1, keepdims=True)
        tln = (x - t) * jax.lax.rsqrt(tv + 1e-6)
        tln = tln * blk["tln_g"].astype(bf) + blk["tln_b"].astype(bf)
        t1 = tln.transpose(0, 2, 1)                        # (N, E, T)
        t1 = jax.nn.relu(t1 @ blk["tm1"][0].astype(bf) + blk["tm1"][1].astype(bf))
        t1 = t1 @ blk["tm2"][0].astype(bf) + blk["tm2"][1].astype(bf)
        x = x + t1.transpose(0, 2, 1)                      # residual

        # channel-mixing：沿 channel 维 MLP
        m = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        cln = (x - m) * jax.lax.rsqrt(v + 1e-6)
        cln = cln * blk["cln_g"].astype(bf) + blk["cln_b"].astype(bf)
        c1 = jax.nn.relu(cln @ blk["cm1"][0].astype(bf) + blk["cm1"][1].astype(bf))
        c1 = c1 @ blk["cm2"][0].astype(bf) + blk["cm2"][1].astype(bf)
        x = x + c1                                          # residual

    g = x.mean(1).astype(jnp.float32)                      # 全局池化 (N, E)
    mv = g @ params["heads"]["wm"][0] + params["heads"]["wm"][1]
    bm = g @ params["heads"]["wb"][0] + params["heads"]["wb"][1]
    v_logits = g @ params["heads"]["wv"][0] + params["heads"]["wv"][1]
    v_scalar = jnp.sum(jax.nn.softmax(v_logits, axis=-1) * BIN_CENTERS, axis=-1)
    return mv, bm, v_scalar, v_logits


# ---------------- dispatch ----------------


INIT = {"mlp": init_mlp, "mlp_bf16": init_mlp, "mlp4": init_mlp4,
        "cnn": init_cnn, "transformer": init_transformer,
        "mixer": init_mlp_mixer}
FWD = {"mlp": mlp_forward, "mlp_bf16": mlp_bf16_forward,
       "mlp4": mlp4_forward,
       "cnn": cnn_forward, "transformer": transformer_forward,
       "mixer": mlp_mixer_forward}


def init_net(key, arch, c, h, w, **kw):
    return INIT[arch](key, c, h, w, **kw)


def net_forward(params, arch, obs, state=None):
    if arch == "transformer":
        return transformer_forward(params, obs, state)
    return FWD[arch](params, obs)


def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params)
               if hasattr(x, "size"))
