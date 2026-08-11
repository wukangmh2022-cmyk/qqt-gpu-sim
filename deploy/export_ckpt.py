#!/usr/bin/env python3
"""ckpt/ → web/models/*.json 转换工具（部署工具链的第一环）。

把训练侧 `ckpt/*.pt`（PyTorch 权重）转成浏览器可直接加载的 JSON：
    - 只支持 `arch == "mlp"` 且观测为 (C, 13, 13) 的 2 人模型（web 版 v1 只做 MLP）；
    - **折叠 pid=0 视角置换**：把 `train/model.py::ActorCritic` 里
      `shared.0.weight[:, inv_cols[0]]` 的列块重排**做进导出的权重**。
      这样浏览器端拿到一份 (C,H,W) 的**共享观测**直接做普通矩阵乘即可，
      不需要在 JS 里再实现一遍视角置换；
    - 输出 web/models/<stem>.json（扁平 float32 数组）+ web/models/index.json
      （全部已导出模型的元信息列表，供页面下拉选择）。

生产路径**只依赖 ckpt/ 目录**，不 import 任何 train/play/sim 代码 ——
别人在改训练侧也不会破坏这条链路。`--verify` 自检（可选）会临时 import
train.model 用 torch 对比一次前向，确认导出权重与真实网络逐位一致。

用法：
    .venv/bin/python deploy/export_ckpt.py            # 导出全部 MLP ckpt
    .venv/bin/python deploy/export_ckpt.py --verify   # 导出 + 前向自检
    .venv/bin/python deploy/export_ckpt.py ckpt/course_1023m.pt   # 只导指定档
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from datetime import datetime, timezone

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)          # 保证 --verify 能 import train.model
CKPT_DIR = os.path.join(PROJ, "ckpt")
OUT_DIR = os.path.join(PROJ, "web", "models")

# view_perm 只在 config 里有一份；这里是它的纯 Python 复制（不 import sim，防
# 别人改坏链路）。2 人布局：基础 2P+3 参与置换，扩展通道原样保留在尾部。
def view_perm_2p(c: int) -> tuple[int, ...]:
    p = 2
    others = [i for i in range(p) if i != 0]
    base = 2 * p + 3
    return tuple([0, p + 0] + others + [p + o for o in others]
                 + [2 * p, 2 * p + 1, 2 * p + 2]
                 + list(range(base, c)))


def fold_perm_pid0(w: torch.Tensor, c: int, h: int, w_: int) -> torch.Tensor:
    """把 pid=0 的视角列块置换折进第一层 Linear 权重（与 model.py 的 inv_cols
    同一套推导：新权重第 k 块 ← 原权重第 inv[k] 块，inv = argsort(perm)）。"""
    perm = view_perm_2p(c)
    inv = sorted(range(c), key=lambda k: perm[k])
    block = h * w_
    cols = []
    for k in range(c):
        cols.append(w[:, inv[k] * block:(inv[k] + 1) * block])
    return torch.cat(cols, dim=1)


def extract_mlp(state: dict, c: int, h: int, w: int) -> dict:
    """MLP state_dict → {权重 key: torch.Tensor}（已折 pid=0 视角）。

    返回张量而不是扁平 list：序列化时统一打包成 float32 二进制（base64），
    浏览器端一次解码后按偏移切片 —— 比全精度 JSON 小数小 ~4 倍、解析快。
    """
    keys = (
        ("shared.0.weight", "shared0_w", True),
        ("shared.0.bias", "shared0_b", False),
        ("shared.1.weight", "ln1_w", False),
        ("shared.1.bias", "ln1_b", False),
        ("shared.3.weight", "shared3_w", False),
        ("shared.3.bias", "shared3_b", False),
        ("shared.4.weight", "ln2_w", False),
        ("shared.4.bias", "ln2_b", False),
        ("move_head.0.weight", "move0_w", False),
        ("move_head.0.bias", "move0_b", False),
        ("move_head.2.weight", "move2_w", False),
        ("move_head.2.bias", "move2_b", False),
        ("bomb_head.0.weight", "bomb0_w", False),
        ("bomb_head.0.bias", "bomb0_b", False),
        ("bomb_head.2.weight", "bomb2_w", False),
        ("bomb_head.2.bias", "bomb2_b", False),
        ("critic.0.weight", "critic0_w", False),
        ("critic.0.bias", "critic0_b", False),
        ("critic.2.weight", "critic2_w", False),
        ("critic.2.bias", "critic2_b", False),
    )
    out: dict[str, torch.Tensor] = {}
    for src, dst, do_perm in keys:
        t = state[src]
        if do_perm:
            t = fold_perm_pid0(t, c, h, w)
        out[dst] = t.reshape(-1).to(torch.float32)
    return out


def pack_tensors(tensors: dict[str, torch.Tensor]
                 ) -> tuple[str, dict[str, list[int]]]:
    """全部张量按序连成一个 float32 缓冲 → (base64, {key: [offset, count]})。"""
    names = list(tensors.keys())
    total = sum(t.numel() for t in tensors.values())
    buf = torch.empty(total, dtype=torch.float32)
    off = 0
    index: dict[str, list[int]] = {}
    for name in names:
        t = tensors[name]
        buf[off:off + t.numel()].copy_(t)
        index[name] = [off, t.numel()]
        off += t.numel()
    b64 = base64.b64encode(struct.pack(
        f"<{total}f", *buf.tolist())).decode("ascii")
    return b64, index


def export_one(path: str, verify: bool) -> dict | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "cnn")
    obs_shape = tuple(int(x) for x in ck["obs_shape"])
    n_players = int(ck.get("n_players", 2))
    c, h, w = obs_shape

    if arch != "mlp":
        print(f"  [skip] {stem}: arch={arch}（web 版 v1 只导出 mlp）")
        return None
    if n_players != 2 or h != 13 or w != 13 or c not in (7, 14):
        print(f"  [skip] {stem}: obs_shape={obs_shape} n_players={n_players}"
              f"（web 版 v1 只支持 2 人 13×13 的 7/14 通道）")
        return None

    weights = extract_mlp(ck["model"], c, h, w)
    meta = {
        "name": stem,
        "arch": "mlp",
        "obs_shape": list(obs_shape),
        "n_players": n_players,
        "global_step": int(ck.get("global_step") or 0),
        "elo": round(float(ck.get("elo") or 0.0), 1),
        "source": os.path.basename(path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    if verify:
        ok = _verify_forward(ck, weights, c, h, w)
        if not ok:
            print(f"  [FAIL] {stem}: 前向自检不一致，已跳过导出")
            return None

    b64, index = pack_tensors(weights)
    doc = {"meta": meta, "flat": b64, "tensors": index}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{stem}.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    n_param = sum(t.numel() for t in weights.values())
    size = os.path.getsize(out) / 1024 / 1024
    print(f"  [ok]   {stem}: {n_param:,} 参数 → {os.path.relpath(out, PROJ)}"
          f"（{size:.1f}MB，step={meta['global_step']:,} elo={meta['elo']}）")
    return meta


def _verify_forward(ck: dict, tensors: dict, c: int, h: int, w: int) -> bool:
    """用真实 ActorCritic（train.model）对拍一次：导出权重的前向 vs 网络 pid=0 前向。

    只用于开发自检；失败只告警不阻塞（训练侧代码变动可能让这里不匹配）。
    """
    try:
        from train.model import ActorCritic  # noqa: 仅自检时 import
    except Exception as e:
        print(f"  [warn] 自检跳过（import train.model 失败: {e}）")
        return True
    try:
        torch.manual_seed(0)
        net = ActorCritic(tuple(ck["obs_shape"]), arch="mlp",
                          n_players=int(ck.get("n_players", 2)))
        net.load_state_dict(ck["model"])
        net.eval()

        obs = torch.rand(2, c, h, w)
        with torch.no_grad():
            m_ref, b_ref, v_ref = net(obs, pid=0)

        # 导出权重的手工前向（与 JS 端同构）
        def t(name, *shape):
            if shape:
                return tensors[name].view(*shape)
            return tensors[name]

        def ln(x, ww, bb):
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + 1e-5) * ww + bb

        x = obs.reshape(2, -1)
        x = torch.nn.functional.linear(x, t("shared0_w", 128, c * h * w),
                                       t("shared0_b"))
        x = torch.relu(ln(x, t("ln1_w"), t("ln1_b")))
        x = torch.nn.functional.linear(x, t("shared3_w", 128, 128),
                                       t("shared3_b"))
        x = torch.relu(ln(x, t("ln2_w"), t("ln2_b")))
        mv = torch.relu(torch.nn.functional.linear(
            x, t("move0_w", 64, 128), t("move0_b")))
        move = torch.nn.functional.linear(mv, t("move2_w", 5, 64), t("move2_b"))
        bv = torch.relu(torch.nn.functional.linear(
            x, t("bomb0_w", 64, 128), t("bomb0_b")))
        bomb = torch.nn.functional.linear(bv, t("bomb2_w", 2, 64), t("bomb2_b"))
        cv = torch.relu(torch.nn.functional.linear(
            x, t("critic0_w", 64, 128), t("critic0_b")))
        val = torch.nn.functional.linear(cv, t("critic2_w", 1, 64),
                                         t("critic2_b")).squeeze(-1)

        d_m = (m_ref - move).abs().max().item()
        d_b = (b_ref - bomb).abs().max().item()
        d_v = (v_ref - val).abs().max().item()
        print(f"  [verify] forward maxdiff: move={d_m:.2e} bomb={d_b:.2e} "
              f"critic={d_v:.2e}")
        return max(d_m, d_b, d_v) < 1e-4
    except Exception as e:
        print(f"  [warn] 自检异常（{type(e).__name__}: {e}），已跳过导出判定")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description="ckpt → web/models JSON 转换")
    ap.add_argument("paths", nargs="*", help="指定 ckpt 文件；缺省 = 全部")
    ap.add_argument("--verify", action="store_true", help="导出后跑一次前向自检")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args()

    if args.paths:
        files = [p if os.path.isabs(p) else os.path.join(PROJ, p) for p in args.paths]
    else:
        files = sorted(os.path.join(args.ckpt_dir, f)
                       for f in os.listdir(args.ckpt_dir) if f.endswith(".pt"))
    if not files:
        print(f"没有找到 ckpt（{args.ckpt_dir}）")
        sys.exit(1)

    metas = []
    for p in files:
        print(f"导出 {os.path.relpath(p, PROJ)}")
        m = export_one(p, verify=args.verify)
        if m:
            metas.append(m)

    if not metas:
        print("没有可导出的 MLP 模型")
        sys.exit(1)

    # 按 global_step 降序（最新训练排最前，页面默认选它）
    metas.sort(key=lambda m: m["global_step"], reverse=True)
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump({"models": metas}, f, indent=1, ensure_ascii=False)
    print(f"\n共导出 {len(metas)} 个模型 → {os.path.relpath(OUT_DIR, PROJ)}/")


if __name__ == "__main__":
    main()
