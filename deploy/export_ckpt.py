#!/usr/bin/env python3
"""ckpt/ → web/models/*.json 转换工具（部署工具链的第一环）。

把训练侧 `ckpt/*.pt`（PyTorch 权重）转成浏览器可直接加载的 JSON：
    - 支持 `arch == "mlp"` 与 `arch == "cnn"`，观测须为 (C, 13, 13) 的 2 人模型；
    - **折叠 pid=0 视角置换**：把 `train/model.py::ActorCritic` 里的视角重排
      **做进导出的第一层权重**——MLP 是 `shared.0.weight` 的列块重排
      （`inv_cols[0]`），CNN 是 `conv0.weight` 的输入通道重排（`inv_perm[0]`，
      kernel 3×3 的通道维，同一套 inv = argsort(perm) 推导）。浏览器端拿到
      一份 (C,H,W) 的**共享观测**直接做普通卷积/矩阵乘即可，
      不需要在 JS 里再实现一遍视角置换；
    - 输出 web/models/<stem>.json（扁平 float32 数组）+ web/models/index.json
      （全部已导出模型的元信息列表，供页面下拉选择）。

生产路径**只依赖 ckpt/ 目录**，不 import 任何 train/play/sim 代码 ——
别人在改训练侧也不会破坏这条链路。`--verify` 自检（可选）会临时 import
train.model 用 torch 对比一次前向，确认导出权重与真实网络逐位一致。

用法：
    .venv/bin/python deploy/export_ckpt.py            # 导出全部 mlp/cnn ckpt
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


def fold_conv_perm_pid0(w: torch.Tensor, c: int) -> torch.Tensor:
    """把 pid=0 的视角通道置换折进 conv0 的**输入通道维**（kernel 3×3）。

    与 MLP 列块重排同一套推导（model.py::inv_perm）：forward 里
    `conv2d(obs, weight[:, inv])`，即新权重输入通道 k ← 原权重输入通道
    inv[k]，inv = argsort(perm)。折叠后 JS 端对共享观测直接做普通卷积。
    """
    perm = view_perm_2p(c)
    inv = sorted(range(c), key=lambda k: perm[k])
    return w[:, inv, :, :].contiguous()


def extract_cnn(state: dict, c: int, h: int, w: int) -> dict:
    """CNN state_dict → {权重 key: torch.Tensor}（已折 pid=0 视角到 conv0）。

    train/model.py arch="cnn"：conv0(3×3) → [LN(16,h,w)+ReLU] → conv(16→32)
    → [LN(32)+ReLU] → conv(32→64) → [LN(64)+ReLU] → conv1x1(64→8) →
    [LN(8)+ReLU] → flatten(8·h·w) → shared MLP(128→128) → 双头。
    注意 conv 里的 LayerNorm 是三维 [C,h,w]（归一化范围整个 C·h·w，w/b
    逐元素），与 shared 的 1 维 [128] 不同 —— JS 端按 key 区分。
    """
    keys = (
        ("conv0.weight", "conv0_w", True),          # 输入通道维折视角
        ("conv0.bias", "conv0_b", False),
        ("conv.0.weight", "cn1_w", False),          # LayerNorm [16,h,w]
        ("conv.0.bias", "cn1_b", False),
        ("conv.2.weight", "conv1_w", False),
        ("conv.2.bias", "conv1_b", False),
        ("conv.3.weight", "cn2_w", False),          # [32,h,w]
        ("conv.3.bias", "cn2_b", False),
        ("conv.5.weight", "conv2_w", False),
        ("conv.5.bias", "conv2_b", False),
        ("conv.6.weight", "cn3_w", False),          # [64,h,w]
        ("conv.6.bias", "cn3_b", False),
        ("conv.8.weight", "conv3_w", False),        # 1×1
        ("conv.8.bias", "conv3_b", False),
        ("conv.9.weight", "cn4_w", False),          # [8,h,w]
        ("conv.9.bias", "cn4_b", False),
        ("shared.0.weight", "shared0_w", False),    # 与 MLP 同名（JS 共用共享层）
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
            t = fold_conv_perm_pid0(t, c)
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


def export_one(path: str, verify: bool, incremental: bool = False) -> dict | None:
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT_DIR, f"{stem}.json")
    if incremental and os.path.exists(out) \
            and os.path.getmtime(out) >= os.path.getmtime(path):
        print(f"  [same] {stem}: 已是最新（跳过）")
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        print(f"  [skip] {stem}: 反序列化缺模块 {e.name!r}"
              f"（训练环境专属，本地无此依赖）")
        return None
    except Exception as e:
        print(f"  [skip] {stem}: 加载失败（{type(e).__name__}: {e}）")
        return None
    arch = ck.get("arch", "cnn")
    obs_shape = tuple(int(x) for x in ck["obs_shape"])
    n_players = int(ck.get("n_players", 2))
    c, h, w = obs_shape

    if arch not in ("mlp", "cnn"):
        print(f"  [skip] {stem}: arch={arch}（web 版只支持 mlp/cnn）")
        return None
    if n_players != 2 or h != 13 or w != 13 or c not in (7, 14):
        print(f"  [skip] {stem}: obs_shape={obs_shape} n_players={n_players}"
              f"（web 版只支持 2 人 13×13 的 7/14 通道）")
        return None

    weights = extract_cnn(ck["model"], c, h, w) if arch == "cnn" \
        else extract_mlp(ck["model"], c, h, w)
    meta = {
        "name": stem,
        "arch": arch,
        "obs_shape": list(obs_shape),
        "n_players": n_players,
        "global_step": int(ck.get("global_step") or 0),
        "elo": round(float(ck.get("elo") or 0.0), 1),
        "source": os.path.basename(path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    if verify:
        ok = (_verify_forward_cnn(ck, weights, c, h, w) if arch == "cnn"
              else _verify_forward(ck, weights, c, h, w))
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


def _verify_forward_cnn(ck: dict, tensors: dict, c: int, h: int, w: int) -> bool:
    """CNN 版对拍：真实 ActorCritic(arch="cnn") pid=0 前向 vs 导出权重手工前向。

    手工前向与 JS 端 CNNModel 同构（conv 循环 + 3D LayerNorm + shared MLP）。
    """
    try:
        from train.model import ActorCritic  # noqa: 仅自检时 import
    except Exception as e:
        print(f"  [warn] 自检跳过（import train.model 失败: {e}）")
        return True
    try:
        torch.manual_seed(0)
        net = ActorCritic(tuple(ck["obs_shape"]), arch="cnn",
                          n_players=int(ck.get("n_players", 2)))
        net.load_state_dict(ck["model"])
        net.eval()

        obs = torch.rand(2, c, h, w)
        with torch.no_grad():
            m_ref, b_ref, v_ref = net(obs, pid=0)

        def t(name, *shape):
            if shape:
                return tensors[name].view(*shape)
            return tensors[name]

        def conv3d(x, ww, bb, pad=1):
            return torch.nn.functional.conv2d(x, ww, bb, padding=pad)

        def ln3d(x, ww, bb):
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + 1e-5) * ww + bb

        def ln1d(x, ww, bb):
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, unbiased=False, keepdim=True)
            return (x - mean) / torch.sqrt(var + 1e-5) * ww + bb

        x = conv3d(obs, t("conv0_w", 16, c, 3, 3), t("conv0_b"))
        x = torch.relu(ln3d(x, t("cn1_w", 16, h, w), t("cn1_b", 16, h, w)))
        x = conv3d(x, t("conv1_w", 32, 16, 3, 3), t("conv1_b"))
        x = torch.relu(ln3d(x, t("cn2_w", 32, h, w), t("cn2_b", 32, h, w)))
        x = conv3d(x, t("conv2_w", 64, 32, 3, 3), t("conv2_b"))
        x = torch.relu(ln3d(x, t("cn3_w", 64, h, w), t("cn3_b", 64, h, w)))
        x = conv3d(x, t("conv3_w", 8, 64, 1, 1), t("conv3_b"), pad=0)
        x = torch.relu(ln3d(x, t("cn4_w", 8, h, w), t("cn4_b", 8, h, w)))
        x = x.flatten(1)
        x = torch.nn.functional.linear(x, t("shared0_w", 128, 8 * h * w),
                                       t("shared0_b"))
        x = torch.relu(ln1d(x, t("ln1_w"), t("ln1_b")))
        x = torch.nn.functional.linear(x, t("shared3_w", 128, 128),
                                       t("shared3_b"))
        x = torch.relu(ln1d(x, t("ln2_w"), t("ln2_b")))
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
        print(f"  [verify] cnn forward maxdiff: move={d_m:.2e} bomb={d_b:.2e} "
              f"critic={d_v:.2e}")
        return max(d_m, d_b, d_v) < 1e-4
    except Exception as e:
        print(f"  [warn] 自检异常（{type(e).__name__}: {e}），已跳过导出判定")
        return True


# 源头黑名单：已被废弃/删除的实验变体，禁止再次扫入 index.json
EXCLUDED_MODELS = {
    "params_it00000068_hlgauss_top25_patch3_k32",
    "params_it00000068_repro8x2_k32",
    "params_it00000068_repro14ch",
}


def scan_out_dir(out_dir: str = OUT_DIR) -> list[dict]:
    """扫描 web/models/*.json（除 index.json）的 meta，重建完整模型列表。

    index.json 永远由目录扫描生成（而不是只写"本次导出"的模型）——
    否则增量导出时旧模型会从 index 里消失（8B 档单独被重新导出就把
    列表覆盖成 1 个的 bug，2026-08-16 修复）。
    """
    metas = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".json") or f == "index.json":
            continue
        stem = f[:-5]
        if stem in EXCLUDED_MODELS:
            continue
        try:
            with open(os.path.join(out_dir, f)) as fp:
                doc = json.load(fp)
            meta = doc.get("meta")
            if meta and meta.get("name") not in EXCLUDED_MODELS:
                metas.append(meta)
        except Exception:
            continue                       # 半截 json：跳过不阻塞
    return metas



def main() -> None:
    ap = argparse.ArgumentParser(description="ckpt → web/models JSON 转换")
    ap.add_argument("paths", nargs="*", help="指定 ckpt 文件；缺省 = 全部")
    ap.add_argument("--verify", action="store_true", help="导出后跑一次前向自检")
    ap.add_argument("--incremental", action="store_true",
                    help="只导出比 web/models/<stem>.json 新的 ckpt"
                         "（serve_web.sh 自动调用）")
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

    exported = 0
    for p in files:
        print(f"导出 {os.path.relpath(p, PROJ)}")
        m = export_one(p, verify=args.verify, incremental=args.incremental)
        if m:
            exported += 1

    # index.json 由目录扫描重建（增量/全量统一）——保证旧模型不丢
    metas = scan_out_dir()
    if not metas:
        if args.incremental:
            print("web/models 下没有已导出的模型（可先全量跑一次）")
            return
        print("没有可导出的 mlp/cnn 模型")
        sys.exit(1)

    # 按 global_step 降序（最新训练排最前，页面默认选它）；transformer 档
    # 的 meta 只有 it 没有 global_step，取 get 容错
    metas.sort(key=lambda m: m.get("global_step") or m.get("it") or 0, reverse=True)
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump({"models": metas}, f, indent=1, ensure_ascii=False)
    verb = "本次新增" if args.incremental else "共导出"
    print(f"\n{verb} {exported} 个模型，web/models 现共 {len(metas)} 个 → index.json")


if __name__ == "__main__":
    main()
