"""CUDA backend 与参考实现的逐 tick 一致性测试。

没有 CUDA 设备时整个模块 skip —— 这不是"测试通过"，而是"没测"。
本机（darwin/arm64）就属于后者。

对齐策略：两侧用同一个 seed 各自 reset，再把参考实现的状态**灌进** CUDA
backend，从而排除地图生成差异；之后每 tick 用同一份随机动作推进，
比较全部状态张量 + reward + done + obs + mask。
"""

from __future__ import annotations

import pytest
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="需要 CUDA 设备才能编译并运行 kernel"
)


def _sample_actions(mm: torch.Tensor, bm: torch.Tensor,
                    gen: torch.Generator) -> torch.Tensor:
    n, p, _ = mm.shape
    mv = torch.multinomial(mm.float().view(-1, mm.shape[-1]), 1, generator=gen)
    bo = torch.multinomial(bm.float().view(-1, bm.shape[-1]), 1, generator=gen)
    return torch.stack([mv.view(n, p), bo.view(n, p)], dim=-1)


def _assert_state_equal(ref: dict, cu: dict, tick: int) -> None:
    for key in ("wall", "fuse", "owner", "alive", "hp", "since_bomb", "t"):
        a, b = ref[key], cu[key].to(ref[key].device)
        assert torch.equal(a.to(b.dtype), b), f"tick {tick}: 状态 {key} 不一致"
    # 位置是 float32，CUDA 侧的 fma 与 torch 的乘加顺序可能差最后一位
    assert torch.allclose(ref["pos"], cu["pos"].to(ref["pos"].device), atol=1e-5), \
        f"tick {tick}: 位置不一致"


@cuda_only
@pytest.mark.parametrize(
    "cfg",
    [
        SimConfig(height=9, width=9, n_players=2),
        SimConfig(height=11, width=11, n_players=2, wall_density=0.7,
                  obs_fp16=False),  # 同时覆盖 CUDA observe 的 fp32 写出分支
        SimConfig(height=11, width=11, n_players=3, blast=3, max_bombs=1,
                  max_hp=2),  # 多角色扣血：血没归 0 不死的路径也要逐位一致
        SimConfig(height=13, width=13, n_players=4, fuse=6, max_chain=4),
        # 胖碰撞盒 + 慢速：radius=0.4 时单 tick 位移必须 < 1-2r = 0.2
        # （10Hz 下 step_len=0.15 满足约束；15Hz 时可以用 speed=2.0）
        SimConfig(height=11, width=11, n_players=2, speed=1.5, radius=0.4),
    ],
)
def test_parity_tick_by_tick(cfg: SimConfig):
    from sim.cuda.wrapper import CudaSim

    num_envs = 64
    ref = BatchedSim(cfg, num_envs, device="cuda", seed=123)
    cu = CudaSim(cfg, num_envs, seed=123)
    cu.load_state_dict(ref.state_dict())          # 强制同一初始局面

    gen = torch.Generator(device="cuda").manual_seed(7)
    for tick in range(300):
        mm_ref, bm_ref = ref.legal_mask()
        mm_cu, bm_cu = cu.legal_mask()
        assert torch.equal(mm_ref, mm_cu), f"tick {tick}: 方向掩码不一致"
        assert torch.equal(bm_ref, bm_cu), f"tick {tick}: 放泡掩码不一致"
        o_ref, o_cu = ref.observe(), cu.observe()
        assert o_ref.shape == o_cu.shape and o_ref.dtype == o_cu.dtype
        # 观测默认存 fp16：中间量两边都是 fp32，但 fp32 差 1 ulp 就可能让
        # fp16 舍到相邻档位，所以这里的容差按 fp16 的分辨率给（约 1e-3），
        # 逐位一致的要求留给 state_dict 里的整数状态和 float 坐标。
        atol = 2e-3 if o_ref.dtype == torch.float16 else 1e-5
        assert torch.allclose(o_ref.float(), o_cu.float(), atol=atol), \
            f"tick {tick}: 观测不一致"

        acts = _sample_actions(mm_ref, bm_ref, gen)
        r_ref, d_ref, i_ref = ref.step(acts)
        r_cu, d_cu, i_cu = cu.step(acts)
        assert torch.allclose(r_ref, r_cu, atol=1e-6), f"tick {tick}: reward 不一致"
        assert torch.equal(d_ref, d_cu), f"tick {tick}: done 不一致"
        assert torch.equal(i_ref["winner"], i_cu["winner"]), \
            f"tick {tick}: winner 不一致"
        # auto_reset 会各自生成新地图，所以终局那一 tick 之后重新对齐一次
        if bool(d_ref.any()):
            cu.load_state_dict(ref.state_dict())
        else:
            _assert_state_equal(ref.state_dict(), cu.state_dict(), tick)


@cuda_only
def test_parity_handcrafted_chain():
    """手工摆一条连锁引线，确认 CUDA 的定轮同步迭代和参考实现同结果。"""
    from sim.cuda.wrapper import CudaSim

    cfg = SimConfig(height=11, width=11, n_players=2, blast=2, max_chain=8)
    ref = BatchedSim(cfg, 1, device="cuda", seed=0)
    cu = CudaSim(cfg, 1, seed=0)
    ref.wall.zero_()
    ref.fuse.zero_()
    ref.owner.fill_(-1)
    ref.pos[0, 0] = torch.tensor([5.5, 0.5], device="cuda")
    ref.pos[0, 1] = torch.tensor([10.5, 10.5], device="cuda")
    for col, f in ((2, 1), (4, 9), (6, 9), (8, 9)):
        ref.fuse[0, 5, col] = f
        ref.owner[0, 5, col] = 1
    cu.load_state_dict(ref.state_dict())

    acts = torch.zeros((1, 2, 2), dtype=torch.long, device="cuda")
    r_ref, d_ref, _ = ref.step(acts, auto_reset=False)
    r_cu, d_cu, _ = cu.step(acts, auto_reset=False)
    assert torch.allclose(r_ref, r_cu)
    assert torch.equal(d_ref, d_cu)
    _assert_state_equal(ref.state_dict(), cu.state_dict(), 0)
    assert int(ref.fuse.sum()) == 0, "四颗泡应被一次连锁全部引爆"


@cuda_only
def test_parity_after_shared_reset():
    """不灌状态、只共享 seed：确认 host 侧地图生成在两个 backend 上一致。"""
    from sim.cuda.wrapper import CudaSim

    cfg = SimConfig(height=11, width=11, n_players=2, wall_density=0.6)
    ref = BatchedSim(cfg, 32, device="cuda", seed=999)
    cu = CudaSim(cfg, 32, seed=999)
    _assert_state_equal(ref.state_dict(), cu.state_dict(), -1)
