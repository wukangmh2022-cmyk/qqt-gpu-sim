"""火焰射线传播 —— 参考实现（batch 向量化，device-agnostic）。

这是整个 simulator 里唯一"有依赖链"的部分：连锁爆炸本质是网格上的波传导。
CPU 上常用队列递归，但那在 GPU 上会退化成串行。这里统一改成
**固定轮数的同步迭代**：每一轮只读上一轮结果、只写本轮结果，
写冲突被彻底消除，CUDA 侧用完全相同的迭代结构（见 bomber_kernels.cu）。
"""

from __future__ import annotations

import torch


def _shift(x: torch.Tensor, drow: int, dcol: int, fill: int = 0) -> torch.Tensor:
    """把 x 的内容整体朝 (drow, dcol) 方向挪一格，越界处补 fill。

    result[i, j] = x[i - drow, j - dcol]
    `fill` 让 owner 类整数图（-1 表示无归属）也能安全移位，不被 0 污染。
    """
    h, w = x.shape[-2], x.shape[-1]
    r_src = slice(max(0, -drow), h - max(0, drow))
    c_src = slice(max(0, -dcol), w - max(0, dcol))
    # F.pad：一次 kernel 完成"切源区 + 目标侧补 fill"，替代 full_like+copy 两个
    # kernel（DCU 上小 kernel 的 launch 开销是大头，rays/danger 每 tick 调几十次）。
    top = max(0, drow)
    left = max(0, dcol)
    bottom = max(0, -drow)
    right = max(0, -dcol)
    return torch.nn.functional.pad(
        x[..., r_src, c_src], (left, right, top, bottom), value=fill)


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def rays(
    sources: torch.Tensor,
    wall: torch.Tensor,
    bombed: torch.Tensor,
    blast: int | torch.Tensor,
    brick: torch.Tensor | None = None,
) -> torch.Tensor:
    """从 sources 出发的十字火焰覆盖范围。

    sources / wall / bombed / brick: (..., H, W) bool。墙体挡火且自身不被覆盖。
    **泡泡挡火**：火焰到达泡泡所在的格会覆盖它（把它点燃），但不再穿过它
    继续延伸 —— 这是炸弹人的经典规则（放泡可以当屏障）。连锁爆炸不靠穿透
    实现：被点燃的泡泡在 resolve_explosions 里成为新的爆源，重新向外扩散。

    `blast` 是 int（全图同威力）或 (..., H, W) int 张量（每颗泡自己的威力，
    成长系统的等级不同）。`brick` 是**可炸墙**：挡火（火焰不穿过），但被
    覆盖（covered 含 brick 格）—— 调用方据此把被烧到的 brick 摧毁。

    **无 host 同步**：不用 bool(any())/max() 早退（DCU 上每次 device→host
    同步 ~ms 级，每 tick 几十次直接卡死训练）。固定轮数多算空轮无妨，
    结果与旧版（有早退）逐位一致。
    """
    if isinstance(blast, int):
        blast_cell = torch.full_like(sources, blast, dtype=torch.int32)
        b_max = blast                       # int：Python 循环无需同步
    else:
        blast_cell = blast
        b_max = int(blast_cell.max())       # 一次同步；值域 ≤ growth_blast_max（默认 7）
    # 预计算（循环外一次）：永久墙不可覆盖、brick 可被覆盖但挡火。
    # ~wall / ~solid 提前算好，循环里只剩 & （每 tick 的 rays/danger 调几十次，
    # 循环内少两次 ~ 就是一个真实 kernel —— DCU 上小 kernel launch 是大头）。
    brick_t = brick if brick is not None else torch.zeros_like(sources)
    not_wall = ~wall
    solid = bombed | brick_t                    # 泡/brick 都吸收火焰
    not_solid = ~solid
    seed = sources & not_wall & ~brick_t
    covered = seed.clone()
    for b in range(1, b_max + 1):
        src = seed & (blast_cell == b)
        for drow, dcol in _DIRS:
            front = src
            for _ in range(b):
                front = _shift(front, drow, dcol) & not_wall   # 永久墙不可覆盖；brick 可
                covered = covered | front
                front = front & not_solid      # 泡/brick 挡火：覆盖它但不穿透
    return covered


def resolve_explosions(
    fuse: torch.Tensor,
    owner: torch.Tensor,
    wall: torch.Tensor,
    blast: int | torch.Tensor,
    max_chain: int,
    brick: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (火焰覆盖 mask, 本 tick 被引爆的泡泡 mask)。

    fuse/owner/wall/brick: (N, H, W)。引信已经在本 tick 递减过，fuse == 0 且
    owner >= 0 的格子是爆源。brick 被覆盖后由调用方摧毁（self.brick &= ~covered）。

    宝箱机制下成长不按"谁炸的"分配（拾取制），所以不再需要归属图——
    见 torch_sim 的 crate（砖炸掉变宝箱，走到才开）。
    """
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    covered = rays(triggered, wall, live, blast, brick)

    # 固定 max_chain 轮（**无 early-exit/同步**，CUDA graph 兼容）：
    # 连锁结束（newly 空）后多算的轮只产生空覆盖，结果与早退版逐位一致。
    for _ in range(max_chain - 1):
        newly = live & covered & ~triggered
        covered = covered | rays(newly, wall, live, blast, brick)
        triggered = triggered | newly

    return covered, triggered


def danger_map(
    fuse: torch.Tensor,
    wall: torch.Tensor,
    blast: int | torch.Tensor,
    fuse_max: int,
    brick: torch.Tensor | None = None,
) -> torch.Tensor:
    """在场所有泡泡的"时空影响范围"，越接近爆炸值越大，落在 (0, 1]。

    这就是聊天里说的"放泡瞬间就把倒计时锥体画进矩阵"：网络直接读这张图，
    不需要自己从泡泡坐标反推威胁方向。多个泡泡覆盖同一格时取最大值。
    **泡泡挡火**：射线遇泡停止（与 rays 同规则），泡自身格由自己的引信
    权重给出危险值；被挡的泡是独立爆源，它自己的 seed 已覆盖自己的影响范围。
    `brick` 挡火传播（不穿过）；其格自身危险 0（玩家不可站立，无意义）。
    """
    weight = torch.where(
        fuse > 0,
        1.0 - (fuse.float() - 1.0) / float(fuse_max),
        torch.zeros_like(fuse, dtype=torch.float32),
    )
    bombed = fuse > 0
    brick_t = brick if brick is not None \
        else torch.zeros_like(bombed, dtype=torch.bool)
    solid = bombed | brick_t
    not_solid = (~solid).float()          # 预计算：循环里只剩乘法（少一个 ~ kernel）
    passable = (~wall).float()
    seed = weight * passable
    danger = seed.clone()
    if isinstance(blast, int):
        for drow, dcol in _DIRS:
            front = seed
            for _ in range(blast):
                front = _shift(front, drow, dcol) * passable
                danger = torch.maximum(danger, front)
                front = front * not_solid   # 泡/brick 挡火：不穿过
    else:
        # 一次同步取最大威力；值域 ≤ growth_blast_max（默认 7）
        max_b = int(blast.max()) if blast.numel() else 0
        for b in range(1, max_b + 1):
            src = seed * (blast == b)
            for drow, dcol in _DIRS:
                front = src
                for _ in range(b):
                    front = _shift(front, drow, dcol) * passable
                    danger = torch.maximum(danger, front)
                    front = front * not_solid
    return danger
