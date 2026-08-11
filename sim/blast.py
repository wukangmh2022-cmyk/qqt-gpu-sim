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

# 早退检查降频：bool(any()) 是 GPU→CPU 同步点（DCU 上每次 ~ms）。
# 连锁/危险传播通常在头几轮结束，多算 1 轮空轮对结果逐位无影响。
# 每 2 轮查一次 → 同步频率减半（风险：连锁恰好在非检查轮结束时多算
# 1 轮空轮，代价是一次空 rays/扩散，可忽略）。
CHECK_EVERY = 2
# danger 阶段 A 波前传播：每 CHECK_EVERY 轮才查一次 newly（同上）
DANGER_CHECK_EVERY = 2


def rays(
    sources: torch.Tensor,
    wall: torch.Tensor,
    bombed: torch.Tensor,
    blast: int | torch.Tensor,
    brick: torch.Tensor | None = None,
    blast_max_hint: int | None = None,
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

    **注意**：不要用 stack(4 方向)+amax 合并 —— 实测本机吞吐 -43%
    （CPU 上每算子有固定 dispatch 开销，合并把 4×pad 换成
    stack×2+pad+amax 反而多算子）。保持每方向独立 _shift 最快。
    """
    if isinstance(blast, int):
        blast_cell = torch.full_like(sources, blast, dtype=torch.int32)
        b_max = blast                       # int：Python 循环无需同步
    else:
        blast_cell = blast
        # 动态取实际档位（值域 ≤ growth_blast_max）：固定上限会让每档空轮
        # 的 pad launch 全跑（实测 max 固定 7 比动态 1-2 慢 3 倍，见 2026-08-10），
        # 一次 max() 同步比多算档位的 pad 空轮便宜。blast_max_hint 仅保留
        # 给确实已知档位恒满的调用方（当前训练端不用）。
        b_max = (blast_max_hint if blast_max_hint is not None
                 else (int(blast_cell.max()) if blast_cell.numel() else 0))
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
    early_exit: bool | None = None,
    blast_max_hint: int | None = None,
    chain_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (火焰覆盖 mask, 本 tick 被引爆的泡泡 mask)。

    fuse/owner/wall/brick: (N, H, W)。引信已经在本 tick 递减过，fuse == 0 且
    owner >= 0 的格子是爆源。brick 被覆盖后由调用方摧毁（self.brick &= ~covered）。

    宝箱机制下成长不按"谁炸的"分配（拾取制），所以不再需要归属图——
    见 torch_sim 的 crate（砖炸掉变宝箱，走到才开）。

    `early_exit`：None = 自动（**训练/推理全设备早退**，省空轮 kernel）；
    True 强制、False 强制关（CUDA graph 捕获段必须 False —— 回放要求
    固定 kernel 序列，早退的 host break 会把轮数钉死在捕获值）。连锁
    结束（newly 空）后多算的轮只产生空覆盖，早退结果与固定轮逐位一致。

    `chain_cap`（同步免除）：非 None 时连锁固定跑 min(max_chain-1, chain_cap)
    轮、无 newly 轮检查同步、并跳过无爆炸守卫 —— 结果对链长 ≤ cap 逐位一致。
    910B 训练分布实测爆炸链深 max≤4（200 tick 抽样），cap=4 覆盖全部分布。
    """
    should_ee = True if early_exit is None else early_exit
    sync_free = chain_cap is not None
    triggered = (fuse == 0) & (owner >= 0)
    live = fuse > 0
    # **无爆炸短路**：本 tick 没有引信走完的泡（大多数 tick 只是倒计时）时
    # 直接返回空覆盖 —— rays 无源也照跑 4×Σb 档 shift（blast 分散时最多
    # ~112 pad/调用），短路用 1 次 sync 换掉 70-80% tick 的整段 rays。
    # 爆炸 tick 占比 ~20-30%（放泡 30 tick 倒计时才爆），净赚。
    if should_ee and not sync_free and not bool(triggered.any()):
        return torch.zeros_like(fuse, dtype=torch.bool), triggered
    covered = rays(triggered, wall, live, blast, brick,
                   blast_max_hint=blast_max_hint)

    # 早退检查每 CHECK_EVERY 轮才做一次（bool(any()) 是 GPU→CPU 同步点；
    # 连锁通常在头几轮就结束，多算 1 轮空轮对结果逐位无影响）。
    # 同步免除模式（chain_cap≠None）：固定轮无检查（结果对链长 ≤cap 逐位一致）。
    rounds = min(max_chain - 1, chain_cap) if sync_free else max_chain - 1
    for i in range(rounds):
        newly = live & covered & ~triggered
        if should_ee and not sync_free and i % CHECK_EVERY == 0 \
                and not bool((newly).any()):
            break
        covered = covered | rays(newly, wall, live, blast, brick,
                                 blast_max_hint=blast_max_hint)
        triggered = triggered | newly

    return covered, triggered


def danger_map(
    fuse: torch.Tensor,
    wall: torch.Tensor,
    blast: int | torch.Tensor,
    fuse_max: int,
    brick: torch.Tensor | None = None,
    max_chain: int = 1,
    exp: float = 2.0,
    early_exit: bool | None = None,
    blast_max_hint: int | None = None,
    chain_cap: int | None = None,
) -> torch.Tensor:
    """在场所有泡泡的"时空影响范围"，越接近爆炸值越大，落在 (0, 1]。

    这就是聊天里说的"放泡瞬间就把倒计时锥体画进矩阵"：网络直接读这张图，
    不需要自己从泡泡坐标反推威胁方向。多个泡泡覆盖同一格时取最大值。
    **泡泡挡火**：射线遇泡停止（与 rays 同规则），泡自身格由自己的引信
    权重给出危险值；被挡的泡是独立爆源，它自己的 seed 已覆盖自己的影响范围。
    `brick` 挡火传播（不穿过）；其格自身危险 0（玩家不可站立，无意义）。

    危险度对引信做**指数压缩**（`(1-(fuse-1)/FUSE)^exp`，exp 默认 2）：
    线性值把"还有 9 tick 才爆"的泡画得跟"3 tick 内要爆"的差不多深，显示
    上糊成一片；平方后危险感集中到真正要响的最后几 tick，刚放的泡几乎
    无色 —— 显示和训练共用这份输出，网络同样受益。

    **连锁时间修正（max_chain > 1）**：resolve_explosions 的连锁是同 tick
    同步传播（固定轮迭代）——互相接壤/被 blast 覆盖的炮组**同时爆炸**。
    但单颗炮自己的 fuse 是"先放的深、后放的浅"，跟实际爆炸时刻不符：
    10 颗完全连着的横向炮，先放的 fuse 小画得深红，最后放的 fuse 大画得
    几乎无色，可它们实际同 tick 一起爆。所以这里先做**炮格间的危险传播**
    （与 resolve 同规则、同 max_chain 轮数）：任何炮被更早爆炸的炮的 blast
    覆盖 → 它的爆炸时刻提前到组内最早 → 整组危险度统一取"组内最危险"。
    网络读到的危险度因此与引擎的真实连锁语义一致（能区分"会被连锁提前
    点燃的炮"和"独立晚爆的炮"），不再被单颗 fuse 骗。

    性能：阶段 A 只在炮格累积（非炮格权重恒 0），扩散直接从 `w` 出发，
    不额外乘 bombed 掩码；`spread` 缓冲循环内 `zero_()` 复用 —— 每 tick
    两次调用（obs 通道 + danger 惩罚）不能带多余 kernel。
    连锁用**波前接力**（每轮只从"本轮新激活的炮格"传播，老炮格不重复）：
    避免"轮数 × blast 格"的距离累加 —— blast=3 间隔 5 格、中间无炮的两颗
    炮在引擎里不会连锁（火焰被无炮格子吸收，波前停），全源逐轮传播会把
    它们误判成一组。空场（无泡）直接短路返回全 0 —— 训练里空场常见，
    固定 max_chain 轮在无泡时全是空转。

    `early_exit`：None = 自动（全设备早退，含 cuda 训练端）；True 强制；
    False 强制关（CUDA graph 捕获段必须 False，回放要求固定 kernel 序列）。
    早退只是纯省空轮，结果与固定轮逐位一致。

    `chain_cap`（同步免除模式）：非 None 时阶段 A **固定跑 min(max_chain,
    chain_cap) 轮、无 newly 轮检查同步**，并跳过空场守卫（bombed.any()）——
    从默认的 5 次 host 同步降到 1 次（档位 max 一次）。链长 ≤ chain_cap 时
    结果与动态早退**逐位一致**（多跑的空轮只产生零波前）；910B 训练分布
    实测链长 max≤4（resolve 200 tick 抽样），默认 cap=4 有 2 倍裕量。
    链长 > cap 的合成场景请保持 chain_cap=None（验证脚本用动态路径）。
    """
    should_ee = True if early_exit is None else early_exit
    sync_free = chain_cap is not None
    bombed = fuse > 0
    if should_ee and not sync_free and not bool(bombed.any()):
        return torch.zeros_like(fuse, dtype=torch.float32)   # 空场：无任何危险
    w_raw = 1.0 - (fuse.float() - 1.0) / float(fuse_max)
    weight = torch.where(fuse > 0, w_raw.clamp_min(0.0).pow(exp),
                         torch.zeros_like(fuse, dtype=torch.float32))
    brick_t = brick if brick is not None \
        else torch.zeros_like(bombed, dtype=torch.bool)
    solid = bombed | brick_t
    not_solid = (~solid).float()          # 预计算：循环里只剩乘法（少一个 ~ kernel）
    passable = (~wall).float()
    # **910B 修正（2026-08-11）**：torch_npu 上"张量±/×/÷ Python 标量"的 op
    # 每次 dispatch 内部 item() 同步（成本 = 队列等待，step 里 ~0.26ms/次，
    # 94/step 的 _local_scalar_dense 大头）。预分配全 1 张量作操作数（张量-张量
    # op 零同步），fd1-1.0 变 fd1-one_buf —— 位级一致（逐元素同算术），省 ~50 次
    # 同步。其他标量 op（>=0、where 0.0）实测不同步，保持标量。
    one_buf = torch.ones_like(passable)
    # 档位上限统一算一次（阶段 A/B 共用同一 blast_f —— 原来各算一次
    # int(max()) 是 2 次 host 同步；合并后结果不变，省 1 次同步）。
    blast_f = (torch.full_like(weight, float(blast)) if isinstance(blast, int)
               else blast.float())
    # 动态取实际档位（值域 ≤ growth_blast_max）：固定上限（hint）会让空档的
    # pad 全跑（实测 max 固定 7 比动态 1-2 慢 3 倍，见 2026-08-10）；一次
    # max() 同步比多算档位的 pad 空轮便宜。blast_max_hint 仅保留给确实已知
    # 档位恒满的调用方（graph 捕获段）。
    max_b = (blast_max_hint if blast_max_hint is not None
             else (int(blast_f.max()) if blast_f.numel() else 0))

    # 阶段 A：炮格间的连锁危险传播（只在炮格累积，非炮格恒 0）。
    # **双缓冲传播（v2，2026-08-10）**：波前权重 fw + 剩余距离 fd 一起挪格，
    # 每方向统一 max_b 步（原版按 blast 档分组 Σb 步）→ pad 数 ÷14；
    # 传播语义逐位一致（verify_danger_v2 25 tick 富泡场景 PASS）：
    #   - 每轮从 newly 激活炮格出发，传播自己的 blast 档距离（fd 递减）；
    #   - 先记录后挡（泡/brick 格被覆盖但 fw/fd 置 0 不穿透）；
    #   - **只有炮格接收权重**（spread × bombed）——非炮格权重混进 w
    #     会让阶段 B 从非炮格 seed 多扩散（v1 的 bug，已修）。
    if max_chain > 1:
        w = weight.clone()
        # （blast_f / max_b 已在上方统一算好，阶段 A/B 共用 —— 省 1 次 max 同步）
        front = torch.where(bombed, w, torch.zeros_like(w))   # 波前权重
        fdist = torch.where(bombed, blast_f, torch.zeros_like(w))  # 剩余距离
        spread = torch.zeros_like(weight)
        # **stack 融合（2026-08-11，910B dispatch-bound 优化）**：fw/fd 两个
        # 张量叠成 (2,n,h,w) 一次 _shift + 一次 passable 乘法（原来 2 pad + 2 mul）。
        # **注意**：not_solid 必须在 maximum() **之后**乘（先记录后挡穿透），
        # 不能并进 shift 的 gate —— 否则 solid 格的记录值被提前清零，语义不同。
        # 逐位一致（F.pad 按通道独立、元素级乘法可结合），verify_danger_v2 对拍。
        # 同步免除模式（chain_cap≠None）：固定轮无 newly 检查（结果对链长
        # ≤cap 逐位一致）；动态早退每 CHECK_EVERY 轮一次 bool(any()) 同步。
        rounds = min(max_chain, chain_cap) if sync_free else max_chain
        for i in range(rounds):
            spread.zero_()
            for drow, dcol in _DIRS:
                fw_p, fd_p = front, fdist
                for _ in range(max_b):
                    # **910B 修正（2026-08-11）**：双 pad 连续张量（v1 结构）——
                    # stack(2,n,h,w) 后 st[0]/st[1] 是**非连续视图**，标量 op
                    # （fd1-1.0 等）在视图上每次触发 torch_npu 内部 item() 同步
                    # （~0.26ms，_local_scalar_dense 大头）。连续张量不同步。
                    # 2×(n,h,w) pad 数据量 ≈ stack+1×(2,n,h,w) pad，还免 item。
                    fw1 = _shift(fw_p, drow, dcol) * passable
                    fd1 = _shift(fd_p, drow, dcol) * passable
                    fd1 = fd1 - one_buf          # 张量操作数免 item 同步
                    keep = fd1 >= 0          # 第 b 格（fd1=0）也记录；耗尽才停
                    fw1 = fw1 * keep          # bool×float32 提升：keep=False→0（省 where dispatch）
                    spread = torch.maximum(spread, fw1, out=spread)  # in-place 省分配
                    fw1 = fw1 * not_solid    # 再挡穿透：泡/brick 记录后不穿
                    fd1 = fd1 * not_solid
                    fw_p, fd_p = fw1, fd1
            spread = spread * bombed.float() # 只有炮格接收（波前落脚点）
            newly = (spread > w) & bombed    # 新激活炮格 → 下一轮波前
            w = torch.maximum(w, spread)
            front = torch.where(newly, spread, torch.zeros_like(w))
            fdist = torch.where(newly, blast_f, torch.zeros_like(w))
            if should_ee and not sync_free and i % DANGER_CHECK_EVERY == 0 \
                    and not bool((newly).any()):
                break                      # 无新激活炮格：后续轮恒空转
        weight = w

    # 阶段 B：从每颗炮（用修正后的权重）扩散危险范围（与旧版同一扩散逻辑）。
    # **双缓冲扩散（v2，2026-08-10）**：fw=权重 + fd=每炮剩余距离一起挪格，
    # 每方向统一 max_b 步（原版按 blast 档分组 Σb 步）→ pad ÷4。
    # 等价性：覆盖记录先于挡火（泡/brick 格被覆盖但 fw/fd 置 0 不穿透），
    # 每炮的传播距离（自己的 blast 档）固定，与档位传播顺序无关 ——
    # 逐炮朴素参考对比 PASS（verify_danger_v2）。
    seed = weight * passable
    danger = seed.clone()
    # （blast_f / max_b 统一来自上方，阶段 B 不再重算）
    fw = seed.clone()
    fd = torch.where(bombed, blast_f, torch.zeros_like(seed))
    for drow, dcol in _DIRS:
            fw_p, fd_p = fw, fd
            for _ in range(max_b):
                fw1 = _shift(fw_p, drow, dcol) * passable   # 同阶段 A：连续张量免 item 同步
                fd1 = _shift(fd_p, drow, dcol) * passable
                fd1 = fd1 - one_buf          # 张量操作数免 item 同步
                keep = fd1 >= 0          # 第 b 格（fd1=0）也记录；耗尽才停
                fw1 = torch.where(keep, fw1, 0.0)   # 标量 0：省 zeros_like 分配
                danger = torch.maximum(danger, fw1, out=danger)  # 先记录（覆盖泡格）
                fw1 = fw1 * not_solid    # 再挡穿透：泡/brick 记录后不穿
                fd1 = fd1 * not_solid
                fw_p, fd_p = fw1, fd1
    return danger
