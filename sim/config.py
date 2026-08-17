"""关卡配置与张量布局定义。

这里是 Python 侧和 CUDA 侧共享的"常量真相"。改这里的默认值时，
`sim/cuda/bomber_kernels.cu` 顶部的编译期上限也要一起看（见 CUDA_LIMITS）。

动作空间是**分解式**的：方向键和放炮键各一个头，而不是拍平成 6 个互斥动作。
理由是规则本身要求二者独立 —— 炸弹人可以边跑边放炮，拍平成互斥动作会
强迫"放炮那一 tick 必须站住"，那是错的规则，不是简化。

观测是**每个 env 一份共享张量**，不是每个角色一份。见下面 OBS_LAYOUT 的说明。
"""

from __future__ import annotations

from dataclasses import dataclass

# 方向头：4 方向 + 松手。松手 = 停下，不是"保持上一次方向"。
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_IDLE = 0, 1, 2, 3, 4
N_MOVES = 5
# 放炮头：0 = 不放，1 = 放（trigger 语义，按下即触发）
N_BOMB = 2

# (dy, dx)，索引与方向编码对齐；MOVE_IDLE 没有位移
DIRS = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))

# CUDA kernel 里用固定大小数组，超过这些上限需要改 .cu 文件
CUDA_LIMITS = {"max_h": 21, "max_w": 21, "max_players": 4}

# ---------------------------------------------------------------- 观测布局
#
# **存储布局（kernel 写出来的东西）：一个 env 一份，(N, 2P+3, H, W)。**
#
#   通道           内容                                    与"我是谁"有关？
#   ------------   -------------------------------------   ----------------
#   0     .. P-1   玩家 i 的位置（双线性 splat，质量 1）    无
#   P     .. 2P-1  玩家 i 名下泡泡的引信，fuse / FUSE       无
#   2P             墙，0/1（整局不变）                      无
#   2P+1           危险图，越接近爆炸越接近 1               无
#   2P+2           局内进度 t / MAX_STEPS（常量平面）       无
#   （以下为扩展通道 OBS_EXTRA = 1 + 3P 个，全部**世界信息**，
#     排在 2P+3 之后，view_perm 对尾部原样保留不做置换）
#   2P+3           宝箱位置，0/1（可炸砖被炸掉后变宝箱）    无
#   .. +P         玩家 i 无敌标记（位置格 1，其余 0）       无
#   .. +P         玩家 i 可用泡泡数（位置格 = 可用/上限档） 无
#   .. +P         玩家 i 泡泡上限（位置格 = 上限/上限档）   无
#
# 所有通道都与"我是谁"无关，所以 P 个角色**共用同一份内存**。
# 旧版给每个角色各写一份（(N, P, C, H, W)），其中 3 个通道字节完全相同、
# 位置通道也只是顺序不同 —— 纯粹的重复写。P=4 时写入量差 6.5 倍，
# 而 observe 的写入量本来就是 step 访存量的 9 倍，是整个模拟器的第一瓶颈。
#
# **视角布局（网络看到的东西）：`view_perm(me, P)` 给出的置换。**
#
#   视角通道 0        自己位置
#   视角通道 1        自己泡泡引信
#   2 .. P            各对手位置（按编号升序，跳过自己）
#   P+1 .. 2P-1       各对手泡泡引信（同上顺序，一一对应）
#   2P, 2P+1, 2P+2    墙、危险图、进度
#
# 关键性质：**视角只是存储的一个置换**，长度相同（2P+3），没有求和、
# 没有拼接、没有条件分支。所以"共享"这件事的全部复杂度就是下面这个函数，
# 而且置换输入通道等价于置换第一层卷积的权重（见 train/model.py），
# 连一次数据搬运都不需要。
#
# 对手泡泡引信是**每个对手一个通道**而不是合并成一个。合并需要求和，
# 那就不再是纯置换了；分开还顺带多给网络一个"这泡是谁放的"的信息。


def obs_extra(n_players: int) -> int:
    """扩展通道数（全部世界信息，纯增益，尾部原样保留不进置换）：
    1（宝箱位置）+ P（各玩家无敌标记）+ P（各玩家可用泡泡数）
    + P（各玩家泡泡上限）
    """
    return 1 + 3 * n_players


def view_perm(me: int, n_players: int, c: int | None = None) -> tuple[int, ...]:
    """视角通道 j ← 存储通道 view_perm(me)[j]。

    基础 2P+3 个通道参与视角置换（"我是谁"相关的玩家通道）；
    尾部扩展通道（宝箱/无敌/可用泡数/泡数上限）是世界信息，对任意
    视角都一样，所以**原样保留在尾部**，不进置换。

    `c` 是实际通道数（旧 7 通道 ckpt 无扩展通道 → 尾部为空；
    缺省时按新布局 2P+3+obs_extra(P) 给全）。
    """
    p = n_players
    others = [i for i in range(p) if i != me]
    base = 2 * p + 3
    if c is None:
        c = base + obs_extra(p)
    return tuple([me, p + me] + others + [p + o for o in others]
                 + [2 * p, 2 * p + 1, 2 * p + 2]
                 + list(range(base, c)))


# 基础通道数 = 2P + 3（位置×P + 泡泡引信×P + 墙/危险/进度）
def n_obs_channels(n_players: int) -> int:
    return 2 * n_players + 3 + obs_extra(n_players)


@dataclass(frozen=True)
class SimConfig:
    """一局"基础关卡"的全部可调项。"""

    height: int = 13
    width: int = 13
    n_players: int = 2

    # --- 时间与运动 ---
    tick_hz: int = 10       # 逻辑帧率（≠ 渲染帧率）；每 tick 一次决策
    speed: float = 3.6      # 角色速度，单位"格/秒"。3.6 格/秒 = 每 tick 0.36 格
                            # （上限 3.9：0.39 < 1-2r=0.4，贴墙滑动不穿模）
    radius: float = 0.3     # 角色碰撞盒半宽（格），必须 < 0.5
    max_steps: int = 600    # 600 tick @10Hz = 60 秒，超时判平局

    # --- 泡泡 ---
    fuse: int = 30          # 放泡后多少 tick 爆炸（30 tick @10Hz = 3 秒，原版手感）
    blast: int = 2          # 十字射线长度（不含中心格）。blast=3 时 AI 铺地雷过猛、
                            # 自伤/对轰频繁；改回 2 让走位与进攻更平衡
    max_bombs: int = 10     # 单角色同时在场泡泡数（4 → 10：放炮上限放宽，布局/封锁空间更大）
    max_chain: int = 16     # 连锁爆炸最多迭代几轮。8 → 16（漏爆修复）：13×13 地图
                            # 一行最长 13 颗泡（blast=1 首尾相接）需要 12 轮连锁，
                            # 旧值 8 让 9+ 颗长链尾部漏爆（danger 预警覆盖但实际
                            # 没引爆）。early_exit（resolve/danger 均早退）保证
                            # 多出的轮在连锁结束即停，不损失性能；CUDA graph
                            # 固定轮模式才按满 16 轮算。
    chain_cap_rounds: int = 8   # danger 阶段 A 的**同步免除**固定轮上限（仅非 graph
                                # 模式生效）：链长 ≤ cap 时结果与动态早退逐位一致。
                                # **DCU 实测（2026-08-16，corridor+open0.5 MLP 随机
                                # 打法 60 tick）：danger 预测链深可达 8**（cap=6 仍
                                # 有 4/60 tick 差异，cap=8 才 maxdiff=0）——成长
                                # blast=7 的长炮能连锁 7-8 颗泡；910B 实测 max=3
                                # 只是它的分布（短 blast）。cap=8 全覆盖 → 与动态
                                # 早退逐位一致（生产行为不变，只是免同步）。代价：
                                # 链短 tick 多跑空波前轮（编译后 fused，成本可忽略）。

    step_penalty: float = 0.001
    wall_density: float = 0.0   # 0 = 纯空场；>0 时按固定图案摆柱子（永久墙）

    # --- 地图与可炸墙 ---
    # map_mode：
    #   "open"      纯空场（现有行为，测试默认）
    #   "corridor"  中间 corridor_width 列初始可通行，左右两侧全是**可炸墙**
    #               （brick）：火能把它烧掉（挡火但被覆盖即摧毁）。网络靠
    #               状态变化（墙通道该格变 0）学会"炸墙开图"。
    # 可炸墙是独立张量 `brick`（永久墙 `wall` 保持不可摧毁）。
    # 观测墙通道 = wall | brick（都不可通行，二值）—— corridor 里格内墙
    # 全是 brick，所以"有墙 = 可炸"对网络是隐式可学的。
    map_mode: str = "open"
    corridor_width: int = 5     # corridor 模式中间可通行列数（13 宽 → 左右各 4 列 brick）
    top_wall_rows: int = 4      # corridor 模式**顶/底障碍行总数**：顶部 top 行 + 底部
                                # (top_wall_rows-top) 行全部永久墙（不可炸），
                                # top 每局随机 ∈ [0, top_wall_rows]（random_wall_rows=True 时）。
                                # 出生点自动下移到空旷区（剩余行）中心
    random_wall_rows: bool = True  # corridor 顶/底障碍行随机：每局 top∈U[0,top_wall_rows]、
                                   # bottom = top_wall_rows-top（总障碍行数恒 = top_wall_rows）。
                                   # False = 固定顶部 top_wall_rows 行（旧行为）。
    open_obstacle_max: int = 0  # open 关随机**单障碍**（永久墙）数量上限：每局 0~N 个
                                # 随机散布（避开出生点四邻），0 = 不添加（纯空场）。

    # --- 成长系统（corridor 用）：**宝箱拾取**驱动 ---
    # 砖被炸掉后原地变成**宝箱**（crate），玩家走到宝箱格时掷 growth_crate_prob
    # 概率开箱：命中则随机升一属性（泡数/威力/速度，各 clamp 到上限）；
    # 未命中宝箱也消失。**开箱奖励与概率解耦**：踩到宝箱**必得** brick_reward
    # （收集是密集正向信号，不管中不中奖都要训练得分 —— 中奖与否只是属性，
    #  不是"值不值得捡"的开关）。**与爆炸归属无关**（谁炸的谁捡都行），
    # 不需要归属图 —— 性能与无成长版一致。AI 靠隐式反馈（放泡被拒/爆炸半径/
    # 移速）感知成长；宝箱不占观测通道（2P+3 保持）。
    growth_crate_prob: float = 0.50   # corridor/ring **炸砖变宝箱**的爆率（踩到掷一次，
                                      # 命中才升属性）—— 0.5 保持原样，**不改全局**。
                                      # 掉血回收箱（hit_attr_penalty 生成的）**单独 100%**：
                                      # 用 recycle_crate_prob，不依赖这个全局值 ——
                                      # 掉多少层补多少箱、踩了必还原，总量守恒可核算
                                      # （用户定：受伤爆出来随机生成的才是 100%）。
    brick_reward: float = 0.15        # 踩箱奖励：踩到即给（人类一局踩 ~11 箱。历史：
                                      # 0.15 过强达 1.4x hit 被学成刷分 → 0.05 太低
                                      # 不吃箱 → 0.10 仍不够(corridor 吃箱不明显) →
                                      # 2026-08-11 用户定回调 0.15，且配合 explore 退火
                                      # 放慢(k=0.6)，有效吃箱信号比 3B 版(α≈0.03)恢复
                                      # 近 10 倍；open 关刷分由 open 关设计防住）
    growth_bombs_start: int = 2
    growth_blast_start: int = 2
    growth_speed_start: float = 1.0   # corridor/ring 初始速度倍率（对打/启动器可调）
    growth_bombs_max: int = 10        # corridor/ring 泡数成长上限（7 → 10：与 max_bombs 对齐）
    growth_blast_max: int = 7
    growth_speed_max: float = 2.1     # 速度上限倍率：base 3.0 × 2.1 = 6.3 格/秒（0.63 格/tick）
    growth_speed_step: float = 0.15   # 每次成功 +0.15 → 更快到上限，速度感知更明显

    # --- 混合地图（修正"横向刷分"退化） ---
    # corridor 里"横向放炮→炸砖→宝箱+0.15→还不容易死"被 AI 学成最优刷分路径
    # （实测 rw5 横向移动占 71%）。所以每局按 open_fraction 随机**交替**两类关：
    #   open 关：纯空场，无墙无砖无宝箱（没得刷），成长**随时间**从起点涨到上限，
    #     逼 AI 学真交战（放炮躲泡追命）—— 治"corridor 里学会抢宝箱刷分、不练格斗"。
    #   corridor 关：保持砖墙/宝箱/成长的收集体验。
    # 训练时网络看不到地图类型（观测无标记），靠状态差异自然学会适配。
    open_fraction: float = 0.0        # 每局 open 关占比（其余 = corridor）；训练用 0.5
    open_growth_bombs: int = 3        # open 关初始泡数上限（= 上限 7 的 ~40%）
    open_growth_blast: int = 3        # open 关初始威力（= 上限 7 的 ~40%）
    open_growth_speed: float = 0.84   # open 关初始速度倍率（= 新上限 2.1 的 40%）
    open_crate_cross: bool = True     # open 关开局**中心十字宝箱**（横竖各 2 排，
                                      # ≈46 格，100% 有东西，踩到必升一属性）—— 属性是
                                      # 稀缺资源，开局给一池，掉血回收补充，总量守恒。
                                      # False = 不撒开局池（纯随机回收）。
    hit_attr_penalty: int = 2         # 掉血惩罚（**全地图模式生效**）：每次被炸到
                                      # 掉血，泡/威/速**各扣此层数**（层 = 1泡 / 1威 /
                                      # growth_speed_step速），clamp 回各自模式的起点
                                      # （open → open_growth_*，corridor/ring →
                                      # growth_*_start）；扣掉的以宝箱形式**随机可通行
                                      # 格回收**生成（总量守恒 —— 炸人 = 抢属性资源）。
                                      # 0 = 关闭（纯踩箱成长，无惩罚）。
                                      # open 回收避开中心十字带，corridor/ring 避开砖墙。.
    opp_boost: bool = False       # **训练难度**：对手初始属性按类型增强 ——
                                  #   历史网络（固定 ckpt 陪练 + 模型池快照）→
                                  #     初始属性 × opp_hist_mult（轻微增强 2-30%）；
                                  #   规则 bot（astar/greedy/random）→ 80% 初始
                                  #     （opp_growth_*，接近满上限）—— 强敌陪练。
                                  # 学习侧（pid 0）仍从各自模式起点起步。
                                  # 掉血惩罚对双方生效（各自 clamp 回**自己的**起点，
                                  # 增强侧起点 = 增强后的值）。
                                  # 由训练侧 build_opponents 后按对手类型 set_opp_boost。
                                  # 默认 False（对打/测试双方同起点，行为不变）。
    opp_hist_mult: float = 1.3    # 历史网络初始属性增强倍数（1.0 = 不增强；取 30%，
                                  # 低起点（2泡/1.0速）也有一层/一档的可见提升）
    opp_growth_bombs: int = 6     # 规则 bot 初始泡数上限（= 上限 7 的 ~80%）
    opp_growth_blast: int = 6     # 规则 bot 初始威力（= 上限 7 的 ~80%）
    opp_growth_speed: float = 1.68# 规则 bot 初始速度倍率（= 上限 2.1 的 80%）
    recycle_crate_prob: float = 1.0  # **掉血回收箱爆率 100%**（用户定）：hit_attr_penalty
                                     # 掉血随机生成的宝箱踩到**必升一属性** —— 掉多少层
                                     # 补多少箱、踩了必还原，总量守恒可精确核算；不吃
                                     # growth_crate_prob（那个是炸砖宝箱的 0.5）

    # --- 环岛地图（第三类，泛化性） ---
    # 中间 ring_center_h × ring_center_w 的**永久墙山体**（不可行走不可炸），
    # 山体外围一圈是**稀疏**可炸墙（brick，密度 ring_brick_density），
    # 场地四角与山体之间留有开阔空地 —— 出生点放四角，玩家有足够空间周旋。
    # 宝箱爆率 = ring_crate_prob（默认 100%，环带砖有限，练"炸墙→吃→变强"）。
    # 三图混合时按 open_fraction / ring_fraction / 余量=corridor 随机分配。
    ring_fraction: float = 0.0        # 每局环岛关占比（0 = 不启用）
    ring_center_h: int = 7            # 中间山体高度（13×13 → 山体 7×7，环带 3 宽）
    ring_center_w: int = 7            # 中间山体宽度
    ring_brick_density: float = 0.4   # 环带可炸墙密度（0.4 = 四成格子有 brick）
    ring_crate_prob: float = 1.0      # 环岛宝箱成长爆率（100%）

    # --- 炸弹雨（hazard）躲避特训 ---
    # 每局按 hazard_fraction 掷"炸弹雨"模式（其余 = 正常对局）。炸弹雨关：
    #   玩家不能放泡 —— bombs_cap 强制 0，_place_bombs 的 (live < 0) 恒 False
    #   天然封死，放泡头也被 legal_mask 屏蔽（观测的可用泡/上限通道全 0）。
    #   环境每 hazard_wave_ticks 播一波炸弹雨：每波 hazard_bombs_min..max 颗
    #   落在**可通行格**（无墙/砖、无在场泡、非活人脚下），威力按局内进度
    #   偏向大值 —— v = u^p、p 从 1（均匀）线性退火到 0.2，约
    #   hazard_ramp_seconds 后 v 几乎总是 > 0.8 → 威力几乎总是 max-1/max
    #   （4..8 → 7/8）。纯生存关：谁活得久谁赢。
    #   实现只进参考后端（torch_sim），CUDA kernel 后端不覆盖 —— 躲避特训
    #   用 --backend torch（与现有训练一致），见 train/dodge_train.py。
    hazard_fraction: float = 0.0      # 每局炸弹雨占比（0 = 不启用；特训用 1.0）
    hazard_wave_ticks: int = 50       # 波间隔（50 tick @10Hz = 5 秒）
    hazard_bombs_min: int = 4
    hazard_bombs_max: int = 30
    hazard_blast_min: int = 4
    hazard_blast_max: int = 8
    hazard_ramp_seconds: float = 60.0 # 威力偏向 max 的坡度：60 秒后几乎总是 7/8
    # 躲避关宝箱只加速度：炸弹雨关里踩宝箱**不再随机升 泡/威/速**，只升速度。
    # 融合训练用 —— 特训项的奖励结构是"躲避能力"，而泡数/威力在禁放泡关里
    # 是死通道（学了也是死值）；只有速度真正影响"能不能躲开"。开着它时
    # 躲避关的成长目标单一，AI 学"加速跑位"，与正常关（三属性成长 + 进攻）
    # 交替出现 → 躲避能力进主策略，不牺牲原有进攻/刷箱行为。
    crate_speed_only: bool = False

    # --- 放泡奖励（"这炮放得值不值"的即时信号） ---
    # 放泡本身**不直接给分**（曾经试过，诱导 corridor 横向刷宝箱）。真正的
    # 进攻回报是"这泡会不会炸到人"，而爆炸要等引信走完才结算 —— 放泡瞬间
    # 用**火焰覆盖预测**给即时信号（依赖危险图同款的 rays 传播，网络有直接监督）：
    #   1. 覆盖敌人（当前火力范围能烧到敌人，无论引信长短）→ 小分 × 覆盖人数。
    #      奖励稀疏：全场只剩"覆盖到敌人"的放泡才得分，乱放地雷/围困不赚。
    #   2. 连锁引爆（火焰能点燃**已有**的、引信快走完的泡）→ 大分 × 剩余时间
    #      因子 × 被连锁泡数。剩余时间因子 ≈ 被连锁泡的引信还差多久爆炸
    #      （剩余越短 → 连锁越快兑现 → 分越高，天然奖励"往快爆的泡上续"）。
    #      **时间差（0.5→0.15）**：因子权重压低后，连**新泡**几乎不赚
    #      （≈0.15×0.15），连**老泡**≈满值（×1.0）——"等它快爆了再续"比
    #      "一股脑连丢"多赚 6 倍以上，专治"啪啪啪啪"贴脸连丢。
    #   3. 近身定位（**不在十字辐射上**的放泡）：火焰覆盖不到敌人时，按炮位
    #      到敌人的**距离**给分 —— 距离越近分越高（越贴脸越有价值）。
    #      **带门槛防刷**（有条件版，与 approach_reward 并存不失控）：
    #      a) 敌人在 place_dist_radius 内才给（距离 ≥ 半径 → 0 分）；
    #      b) 冷却：放炮前已连续 place_dist_cooldown tick 没放炮才给
    #         （连续快速乱放不给，只有冷静瞄准的放炮才赚）；
    #      c) 限频：冷却即天然限频（两次得分至少间隔 cooldown tick），
    #         且每个敌人每 tick 至多算一次 —— 乱按刷不了分。
    #   4. **爆炸时刻的连锁兑现**（chain_blast_bonus）：放置预测只是"预告"，
    #      真正"连起来一起爆"的那一刻也要给分 —— 本 tick **被连锁提前点燃**
    #      的泡每颗 +0.08，归**点火源**（引信自然走完的那颗泡的主人）。
    #      好处：a) 直接奖励"先放→别处续→最后连起来"的布网 + 牵引；
    #      b) **天然免疫连丢** —— 同一 tick 自然走完的一排泡 k=0，一分不赚，
    #         只有"等别人/别的泡先爆、自己这颗把它点燃"才得分（这正是 420M
    #         连丢学不到、5X 该放不放的核心玩法）。
    # 三个信号都在**放置成功**的那一刻按当前在场状态评估，一次性计入。
    # 量级：覆盖 1 敌人 +0.02/次；连锁到 1 颗剩余 5 tick 的泡 ≈ +0.13；
    # 近身满值 +0.01/次（1 格外 ≈ +0.0075），一局至多几十次累计 <0.3。
    # 都远小于命中 1.2 与终局 8，是塑形信号不是主回报 —— 主回报仍是
    # "炸到/赢了"。
    place_cover_reward: float = 0.05  # 放泡覆盖到敌人：每人 +0.05（稀疏 0.9% 触发，
                                      # 旧 0.01 只有 0.02x hit 被淹没；加大到进攻核心信号）
    place_chain_reward: float = 0.20  # 放泡连锁到已有泡：每泡 +0.20 × 时间因子
                                      # （半稀疏 6.1%，旧 0.15 ≈ 0.8x hit → 加大到 ~1x）
    chain_time_factor: float = 0.15   # 时间因子 (0,1]：连老泡≈1.0、连新泡≈0.15
    place_dist_reward: float = 0.0    # 近身定位（删除：人类录像 0.0% 触发，死信号）
    place_dist_radius: float = 4.0    # 近身门槛：炮距敌人 < 此值（格）才给分
    place_dist_cooldown: int = 15     # 近身冷却：放炮前 ≥15 tick 没放炮才给分
    chain_blast_bonus: float = 0.0    # 爆炸时刻连锁兑现（删除：人类录像 0.0% 触发，
                                      # 跨 owner 连锁+点火人类从不主动做，死信号）

    # --- 自杀重罚（防"放泡后站自己泡上炸死"） ---
    # 死亡 tick 自己名下有在场泡（own_live_snap 死前快照）→ 额外负奖励。
    # 实测（course8 1023M vs 597M）：98% 死亡是自爆 —— 泡是几 tick 前放的，
    # 终局 -8 的 credit 归因太弱，模型把"激进放炮"和"几秒后的死"脱钩。
    # 死亡时刻即时重罚让"站自己泡上"本身变贵。量级：-2 = 命中奖励的 1.7 倍，
    # 明显疼但不至于让模型完全不敢放炮（-4 会矫枉过正）。**默认 0（关闭）**：
    # 对打/测试不受影响；训练侧 --suicide-penalty 显式开启。
    suicide_penalty: float = 0.0      # 自爆死亡额外扣分（0 = 关；建议 2.0）

    # --- combo 连击奖励（无伤压制，像格斗连段） ---
    # **不掉血**连续造成伤害 = 连击：连击数越高分越多（combo × combo_reward ×
    # 间隔因子），间隔越短分越多（factor = combo_gap_factor^间隔tick）——
    # 连续压制拿高分，磨磨蹭蹭不连。**掉血（被打）打断连击** → 防"互怼刷连击"，
    # 逼出"不掉血打伤害"的干净压制。量级：单 tick 一击 ~0.05，5 连击 ~0.25，
    # 10 连击 ~0.5 —— 是塑形不是主回报（命中 1.2 / 终局 8 才是）。默认 0 关，
    # 训练侧 --combo-reward 显式开启。
    combo_reward: float = 0.10        # 每级连击给的分（稀疏 0.3%，旧 0.05 死信号 → 加大；
                                      # 注：与 dealt 相关 r=0.78 有冗余，可后续评估）
    combo_gap_factor: float = 0.9     # 间隔因子：间隔每 +1 tick 分 ×0.9
                                      # （连击密 → 分高；间隔 20 tick → ×0.12）

    # --- 接近奖励（"贴上去打"的塑形信号） ---
    # 朝最近对手移动（距离在 approach_dist 内且**正在缩短**）→ 每 tick
    # +approach_reward × 缩短量。
    # **量级 0.1→0.02（防 reward hacking）**：0.1 时贴脸追逐每 tick +0.03、
    # 一局 ~200 tick 接近 ≈ +6 分，与命中奖励（1.2×5 次 = 6 分）相当 —— 模型
    # 学成"永远朝对手跑"（实测移动率 97% / IDLE 3%，人类是 62%/38%），无脑连跑
    # 刷接近分而不是布局击杀。降到 0.02 后接近分 ≈ +1.2/局，纯塑形不压主回报。
    # **贴脸门控 approach_gate**：接近后距离仍 < 此值才给分 —— 隔半场空跑白跑
    # 不给（治无脑冲），贴脸纠缠（放泡能形成威胁）才值钱。
    approach_reward: float = 0.0      # 每接近 1 格给的分（重头训删除：0 = 关）
    approach_dist: float = 5.0        # 距离 < 此值才算"接敌区"，接近才给分
    approach_gate: float = 3.0        # 接近后距离仍 < 此值才给（贴脸门控）
    # --- 主动追击奖励（治"追不上逃跑的对手"，与 approach 互补） ---
    # approach 只在**距离缩短**时给分 —— 对手逃跑时追击零收益，模型学会"敌人躲远
    # 就不追"（实测 astar flee 躲顶部磨平）。追击项按方向余弦：位移朝"最近存活
    # 对手"方向的分量 ≥0 → 每朝对手推进 1 格给 chase_reward（距离不缩短也给）。
    # **只在对手逃跑时给**（对手位移朝"离开我"方向）：flee 的 astar 跑到哪追到哪
    # （治"击杀不了躲避形态"）；对手不逃不给（approach 管近距接近，防无脑冲）。
    # chase_adj 是距离阻力 1/(1+d×adj)，就近追更赚、跨场追也有正分。
    # **默认 0（关闭）**：对打/测试不受影响；训练侧 --chase-reward 显式开启。
    chase_reward: float = 0.0         # 每朝对手推进 1 格给的分（默认关，建议 0.02）
    chase_adj: float = 0.05           # 距离阻力 1/(1+d×adj)，就近追更赚

    # --- 无敌保护期 ---
    # 被炸掉 1 血后进入 invuln_ticks 的无敌期（期间被炸不掉血、不触发对方 hit
    # 奖励）。目的是打断"连炮往死里整对手"：对手在无敌期挨炮零收益，
    # 但 danger 图照常显示（无敌只挡掉血，不挡威胁感知）。
    invuln_ticks: int = 30            # 30 tick @10Hz = 3 秒

    # --- 血量（界面/手感用，**不进观测**）---
    # 中心格被火焰覆盖每 tick 扣 1 血，归 0 才死。网络看不到血量（观测布局
    # 仍是 2P+3），它靠危险图 + 价值函数判断威胁；血条只画在 UI 上。
    # max_hp=1 即等价于旧版"一碰就死"。
    max_hp: int = 5

    # --- 超时结局 ---
    # **退火语义（重头训，默认 True）**：一开始超时"谁活着血更多谁赢"（血多者胜
    # 有奖励），随训练击杀能力上来，超时不再有正回报（引导"真击杀而不是拖到
    # 超时拿血差"）。实现：超时终局给分 × `self._explore_coef`（与放炮塑形同一
    # 退火系数，击杀率上来 α→0 → 超时奖励归零）。死亡终局（有人死）**不受退火**
    # 影响 —— 击杀永远有回报。False = 旧行为超时血多者胜不退火。
    timeout_draw: bool = True

    # --- 奖励结构：微观小分（每 step 塑形）+ 宏观胜负（终局一次性）---
    # 分数分两个层面，互不混用：
    #   **微观**（每 step 的 +-）：掉血/打中 ±hit_reward、放泡覆盖/连锁
    #     place_cover/chain_reward、吃箱 brick_reward、危险区站桩罚、
    #     步罚。这些只进 rollout buffer 做策略梯度（"局内学习"），局终就丢弃，
    #     **不累计进 ELO**。
    #   **宏观**（终局一次性 ±win_bonus，平局 0）：局终胜负折算成一条
    #     terminal reward 也进梯度 —— 否则策略收不到"要赢"的信号，只会龟缩
    #     角落（实测的坍塌行为）。ELO 仍只由胜负统计驱动，与上述分数完全独立。
    # 多因子权重（"街霸人格"思路：进攻/收集/生存/效率/终局各司其职）：
    #   **进攻**：造成 1 伤害 +hit_reward；放泡当 tick 按火焰覆盖预测给
    #     place_cover_reward / place_chain_reward / place_dist_reward
    #     （即时信号，见上方注释）。
    #   **收集**：踩宝箱 +brick_reward（与概率解耦，密集正向）。
    #   **生存**：掉 1 血 -hit_reward；危险区每 tick -danger_penalty×danger值。
    #   **效率**：每 tick -step_penalty（防磨洋工）。
    #   **终局**（固定值，重头训）：**对手 hp=0（击杀）** 才给 win_bonus 固定值
    #     （±win_bonus，不看血量差距）。超时全员存活按"血多者胜"给分 × 退火
    #     （击杀能力上来 → α→0 → 超时奖励归零，只留真击杀的固定回报）。
    #     固定值理由：按血量比例的"残血险胜 ≈1.6"引导太弱（用户实测），
    #     且"领先龟缩到超时"在退火归零后无利可图。
    #     必须 > 其他因子总和，否则 AI 停在"捡箱子/龟缩"的局部最优。
    # 量级校验：一局 1800 tick 步罚 −1.8；危险罚量级 ~−0.0x/tick；
    # 打满 5 血击杀 = 5×hit_reward=6；win_bonus=8 高于任何"被动磨一局"的累积，
    # 让"进攻赢"始终优于"活着耗完"。
    #   掉 1 血 -hit_reward / 造成 1 伤害 +hit_reward（1v1 里对方掉血 = 我的泡干的）
    #   站在**危险区**（被在场泡泡爆炸范围覆盖的格）：每 tick
    #     -danger_penalty × danger值 —— danger值 = 1-(fuse-1)/FUSE，越接近爆炸越疼
    #   终局：见 win_bonus / win_hp_scaled（判血逻辑见 torch_sim.step）
    hit_reward: float = 1.5       # 掉 1 血 -1.5 / 造成 1 伤害 +1.5（稀疏主信号 0.5~1.6%
                                  # 触发；旧 1.2 略弱，加大 25% —— 命中是唯一"打中"回报）
    win_bonus: float = 10.0       # 终局固定值：对手 hp=0（击杀）才给 ±win_bonus（稀疏，
                                  # 每局 1 次；旧 8 → 加大到 10，> 所有塑形总和防龟缩）
    win_hp_scaled: bool = False  # 重头训：False = 击杀给固定 ±win_bonus（用户定：对手 hp=0
                                 # 才是奖励，固定值）；超时血多者胜 × 退火。旧 True =
                                 # 按剩余血量比例给（可回退对比）
    danger_penalty: float = 0.015   # 危险区站桩（**稠密**：人类 65% 时间站危险区，旧 0.05
                                    # 达 2.5x hit 过强 → 降到 0.015 ≈ 0.75x hit；
                                    # 乘 _explore_coef 随探索退火衰减）
    passivity_ticks: int = 20      # 2 秒没放泡开始算被动（60→20：旧值 6 秒太松，
                                   # 约束不住"满预算一股脑全丢"；收紧后"没在放炮"
                                   # 有成本 → 学选择性放炮/留炮节奏）
    passivity_penalty: float = 0.0    # 久不放炮罚（重头训删除：0 = 关）

    # --- 观测存储 ---
    obs_extra_enabled: bool = True
    """是否编码**扩展观测通道**（宝箱位置/无敌标记/可用泡数/泡数上限）。

    新训练默认开（P=2 → 14 通道，决策信息更全）；旧 7 通道 checkpoint 的
    网络只能吃 7 通道输入，评估/试玩旧档时要关掉（`obs_extra_enabled=False`）。
    """
    obs_fp16: bool = True
    """观测张量用 fp16 存。通道值全在 [0,1]，fp16 的 10 位尾数远够用。

    省下的是**写入带宽**（observe 是模拟器第一瓶颈）和 rollout buffer 的显存
    （(T,N,C,H,W) 直接减半，等于同样显存能开更大的 batch）。

    诚实说明：要拿到**端到端**的收益，策略网络必须原生吃 fp16（`torch.autocast`）。
    否则 forward 入口那次 fp16→fp32 的 cast 会把省下的读带宽还回去一部分。
    模型侧的 cast 见 `train/model.py::ActorCritic.forward`。
    """

    def __post_init__(self) -> None:
        if self.n_players < 2 or self.n_players > CUDA_LIMITS["max_players"]:
            raise ValueError(f"n_players 必须在 2..{CUDA_LIMITS['max_players']}")
        if self.height > CUDA_LIMITS["max_h"] or self.width > CUDA_LIMITS["max_w"]:
            raise ValueError("地图超出 CUDA kernel 编译期上限")
        if self.height < 5 or self.width < 5:
            raise ValueError("地图至少 5x5，否则四角出生点会互相重叠")
        if self.map_mode not in ("open", "corridor"):
            raise ValueError(f"map_mode 必须是 open/corridor，收到 {self.map_mode}")
        if self.map_mode == "corridor":
            if not 3 <= self.corridor_width <= self.width - 2:
                raise ValueError(
                    f"corridor_width 必须在 [3, width-2]=[3,{self.width - 2}]，"
                    f"两侧才能各留至少 1 列可炸墙")
            if self.corridor_width < 2 * self.n_players - 1:
                raise ValueError("corridor_width 太小，放不下 n_players 个出生点")
            if not 1 <= self.top_wall_rows <= self.height - 3:
                raise ValueError(
                    f"top_wall_rows 必须在 [1, height-3]=[1,{self.height - 3}]，"
                    f"顶部永久墙要留出至少 2 行空旷区给出生点")
            if self.ring_fraction > 0:
                # 环岛山体必须居中且至少留 1 格环带（可炸墙）+ 1 格外缘空地
                if not 3 <= self.ring_center_h <= self.height - 4:
                    raise ValueError(
                        f"ring_center_h 必须在 [3, height-4]=[3,{self.height - 4}]，"
                        f"否则环带 + 外缘放不下出生点")
                if not 3 <= self.ring_center_w <= self.width - 4:
                    raise ValueError(
                        f"ring_center_w 必须在 [3, width-4]=[3,{self.width - 4}]")
                if not 0.0 < self.ring_brick_density <= 1.0:
                    raise ValueError("ring_brick_density 必须在 (0, 1]")
            if not 0.0 <= self.open_fraction + self.ring_fraction <= 1.0:
                raise ValueError("open_fraction + ring_fraction 必须 ≤ 1（余量为 corridor）")
        if not 0.0 < self.radius < 0.5:
            # 半宽 ≥ 0.5 意味着碰撞盒能同时压到三行，2x2 的邻格检查就不够了
            raise ValueError("radius 必须在 (0, 0.5) 开区间内")
        if self.step_len >= 2.0:
            # 单 tick 位移超过 2 格才报错（> 两倍格宽，任何合理手感都用不到）。
            # 位移 > 1-2r 不再阻止：move.py 用 substep 拆解大位移防穿模
            # （成长 1.5× → 0.45 格/tick 也能安全贴墙停）。
            raise ValueError(
                f"单 tick 位移 {self.step_len:.3f} 太大，请降低 speed 或提高 tick_hz"
            )
        # 成长速度上限：corridor 才用成长；move.py substep 已处理大位移，
        # 这里只做 sanity（成长后每 tick 超过 2 格仍不合理）
        if self.map_mode == "corridor" and \
                self.step_len * self.growth_speed_max >= 2.0:
            raise ValueError(
                f"成长后速度 {self.speed * self.growth_speed_max:.2f} 格/秒"
                f"（每 tick {self.step_len * self.growth_speed_max:.3f} 格）太大，"
                f"请降 speed 或降成长上限"
            )
        if self.place_dist_radius <= 0:
            raise ValueError("place_dist_radius 必须是正数（近身定位门槛）")
        if not 0 <= self.place_dist_cooldown < self.passivity_ticks:
            raise ValueError(
                f"place_dist_cooldown 必须在 [0, passivity_ticks)="
                f"[0,{self.passivity_ticks})，否则重置时 since_bomb=冷却会直接触发"
                f"被动罚")
        if not 1 <= self.hazard_bombs_min <= self.hazard_bombs_max:
            raise ValueError(
                f"hazard_bombs_min/max 必须满足 1 ≤ min ≤ max，收到 "
                f"{self.hazard_bombs_min}/{self.hazard_bombs_max}")
        if not 1 <= self.hazard_blast_min <= self.hazard_blast_max:
            raise ValueError(
                f"hazard_blast_min/max 必须满足 1 ≤ min ≤ max，收到 "
                f"{self.hazard_blast_min}/{self.hazard_blast_max}")
        if self.hazard_wave_ticks <= 0:
            raise ValueError("hazard_wave_ticks 必须为正")
        if self.hazard_ramp_seconds <= 0:
            raise ValueError("hazard_ramp_seconds 必须为正")

    # --- 派生量 ---

    @property
    def step_len(self) -> float:
        """一个 tick 的位移（格）。speed=3、10Hz ⇒ 0.3 格/tick。"""
        return self.speed / self.tick_hz

    @property
    def n_cells(self) -> int:
        return self.height * self.width

    @property
    def n_channels(self) -> int:
        """**共享**观测的通道数：2P + 3 + (obs_extra_enabled ? obs_extra : 0)。

        基础 2P+3（位置×P + 泡泡引信×P + 墙/危险/进度）参与视角置换；
        扩展通道（宝箱/无敌/可用泡数/泡数上限）是世界信息，尾部原样保留，
        可由 obs_extra_enabled 关闭（兼容旧 7 通道 checkpoint 的评估/试玩）。
        没有"当前朝向"通道：移动是无惯性的，按下即生效，朝向完全由本 tick
        的动作决定，不构成需要观测的状态。
        """
        base = 2 * self.n_players + 3
        return base + (obs_extra(self.n_players) if self.obs_extra_enabled else 0)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """(C, H, W)。注意这是**一个 env 一份**的形状，不再乘 P。"""
        return (self.n_channels, self.height, self.width)

    def view_perm(self, me: int) -> tuple[int, ...]:
        """角色 me 的视角置换，见 config 文件头的说明。"""
        return view_perm(me, self.n_players)

    @property
    def game_seconds(self) -> float:
        return self.max_steps / self.tick_hz

    # --- 成长系统：炸砖驱动（corridor 用） ---

    def spawn_pos(self) -> list[tuple[float, float]]:
        """出生点：**空旷区中心**（不再是四角）。

        open 模式：整宽均分（P=2 时 13x13 → (4.5, 6.5) 与 (8.5, 6.5)）。
        corridor 模式：空旷区 = 行 top_wall_rows..height-1 × 中间
        corridor_width 列（顶部 top_wall_rows 行是永久墙，左右是 brick）。
        出生点取空旷区**行列中心**，避开墙；P=2 且 corridor_width=5、
        top_wall_rows=4、13x13 时 → (5.0, 8.5) 与 (7.0, 8.5)，相隔 2 列、
        落在空旷区正中央 —— 开局贴脸，炸墙/接敌样本密集。
        mapgen 会清空出生点周围一格，保证不会被墙堵住。
        """
        h, w = float(self.height), float(self.width)
        if self.map_mode == "corridor":
            # 空旷区：行 [top_wall_rows, height-1]，列 [c0, c0+corridor_width-1]
            row = (self.top_wall_rows + (self.height - 1)) / 2.0 + 0.5   # 空旷区中线行中心
            c0 = (w - self.corridor_width) / 2.0 + 0.5    # 可通行区左边界中心
            # 在 [c0, c0+corridor_width-1] 内给 P 个玩家均分，首末各留 1 格
            step = (self.corridor_width - 2) / max(1, self.n_players - 1)
            cols = [c0 + 1.0 + i * step for i in range(self.n_players)]
            return [(row, c) for c in cols]
        row = (h - 1) / 2.0 + 0.5                # 中线格中心，如 6.5
        cols = [(w - 1) * (i + 1) / (self.n_players + 1) + 0.5
                for i in range(self.n_players)]
        return [(row, c) for c in cols]

    def spawn_cells(self) -> list[tuple[int, int]]:
        """出生点所在格（整数），地图生成时用来清空周边。"""
        return [(int(y), int(x)) for y, x in self.spawn_pos()]

    def ring_spawns(self) -> list[tuple[float, float]]:
        """环岛关出生点：场地**四角**（山体在中间，玩家围着它绕圈）。

        P=2 → (1.5,1.5) 与 (1.5,11.5)（13×13 的左上/右上角）；P=3/4 时
        依次取逆时针四个角。四角彼此不相邻，出生点及四邻由 mapgen 清空，
        开局脚下必无 brick。
        """
        h, w = float(self.height), float(self.width)
        corners = [(1.5, 1.5), (1.5, w - 0.5), (h - 0.5, w - 0.5), (h - 0.5, 1.5)]
        return corners[: self.n_players]

    def ring_spawn_cells(self) -> list[tuple[int, int]]:
        """环岛出生点所在格（整数），环带生成时清空周边。"""
        return [(int(y), int(x)) for y, x in self.ring_spawns()]
