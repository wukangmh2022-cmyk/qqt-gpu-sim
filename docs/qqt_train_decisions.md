# QQ堂 260B 训练前评估与设计决策（2026-08-19 定稿）

> 本文件沉淀本轮（13×15 + 标准化关卡 + 宝箱语义对齐 JS）的所有决策与依据，
> 供后续续训/复盘查阅。配套代码：`jax_bomb/`，部署：`deploy_10node/`，校验：`scripts/quick_check_*.py`。

## 1. 训练前 5 问（结论速查）

### 1.1 计算画像（Compute Profile）
- 论文训练量：L_7d_gae90 配置 = 100K iters × 512 envs × 512 steps × 2 × 4 卡 ≈ **210–260B 步**，
  端到端 4×H200 ≈ 35–40 万 sps，历时 7 天。**260B 是总训练步数，不是参数规模**。
- 我们的硬件：10 账号 × 10 notebook × 2 卡 = **20 张 BW1000 DCU**（单卡算力 ≈ H200 的 1/3）。
  本地 Local SGD param mode K=256 实测 25.8s/iter（2 notebook A/B，baseline 无损 132.5s），
  20 卡全开目标 ≈ 30 万 sps → 260B ≈ 6.9 天（见 docs/multicard_lossy_sync.md）。
- **模拟器裸速 50M sps 没有意义**：训练吞吐由"环境推进 + 网络前向 + PPO 更新"整链路决定，
  模拟器再快也只是其中一小段；论文与我们的瓶颈一致。
- 结论：算力只有论文 ~1% 的说法不成立——按步数口径我们与论文同量级（260B、~7 天），
  单卡算力差距已被 20 卡并行 + Local SGD 通信优化（5.1×）补回。

### 1.2 蒸馏
- 已有 CNN 学生课程（resume duel_cnn，40%→80% 成长递进，s2a 40% 过线 0.80）。
- 本轮 260B 大训直接用 transformer（embed 392 depth 4 patch 4 ≈ 7,461,336 参数），
  蒸馏不是本轮目标；CNN 课程用于快速验证环境/奖励正确性。

### 1.3 地图泛化
- **241 张 QQ堂标准图**（levels.json，13×15）按权重混合（默认 240=0.2 空场景）：
  每局随机抽关 + 确定性随机选两个不同出生点 + 关卡初始泡/威/速 + 预置宝箱。
- 地图统计：226/241 关 >50 砖（均值 84），77.1% 出生点对（4141/5368）被墙+砖隔开 → 开局隔离常见，
  直接证据支持"吃宝箱 bootstrap 奖励"（见 §2.3）。
- 多机一致性：环境 RNG（地图/宝箱）是"同一分布的 i.i.d. 采样"，不破坏 Local SGD；
  对手池会破坏（不同分布）→ 采用 A 方案自对弈（见 §2.5）。

### 1.4 Reward Hacking
- 终局奖励最终配方 = 稀疏 ±1（论文同款），不叠中间过程奖励干扰 real-world shaping。
- 风险点与对策：
  - 龟缩（一血领先龟缩）→ 对战模式同时行动 + 胜负 ±1，长期收益由价值函数学，
    不人为加"进攻惩罚"式 shaping。
  - 宝箱奖励被滥用 → 只做 bootstrap：`CRATE_REWARD_COEF=0.5` 前期加，
    5e8 全局步（≈半小时 54 万局）线性退火到 0，训练早期加速学习、后期归零。
  - 掉血爆属性守恒：HIT_ATTR_PENALTY=2，扣 N 层补 N 箱（rec_crate 必升），无凭空奖励/损失。

### 1.5 Pre-flight（部署把关）
- 部署包 v2 已在 10 节点一键化：部署+自检（jax/optax/代码版本/2 卡）+ 取 IP + 同步启动 +
  30s 健康检查 + watch 监控（磁盘告警）+ rank0 参数快照拉回。
- 已知坑全部内置规避：DTK env + LD_PRELOAD=libmpi.so、代理 unset、envs/minibatch 自动均分、
  `--iters 20000`（≈335B，含 260B 余量）、`--ff-factor` 显式传、RCCL 调优不要碰。
- **部署后必跑**：`quick_check_obs_move.py`（54 项）/ `quick_check_levels.py` / 
  `quick_check_crate_semantics.py` 三脚本全 PASS + 本地 `web/test_levels.js` 对照。

## 2. 本轮设计决策

### 2.1 13×15 地图 + ViT patch 不变
- 宽 +2 格（H=13, W=15）。ViT patch 4×4：ceil(13/4)=4、ceil(15/4)=4 → 16 token 不变，
  参数 7,461,336 不变（实测）。出生点/走廊格按 torch spawn 公式推 13×15 值。

### 2.2 标准化关卡（levels.json 为标准真相源）
- 241 关从 `levels_qqt/*.pt` 导出为 torch-free 的 levels.json（export_web.py 与 JS Web 同源）。
- LevelSet 栈：wall/brick/crate(预置)/rec/lo(掉血下限)/rate(炸砖爆率)/spawns(多维数组,
  S_MAX=12)/cnt/logw/is_open。权重解析 `"240=0.2"` / `"empty=0.2"`（空场景 = 名字含"空"）。
- 出生点双人不同：`randint(0,cnt)` + `randint(0,cnt-1)` + bump 技巧（vmap 安全，无动态 shape），
  与 JS Fisher-Yates 打乱取前二 = 同一均匀分布。
- 无 levels.json → 回退过程式生成（torch 等价路径，level_id=-1 防御性回退）。

### 2.3 宝箱语义（本版修正：与 JS Web 逐项对齐）
- **炸砖→生箱**：炸砖**瞬间**按本关 `crate_rate` 掷爆率（step 4b），`<=0/缺失 → 1.0` 钳制（JS 同款）。
  意义：威力大时近处炸开可能没箱、远处有箱是真实的——若炸砖必生箱+拾取必升，AI 会确定性地
  就近开箱，失去"去哪边捡"的对局随机性。
- **拾取必升**：踩到宝箱 `rng() < 1.0` 无条件成长（三属性均匀 +1/+1/+0.15），
  预置箱/炸砖箱/掉血回收箱同一规则（step 6b `hits = stood & alive0`，死人不开箱同 JS）。
- 双端验证：JAX 实测 level4(0.6)→0.6027、rate1.0 关→1.0000、corridor(0.5)→0.5011、拾取必升
  8192/8192；JS 同关 0.6038 / 1.0000 / 0 失败。`quick_check_crate_semantics.py` 全绿。
- **测试策略备忘**：原地连按放泡会无限刷新引信（放泡判定 `fuse<=0` 在爆炸结算前，JS sim.js:291
  同款顺序）——统计炸砖爆率必须"只放一次泡"等引信自然引爆。

### 2.4 掉血爆属性（hit-attr-penalty）
- JAX 版已实现：被炸掉血者泡/威/速各扣 2 层（clamp 回关卡初始 lo），守恒撒箱
  `_scatter_recycle`（掉 N 层补 N 箱，rec_crate 必升）。
- 性能：`jax.lax.cond` 跳过无掉血 tick 的 permutation（绝大多数 tick 零开销）。

### 2.5 对战模式：A 方案自对弈（无对手池）
- 论文 = 纯自对弈，无池（"No population, no pool"）；决定因素不是迷雾，
  是"自对弈时同一权重要不要打两边"——A 方案同一权重打两边，梯度互为对手，
  与 Local SGD 的 i.i.d. 采样兼容；历史对手池会引入不同分布破坏收敛。
- EMA：训练不需要（论文代码无 EMA）；EMA 只在部署评估时用（V-D 原文），
  我们评估侧已固化权重做 ELO/胜率，不切训练。

## 3. 已知边界与后续
- crate_rate 原始值直接来自 levels.json；JS 的 `>0?rate:1.0` 钳制已镜像到 levels.py。
  **levels.json 每次重新导出（export_web.py）会改 crate_rate**——测试断言必须从
  ls.rate 动态读，不能硬编码（level4 曾因 0.6→0.6316 误报）。
- 1 出生点的关不存在（241 关全部 ≥2），`max(cnt,2)` 采样安全。
- `pushable` 字段 JS/JAX 两侧都未参与玩法，仅渲染用，两端一致忽略。
- 部署包 v3（md5 18bb93a1）已含 wheels/（v1 缺失，setup_notebook.sh 依赖它离线装 optax）。

## 4. 灌木丛（bush）特性（2026-08-20）
- **数据**：levels.json 新增 `bush` 布尔层（25 关：21 + 28~55 野外），与 wall/brick
  零重叠（levels.py 加载时断言）。
- **玩法**：可通行（blocked = 泡|墙|砖，不含 bush）+ 可炸毁（爆炸覆盖即摧毁），
  炸毁瞬间按本关 crate_rate 掷爆率出宝箱（与炸砖同规则，bush 与 brick 互斥统一处理）。
- **AI 理解**：obs 新增 **ch8 = 灌木**（N_OBS_CH 8→9，ViT patch embed 输入随之变，
  训练未开始无续训负担）。AI 学到"ch8=1 的格子可站、会被烧、烧完可能出宝箱（ch7）"，
  与砖（ch4 不可通行）明确区分。
- **回收箱规则**：掉血回收箱**只落纯粹地面**——`avail = ~wall & ~brick & ~bush`
  （灌木可通行但非地面，JS 侧同口径；JS 实现由 Web 侧维护，本仓库不改 sim.js）。
- **验证**：`scripts/quick_check_bush.py`（加载/零重叠/可站/炸灌木爆率 0.7286≈0.73/
  回收箱 36480 个 0 落灌木/obs ch8 逐格）。

## 5. 穿墙 bug 修复（测试抓到，2026-08-20）
- **现象**：quick_check_levels 随机动作检查发现玩家进入墙/砖格（speed≥1.33 的关）。
- **根因**：JAX `STEP = 7.56/10 = 0.756` 格/tick 是 torch `speed=3.6 → 0.36` 的 2.1 倍
  （移植时把"成长满速 3.6×2.1"误当基础速度，位移公式再乘 spd_g → 双重计费）。
  单 tick 位移达 1.59 格，`_resolve_axis` 只查两端前沿 → 跨格跳过中间砖 → 穿墙。
- **修复**：`_resolve_axis` 沿移动方向从起点侧逐格扫描（MAX_SWEEP=5）取第一个障碍
  贴停；非跨格轨迹逐位不变（obs_move 54 项 PASS 证明）。torch/JS 位移 <1 格无此 bug。
- **待定**：JAX 速度模型 0.756×spd 仍是 torch(0.36×spd)/JS(0.3×spd) 的 2.1 倍，
  穿墙已修但"速度手感"是否统一需用户决策（训练未开始，改 SPEED=3.6 无续训负担）。

## 6. 预留位布局（2026-08-20，后训练增强用；当前全部默认值不参与玩法）
> 用户要求：为下一版道具/竞技模式扩展预留参数位，现在数据结构就位，届时零结构改动。

| 位置 | 字段 | 位宽 | 语义（当前/预留） |
|---|---|---|---|
| BombState | `crate` | int4（int8 存储） | 炸砖/炸灌木/预置/回收箱道具种类：0=无，1=随机宝箱（唯一现道具），2-16 预留 |
| BombState | `buffs` (2,) | 3 bit | 变身 buff：0=无，1-7（熊猫/螃蟹等） |
| BombState | `debuffs` (2,) | 2 bit | debuff：0=无，1-3（慢慢胶减速等） |
| BombState | `items` (2,4) | 每槽 int6 | 道具栏 4 槽：0=空，1-63（夺宝=宝石数量、飞镖等按竞技类型解释） |
| BombState | `gametype` () | int4 | 竞技类型：0=普通对抗（当前唯一），1=夺宝，2=... |
| BombState | `pushable` (H,W) | bool | 可推墙（推箱子关 231-239，733 格）：**必 ⊆ brick**（加载断言"可推墙必须是障碍物"），预留不参与玩法 |
| BombState | `bush` (H,W) | bool | 灌木 = 道具标志位：可通行 + 可炸 + 按 crate_rate 掉宝，与墙（ch4）区分（ch8） |
| global_vec | G=11→24 | - | +13 维预留：我/敌 buff、我/敌 debuff、我/敌道具栏 4 槽、竞技类型（归一化） |
| obs 格子通道 | ch8 | - | 灌木（ch7 宝箱存在性 = crate>0；道具种类不进 obs，进 state） |

- **crate 升级 int8 的意义**：把"宝箱"从布尔升级为"道具种类"枚举，后续宝箱分离成
  多种道具（随机宝箱/超级道具/变身道具）时，只改掉落处的种类值，不动结构。
- **可推墙约束**：`pushable ⊆ brick`（数据实测 733/733），它天然是障碍物（brick 已挡），
  未来实现"推动"时移动逻辑在 blocked 上叠加 pushable 语义即可。
- 权重默认值：`empty=0.05,功夫=0.1,比武=0.15`（空场景 5% + 功夫主题 22 关 10% +
  比武主题 36 关 15%，其余 70% 均分随机）。`_parse_weights` 支持：`240=0.2`（id）、
  `empty=0.2`（名字含"空"）、`功夫=0.1`（theme 精确，**类总占比**类内均分，匹配失败
  回退名字包含）。实测 5.1%/9.9%/15.3%/69.8%。

## 7. 道具系统（2026-08-20，对齐 Web sim.js）
- **levels.json 新字段**：`bombs_max`/`blast_max`/`speed_max`（每关成长上限，bombs_max
  10 或 7、speed_max 2.1 或 2.2）、`crate_super_fraction`（超级道具占比，0.0909）、
  `crate_expect`（统计信息，不用）。
- **crate int8 编码 7 种**：1/2/3 = 泡泡/威力/速度 +1 档；4/5/6 = 超级（+4 档）；
  7 = 问号随机（空场景预置宝箱 / 掉血回收箱，踩到才掷种类）。
- **生成**（4b，与 Web sim.js:349-359 一致）：炸墙/灌木 → crate_rate 判定掉落 →
  super_fraction 判定超级 → uniform×3 定种类。灌木掉宝是 JAX 保留特性（JS 灌木不掉）。
- **拾取**（6b，Web sim.js:410-448）：踩到必升；编码解释 `(kind-1)%3` 得种类
  （1-6），问号随机；超级 +4 档；成长 clamp 到**每关上限**（levels.caps，过程式全局）。
- **obs**：ch7 = 道具存在性；ch9/10/11 = 泡/威/速 one-hot；ch12 = 超级标志；
  N_OBS_CH 9→13（问号宝箱只亮 ch7）。
- **修过的 bug**：超级道具种类映射 `min(kind-1,2)` 会把 4/5/6 全映射成速度 →
  改 `(kind-1)%3`。
- 验证：crate_semantics 超级占比 0.0814≈0.0909、码4 +4 档、clamp 到 bombs_max=10、
  问号随机；levels 预置宝箱编码=7。部署包 v5（md5 8716931f）。

## 8. 碰撞盒半径对齐 WebGL（0.3 → 0.45，2026-08-20）
- **背景**：WebGL（sim.js）碰撞盒半径已改大到 `radius: 0.45`（盒 54x54px），JAX/torch 仍为
  0.3，且**从未做过移动侧的双端验证**（之前的双端验证只覆盖宝箱语义）。
- **对拍验证**（`scripts/quick_check_js_jax_move.py`）：node 加载真实 `web/sim.js` 的
  `resolveAxis`，与 JAX `_resolve_axis` 在 25920 组确定性场景（9 位置 × 8 步长 × 4 方向 ×
  10 墙型）逐位对比。
  - 同半径 0.45：**25920/25920 逐位一致**（公式同构：old/new_lead + span + 盒覆盖豁免）。
  - 保持旧半径 0.3 会差 **5560/25920（21.5%）**——贴墙停位整体偏移。
  - JS 步长范围（≤0.63）16200 用例全一致；JAX 大步长（1.588）由 MAX_SWEEP 全扫防穿墙，
    2 格厚墙拦截通过（JS 两点检查在 JS 步长下同样拦截，无分歧场景）。
- **顺带修复的扫描方向分歧**：贴地图边（other+rad 恰等于 H/W 越界）时 span 全判堵，
  JS/torch 是"终点侧优先"（`firstLead = sgn>0 ? hi : lo`），JAX 原为起点侧优先 →
  两 lead 同堵时停位差 1 格。已把 JAX `_resolve_axis` 扫描改为终点侧优先（对齐
  JS/torch），跨格中间砖由中心路径硬约束兜底，穿墙防护不回退。
- **改动**：`jax_env.py RADIUS = 0.45`、`sim/config.py radius = 0.45`（默认值）、
  `tests/test_rules.py` 贴墙停位断言 4.30→4.45（半径变大的正确贴停）。四套 JAX 回归
  全绿（obs_move 55 项含防穿炮）+ torch 91 passed。
- **注意**：JS `stepLen=0.3` vs JAX `STEP=0.756` 的速度模型差异仍是待决策项（见 §5），
  半径对齐不涉及速度。

## 9. 穿炮防护双端对齐（2026-08-20，JS+JAX+torch）
- **用户场景（三格穿炮）**：格1放泡 → 左移到格0（盒仍蹭格1）→ 格0放泡 → 右移回格1
  （中心扫过有泡的格1）→ 穿到格2。旧行为：JS/torch 盒覆盖豁免用 `floor+闭区间`，
  盒右/下缘**恰好贴格边界**（y+R=整数，如 x=0.6 时 ceil(0.6+0.45)=2）时误判"压着"
  泡格 → 放行 → 穿炮（sim.js 注释曾称"Feature 保留"）。
- **对拍抓到（quick_check_js_jax_move.py 补整数边界坐标后）**：JS `impassable` 是
  `ceil + 严格小于`（左闭右开：`row>=floor(y-R) && row<ceil(y+R)`），JAX/torch 是
  `floor + <=`（闭区间）→ 恰贴边界时不一致，对拍 1364/92480 失败。
- **修复**：JAX `_impassable_pair`、torch `_impassable`/`_impassable_pair`/
  `_impassable_pair_batch` 全部改为 `ceil + 严格小于`（与 JS sim.js 一致）。
- **中心路径硬约束**：JAX `_move_player` 已有（中心扫过格必须可通行，起点格豁免，
  obs_move 55 项防穿炮验证）；**Web 端本次补上，两条移动路径都改**：
  - `web/sim.js` `Sim.step` 移动段（tick 级，训练/测试用）
  - `web/main.js` `frameMove`（浏览器实际游戏每帧移动路径——用户复现的穿炮
    走的就是这条：60Hz 帧移动 + 10Hz tick 放泡，放泡后帧级 resolveAxis 盒覆盖
    豁免放行泡格 → 穿炮）
  两处都加同样逻辑：先 resolveAxis 再查中心路径，含起点格豁免 + 逐轴扫 lo..hi。
  放泡后能离开泡格（起点格豁免），但不能踩回泡格中心 / 穿过泡格。
- **复现确认**：按用户操作（每个 tick 左移+按住放泡，泡放出来立刻右转）逐帧验证，
  修复前 x 一路穿到格2；修复后右移卡在 0.55（格0），穿不回格1。无泡/贴墙移动
  不受影响（空场 5.04 格/秒 = speed 3.0 × open 初始 spdG 1.68，贴墙停 7.55）。
- **验证**：quick_check_js_jax_move.py 全量 92480 用例 → JS 可达步长（≤0.63）
  **0 不一致**，大步长（JAX 特有，JS 步长达不到）由 MAX_SWEEP 全扫防穿墙 4/4；
  新增 quick_check_anti_tunnel.py 双端穿炮对拍 PASS（放泡能离/踩回被拦/无泡正常）；
  JAX 四套回归全绿；torch 91 passed；JS test_levels.js 全过。
- **注意**：穿炮防护是**中心路径硬约束**，盒覆盖豁免（盒压泡格擦边走）仍是合法
  玩法（贴泡滑动不穿中心），两端一致。
