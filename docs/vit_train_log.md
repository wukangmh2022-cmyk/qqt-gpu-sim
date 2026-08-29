# ViT 模型专项训练记录

> QQ堂 13×15 标准关卡 · transformer(ViT) 自对弈 PPO · DCU 8 卡
> 本文档沉淀：两轮迭代的参数设置与原因、踩过的坑、无头评估方法、监控 checklist。
> 配套代码：`jax_bomb/`（训练）`scripts/headless_test.js`（无头评估）`scripts/analyze_maps.py`（地图统计/课程）。

## 0. 模型与训练基线（不变项）

- 架构：`--arch transformer --embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4` ≈ 7,461,336 参数
- 观测：14×13×15（**14 通道** JAX 视角，2026-08-19 起 ch13=可推箱；第一轮为
  13 通道，旧 ckpt 不兼容）+ 24 维 state token（`encodeObsJAX`/`encodeStateJAX`）
- 环境：241 张 QQ堂标准图（13×15），`levels.json`；`--num-envs 16384 --num-steps 256 --minibatch 16384 --epochs 2`
- 步数口径：8 卡 16384 envs → 8.39M 步/iter；全局 260B 步是总训练量（论文 Average Joe 同量级）
- PPO：`--gamma 0.995 --lam 0.95 --clip-eps 0.2 --vf-coef 0.5 --ent-coef 0.01`
- 奖励常量：`HIT_REWARD 1.5 / STEP_PENALTY 0.001 / WIN_BONUS 10`（jax_train.py）
- 稠密奖励口径（jax_train.py collect_rollout）：掉血 -1.5 = 对方命中 +1.5（**守恒**），每 tick 步罚 -0.001，终局击杀 ±10，超时按血差 × EXPLORE_COEF

## 1. 第一轮迭代（试水，2026-08-20 ~ 21）：500 iter / 4.2B 步

### 1.1 参数
- 无探索/无炸墙塑形（只有 crate 奖励 0.5，退火 30B 步——本轮中途把原 500M 退火改为 30B）
- `train_8gpu_20260820_180405.log`：iter 500/500，155,437 sps，53.97s/iter，loss 收敛 ~-0.007（负值是 PPO 正常形态，见 §3.1）

### 1.2 结果（headless 复测，见 §4 方法）
| ckpt | 探索率 | 放炮/局 | 炸墙/局 | 吃箱/局 | 隔离图放炮 |
|---|---|---|---|---|---|
| it68 | 6% | 6 | 1.3 | 1.4 | **0** |
| it102 | 4% | **0** | **0** | 0.1 | **0** |
| it204 | 8% | 9 | 3.8 | 2.3 | 3 |
| it340 | 8% | 11 | 4.9 | 2.3 | 4 |
| it500 | **10%** | **16** | **7.1** | **3.3** | **8** |

- 进化方向健康（放炮/炸墙/吃箱持续涨），但**探索率全线 3-10%**（走不到 20 格/195）
- 交叉对打（ckpt vs ckpt）**100% 平局**：400-1800 tick 内打不死对方，命中率仅 0.2-0.9/局
- 空场景对打能决出胜负：68 vs 500 = 20%/80%；500 vs 旧 MLP(5.95B) = 33%/67% —— 进化存在但极慢
- vs Hunter（规则 AI）：全部 ≤11% 胜率（Hunter 纸面太强，不适合做区分度评估基准，只做对抗压力参考）

### 1.3 结论（驱动第二轮设计）
1. **踱步是普遍现象**（不是个别地图）：出生点 3 格处放炮 0/局 → 冷启动死锁（见 §3.3）
2. **炸墙没学会**：crate 奖励给的是"吃到箱子"（+0.5），不是"炸墙"（见 §3.4）
3. **训练量不足**：4.2B 步 = 260B 的 1.6%，进化方向对，速度慢
4. **it102 出现能力退化**（放炮 6→0）：早期训练不稳定，需监控（见 §3.6）
5. **需要**：探索奖励 + 炸墙奖励 + 课程 + 动态退火（第二轮全部落地）

## 2. 第二轮迭代（260B 完整训练，设计定稿）

### 2.1 决定：从头训（--fresh），不接续 ckpt_00000500
原因：
- 配置大改（+探索 +炸墙 +课程），接续有分布偏移残留
- 4.2B 步只占 260B 的 1.6%，"保留基础"收益小（基础链条 it204 后才成形，从头+塑形更快）
- cfg 校验对新字段不兼容（旧 ckpt 无 explore/brick 字段 → 校验拒绝），接续需放宽有风险
- 论文路径（Average Joe）就是从头 + 课程 + 260B

### 2.2 奖励塑形（统一乘动态退火 α）
| 塑形 | 参数 | 为什么 |
|---|---|---|
| crate 吃箱 | `--crate-reward-coef 0.5` | 成长链条 bootstrap（已有） |
| 探索 novelty | `--explore-reward-coef 0.01` | 每 tick 新格 +0.01（visited 掩码，done 清零，单局封顶 0.01×195≈1.95 << 胜负 10）；治踱步 |
| **炸墙** | `--brick-reward-coef 0.05` | 每炸一块砖双方各 +0.025——crate 链路（炸→掷爆率→吃到）太长太弱，PPO 学不会；给"炸墙"即时正反馈打破死锁 |

### 2.3 动态退火（代替固定步数拍脑袋）
```
α = α_fix × α_dyn = max(0, 1 − gs/30B) × max(0, 1 − tanh(k·每局击杀率))
```
- `--reward-anneal-k 1.2`（Pommerman 论文 α=1-tanh(k·x) 同款）
- x = 训练内每局击杀率（collect_rollout 的 kills 统计，日志 `kill=` 列）
- **击杀率上来（会打架了）→ 塑形自动归零 → 只剩纯胜负**；α_fix 30B 兜底（防击杀率长期停滞）
- 日志新增 `kill=X.XXX α=0.XX`，监控退火进度

### 2.4 课程：**最终决定不启用**（260B 全图均匀采样）
地图统计（scripts/analyze_maps.py）：**239/241 张真实图出生点被墙砖完全隔开**（无一张"出生点可通"），room（出生点房间大小）1-4 为主，21 张符合 Pommerman 式起点（room=4 即 2×2 房间 + 出生点距离≤4）。曾设计 S1(21张,<1%) → S2(71张,1-4%) → S3(52张,4-15%) → S4(241张,15-100%) 的步数门控课程，**开训前砍掉**。理由：

1. **论文（Average Joe/Generals.io）的课程职能已被我们的稠密塑形接管**。复核论文代码：`ppo.py:227 reward_fn = win_lose_reward` 全程硬编码（YAML 里早期 stage 的 `composite_reward` 是死配置）——纯胜负下远距图没有任何梯度，课程（按 vs random 胜率 ≥0.6 晋级的 spawn-distance 分档）是它**唯一的早期梯度来源**。我们直接加了 explore/brick/crate 三路稠密奖励，每张图从第 0 步就有梯度，这个缺口不存在。
2. **第一轮实证：bootstrapping 不需要课程保险**。第一轮（5B，无塑形，crate 早期退火）出来的模型在大多数非联通图上已自主放炮——放炮开路从稀疏奖励就能涌现，S1"保证早期会炸墙"的保险多余。
3. **口袋图（出生 2-3 格）需要最大暴露量而非延迟出场**。原课程把它们压到全局步 15% 后；它们恰是最难的（推箱开格），uniform 给它们 100% 训练时长。
4. 260B 量级下课程的前期效率收益占比可忽略，还引入阶段切换重编译。

保留物：`web/assets/maps/curriculum.json` 与 `multicard_train --curriculum-json` 实现完整可用，要启用时 `CURRICULUM_JSON=web/assets/maps/curriculum.json` 环境变量传回即可（launch 脚本默认空）。

**与论文有意偏离的另两处**（都是因为论文纯胜负、我们有塑形）：TAF（adv_top_frac 0.25，论文用来对抗优势稀疏——我们的塑形让优势不再稀疏，先不用，学习停滞再启用）；熵退火（论文 0.01→0.001/5000iter——我们固定 0.01，260B 自博弈防策略塌缩更稳，后期行为发散再考虑）。

### 2.5 启动命令（平台双任务脚本，2×8 卡）
本轮实际采用平台「训练任务启动脚本」方式，而不是本地 SSH 编排：将
`scripts/scnet_model_train_2x8.sh` 的全文分别粘贴到两个 8 卡训练任务中。
两个任务使用同一份 `upload_2x8gpu/qqt_upload/` 上传代码；平台需要为两个实例提供
同一分布式网络和 `WORLD_SIZE=2`、`RANK=0/1`、`MASTER_ADDR`、`MASTER_PORT`。
先启动 rank0，再启动 rank1；rank0 会阻塞在 `jax.distributed.initialize()` 等待 rank1，
不是各自跑单机训练。脚本同时兼容平台只注入 `worker0/worker1` 的情况，会自动补齐变量。

脚本默认值：
| 项 | 默认 | 备注 |
|---|---|---|
| ITERS | 15500 | 2×8 卡、32768 env、256 steps：约 16.78M 步/iter，≈260B |
| fresh | `--fresh` | 本轮 obs 13→14 通道；接续时设 `FRESH_FLAG=""` |
| 地图权重 | `empty=0.05,功夫=0.1,比武=0.15` 其余均分 | 第一轮验证过的对抗基本功配比 |
| crate / explore / brick / k | 0.5+30B / 0.01+30B / 0.05 / 1.2 | §2.2–2.3 |
| 课程 | 关闭 | 全图均匀采样，§2.4 |

网页上传后的目录应包含 `jaxbomb.tgz`、`wheels/`、`setup_upload.sh` 和
`scnet_model_train_2x8.sh`。脚本会自动解包代码、检查 14 通道/推箱实现、检查本机
8 卡，再启动相同训练命令。两边的 checkpoint 和日志保存在各自账号的 `qqt-gpu-sim/`
目录，不能让两个账号共享同一个本地 checkpoint 目录。

### 2.6 推箱子玩法（2026-08-19 加入，JAX ↔ Web 等效实现）
数据：123/241 张图含可推箱（15 种元素：中国城酒罐 3004、雪地 2003/2012、
探险 10007/10008、比武 1006、功夫 9012、box 系列 9 个），全部是 **1×1 单格箱**
（levels.json 的 push_boxes，加载进 `LevelSample.pushable`）。

机制（对齐 web/sim.js:373-410，触发条件逐条等效）：
- **触发**：玩家存活 + 方向 ≠ IDLE + **前缘格**是可推箱。前缘格公式与 Web
  一致（dy>0 → `floor(y+R+EPS*8)`；dy<0 → `ceil(y-R)-1` ≡ Web 的
  `floor(y-R-EPS*8)`）；前缘出界 = 无箱（Web `pi` 越界 → bi=-1 同款）。
- **计时**：每 tick 推 +0.1s，累计 ≥PUSH_TIME(0.3s=3 tick) 当 tick 移一格。
  计时挂在**箱子**上（JAX 用 per-cell push_t，因箱子单格一一等价）：玩家走开
  计时**保留**（推 2 停 N 再推 1 → 照样移动）；被挡/被炸才清零。
- **目标格必须全空**：墙/砖/泡(fuse>0)/道具(crate)/其他箱 任一存在 → 推不动
  （该 tick 计时清零）。道具推不过去 ✓。
- **箱子占格 = brick**：挡玩家挡爆炸；本 tick 玩家顶箱不动，箱子移走下 tick
  跟进（blocked 在推箱段之后计算，天然实现）。
- **被炸整箱消失**：`pushable &= ~destroy` 且该格 push_t 清零。
- 双玩家按 P0→P1 顺序处理（与 Web for p in [0,1] 相同），同 tick 写入对后者可见。

观测（关键！否则 AI 根本"看不见"箱子，无法学会主动推）：
- **N_OBS_CH 13 → 14**，ch13 = pushable 二值通道。旧 ViT ckpt（13ch）在
  Web/headless 由模型 obs_shape[0] 自动走旧编码，不受影响。
- 全链路同步点：`web/sim.js encodeObsJAX(pid, C)` 参数化；
  `deploy/export_jax_ckpt.py` / `export_jax_onnx.py` 从 tok_w 反推通道数
  （tok_w: [patch²·C, embed]），不再硬编码 13。

验证（scripts/quick_check_push.py，4 项 ALL PASS + vmap/jit 冒烟）：
推动一格 / 目标格有砖推不动 / 爆炸清箱 / 中断保留计时。
JS↔JAX 观测逐位对拍（quick_check_js_jax_transformer.py）扩到 14 通道，
比赛02 ch13=26 格两侧一致。

实现坑：step() 里 scatter 写入曾用扁平索引 `push_t.at[pi]`（pi=r*W+c）作用在
(H,W) 二维数组上——JAX 静默 clamp 到末行，读对了写错了，测试才发现。写入一律
2D 索引 `.at[pr, pc]` / `.at[tr, tc]`。

### 2.7 本轮设计思路汇总（为什么是这套配置）

**第一轮复盘的修正理解**（驱动本轮的关键证据，修正了"全局死锁"的早期判断）：
- **空场景最像人的原因**：初始属性直给，格斗技能（走位/连炮/卡时间）可以从
  胜负信号**直接**归因，不经过成长链。
- **有砖图被成长链锁死**：不吃道具 → 永远基础属性 → 放炮再多也是"基础属性
  互啄"，胜负归因不到操作上 → 这些图的经验对格斗学习贡献极低（训练浪费）。
- **吃道具没学会的根因** = §3.2 的提前退火（500M）+ 无探索概念，不是 5B 训练
  量不够（5B 连 260B 的 2% 都不到，拿试水行为反推设计会过度矫正）。
- **但放炮开路在大多数非联通图上自己涌现了**（稀疏奖励下）→ 死锁是口袋图特有
  局部最优，bootstrapping 不需要课程保险。

**课程砍掉 vs 地图配比保留——两个维度，不矛盾**：
- 课程 = **时间维**（先训什么后训什么）。砍掉：稠密塑形让每张图第 0 步就有
  梯度，论文靠课程制造的"早期梯度"职能已不存在（§2.4）。
- 配比 = **空间维**（经验往哪积累），服务"1v1 对战强度"目标：空场景/比武/
  功夫这类少障碍图是连炮封锁、卡时间、走位基本功信号最干净的地方——基本功
  在干净环境学、到复杂图是泛化，反过来难得多。70% 经验仍在其他图，泛化暴露不丢。
- 新奖励不扭曲配比：采样权重决定**经验分布**（配比控制），奖励大小只决定哪些
  **行为**被强化（advantage 按批归一化）；塑形退火后永久信号是胜负，地图无关。

**动态退火的自锁结构**：有砖图上发生击杀的前提是成长、成长的前提是会吃 →
kill= 趴 0 则 α 恒 1、塑形一分不减（"没学会就撤火"在结构上不可能发生）；
打起来后 α 平滑让位；α_fix 30B 线性兜底防病态停滞（§2.3/§3.2）。

**本轮泛化预期**（验收基准）：
- 每张图具备底线三件事：会吃道具（有砖图 crates/局 > 0）、会开路（放炮率保持
  第一轮水平）、敢接敌（kill= 从 0 爬升）。
- 空场景水平不回退（5% 配比保留；新技能是超集不是替换）。
- 吃道具可能先以"单纯捡"形态出现（尚未与格斗关联）——正常脚手架阶段，α 退火
  自然交接，不干预。
- 判定节点与兜底方案见 §5.2（~iter 300 headless 分图型 crates/局；推动率中段查）。

## 3. 踩过的坑（按时间线）

### 3.1 负 loss 不是 bug（易误判）
分类任务 loss ≥ 0；PPO 的 loss = pol(advantage 加权，正 advantage 时负) + vf(≥0) − ent(恒负)。收敛后 loss 稳定在 -0.006~-0.001 是**熵项贡献的正常形态**。判别标准：NaN/Inf 才是问题；rew/explore/ep_len 健康即可。外行人（套监督学习思维）容易误报，勿停训。

### 3.2 退火窗口拍脑袋会翻车（第一轮最大教训）
原 crate 退火 500M 步（≈1 小时）→ 训练到 2.3B 步时塑形早已归零，模型还没学会。改为 30B 全覆盖 + 动态击杀率退火。**教训：退火窗口必须 > 预期学习周期，或完全动态化**。
机理补充：塑形必须在**行为成形之后**才能撤——行为没出现就撤火，纯胜负永远穿不透"吃→成长→赢"的长链，剩余 90% 训练量（4.5B）与吃道具彻底断开关联，不是训练量不够而是梯子被提前抽走。行为什么时候成形事先不可知 → 固定窗口本质是赌博 → 动态化（§2.3）是唯一稳解：kill= 趴 0 则 α 恒 1 一分不减；打起来了 α 平滑让位；30B 线性项兜底防病态停滞。

### 3.3 冷启动死锁：为什么出生点 3 格"一个炮都不敢放"
- 随机初始化确实会尝试放炮；但放炮即时期望回报为负（炸不到隔离的对方 + 可能自伤 -1.5 + 炸墙大概率无箱 0 奖励）
- PPO 快速收敛到"不放炮"（踱步 = 安全局部最优），ent_coef=0.01 太小不再随机尝试 → 永久困死
- **解法**：探索奖励（动起来有分）+ 炸墙奖励（破墙有分）——两个都是"即时正反馈"，打破死锁

### 3.4 crate 奖励 ≠ 炸墙奖励
`crate_coef` 只在"吃到箱子"（站在箱子上）给分。完整链路 放炮→引信 30 tick→炸→掷爆率(0.5-0.8)→走过去→吃到 → +0.5，任何一环断就 0 奖励，且被 -1.5 掉血淹没，PPO credit assignment 极差 → "炸墙根本没学会"。**教训：链路型奖励学不会时，给中间环节即时塑形**。

### 3.5 位置偏置（P0 强 P1 弱）是"镜像同步"假象，训练无需处理
- 现象：同权重自博弈 self 对打，P0 位置胜率 90-100% vs P1 0%
- 排查：观测/state 编码对称（encodeObsJAX/encodeStateJAX 按 pid 互换）；但 sim.step 放泡是 `for me in range(2)` P0 先（同格竞争先到先得）
- 机制：同权重 + 对称观测 + 对称出生点 → **镜像同步**（双方做镜像决策 → 总是同格 → 放泡竞争高频化 → P0 先手被放大）。实验证实：出生点不对称后 P0 胜率从 90% 掉到 40-60%
- 关键：**异权重对打（cross）完全无偏置**（340 在 P0=P1=20%）→ 模型没有视角缺陷；self 的 P0 优势是评估工具的构造产物
- **结论：训练无需交替/身份随机化**（训练数据出生点随机，对称状态罕见，先手影响弱）

### 3.6 it102 能力退化（需监控的早期不稳定）
self 对打中 it68→it102 放炮 6→0、炸墙 1.3→0（it204 才恢复）。原因未完全定位（疑似早期价值函数/GAE 不稳）。第二轮若再次出现"能力归零"需查 vf loss/advantage 分布。

### 3.7 Hunter 不适合做胜率基准
规则 Hunter（纯进攻寻路）纸面太强：强模型 MLP(elo 4537) vs Hunter 也仅 5.6% 胜率。**评估模型相对强度用 ckpt 互打（cross），Hunter 只做对抗压力下的行为参考**（自杀率/探索率）。

### 3.8 levels.py 课程权重两个 bug（本轮修复）
- 21 个 `1/21` 浮点累加略超 1 → 未指定图权重为**负** → `np.log(负)=nan` 毒化 categorical 采样（采样乱出图）。修复：`w_rest = max(0, ...)` clamp
- 全图都指定（S4 241 张）时 `(1-total)/(n-len(specs))` 除零。修复：`len(specs)==n` 直接返回

## 4. 无头评估方法（headless_test.js）

```bash
# 全部 transformer ckpt 自测（self）+ 相邻互打（cross）
node scripts/headless_test.js --opp self,cross --maps 2 --ep 2
# 指定模型对打（进化对比）
node scripts/headless_test.js --models params_it00000340,params_it00000500 \
  --pairs "0,1" --map-source empty_scene --ep 5
# 旧模型兼容（空场景 oldMode：13 宽 obs 自动触发）
node scripts/headless_test.js --models duel_nobc_5.95B --map-source empty_scene --opp hunter
# 位置偏置检查（--swap 交替位置）
node scripts/headless_test.js --models <m> --map-source empty_scene --opp self --swap --ep 10
# 出生点实验（--spawns 固定覆盖）
node scripts/headless_test.js --models <m> --map-source empty_scene --opp self --spawns "6,5;1,5" --ep 10
```
指标口径：探索率 = visited/195；自杀 = 被自己泡炸死（fuse==1 引爆覆盖死亡格）；命中 = 造成对方掉血；炸墙/吃箱 = 本局计数。
**重要**：self 对打的 P0 胜率无意义（镜像同步假象），看 cross（异权重）和 vs Hunter 的行为列。

## 5. 监控 checklist（每轮训练）

### 5.1 每 iter（日志行）
- `sps` ≈ 15 万量级（8 卡）；骤降 = 异常
- `loss` 负值正常（§3.1），NaN/Inf = 立即停
- `explore=X.XXX` 探索分/帧（应早期上升、随 α 下降）
- `kill=X.XXX` 每局击杀率（动态退火 x；应从 0 缓慢爬升）
- `α=0.XX` 塑形总退火系数（击杀率上来应下降；长期恒 1 = 击杀率停滞异常）
- `rew`、`ep_len`（~1700 tick 上限）

### 5.2 周期性
- ckpt 落盘（`~/qqt-v8/ckpt/ckpt_<iter>_r0.pkl`，整点）；ckpt_local 参数快照
- 无 Traceback/Error/Exception（grep 计数 = 0）
- **headless 行为锚点**（拉回 ckpt 跑 `scripts/headless_test.js`，判定塑形是否生效）：
  - ~iter 300（≈2.6B，全局步 1%）：**吃道具率**（crates/局）应从第一轮的 0 抬头——crate 0.5×30B 窗口生效的直接证据；放炮率应保持（第一轮已涌现）
  - 中段：含箱图**推动率**（推箱是否涌现；为 0 且吃道具正常 → 考虑推箱奖励兜底）
  - `kill=` 爬升 → `α` 跟随下降 = 动态退火正常；`α` 长期恒 1 且 kill 恒 0 → 检查是否卡口袋图
- **it102 退化复现监控**：50-150 iter 段留意放炮/炸墙是否归零（对比 headless）

### 5.3 里程碑评估（数据驱动决策）
- 每 ~500 iter：拉 `params_itNNNN.pkl` → 导出 JSON+ONNX（`deploy/export_jax_ckpt.py` / `export_jax_onnx.py` --verify）→ `quick_check_js_jax_transformer.py` 对拍 → headless 复测
- 对比指标：探索率（应 >10%）、炸墙/局（应 >7）、交叉对打击杀（应出现非平局）、vs MLP 胜率（应 >33%）
- 达标 → 下一轮退火 k 可调大（塑形更早退）/ 提前进 S4；不达标 → 查 α 是否过早归零 / 课程切换是否正常

### 5.4 第二轮训练启动前
- `--fresh` 从头（清 ckpt/ 或加 --fresh）
- 确认日志出现：`课程模式 ... 初始 Stage1（21 张）`、`关卡模式 ... 权重 ...`
- 首 iter 后确认 `kill=`/`α=`/`explore=` 列存在且数值合理

## 6. 工具清单
| 工具 | 用途 |
|---|---|
| `scripts/headless_test.js` | 无头对战评估（self/cross/hunter/swap/spawns/map-source/oldMode） |
| `scripts/analyze_maps.py` | 241 图开局形态统计 + `--curriculum` 生成课程文件 |
| `web/assets/maps/curriculum.json` | 课程 4 阶段 id 列表 + 阈值 |
| `deploy/export_jax_ckpt.py` / `export_jax_onnx.py` | ckpt → Web JSON/ONNX（--verify 对拍） |
| `scripts/quick_check_js_jax_transformer.py` | JS↔JAX 前向对拍 |
| `deploy_10node/launch_8gpu.sh` / `launch_10nodes.sh` | 训练部署（已内置第二轮全部参数） |
