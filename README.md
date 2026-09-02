# qqt-gpu-sim — 泡泡堂风格 1v1 格斗：GPU 批量模拟器 + 自博弈 PPO

一句话：**写一个全张量化炸弹人模拟器（GPU 一次跑几千局），用 PPO 自博弈训出能打赢手写寻路 AI 的模型。** 方案从 torch 时代的 CNN/MLP 课程化训练起步，08 月中旬用 JAX 重写全链路并换成 7.5M 参数 ViT，**目前已迭代到第三次大训练（48 卡 × 48 小时，正在进行）**。

---

## 🎮 试玩（浏览器版）

**▶️ [点这里在线玩 QQT 格斗](https://wukangmh2022-cmyk.github.io/qqt-gpu-sim/)**

原版 `res/` 素材渲染（角色精灵/炸弹/爆炸/场景皮肤/音效），方向键/WASD 移动、空格放泡、支持推箱。模型下拉可选：

- **ViTModel2 系列**（JAX ViT，按累计训练步数命名）：`1.1B / 7.5B / 22.6B / 31.9B`（第二次大训练的快照）；
- **Pre-Train Test**（Patch3 ViT 7.5M）：第三次大训练前的试水模型（双卡 30 分钟预训练）；
- torch 时代遗产模型（`duel_course / duel_cnn / duel_nobc` 等）；
- 规则 AI（Hunter 等）。

页面含实时 AI 胜率对峙条（Win Probability Gauge）、录像回放与 60FPS 视频导出。

- 本仓库已启用 GitHub Pages（`main@/` 分支直发 `web/`），**push 即自动上线**；
- 本地启动：`bash scripts/serve_web.sh [端口]`（默认 8080，自动增量导出新模型 + 开服）；
- 手动工具链：`.venv/bin/python deploy/export_ckpt.py --verify`（ckpt→web 权重，含前向自检）→ `git add web && git push`；
- 引擎是 `sim/torch_sim.py` 的纯 JS 标量移植，与 Python 参考实现逐元素对拍一致（60 随机状态 maxdiff < 1e-7）。

---

## 实验历程（按时间）

### 阶段一：Torch 批量模拟器 + CNN/LSTM/MLP 课程化 PPO（2026-08-06 ~ 08-15）

**方案**：`sim/torch_sim.py` 纯 PyTorch 全张量化批量对战环境（5632 env 一批，无逐环境循环），网络 MLP 345K（CNN 281K 因 DCU 小卷积慢而落选；LSTM 试验线并存），PPO + **课程化对手管线**：规则 bot 启蒙（random/greedy/astar）→ 固定陪练锚点（rw8/5x2/5x3/cnn 冻结档，ELO 绝对锚定）→ 模型池自博弈。奖励用 71 局人类录像逐 tick 校准；修复了后期"钟摆效应"（熵下限 0.03 破确定性对称均衡）。

**成绩（DCU 实测，corridor 70%，256~512 局/组）**：

| 对战 | v1 纯自博弈（400M 步） | **v2 课程化（230M 步）** |
|---|---|---|
| vs 手写寻路 AI（astar） | 24%（大败） | **85~88%** |
| vs rw8（420M，上一代最强） | 32%（输） | **74%** |
| vs cnn（ring 系最强） | 18.8%（输） | **44.9%**（转平） |

**核心教训**：纯自博弈从近零起步只打"上一版自己"→ ELO 自指学不到强者打法；课程化（规则 bot + 固定锚点 + 池子）样本效率高一个量级。

**副产品**：引擎/算子层优化到 **DCU 36~41k sps**；Ascend 910B 上 Triton 化 danger_map 后峰值 **22.2 万 sps**（N=65536 物理上限）；08-12 浏览器版上线 GitHub Pages（JS 移植 + 原版素材）。

### 阶段二：JAX 重写 + 原版 241 关卡 + 多卡 LSGD（2026-08-17 ~ 08-21）

- **JAX 环境重写**（`jax_bomb/`）：与 torch/JS **逐位对拍一致**（修 5 处环境差异；PPO NaN 根因 = 熵项 0×(−inf)，p>0 门控修复）；obs 13→**14 通道**（ch13=可推箱），实现推箱玩法（JAX↔Web 等效，4 项 quick check 全过）；
- **原版关卡**：QQ 堂原版地图/素材导入工具链接入 **241 张原版关卡**（`levels.json`，含出生点对/可推箱/宝箱率数据）；
- **架构定稿**（2026-08-19 拍板）：transformer(ViT) `embed 392 / depth 4 / patch 4 / heads 4 / FF×4` ≈ **7.46M 参数**；单卡实测 21.8K sps（48.1s/iter）；
- **多卡 LSGD**：Local SGD 有损同步（每 K 个 minibatch 平均一次参数，通信量 1/K），跨机可扩展；`launch_8gpu.sh`（单机 8 卡）与 `launch_10nodes.sh`（10 机×2 卡 SSH 编排）落地；
- 教师蒸馏链（torch teacher → JAX student）打通，作为中途验证手段。

### 阶段三：ViT 自博弈三次大训练

#### 第一次大训练（2026-08-20 ~ 21，试水）：8 卡 · 500 iter · 4.2B 步

- 配置：crate 奖励 0.5（唯一塑形），无探索/无炸墙/无课程；155k sps，54s/iter；
- headless 行为：放炮 16/局、炸墙 7.1/局、吃箱 3.3/局（it500）——**进化方向健康但极慢**；探索率全线 3-10%，交叉对打 400-1800 tick **100% 平局**；
- 结论（驱动第二次设计）：出生点 3 格死锁（冷启动局部最优）；crate 奖励给的是"吃到"不是"炸墙"，链路型奖励学不会；4.2B 仅 260B 的 1.6%；
- **最大教训：退火窗口拍脑袋会翻车**——crate 退火 500M 步 ≈ 1 小时归零，模型还没学会。教训固化为"退火必须 > 学习周期或完全动态化"。

#### 第二次大训练（2026-08-21 ~ 08-27）：260B 完整配置

- 配置定稿（`docs/vit_train_log.md` §2）：`--fresh` 从头，三路稠密塑形统一乘动态退火 **α = max(0,1−gs/30B) × max(0,1−tanh(1.2×击杀率))**（击杀率上来塑形自动归零，只剩纯胜负）：
  - crate 吃箱 0.5（成长链 bootstrap）
  - 探索 novelty 0.01（本局首达格 +0.01，治踱步）
  - 炸墙 0.05（每炸一砖双方 +0.025，即时正反馈破死锁）
- 主跑道 8 卡（16384 envs，8.39M 步/iter，ITERS=31000 ≈ 260B）；10 机×2 卡 SSH 编排同期就绪；
- 08-26 起以 10 机×2 卡从 it889（7.46B 步）**热启动 48 小时续训**（`launch_warmstart_889_48h.sh`，`--reward-anneal-step-offset` 继承退火进度）；
- 进度：快照拉到 it1504 ≈ 12.6B 步，web 权重按累计步数命名导出（37 档，后精简为 4 档：`1.1B/7.5B/22.6B/31.9B`，最高档 ≈ 260B 的 12%）。

#### 第三次大训练（2026-08-29 启动，**进行中**）：48 卡 · 48 小时

**Pre-Train Test 试水**（双卡 30 分钟预训练，Patch3 ViT 7.5M）先验证 P0 全链路（新奖励/新价值头/课程门禁端到端跑通），其 ONNX 已部署进 web；随后 **24 机 × 2 卡 = 48 卡主训练启动，平台配额 48 小时，正在跑**。

驱动本轮的 P0 改造（commit `7eaa997` 及当天跟进，设计动机详见提交说明与 `docs/vit_train_log.md`）：

| 改造 | 内容 | 为什么 |
|---|---|---|
| **零和生命演进奖励** | 废除 ±1.5 掉血/±10 击杀/0.001 步罚/超时血差全部人工项；`r = (造成−受到)/5`，逐 tick 严格零和，终局回报 = 最终血差/5 ∈ [−1,1] | Bitter Lesson：塑形信号全部退场，不可刷分、天然无偏；回报尺度统一且有界 |
| **HL-Gauss 分布式价值头** | 标量 critic → 128 桶分类头（[−1,1]，σ=0.04，交叉熵） | 自博弈回报强双峰，分布回归梯度信号远好于 MSE；γ=1.0 下回报精确落在 [−1,1]，桶范围零截断 |
| **γ 0.99 → 1.0** | 无折扣 GAE | 配合有界零和回报，credit assignment 不再被折扣压缩 |
| **Actor Top-25% 优势过滤** | 每 minibatch 前半 = \|A\| 前 25% 高信噪比决策帧；**后半 = Critic 均匀无偏抽样** | 大量无事件帧优势近零是噪声；Actor 只吃高信号帧，Critic 保持全分布无偏（方案 1，`2087fae`） |
| **优势标准化** | 整批 (A−μ)/σ | 奖励尺度从 ±10 塌到 ±1 后，策略/价值/熵三项损失重新平衡 |
| **Patch 4 → 3** | 25 patch token（13×15 非方形独立高宽切块 + 补零，`932bafd`） | 3×3 格/token 对墙群走廊分辨率更合适，attention 开销仍便宜 |
| **空间课程 + 动态胜率门禁** | 出生点对按曼哈顿距离分阶段放开；晋级看"vs 阶段起点冻结参数胜率" | 239/241 张原图出生点被隔开——第一次大训练的死锁改用**塑形环境**而非塑形奖励来解决 |
| **Tick 级先手对称** | 放泡/推箱/宝箱争抢按 tick 奇偶轮换优先级 + 出生点 50/50 翻转 | 旧版 P0 恒定先结算的系统性偏置清零 |
| **EMA 0.999** | 参数指数滑动平均随 ckpt 快照 | 平稳快照供评估/导出挑选 |

当天评审跟进（把"过滤耦合 Critic / 门禁统计"等风险当场修掉）：Actor/Critic 损失解耦（`c95c862`）→ 方案 1 定稿（Critic 全分布 + Actor Top25% + 优势标准化，吞吐恢复 3.35 万 sps，`2087fae`）；课程升级为**累积包含图池**（Stage N 包含之前全部图，杜绝灾难性遗忘）+ 阶梯胜率门禁 85/80/75/65%（`c65f187`）；门禁严格执行、步数兜底不抢晋级（`d94cacd`）；加入 **Stage 0 纯空场景道场**构成 5 阶段渐进（`929145d`）；`EPOCHS=1` 对齐 Generals.io 分布式标准（`17bf2d8`）；`--init-params` 迭代式热启动支持（`9b7b44d`）。

---

## 第三次大训练：当前运行配置

| 项 | 值 |
|---|---|
| 拓扑 | 24 机 × 2 卡 = 48 副本（`deploy_10node/launch_24nodes.sh`，LSGD 跨机 pmap） |
| 模型 | transformer embed 392 / depth 4 / **patch 3** / heads 4 / FF×4 ≈ 7.5M |
| 负载 | 32768 envs（全局）→ 每卡 682（48 卡自动取整 32736）× 256 steps，epochs 1 |
| 奖励 | 零和生命演进 `r=(造成−受到)/5`，**无任何塑形项** |
| 价值头 | HL-Gauss 128 桶，[−1,1]，σ=0.04，交叉熵 |
| PPO | γ=1.0，λ=0.95，clip 0.2，vf 0.5，ent 0.01，Adam 3e-4 恒定 |
| 更新 | 每 iter 256 次 minibatch 更新：前半 Actor（Top-25% \|A\| 帧）+ 后半 Critic（均匀无偏），优势整批标准化 |
| 分布式 | LSGD K=256 / mode=param（fp32 全量参数 pmean，约每 iter 一次同步）；跨副本参数摘要 all_gather 校验 |
| 课程 | 5 阶段累积图池 [1, 22, 93, 145, 241] 张（Stage0 道场空景 → Stage4 全图 241 张）；步数兜底 [0.5%, 2%, 6%, 18%]；阶梯胜率门禁 [85%, 80%, 75%, 65%]（每 50 iter vs 阶段起点冻结参数，阶段内 ≥50 iter 方可晋级） |
| 对称性 | tick 奇偶轮换先手 + 出生点 50/50 翻转 |
| 存档 | `ckpt/` 每 30 分钟（重跑自动接续）；rank0 `params_it*.pkl` + **EMA(0.999)** 快照每 30 分钟 |
| 总量口径 | 15500 iter × 16.76M 步/iter ≈ 260B 全局步 |

监控：`bash deploy_10node/watch_24nodes.sh deploy_10node/nodes_24x2.txt`（60s 刷新各 rank 日志/磁盘/卡死检测）；快照回拉 `bash deploy_10node/pull_ckpt_local.sh deploy_10node/nodes_24x2.txt`。

---

## 做法（现行 JAX 管线速览）

- **模拟器**（`jax_bomb/jax_env.py`）：13×15 网格、10Hz、双玩家 5+2 双头动作；`lax.scan` 全张量 rollout，auto-reset 就地开新局；危险图/推箱/宝箱/引信连锁全部纯张量；与 Web JS 编码逐位对拍（`quick_check_js_jax_*.py`）。
- **网络**（`jax_bomb/jax_net.py`）：ViT 式 patch token + 全局状态向量作 state token（双序列输入），bf16 计算 / fp32 输出；策略头（move 5 × bomb 2，非法动作 −inf 掩码）+ HL-Gauss 分类价值头。
- **训练**（`jax_bomb/multicard_train.py`）：`pmap` 跨卡跨机（`jax.distributed.initialize` + RCCL），rollout→GAE→minibatch PPO→LSGD 周期参数同步；课程/门禁/退火在训练循环里热切换（同 shape 不重编译）。
- **torch 时代遗产**（`sim/` `train/` `play/`）：课程化 PPO + ELO 模型池 + 规则 bot 全套仍在，可复跑（见下方快速上手）。

**在哪儿训练**：

| 环境 | 用途 | 实测 |
|---|---|---|
| **SCNet DCU 集群（DTK 26.04）** | **JAX 正式训练**（第一次 8 卡 → 第二次 2×8/10 机 → **现在 24 机×2 卡**） | 第二次 155k sps/8 卡；第三次见 `watch_24nodes` 日志 |
| DCU 单机（torch 后端） | 阶段一训练/回归 | 36~41k sps（5632 env × 128） |
| BW-1（SCNet 910B） | torch 时代正式训练 | 249k sps（N=16384），见 `docs/bw1_notes.md` |
| 本地 MPS（macOS） | 开发/对拍/验收 | ~2.2k sps |

---

## 快速上手

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
pytest tests -q                                    # 规则/训练侧/parity 测试

# ── JAX 现役：多卡部署（第三次大训练同款）──
bash deploy_10node/launch_24nodes.sh deploy_10node/nodes_24x2.txt --deploy  # 首次
bash deploy_10node/launch_24nodes.sh deploy_10node/nodes_24x2.txt          # 续跑（自动接续断点）
bash deploy_10node/watch_24nodes.sh deploy_10node/nodes_24x2.txt           # 监控

# ── JAX 单机 8 卡（第一/二次大训练同款）──
bash deploy_10node/launch_8gpu.sh

# ── JAX 本地小规模（单卡 1.5M 参数 MLP 调试/对拍）──
python3 -m jax_bomb.jax_train --num-envs 2048 --num-steps 256 --minibatch 2048 --iters 5

# ── 模型导出 + 无头评估 ──
.venv/bin/python deploy/export_jax_ckpt.py --verify   # ckpt → web JSON
node scripts/headless_test.js --opp self,cross --maps 2 --ep 2
bash scripts/serve_web.sh                             # 浏览器版试玩

# ── torch 时代（阶段一，仍可复跑）──
python -m train.train --backend torch --device cuda --arch mlp --single-stage \
  --map-mode corridor --open-fraction 0.3 --total-steps 1_600_000_000 \
  --warmup-steps 150_000_000 --fixed-opp-prob 0.4 --bot-opponents astar,greedy \
  --time-budget 43200 --explore-anneal --bc-data recordings/ --bc-coef 0.3
python scripts/duel_arena.py --ckpt ckpt/duel_course_*.pt --map-mode corridor  # 对战验收
```

---

## 目录

```
jax_bomb/      现役 JAX 训练栈：jax_env(模拟器) / jax_net(ViT+HL-Gauss) /
               jax_train(rollout/PPO/LSGD) / multicard_train(多卡主循环) /
               levels(241 关卡+出生点对课程) / train_real(长训入口)
deploy_10node/ 多卡部署：launch_24nodes(48卡现役) / launch_8gpu / launch_10nodes /
               launch_warmstart_889_48h(热启动) / watch_* / pull_ckpt_local /
               nodes_24x2.example.txt(节点清单模板)
web/           浏览器版（sim.js 引擎 + ViT/MLP 权重 + 原版素材），Pages 直发
               assets/maps/levels.json(241 关卡) + curriculum.json(5 阶段课程)
sim/ train/    torch 时代遗产：批量模拟器 / CNN·MLP·LSTM 课程化 PPO / ELO 模型池
play/          对局核心（duel CLI / 录像回放）
deploy/        ckpt→web 导出（export_ckpt/export_jax_ckpt/export_jax_onnx）
scripts/       headless_test.js(无头评估) / analyze_maps.py(地图统计/课程) /
               quick_check_*(JS↔JAX 对拍) / duel_arena.py(对战矩阵)
docs/          vit_train_log.md(ViT 三轮训练记录) / multicard_lossy_sync.md(LSGD 设计) /
               bw1_notes.md / performance.md
tests/         规则/训练侧/parity 测试
RULES.md       规则唯一权威定义（torch/JAX/JS 三端以此为准）
```

---

## 经验教训（跨阶段沉淀）

- **纯自博弈 ELO 自指**（阶段一）：对手只有"上一版自己"学不到强者打法 → 课程化/课程环境是样本效率的关键。
- **退火窗口拍脑袋会翻车**（第一次大训练）：塑形必须在行为成形之后才能撤，固定窗口本质是赌博 → 动态化（击杀率驱动）是唯一稳解。
- **链路型奖励学不会**（第一次大训练）：放炮→炸墙→吃到相隔太远，credit assignment 断链 → 给中间环节即时塑形，或干脆塑形环境（空间课程）。
- **位置先手偏置要分清真假**：self 对打 P0 胜率 90% 是"镜像同步"评估假象（异权重对打无偏置）；但 sim 里 P0 恒定先结算的系统性偏置是真的——第三次大训练用 tick 轮换 + 出生点翻转根治。
- **PPO 负 loss 不是 bug**：收敛后 −0.006~−0.001 是熵项正常形态，NaN/Inf 才停训。
- **规则 bot 当对照实验极好用**：行为异常先跑 `--opp-bot idle` / headless 对照，别急着怪训练。
- **JAX 静默 clamp 坑**：`push_t.at[扁平索引]` 作用在二维数组上读对写错——`.at` 写入一律 2D 索引。

> 更完整的逐轮记录：**[docs/vit_train_log.md](docs/vit_train_log.md)**（ViT 三轮参数/坑/监控 checklist）、**[docs/multicard_lossy_sync.md](docs/multicard_lossy_sync.md)**（LSGD 与生产配置决策）、**[docs/performance.md](docs/performance.md)**（引擎/算子优化）。
