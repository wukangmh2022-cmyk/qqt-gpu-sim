# 10 机 × 2 卡有损同步（Local SGD）部署包（v2）

## 背景（实测数据，详见 docs/multicard_lossy_sync.md）
- 单机 8 卡无损 = 304K sps（要审批）；10 账号 × 10 notebook = 20 卡是能自由支配的最大配置。
- 无损每 minibatch 同步 = 每迭代 1024 次全量 all-reduce ≈ 50GB/迭代/卡（20 卡）→ 不可行。
- RCCL 调优实测是死路（sysctl 不可用；BUFFSIZE/NTHREADS 反而慢 2×；接口本就是 eth0）。
- 唯一出路：Local SGD（每 K 个 minibatch 同步一次）。K=256 → 4 次同步/迭代 ≈ 370MB/节点 ≈
  1.5s → 效率损耗 ~7%（fp32）/ ~3.5%（+`--lsgd-bf16`）。
- **实测（2 notebook 跨机 A/B）**：baseline 132.5s/iter → param k=256 25.8s/iter（**5.1×**），
  通信 0.4s = 1.6% 损耗；30 iter 长训 loss 0.83→0.05 收敛健康，consistency 跨机 PASS。

## 三个模式（`jax_bomb/multicard_train.py` / `train_real.py`）
| 模式 | 命令 | 语义 | 内存限制 |
|---|---|---|---|
| 无损 | `--lsgd-k 0` | 每 minibatch pmean 梯度，逐位一致 | 无 |
| **param（推荐）** | `--lsgd-k K --lsgd-mode param` | 每 K 步平均参数，保持 1024 次更新/迭代 | 无（任意 K）|
| grad | `--lsgd-k K --lsgd-mode grad` | K 个 minibatch 拼大 batch 一次梯度同步（零漂移；K=1 与无损逐位一致）| K 受限（65536 样本护栏）|

- `--lsgd-bf16`：同步流量减半。`--lsgd-sync-state`（param）：连 Adam 动量平均（流量 ×3）。

## 生产配置（2026-08-19 拍板，launch 脚本已内置）
- **transformer：`--embed 392 --depth 4 --patch 4 --heads 4 --ff-factor 4` ≈ 7,461,336 参数**。
- `--num-envs 32768 --minibatch 32768` 是**全局**值，按 20 卡自动均分（32768//20=1638/卡，
  不能整除自动下调并打印）。`--iters 20000` ≈ 335B 步（130B 停检点在 ~7750 iter，约 3.5-4.5 天；
  中途手动停 = Ctrl-C/SIGTERM，断点续训自动接上）。
- ⚠️ `--ff-factor` 必须显式传 4（脚本已带）；shape float bug 已修（jax_net.py int() 包 6 处）。

## 标准化关卡（241 张 QQ堂地图）+ 宝箱语义（本版核心）
- 地图 13×15（宽 +2 格）；ViT patch 4×4=16 token 不变，参数 7,461,336 不变。
- `levels.json` 随部署包分发，`jax_bomb.train_real` auto-detect（`./levels.json`）找到即启用关卡模式：
  241 张图按权重混合（默认 `LEVEL_WEIGHTS="empty=0.05,功夫=0.1,比武=0.15"` =
  空场景 5% + 功夫主题 10% + 比武主题 15%，其余 70% 均分随机；语法支持 id/empty/主题名），每局随机抽关 +
  **确定性随机选两个不同出生点**，初始泡/威/速用关卡 `initial_stats`（同时写掉血 clamp 下限 lo），
  预置宝箱 `initial_crates` 逐格落地。**没有关卡文件 = 回退过程式生成**（torch 等价路径）。
- **宝箱语义已与 JS Web（web/sim.js）逐项对齐**（本版修正）：
  - 炸砖→生箱：在**炸砖瞬间**按本关 `crate_rate` 掷爆率（`<=0/缺失 → 1.0` 钳制，与 JS 同款）——
    近处炸开没箱、远处才有箱的随机性是真实存在的，AI 学到"箱子在哪"才可信。
  - 拾取：踩到**必升**（三属性均匀，+1 泡/威 或 +0.15 速），预置箱/掉血回收箱同规则。
  - 掉血回收（hit-attr-penalty=2，守恒撒箱）：`lax.cond` 跳过无掉血 tick 的 permutation；
  **回收箱只落纯粹地面**——墙/砖/灌木都不可落。
- **灌木丛（bush，野外关新特性）**：独立布尔层（25 关，与墙/砖零重叠），
  **可通行 + 可炸毁**，炸毁瞬间按本关 crate_rate 掷爆率出宝箱（与炸砖同规则）；
  obs 新增 **ch8 = 灌木**（N_OBS_CH=9）让 AI 学到"空地会炸出东西"。
- **预留位（后训练增强，当前默认值不参与玩法）**：`buffs`(3bit)/`debuffs`(2bit)/
  `items`(4槽×int6)/`gametype`(int4) 字段 + global_vec 13 维预留（G=11→24）；
  `pushable` 可推墙（推箱子关 231-239，必 ⊆ brick 即天然障碍物，加载断言）。
- **道具系统（与 Web sim.js 一致，2026-08-20）**：`crate` int8 编码 7 种道具——
  1/2/3 = 泡泡/威力/速度 +1 档，4/5/6 = 超级（+4 档），7 = 问号随机（预置宝箱/回收箱）。
  炸墙/灌木时：crate_rate 判定掉落 → super_fraction 判定超级 → 均匀定种类（与 Web
  同款三重随机）；成长 clamp 到**每关上限** `bombs_max/blast_max/speed_max`（levels.json
  新字段，过程式用全局 10/7/2.1）。obs 新增 ch9-12（泡/威/速 one-hot + 超级标志），
  N_OBS_CH 9→13，让 AI 区分道具种类。
- **穿墙修复（测试抓到）**：JAX 单 tick 位移 = STEP×spd 可达 1.59 格（spd 2.1），
  `_resolve_axis` 原只查两端前沿会把中间砖跳过 → 穿墙。已改为沿移动方向
  逐格扫描第一个障碍贴停（非跨格轨迹逐位不变）。注意：JAX 速度模型
  `SPEED=7.56`（0.756 格/tick×spd）是 torch（0.36）的 2.1 倍，手感更快，
  穿墙已修但速度值是否对齐 torch/JS 待定。
- 开箱成长 bootstrap 奖励：`CRATE_REWARD_COEF=0.5` 前期加，`CRATE_REWARD_ANNEAL=5e8`
  全局步（≈半小时）线性退火到 0（launch 已内置透传；不想用设 `CRATE_REWARD_COEF=0`）。
- **部署后验证**（qqt-gpu-sim 下，六个脚本全部 PASS 才开训）：
  ```
  python3 scripts/quick_check_obs_move.py        # 55 项：行走/输入 patch 格子信息 + 防穿炮
  python3 scripts/quick_check_levels.py levels.json   # 241 关：出生点/属性/宝箱/权重/碰撞
  python3 scripts/quick_check_crate_semantics.py levels.json  # 宝箱：炸砖爆率统计/拾取必升/钳制
  python3 scripts/quick_check_bush.py levels.json # 灌木：可通行/可炸/掉宝率/回收箱不落灌木/obs ch8
  python3 scripts/quick_check_js_jax_move.py     # JS↔JAX 碰撞半径 0.45 对拍（JS可达步长内逐位一致）
  python3 scripts/quick_check_anti_tunnel.py     # 双端穿炮防护对拍（放泡能离/踩回被拦/无泡正常）
  ```
  本地对照：`cd web && node test_levels.js`（JS 侧同关统计爆率 + 拾取必升）。
- **v8（2026-08-20）**：盒覆盖豁免对齐 JS（`ceil + 严格小于`，恰贴格边界不误判"压着"
  泡格）；中心路径硬约束 Web 端补齐（sim.js `Sim.step` + main.js `frameMove` 两条
  移动路径，防穿炮）；
  新增 js_jax_move（整数边界坐标）与 anti_tunnel（含帧级路径）对拍。含 v5-v7 全部
  内容（半径 0.45/终点侧优先/道具系统/灌木/穿墙修复/预留位）。
  包 md5 以部署时 `md5sum dcu_deploy_10node.tar.gz` 实际值为准（tar 含 mtime，重建会变）。

## 两套启动代码（本机编排，只需 expect + 密码）

### A. 十机二卡主战场（20 卡，Local SGD param K=256，260B 目标）
**nodes.txt**（第 1 行 = rank0；密码只用字母/数字，含 `$ [ ] "` 会炸）：
```
10326 ssh.zzai.scnet.cn <rank0密码>
10831 ssh.zzai.scnet.cn <rank1密码>
...
```
```
bash launch_10nodes.sh nodes.txt --deploy    # 首次：部署 + 推最新代码 + 自检 + 取 IP + 同步启动 + 健康检查
bash launch_10nodes.sh nodes.txt             # 之后：只推最新代码 + 自检 + 启动（自动接续断点）
bash watch_10nodes.sh nodes.txt              # 监控（60s 刷新各 rank 状态 + 磁盘余量）
bash pull_ckpt_local.sh nodes.txt            # 把 rank0 30min 参数快照拉回本地
```
- **每次 launch 都自动推送仓库根最新 jax_bomb 代码**（`--deploy` 只补 wheels/setup，
  代码始终以仓库当前为准，不依赖包里旧代码）。
- 部署包路径自动取：仓库根 `dcu_deploy_10node.tar.gz` > `/tmp/dcu_deploy/` 旧包。

### B. 单机八卡（8 卡 pmap DP，默认 LSGD_K=0 逐位一致；模型/关卡/奖励参数同主战场）
**nodes_8gpu.txt**（只读第 1 行）：
```
10832 ssh.zzai.scnet.cn <8卡机密码>
```
```
bash launch_8gpu.sh nodes_8gpu.txt --deploy  # 首次：完整包 + setup + 推代码 + 自检(卡数=8) + 启动
bash launch_8gpu.sh nodes_8gpu.txt           # 之后：只推代码 + 自检 + 启动（自动接续断点）
bash watch_8gpu.sh nodes_8gpu.txt            # 监控（60s 刷新 + 磁盘余量）
bash pull_ckpt_local.sh nodes_8gpu.txt       # 拉 rank0 断点快照
```
- 默认 `num-envs 16384`（8×2048）、`iters 500`（≈4.2B 步 ≈ 8h；长训用 ITERS= 覆盖）；
  同模型 `embed 392 depth 4 patch 4 heads 4 ff 4`，断点可跨 A/B 两套续训。
- `LSGD_K` 默认 0（单机机内通信快）；要模拟主战场行为可 `LSGD_K=256 bash launch_8gpu.sh ...`。

#### 首次 8 小时训练（CKPT 验证）操作步骤
```
# 1) 本地：写 8 卡机入口（一行：端口 主机 密码）
printf '10832 ssh.zzai.scnet.cn <8卡机密码>\n' > nodes_8gpu.txt

# 2) 首次部署 + 启动（自动：推包→setup→推最新代码→自检卡数=8→nohup 启动→30s 健康检查）
bash deploy_10node/launch_8gpu.sh nodes_8gpu.txt --deploy

# 3) 监控（另开终端）
bash deploy_10node/watch_8gpu.sh nodes_8gpu.txt

# 4) 断点/快照
bash deploy_10node/pull_ckpt_local.sh nodes_8gpu.txt   # 30 分钟参数快照拉回本地 ckpt_local/

# 5) 8 小时后收尾：进程会自动跑满 ITERS=500（≈8h）后优雅存盘退出；
#    要提前停：ssh 上去 kill -TERM <pid>（当前 iter 结束后存盘退出，下次 launch 自动接续）
```
- **8 小时口径**：steps/iter = 2×16384×256 = 8.39M；8 卡 ≈ 120k sps（单卡 21k
  实测外推×8 扣 pmap 同步）→ 8h ≈ 3.4B 步 ≈ 400 iter，默认 ITERS=500 有 ~20% 余量。
- 首 iter 含 jit 编译（几十分钟量级），sps 从 iter 2-3 起才是稳定值，判断吞吐看
  watch 里的 `avg` 列。

## 部署步骤（每台新 notebook，创建时设备数选 2 / 选 8）
1. 拷 `dcu_deploy_10node.tar.gz` 到 `/root/private_data/`，解压：
   `cd /root/private_data && tar xzf dcu_deploy_10node.tar.gz`
   （包里是 `dcu_deploy/` 目录；launch 也兼容旧版裸文件布局）
2. 一键初始化：`cd dcu_deploy && bash setup_notebook.sh`
   （解代码到 qqt-gpu-sim/、离线装 optax wheels `--no-deps`、DTK env + LD_PRELOAD 自检；
   幂等可重跑；卡数≠2 只警告不中断，launch 自检才会硬性拒绝）
3. 本地编排（见上方 A/B 两套）。

4. **launch 的自动把关**（任何一台不过就中止，不会让全部节点白等）：
   - 自检：`import jax/optax/jax_bomb` + `jax.local_device_count()==2`（十机）/ `==8`（单机八卡）
     + 代码版本（ppo_update_gradsync）。
   - 启动 30s 后健康检查：空日志或 Traceback → 报错中止（其余情况看 watch）。
   - 手动单台启动（平台训练任务用 scripts/scnet_model_train.sh，支持 LSGD_* 透传）：
   ```
   export WORLD_SIZE=10 RANK=<N> MASTER_ADDR=<rank0-172.31.x-IP> MASTER_PORT=29500
   export LSGD_K=256 LSGD_MODE=param CKPT_DIR=ckpt CKPT_EVERY=30
   export CKPT_LOCAL_DIR=ckpt_local CKPT_LOCAL_EVERY=30
   cd /root/private_data/qqt-gpu-sim
   python3 -m jax_bomb.train_real --arch transformer --embed 392 --depth 4 --patch 4 \
     --heads 4 --ff-factor 4 --num-envs 32768 --num-steps 256 --minibatch 32768 \
     --epochs 2 --iters 20000 --lsgd-k $LSGD_K --lsgd-mode $LSGD_MODE
   ```

## 落盘策略
- 每台：`ckpt/ckpt_<iter>_r<rank>.pkl`（每 30 分钟，断点续训；rank 各自 env states）。
  ≈110MB/30min/节点 → 一周约 35-40GB，watch 已带磁盘余量告警（<10G 标 ⚠）。
- **rank0**：额外 `ckpt_local/params_it*.pkl`（~25MB，30 分钟一次，供拉回本地/评估），
  用 `pull_ckpt_local.sh` 拉回本机 `ckpt_local/`。
- 校验：每迭代 `consistency PASS`（跨全部 20 replica 参数逐位一致）；启动打印
  `LSGD: k=… → N 次同步/迭代 ≈ XMB`。

## 环境坑速查
- notebook JAX 需 `source /opt/dtk/env.sh` + `LD_PRELOAD=libmpi.so`（setup 已含；launch 用 glob 取版本）。
- 平台容器代理让 rendezvous 超时 → 启动命令已 unset。
- 总卡数不能整除时 envs/minibatch 自动下调（multicard 内置）。
- RCCL 调优不要碰；稀疏化/量化不需要（K 旋钮已够 100×+ 降流量）。
- 重新 launch 前先停 watch_10nodes.sh（launch 会 rm -rf /tmp/ndrun，运行中的 watch 会失效）。
