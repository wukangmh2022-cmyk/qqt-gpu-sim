"""SCNet 超算互联网「模型训练」多卡 DP 入口（JAX + RCCL，跨实例走平台 RDMA）。

平台「模型训练」创建任务时自动注入环境变量（见文档 模型训练>最佳实践）：
  WORLD_SIZE   实例（容器）数
  RANK         本实例序号，0..WORLD_SIZE-1
  MASTER_ADDR  process 0 的 IP（coordinator 由 process 0 自动拉起）
  MASTER_PORT  分布式 rendezvous 端口（所有实例一致即可）

DP 语义与 jax_train.build_dp_one_iter 一致：
  - 每个 (实例, 卡) 副本独立 rollout，通信为零
  - ppo_update 内对全局 replica 轴 pmean allreduce 梯度（跨实例由 JAX
    distributed runtime + RCCL 完成，平台已预置 NCCL_IB_* RDMA 环境变量，
    RCCL 兼容读取）
  - 全部副本参数逐位一致；每轮用 all_gather 拉取参数摘要做跨副本校验
    （pmean/all_gather 正确性的直接证据，失败即退出）

有损同步（Local SGD，--lsgd-k > 0，通信量降到 ~1/K）：
  - --lsgd-mode param：每 K 步 pmean 参数（保持每迭代更新次数不变，
    代价是 K 步本地漂移）
  - --lsgd-mode grad：冻结参数上累加 K 个 minibatch 梯度 → 一次平均梯度
    同步 → 一次更新（零漂移、参数始终逐位一致；K=1 时与现状逐位一致）
  - --lsgd-bf16：同步时 bf16 半精度（流量再减半；通道仍无损，无稀疏/
    量化）。跨机 20 卡（10 机×2 卡）K=256 ≈ 194MB/迭代 ≈ 0.5-1.5s

WORLD_SIZE=1 时退化为单机 pmap DP，可在任意 DCU 节点直接运行。

示例（控制台启动脚本内）：
  python3 -m jax_bomb_v2.multicard_train --arch transformer --embed 512 \
    --depth 2 --patch 4 --num-envs 4096 --num-steps 256 --minibatch 4096 \
    --iters 5
"""
import argparse
import glob
import json
import os
import pickle
import signal
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

# 注意：jax_bomb 模块（jax_env.py:42 的模块级 jnp.array）在 import 时会
# 初始化 XLA 后端，而 jax.distributed.initialize 必须在任何后端调用之前 ——
# 因此 jax_bomb 的 import 放在 main() 里、initialize() 之后。


def tree_stack(trees):
    """[(n, ...) pytree] -> (k, n, ...) pytree，沿新首轴堆叠（pmap 输入用）。"""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def param_digest(params):
    """逐层求和（fp32；DCU 无 x64，fp32 求和顺序确定 → 逐位一致语义不变）。"""
    s = jnp.zeros((), jnp.float32)
    for leaf in jax.tree.leaves(params):
        s = s + jnp.sum(leaf)
    return s


def reward_schedule_global_steps(iteration, steps_per_iteration, step_offset=0):
    """Global reward-schedule progress for the current training update."""
    return step_offset + (iteration - 1) * steps_per_iteration


def fixed_reward_alpha(iteration, steps_per_iteration, anneal_steps,
                       step_offset=0):
    """Fixed linear component of the shared reward-shaping schedule."""
    if anneal_steps <= 0:
        return 1.0
    gs = reward_schedule_global_steps(iteration, steps_per_iteration, step_offset)
    return max(0.0, 1.0 - gs / anneal_steps)


def shared_reward_anneal_steps(crate_coef, crate_steps, explore_coef,
                                explore_steps):
    """Select the one fixed schedule used by all reward-shaping signals."""
    if crate_coef != 0.0 and explore_coef != 0.0 and crate_steps != explore_steps:
        raise ValueError(
            "crate and explore anneal steps must match when both shaping rewards "
            "are enabled"
        )
    return explore_steps if crate_coef == 0.0 and explore_coef != 0.0 else crate_steps


def write_result(rank, lines):
    """把关键结果落盘（平台 stdout 日志不一定可见）。默认写到 cwd 下
    multicard_result.txt（挂载目录里两端都能看到），可用 SCNET_RESULT_FILE
    覆盖路径。仅 rank 0 写；失败只警告不中断。"""
    if rank != 0:
        return
    path = os.environ.get("SCNET_RESULT_FILE",
                          os.path.join(os.getcwd(), "multicard_result.txt"))
    try:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        print(f"[0] WARN: 结果文件写入失败 {path}: {e}", flush=True)


# ---------------- 检查点（断点续训） ----------------

def ckpt_file(ckpt_dir, rank, it):
    return os.path.join(ckpt_dir, f"ckpt_{it:08d}_r{rank}.pkl")


def newest_ckpt(ckpt_dir, rank):
    """该 rank 最新检查点文件（按 iter 号取最大），没有返回 None。"""
    files = glob.glob(os.path.join(ckpt_dir, f"ckpt_*_r{rank}.pkl"))
    if not files:
        return None
    return max(files, key=lambda p: int(os.path.basename(p).split("_")[1]))


def save_ckpt(ckpt_dir, rank, it, params, opt_state, states, keys, cfg):
    """每个 rank 存自己的文件（states/keys 每 rank 不同；params/opt_state
    跨 rank 一致）。host 数组序列化，加载时转回 jax 数组。"""
    path = ckpt_file(ckpt_dir, rank, it)
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        payload = {"it": it, "cfg": cfg,
                   "params": jax.tree.map(np.asarray, params),
                   "opt_state": jax.tree.map(np.asarray, opt_state),
                   "states": jax.tree.map(np.asarray, states),
                   "keys": np.asarray(keys)}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        print(f"[{rank}] WARN: 检查点保存失败 {path}: {e}", flush=True)
        return False
    return True


def load_ckpt(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    p["params"] = jax.tree.map(jnp.asarray, p["params"])
    p["opt_state"] = jax.tree.map(jnp.asarray, p["opt_state"])
    p["states"] = jax.tree.map(jnp.asarray, p["states"])
    p["keys"] = jnp.asarray(p["keys"])
    return p


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="transformer")
    ap.add_argument("--embed", type=int, default=392)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff-factor", type=float, default=4)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--num-steps", type=int, default=256)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--adv-top-frac", type=float, default=0.25,
                    help="Top-Advantage Filtering 比例 (保留前 25%% |A_t|)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="参数 EMA 指数移动平均衰减系数")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--obs-quant", action="store_true")
    ap.add_argument("--checkpoint", action="store_true")
    # ---- Local SGD（有损同步，跨机降通信）----
    ap.add_argument("--lsgd-k", type=int,
                    default=int(os.environ.get("LSGD_K", "0")),
                    help="Local SGD 同步周期：每 K 个 minibatch 同步一次参数"
                         "（0=现状：每个 minibatch pmean 梯度，逐位一致）。"
                         ">0 时 minibatch 循环内零通信、每 K 步 pmean 参数，"
                         "通信量降到 ~1/K。20 卡（10 机×2 卡）下 K=256 ≈ "
                         "4 次同步/迭代 ≈ 0.5-1.5s，K=128 ≈ 8 次 ≈ 1-3s。"
                         "环境变量 LSGD_K 同效")
    ap.add_argument("--lsgd-mode", default="param",
                    choices=["param", "grad"],
                    help="Local SGD 同步对象：param=每 K 步平均参数（保持 "
                         "1024 次更新/迭代，代价是 K 步本地漂移）；grad=冻结"
                         "参数上累加 K 个梯度、一次平均梯度同步、一次更新"
                         "（零漂移、参数始终逐位一致，代价是只有 1024/K 次"
                         "更新/迭代；K=1 时与现状逐位一致）")
    ap.add_argument("--lsgd-bf16", action="store_true",
                    help="Local SGD 同步时用 bf16 半精度传输（流量减半，"
                         "尾数损失可忽略）")
    ap.add_argument("--lsgd-sync-state", action="store_true",
                    help="Local SGD 同步时连 Adam 动量/方差一起平均"
                         "（防本地漂移，流量×3）")
    ap.add_argument("--tolerate-inconsistent", action="store_true",
                    help="一致性校验失败时继续跑（诊断用），默认退出")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "ckpt"),
                    help="检查点目录；不设=不存盘（benchmark 模式）。"
                         "真实训练设 ckpt/（自动接续最新检查点）")
    ap.add_argument("--ckpt-every", type=int,
                    default=int(os.environ.get("CKPT_EVERY", "60")),
                    help="周期存盘间隔（分钟）；0=仅在结束/收到信号时存")
    ap.add_argument("--ckpt-local-dir",
                    default=os.environ.get("CKPT_LOCAL_DIR", "ckpt"),
                    help="rank0 轻量参数快照目录（供拉回本地/评估，params 仅 "
                         "~25MB pickle，不拖速度）。")
    ap.add_argument("--ckpt-local-every", type=int,
                    default=int(os.environ.get("CKPT_LOCAL_EVERY", "30")),
                    help="参数快照间隔（分钟）；0=仅在结束/信号时存")
    # ---- 评估（当前策略 vs 冻结基线，两策略对打）----
    ap.add_argument("--eval-vs", default=os.environ.get("EVAL_VS", None),
                    help="冻结基线 ckpt/params 路径（支持 {RANK} 占位，各 rank "
                         "用自己那份：如 ckpt/ckpt_00001000_r{RANK}.pkl）。"
                         "每 --eval-every 迭代打一次当前 vs 基线胜率")
    ap.add_argument("--eval-every", type=int,
                    default=int(os.environ.get("EVAL_EVERY", "0")),
                    help="评估间隔（迭代数）；0=关闭")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已有检查点全新开始（默认自动接续；"
                         "也可删除检查点目录实现不接续）")
    ap.add_argument("--levels", default=os.environ.get("LEVELS_FILE", None),
                    help="标准化关卡数据 levels.json 路径（241 张 QQ堂地图；"
                         "不设时自动探测 ./levels.json / ./web/assets/maps/"
                         "levels.json，都没有则回退过程式生成）")
    ap.add_argument("--level-weights", default=os.environ.get("LEVEL_WEIGHTS", "empty=0.05,功夫=0.1,比武=0.15"),
                    help="关卡采样权重，如 '240=0.2' 或 'empty=0.2'（空场景关"
                         "占 20%%）；逗号分隔多个；其余关均分剩余概率")
    ap.add_argument("--crate-reward-coef", type=float, default=0.0,
                    help="开箱成长 bootstrap 奖励系数（0=关）。关卡模式多数"
                         "地图出生点被砖隔开，前期无交战通道——短促正奖励加速"
                         "前期学习，随 --crate-reward-anneal-steps 线性退火到 0")
    ap.add_argument("--crate-reward-anneal-steps", type=int, default=0,
                    help="开箱奖励退火步数（全局环境步；长训默认 300 亿，8.39M 步/iter 下覆盖整轮）。"
                         "0=不退火（保持恒定，不推荐）")
    ap.add_argument("--explore-reward-coef", type=float, default=0.0,
                    help="探索 novelty 奖励系数（0=关）。每 tick 玩家中心格若是本局首次"
                         "到达 → +coef；走过的格不再给分，坐桩/困出生点零探索分。与开箱"
                         "奖励同为 bootstrap 信号，随 --explore-reward-anneal-steps 退火到 0")
    ap.add_argument("--explore-reward-anneal-steps", type=int, default=0,
                    help="探索奖励退火步数（全局环境步；长训默认 300 亿）。0=不退火")
    ap.add_argument("--brick-reward-coef", type=float, default=0.0,
                    help="炸墙奖励系数（0=关）。每炸毁一块砖双方各 +coef/2——给"
                         "'炸墙'本身即时正反馈，治出生点 3 格死锁（crate 链路"
                         "炸→掷爆率→吃到太长太弱学不会）。乘统一退火 α")
    ap.add_argument("--reward-anneal-k", type=float, default=1.2,
                    help="动态退火斜率 k（α_dyn = max(0, 1-tanh(k·x))，x=训练内"
                         "每局击杀率）。击杀率上来（模型会打架了）塑形自动归零，"
                         "只剩纯胜负——Pommerman 论文机制，代替固定步数拍脑袋")
    ap.add_argument("--reward-anneal-step-offset", type=int, default=0,
                    help="奖励固定退火的已完成全局环境步数。参数 warm start 用它继承"
                         "源模型的固定退火进度；不改变 checkpoint iteration、Adam、"
                         "环境或 RNG 的恢复语义。")
    ap.add_argument("--curriculum-json", default=None,
                    help="Spawn-Distance 课程文件：按阶段自动切换图集与出生点距离 Stage1→4")
    ap.add_argument("--curriculum-winrate-gate", type=float, default=0.65,
                    help="课程自动晋级胜率门禁（默认 0.65：对战本阶段初始基准胜率 >= 65%% 时自动晋级）")
    ap.add_argument("--curriculum-eval-every", type=int, default=50,
                    help="课程对战评估间隔（迭代数，默认 50：每 50 轮评估一次 vs 阶段基线胜率）")
    ap.add_argument("--curriculum-min-iters", type=int, default=50,
                    help="每个课程阶段最小训练迭代数（默认 50：防止阶段跳变过快）")
    args = ap.parse_args()
    if args.reward_anneal_step_offset < 0:
        raise SystemExit("--reward-anneal-step-offset must be non-negative")
    try:
        shared_anneal_steps = shared_reward_anneal_steps(
            args.crate_reward_coef, args.crate_reward_anneal_steps,
            args.explore_reward_coef, args.explore_reward_anneal_steps)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # ---- 平台注入 / 单机默认 ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world > 1:
        # 平台容器默认代理会让 rendezvous 超时（JAX 官方 docstring 明示）
        for v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                  "ALL_PROXY", "all_proxy"):
            os.environ.pop(v, None)
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port}",
            num_processes=world, process_id=rank,
            initialization_timeout=600)
        print(f"[{rank}] jax.distributed OK: world={world} "
              f"coordinator={master_addr}:{master_port}", flush=True)

    # initialize 之后才能 import（见模块顶部注释）
    from jax_bomb_v2.jax_env import H, W, N_OBS_CH, RADIUS, init_batch
    from jax_bomb_v2.jax_net import count_params, init_net
    from jax_bomb_v2.jax_train import (both_masks, both_perspectives,
                                    both_states, collect_rollout,
                                    collect_rollout_two, compute_gae,
                                    ppo_update, ppo_update_gradsync,
                                    ppo_update_lsgd, sample_actions)

    # ---- 标准化关卡（可选项；任何 jit/vmap 之前激活）----
    levels_path = args.levels
    if not levels_path:
        for cand in ("levels.json", "web/assets/maps/levels.json"):
            if os.path.isfile(cand):
                levels_path = cand
                break
    curriculum = None
    cur_stage = -1
    if levels_path:
        from jax_bomb_v2 import levels as _levels

        def _set_stage(si: int):
            """课程：把 active 图集切到 stage si，并设置该阶段的出生点最大距离限制。"""
            ids = curriculum['stages'][si]
            w = 1.0 / len(ids)
            # 空间几何距离：Stage 0 空场景(<=4) → Stage 1 贴脸(<=4) → Stage 2 近距破障(<=6) → Stage 3 中距迷宫(<=10) → Stage 4 全图无限制(0)
            stage_dists = [4, 4, 6, 10, 0]
            max_d = stage_dists[si] if si < len(stage_dists) else 0
            _levels.set_active(levels_path,
                               weights=','.join(f"{i}={w:.8f}" for i in ids),
                               max_spawn_dist=max_d)

        if args.curriculum_json:
            with open(args.curriculum_json, encoding="utf-8") as f:
                curriculum = json.load(f)
            _set_stage(0)
            cur_stage = 0
            print(f"[{rank}] 课程模式: {args.curriculum_json} 初始 Stage 0 (纯空场景道场)"
                  f"（{len(curriculum['stages'][0])} 张）阈值"
                  f" {curriculum['thresholds']}", flush=True)
        else:
            _levels.set_active(levels_path, weights=args.level_weights)
            print(f"[{rank}] 关卡模式: {levels_path}"
                  f"{' 权重 ' + args.level_weights if args.level_weights else ''}"
                  f" → {_levels.active().summary()}", flush=True)
    else:
        print(f"[{rank}] 关卡模式未激活 → 过程式生成地图", flush=True)

    devs = jax.devices()
    # 多进程时 pmap 输入轴 = 每进程本地设备数（= devices/num_processes），
    # 而 pmap 轴本身跨进程（axis_size = n_total，pmean/all_gather 全局归约）
    n_local = jax.local_device_count()
    n_total = n_local * world      # 全局 replica（pmap 轴）总数
    print(f"[{rank}] devices={[str(d) for d in devs]} "
          f"n_local={n_local} n_total={n_total}", flush=True)

    # ---- 全局 minibatch / envs 按总卡数均分（与 build_dp_one_iter 同语义）。
    # 不能整除时自动下调到最近整除值（一份脚本适配任意 卡数×实例数 组合）。
    envs_per = args.num_envs // n_total
    if envs_per * n_total != args.num_envs:
        print(f"[{rank}] --num-envs {args.num_envs} 不能整除总卡数 {n_total}"
              f"，已自动下调到 {envs_per * n_total}", flush=True)
        args.num_envs = envs_per * n_total
    mb_local = args.minibatch // n_total
    if mb_local * n_total != args.minibatch:
        print(f"[{rank}] --minibatch {args.minibatch} 不能整除总卡数 {n_total}"
              f"，已自动下调到 {mb_local * n_total}", flush=True)
        args.minibatch = mb_local * n_total
    if envs_per < 1 or mb_local < 1:
        raise SystemExit(f"配置过小：envs/副本={envs_per} mb/副本={mb_local}，"
                         f"请加大 --num-envs/--minibatch")

    # ---- 每副本独立环境与随机流（全局副本序号 g = rank*n_local+l）----
    steps = args.num_steps
    opt = optax.adam(args.lr)

    # 信号处理：平台抢占/手动停时，当前 iter 结束后存盘退出
    stop_flag = {"v": False}

    def _on_signal(signum, _frame):
        stop_flag["v"] = True
        print(f"[{rank}] 收到信号 {signum}，当前 iter 结束后保存检查点并退出",
              flush=True)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ---- 断点续训：默认自动接续该 rank 最新检查点（--fresh 关闭）----
    it_done = 0
    params = opt_state = states = keys = None
    if args.ckpt_dir and not args.fresh:
        ck = newest_ckpt(args.ckpt_dir, rank)
        if ck is not None:
            p = load_ckpt(ck)
            cfg_old = p["cfg"]
            if "envs_per" in cfg_old:
                want = {"arch": args.arch, "embed": args.embed,
                        "depth": args.depth, "patch": args.patch,
                        "heads": args.heads, "ff_factor": args.ff_factor,
                        "num_steps": args.num_steps, "epochs": args.epochs,
                        "lr": args.lr, "seed": args.seed,
                        "radius": RADIUS,
                        "map": f"{H}x{W}",
                        "levels": os.path.basename(levels_path)
                        if levels_path else "",
                        "crate_coef": args.crate_reward_coef,
                        "crate_anneal": args.crate_reward_anneal_steps,
                        "explore_coef": args.explore_reward_coef,
                        "explore_anneal": args.explore_reward_anneal_steps,
                        "brick_coef": args.brick_reward_coef,
                        "anneal_k": args.reward_anneal_k,
                        "reward_anneal_step_offset": args.reward_anneal_step_offset,
                        "envs_per": envs_per, "mb_local": mb_local}
                bad = [k for k, v in want.items()
                       if cfg_old.get(k, 0 if k == "reward_anneal_step_offset"
                                      else None) != v]
                if cfg_old.get("n_total") != n_total:
                    print(f"[{rank}] WARN: n_total {cfg_old.get('n_total')} → "
                          f"{n_total}（机器数变化，per-replica 负载不变，"
                          f"允许接续。--num-envs/--minibatch 请按新卡数重设）",
                          flush=True)
            else:
                want = {"arch": args.arch, "embed": args.embed,
                        "depth": args.depth, "patch": args.patch,
                        "num_envs": args.num_envs,
                        "num_steps": args.num_steps,
                        "minibatch": args.minibatch, "epochs": args.epochs,
                        "map": f"{H}x{W}",
                        "levels": os.path.basename(levels_path)
                        if levels_path else "",
                        "crate_coef": args.crate_reward_coef,
                        "crate_anneal": args.crate_reward_anneal_steps,
                        "explore_coef": args.explore_reward_coef,
                        "explore_anneal": args.explore_reward_anneal_steps,
                        "brick_coef": args.brick_reward_coef,
                        "anneal_k": args.reward_anneal_k,
                        "reward_anneal_step_offset": args.reward_anneal_step_offset}
                bad = [k for k, v in want.items()
                       if cfg_old.get(k, 0 if k == "reward_anneal_step_offset"
                                      else None) != v]
                if cfg_old.get("n_total") != n_total:
                    bad.append("n_total")
            if bad:
                raise SystemExit(
                    f"[{rank}] 检查点 {ck} 配置不匹配：{bad}，"
                    f"如确要新跑请加 --fresh 或删除检查点目录")
            it_done, params, opt_state = p["it"], p["params"], p["opt_state"]
            states, keys = p["states"], p["keys"]
            print(f"[{rank}] 接续检查点 {ck}（iter {it_done}）", flush=True)
            write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                f"RESUME from {ck} iter={it_done}"])
    if it_done >= args.iters:
        raise SystemExit(f"[{rank}] 检查点已是最终 iter {it_done} "
                         f">= iters {args.iters}，无需继续（要重跑用 --fresh）")
    if params is None:
        states = tree_stack([
            init_batch(jrandom.PRNGKey(args.seed * 1000 + rank * n_local + l),
                       envs_per)
            for l in range(n_local)])
        pkey = jrandom.PRNGKey(args.seed + 9999)
        params = init_net(pkey, args.arch, N_OBS_CH, H, W,
                          embed=args.embed, depth=args.depth, patch=args.patch,
                          heads=args.heads, ff_factor=args.ff_factor)
        opt_state = opt.init(params)
        keys = jrandom.split(jrandom.PRNGKey(args.seed * 7919 + rank), n_local)
        write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN start: "
                            f"world={world} rank={rank} n_total={n_total} "
                            f"arch={args.arch} embed={args.embed} depth={args.depth} "
                            f"patch={args.patch} envs={args.num_envs} "
                            f"steps={steps} minibatch={args.minibatch} "
                            f"epochs={args.epochs} iters={args.iters} "
                            f"lsgd_k={args.lsgd_k} lsgd_mode={args.lsgd_mode} "
                            f"lsgd_bf16={args.lsgd_bf16} "
                            f"lsgd_sync_state={args.lsgd_sync_state} "
                            f"adv_top_frac={args.adv_top_frac} "
                            f"params={count_params(params):,}"])
    cfg = {"arch": args.arch, "embed": args.embed, "depth": args.depth,
           "patch": args.patch, "heads": args.heads,
           "ff_factor": args.ff_factor,
           "num_envs": args.num_envs, "num_steps": args.num_steps,
           "minibatch": args.minibatch, "epochs": args.epochs,
           "lr": args.lr, "seed": args.seed, "radius": RADIUS,
           "n_total": n_total,
           "map": f"{H}x{W}",
           "levels": os.path.basename(levels_path) if levels_path else "",
           "crate_coef": args.crate_reward_coef,
           "crate_anneal": args.crate_reward_anneal_steps,
           "explore_coef": args.explore_reward_coef,
           "explore_anneal": args.explore_reward_anneal_steps,
           "brick_coef": args.brick_reward_coef,
           "anneal_k": args.reward_anneal_k,
           "reward_anneal_step_offset": args.reward_anneal_step_offset,
           "envs_per": envs_per, "mb_local": mb_local,
           "lsgd_k": args.lsgd_k, "lsgd_mode": args.lsgd_mode,
           "lsgd_bf16": args.lsgd_bf16, "lsgd_sync_state": args.lsgd_sync_state}
    print(f"[{rank}] arch={args.arch} embed={args.embed} depth={args.depth} "
          f"patch={args.patch} params={count_params(params):,} "
          f"envs/replica={envs_per} mb_local={mb_local} it_done={it_done}",
          flush=True)
    if args.lsgd_k > 0:
        n_mb = 2 * envs_per * steps // mb_local     # 每 epoch minibatch 数
        n_sync = (n_mb // args.lsgd_k
                  + (1 if n_mb % args.lsgd_k else 0)) * args.epochs
        s_bytes = count_params(params) * 4 * 2 * (n_total - 1) / n_total
        if args.lsgd_bf16:
            s_bytes /= 2
        print(f"[{rank}] LSGD: k={args.lsgd_k} mode={args.lsgd_mode} "
              f"bf16={args.lsgd_bf16} sync_state={args.lsgd_sync_state} → "
              f"{n_sync} 次同步/迭代 ≈ "
              f"{n_sync * s_bytes / 1e6:.0f}MB/迭代/卡（fp32 稠密，不含 "
              f"稀疏/量化）", flush=True)

    def shard(params, opt_state, states, key, crate_coef, explore_coef,
              brick_coef, timeout_alpha):
        """单个 replica 的一次迭代（与 jax_train.build_dp_one_iter 一致）。"""
        states, batch, nov, kills = collect_rollout(
            params, args.arch, states, key, steps,
            args.no_mask, args.obs_quant, args.checkpoint, crate_coef,
            explore_coef, brick_coef, timeout_alpha)
        obs, state, acts, lps, vals, rew, done, masks = batch
        fobs = both_perspectives(states)
        fmasks = both_masks(states)
        fstate = both_states(states)
        fkey = jrandom.split(key)[0]
        _, _, fval = sample_actions(params, args.arch, fobs, fmasks, fkey,
                                    state=fstate)
        next_val = jnp.concatenate([vals[1:], fval[None]], axis=0)
        advs = compute_gae(rew, vals, next_val, done, args.gamma, args.lam)
        rets = advs + vals
        if args.lsgd_k > 0:
            if args.lsgd_mode == "grad":
                params, opt_state, last_loss = ppo_update_gradsync(
                    params, opt, opt_state, args.arch,
                    (obs, state, acts, lps, advs, rets, masks),
                    key, mb_local, args.clip_eps, args.vf_coef,
                    args.ent_coef, args.epochs, axis_name="dev",
                    sync_k=args.lsgd_k, bf16_sync=args.lsgd_bf16,
                    return_loss=True)
            else:
                params, opt_state, last_loss = ppo_update_lsgd(
                    params, opt, opt_state, args.arch,
                    (obs, state, acts, lps, advs, rets, masks),
                    key, mb_local, args.clip_eps, args.vf_coef,
                    args.ent_coef, args.epochs, axis_name="dev",
                    sync_k=args.lsgd_k, bf16_sync=args.lsgd_bf16,
                    sync_state=args.lsgd_sync_state, return_loss=True,
                    adv_top_frac=args.adv_top_frac)
        else:
            params, opt_state, last_loss = ppo_update(
                params, opt, opt_state, args.arch,
                (obs, state, acts, lps, advs, rets, masks),
                key, mb_local, args.clip_eps, args.vf_coef, args.ent_coef,
                args.epochs, axis_name="dev", return_loss=True,
                adv_top_frac=args.adv_top_frac)
        key = jrandom.split(key)[0]
        digest = jax.lax.all_gather(param_digest(params), "dev")
        ep_cnt = jnp.maximum(done.sum().astype(jnp.float32), 1.0)
        kill_rate = 2.0 * kills.sum() / ep_cnt
        stats = jnp.stack([jnp.mean(rew),
                           jnp.mean(done.astype(jnp.float32)),
                           explore_coef * jnp.mean(nov) / steps,
                           kill_rate])
        return params, opt_state, states, key, digest, last_loss, stats


    one_iter = jax.pmap(shard, axis_name="dev",
                        in_axes=(None, None, 0, 0, None, None, None, None),
                        out_axes=(None, None, 0, 0, 0, 0, 0))

    # ---- 评估：两策略对打（用于课程胜率门禁与 --eval-vs 基线）----
    eval_fn = None
    if (args.eval_vs and args.eval_every > 0) or (curriculum is not None and args.curriculum_eval_every > 0):
        def eval_shard(params, frozen, states, key):
            states, (w, l) = collect_rollout_two(
                params, frozen, args.arch, states, key, steps,
                args.no_mask, args.obs_quant)
            return states, key, w, l

        eval_fn = jax.pmap(eval_shard, axis_name="dev",
                           in_axes=(None, None, 0, 0),
                           out_axes=(0, 0, 0, 0))

    # ---- warmup（编译 + 预热，输出丢弃）----
    _w = one_iter(params, opt_state, states, keys,
                  jnp.maximum(0.0, args.crate_reward_coef),
                  jnp.maximum(0.0, args.explore_reward_coef),
                  jnp.maximum(0.0, args.brick_reward_coef), 1.0)
    jax.block_until_ready(_w)

    # ---- 计时 + 逐轮指标 + 周期存盘（stdout/结果文件同步输出，长训监控）----
    t0 = time.time()
    last_save = t0
    last_local_save = t0
    n_run = 0
    cur = (params, opt_state, states, keys)
    ema_params = jax.tree.map(np.asarray, params)    # EMA 参数副本
    stage_baseline_params = cur[0]                    # 当前 Stage 的初始冻结基准 (内存零 IO)
    stage_start_iter = it_done + 1                    # 当前 Stage 开始的 iter
    steps_per_iter_g = 2 * args.num_envs * steps     # 全局环境步/迭代
    kill_rate_prev = 0.0                              # 上一 iter 击杀率（首轮未知 → α_dyn=1）
    for i in range(it_done + 1, args.iters + 1):
        ti = time.time()
        gs = reward_schedule_global_steps(
            i, steps_per_iter_g, args.reward_anneal_step_offset)
        curriculum_gs = (i - 1) * steps_per_iter_g

        # ---- 课程自适应胜率门禁提档 (Dynamic Winrate Gating) ----
        if curriculum is not None and cur_stage < len(curriculum['stages']) - 1:
            iters_in_stage = i - stage_start_iter
            gate_thresh = (curriculum.get('winrate_gates', [])[cur_stage]
                           if curriculum and 'winrate_gates' in curriculum and cur_stage < len(curriculum['winrate_gates'])
                           else args.curriculum_winrate_gate)

            winrate_passed = False
            wr = 0.0
            if eval_fn is not None and i % args.curriculum_eval_every == 0 and iters_in_stage >= args.curriculum_min_iters:
                est, ekey, ew, el = eval_fn(cur[0], stage_baseline_params, cur[2], cur[3])
                jax.block_until_ready(est)
                cur = (cur[0], cur[1], est, ekey)
                ew_sum = int(np.sum(np.asarray(ew)))
                el_sum = int(np.sum(np.asarray(el)))
                total_games = ew_sum + el_sum
                wr = ew_sum / max(total_games, 1)
                print(f"[{rank}] [Curriculum Gate] Stage {cur_stage} 评估: "
                      f"vs StageBaseline win={ew_sum} lose={el_sum} (总完局={total_games}) winrate={wr:.1%} "
                      f"(晋级门禁={gate_thresh:.1%}, 已训={iters_in_stage} iters)", flush=True)
                write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"[Gate] Stage {cur_stage} vs Baseline "
                                    f"winrate={wr:.1%} ({ew_sum}/{total_games})"])
                if total_games >= 30 and wr >= gate_thresh:
                    winrate_passed = True

            # 步数兜底：仅在未开启评估时使用全局步数比例；若开启评估，则仅在本阶段超长未晋级时触发超时兜底
            step_fallback = False
            if eval_fn is None:
                step_frac = curriculum_gs / max(1, steps_per_iter_g * args.iters)
                step_fallback = (step_frac >= curriculum['thresholds'][cur_stage]) if cur_stage < len(curriculum['thresholds']) else False
            elif iters_in_stage >= 100:
                step_fallback = True

            if winrate_passed or step_fallback:
                cur_stage += 1
                _set_stage(cur_stage)
                stage_start_iter = i
                stage_baseline_params = cur[0]  # 锁定当前学成的新模型作为下一阶段 Baseline
                reason = f"胜率达标 (winrate={wr:.1%} >= {gate_thresh:.1%})" if winrate_passed else f"阶段超时兜底 (iters_in_stage={iters_in_stage})"
                print(f"[{rank}] 🏆 课程晋级 → Stage {cur_stage}（原因: {reason}，"
                      f"{len(curriculum['stages'][cur_stage])} 张图）", flush=True)
                write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"PROMOTION -> Stage {cur_stage} ({reason})"])

        alpha_fix = fixed_reward_alpha(
            i, steps_per_iter_g, shared_anneal_steps,
            args.reward_anneal_step_offset)
        alpha_dyn = jnp.maximum(
            0.0, 1.0 - jnp.tanh(args.reward_anneal_k * kill_rate_prev))
        alpha = alpha_fix * alpha_dyn
        crate_coef = args.crate_reward_coef * alpha
        explore_coef = args.explore_reward_coef * alpha
        brick_coef = args.brick_reward_coef * alpha
        res = one_iter(cur[0], cur[1], cur[2], cur[3], crate_coef,
                       explore_coef, brick_coef, alpha)
        jax.block_until_ready(res)
        cur = res
        n_run += 1
        # 更新参数 EMA
        cur_p_np = jax.tree.map(np.asarray, res[0])
        d_ema = float(args.ema_decay)
        ema_params = jax.tree.map(lambda e, p: d_ema * e + (1.0 - d_ema) * p,
                                  ema_params, cur_p_np)
        dt_i = time.time() - ti
        dt_avg = (time.time() - t0) / n_run
        sps_i = 2 * args.num_envs * steps / dt_i
        sps_avg = 2 * args.num_envs * steps / dt_avg
        loss_i = float(np.mean(np.asarray(res[5])))
        stats = np.mean(np.asarray(res[6]), axis=0)  # (平均回报, 结束率, 探索分/帧, 击杀率)
        rew_m, done_m, exp_m = (float(stats[0]), float(stats[1]),
                                float(stats[2]))
        kill_r = float(stats[3])
        kill_rate_prev = max(0.0, kill_r)
        ep_len = 1.0 / done_m if done_m > 1e-6 else float("nan")
        line = (f"iter {i}/{args.iters} {dt_i:.3f}s ({dt_avg:.3f}s avg) "
                f"{sps_i:,.0f} sps (avg {sps_avg:,.0f}) loss={loss_i:.4f} "
                f"rew={rew_m:.3f} explore={exp_m:.3f} kill={kill_r:.3f} "
                f"α={float(alpha):.2f} gs={gs:,} ep_len={ep_len:.1f}")
        print(f"[{rank}] {line}", flush=True)
        write_result(rank,
                     [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}"])
        # 周期存盘（间隔分钟；0=只在结束/信号时存）；末 iter / 信号也存
        if args.ckpt_dir and (i == args.iters or stop_flag["v"]
                              or (args.ckpt_every > 0
                                  and time.time() - last_save
                                  >= args.ckpt_every * 60)):
            if save_ckpt(args.ckpt_dir, rank, i,
                         res[0], res[1], res[2], res[3], cfg):
                print(f"[{rank}] ckpt saved: "
                      f"{ckpt_file(args.ckpt_dir, rank, i)}", flush=True)
                write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"ckpt saved iter {i}"])
                last_save = time.time()
        # rank0 轻量参数快照 + EMA 快照
        if (args.ckpt_local_dir and rank == 0
                and (i == args.iters or stop_flag["v"]
                     or (args.ckpt_local_every > 0
                         and time.time() - last_local_save
                         >= args.ckpt_local_every * 60))):
            try:
                os.makedirs(args.ckpt_local_dir, exist_ok=True)
                p = os.path.join(args.ckpt_local_dir,
                                 f"params_it{i:08d}.pkl")
                p_ema = os.path.join(args.ckpt_local_dir,
                                     f"params_it{i:08d}_ema.pkl")
                with open(p, "wb") as f:
                    pickle.dump(cur_p_np, f)
                with open(p_ema, "wb") as f:
                    pickle.dump(ema_params, f)
                print(f"[{rank}] params & EMA snapshot -> {p}", flush=True)
                write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"params & EMA snapshot iter {i}"])
                last_local_save = time.time()
            except Exception as e:
                print(f"[{rank}] WARN: 参数快照失败: {e}", flush=True)
        # ---- 评估：当前策略 vs 冻结基线（两策略对打，--eval-vs/--eval-every）----
        if eval_fn is not None and args.eval_vs and args.eval_every > 0 and i % args.eval_every == 0:
            fpath = args.eval_vs.replace("{RANK}", str(rank))
            try:
                with open(fpath, "rb") as f:
                    obj = pickle.load(f)
                frozen = (obj.get("params") if isinstance(obj, dict)
                          and "params" in obj else obj)
                frozen = jax.tree.map(jnp.asarray, frozen)
                est, ekey, ew, el = eval_fn(cur[0], frozen, cur[2], cur[3])
                jax.block_until_ready(est)
                cur = (cur[0], cur[1], est, ekey)   # eval 推进了 env states/keys
                ew = int(np.sum(np.asarray(ew)))
                el = int(np.sum(np.asarray(el)))
                wr = ew / max(ew + el, 1)
                print(f"[{rank}] EVAL vs {os.path.basename(fpath)}: "
                      f"win={ew} lose={el} winrate={wr:.3f}", flush=True)
                write_result(rank, [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"EVAL vs {os.path.basename(fpath)} "
                                    f"winrate={wr:.3f} ({ew}/{ew + el})"])
            except Exception as e:
                print(f"[{rank}] WARN: 评估跳过: {e}", flush=True)
        if stop_flag["v"]:
            print(f"[{rank}] 已按信号停止，完成 {n_run} 个 iter", flush=True)
            break
    dt = (time.time() - t0) / max(n_run, 1)
    sps = 2 * args.num_envs * steps / dt
    it_end = it_done + n_run

    params, opt_state, states, keys, dg, _, _ = res
    print(f"[{rank}] {dt:.3f} s/iter × {n_run} → "
          f"{sps:,.0f} sps ({args.num_envs} envs × {steps} steps) "
          f"iter {it_end}/{args.iters}", flush=True)

    # ---- 一致性：全部 replica 摘要逐位相同 ----
    darr = np.asarray(dg)                  # (n_local, n_total)
    d0 = darr[0]
    ok = bool(np.all(darr == d0[None]))
    if ok:
        print(f"[{rank}] consistency PASS: {n_total} 个 replica 参数逐位一致 "
              f"(digest={d0[0]:.17e})", flush=True)
    else:
        bad = np.abs(darr - d0[None])
        print(f"[{rank}] consistency FAIL: max delta={bad.max():.3e} "
              f"n_total={n_total}", flush=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    write_result(rank, [
        f"[{ts}] RUN end: dt={dt:.3f} s/iter × {n_run} "
        f"sps={sps:,.0f} (2×{args.num_envs}×{steps} frames/iter) "
        f"iter={it_end}/{args.iters}",
        f"[{ts}] consistency={'PASS' if ok else 'FAIL'} "
        f"n_total={n_total} digest={d0[0]:.17e}",
    ])
    if not ok and not args.tolerate_inconsistent:
        raise SystemExit("跨卡参数不一致 —— RCCL/pmean 异常，训练结果无效")

    # ---- rank0 保存最终训练模型与 EMA 快照 ----
    if rank == 0:
        save_dir = args.ckpt_local_dir or args.ckpt_dir or "ckpt"
        os.makedirs(save_dir, exist_ok=True)
        final_pkl = os.path.join(save_dir, f"params_it{it_end:08d}.pkl")
        final_ema_pkl = os.path.join(save_dir, f"params_it{it_end:08d}_ema.pkl")
        cur_p_np = jax.tree.map(np.asarray, params)
        with open(final_pkl, "wb") as f:
            pickle.dump(cur_p_np, f)
        with open(final_ema_pkl, "wb") as f:
            pickle.dump(ema_params, f)
        print(f"[{rank}] 💾 训练完成！已保存最终模型 -> {final_pkl} 以及 EMA 模型 -> {final_ema_pkl}", flush=True)
        write_result(rank, [f"[{ts}] Saved final models: {final_pkl}, {final_ema_pkl}"])


if __name__ == "__main__":
    main()
