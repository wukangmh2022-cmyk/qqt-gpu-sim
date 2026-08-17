"""离线蒸馏数据收集（批量版）：teacher（最强 ckpt）打 10 种对手，逐 tick
记录 (obs7, teacher_logits, masks)。

设计（与 jax 训练侧对齐，全批量向量化跑 DCU）：
  - 地图构成 = **纯空场 50% + 变换 50%**（用户定案）：
      pure_open_fraction=0.5 → 原版纯空场 50%；
      变换 50% 内部再切：open 带随机单障碍（open_obstacle_max=5）25%
      + corridor（random_wall_rows + wall_density 连续段，copy 仓库同款）25%。
      变换参数与 copy 仓库（11B 训练环境）逐位一致（make_walls/make_bricks/
      make_open_obstacles 已对拍 bit-identical）。
  - 数值对齐 jax 环境（speed=7.56/blast=7/max_steps=1800/invuln_ticks=30/
    max_hp=5/max_bombs=10），**全图满级**：open/corridor 的初始属性都拉满
    （growth/open_growth_* = 上限），与 jax student 能做的动作空间一致。
  - 宝箱/成长**后置**：growth_crate_prob=0 + open_crate_cross=False，
    地图只有墙/砖结构，无属性成长交互。
  - num_envs=N 并行：每 tick 推进 N 局，teacher + 对手都是批量前向
  - 每 tick 存 N 组、双视角各一：
      obs7 : jax make_obs 的 7 通道布局（**不含危险图**）
             ch0 我位置 splat  ch1 我泡 fuse/FUSE  ch2 对手位置
             ch3 对手泡 fuse  ch4 墙|砖(wall|brick)  ch5 泡威力 blast/BLAST
             ch6 t/1800
      logits : teacher 双视角的 (move_logits[5], bomb_logits[2])，mask 后
               （每局面 2 视角各一帧，同 student 2N 批量格式）
      move_mask/bomb_mask : 对应视角当时的合法动作掩码
        （bool (T,2,5)/(T,2,2)）—— 学生离线 KL 用 mask 后的 teacher 分布；
        后续 PPO 微调也用它保证采样只在合法动作里。
  - teacher 恒为 player0 视角（与 PPO learner 一致），对手为 player1。
  - obs7 复用 sim/obs.encode_obs 的共享 14 通道，再按 view_perm 取视角
    通道、ch5 用泡威力图替换危险图 —— 全张量运算，无逐 env Python 循环。
  - 存储：npz，每对手一个文件：obs7 (T,2,7,13,13) uint8×255、
    logits (T,2,7) float16、move_mask (T,2,5) bool、bomb_mask (T,2,2) bool
    （T = 总 tick×N 帧，双视角排成 2 块，jax 侧按 env×view 展平成帧）。

用法：
  python deploy/collect_distill.py --teacher ckpt/duel_nobc_8b_live.pt \
      --out distill_data --ticks 500 --num-envs 2048 --seed 0
"""
import argparse
import os
import sys
import time

# deploy/ 下的脚本：仓库根加入 sys.path 才能 import sim/train/play
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.bots import make_bot
from train.model import ActorCritic

H, W, P = 13, 13, 2
TICK_HZ = 10
FUSE = 30
BLAST = 7
MAX_STEPS = 1800

# jax make_obs 视角通道 ← 共享 14 通道的索引（P=2）：
#   jax ch0 我位置=共享0  ch1 我泡=共享2  ch2 对手位置=共享1
#   ch3 对手泡=共享3  ch4 墙=共享4  ch5(自行计算) ch6 进度=共享6
# 前 5 通道直接取共享通道；ch5 泡威力、ch6 进度单独拼（不能只取 6 通道再覆盖，
# 那样会把进度通道覆盖掉 —— 旧版 bug，收集的 6 通道数据格式不对）。
VIEW_MAP5 = [0, 2, 1, 3, 4]     # pid0 视角前 5 通道
VIEW1_MAP5 = [1, 3, 0, 2, 4]    # pid1 视角前 5 通道


def make_cfg() -> SimConfig:
    """蒸馏收集环境：纯空场 50% + copy 变换 50%，全图满级、无宝箱。

    - 地图变换参数与 copy 仓库（11B 训练环境）逐位一致：corridor 的
      random_wall_rows + wall_density 连续段、open 的 open_obstacle_max。
    - 数值对齐 jax（speed=7.56/blast=7/1800 步），open/corridor 初始属性
      全部拉满（= 上限）—— student 在 jax 里就是固定满级，动作空间一致。
    - 宝箱后置：growth_crate_prob=0（炸砖不出箱）+ open_crate_cross=False
      （不撒开局池）—— 纯地图结构，无成长交互。
    """
    return SimConfig(
        height=H, width=W, n_players=P,
        map_mode="corridor",                 # 混合关机制在 corridor 分支
        pure_open_fraction=0.5,              # 原版纯空场 50%
        open_fraction=0.25, ring_fraction=0.0,  # 变换 50% 内部再切
        open_obstacle_max=5,                 # copy open 变换：随机单障碍
        random_wall_rows=True,               # copy corridor 变换：顶/底墙行随机
        wall_density=0.45,                   # copy corridor 连续段
        open_crate_cross=False,              # 宝箱后置
        growth_crate_prob=0.0,               # 炸砖不出宝箱（后置）
        # 全图满级（对齐 jax 固定满级：泡/威/速不随地图类型打折）
        growth_bombs_start=10, growth_blast_start=7, growth_speed_start=1.0,
        open_growth_bombs=10, open_growth_blast=7, open_growth_speed=1.0,
        speed=7.56, blast=BLAST, max_steps=MAX_STEPS,
        invuln_ticks=30, max_hp=5, max_bombs=10,
    )


def load_net(path: str, device) -> ActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic(tuple(ck["obs_shape"]), arch=ck["arch"],
                      n_players=ck.get("n_players", P)).to(device)
    sd = ck["model"] if isinstance(ck["model"], dict) else ck["model"].state_dict()
    net.load_state_dict(sd)
    net.eval()
    return net


def teacher_logits(net: ActorCritic, obs14, mm, bm, pid: int):
    """teacher 批量前向，返回 mask 后 logits (N,5)/(N,2)。

    pid=0：obs14 原样（P0 视角）；pid=1：物理通道互换（P1 视角）—— 两视角
    各收集一份 (obs7, logits, masks)，学生 2N 批量格式直接对齐。
    """
    with torch.no_grad():
        o = _swap_player_channels(obs14) if pid == 1 else obs14
        dm, db, _ = net.dists(o, mm[:, pid].bool(), bm[:, pid].bool(), pid)
    return dm.logits, db.logits


def net_actions(net: ActorCritic, obs14, mm, bm, pid: int, device):
    """对手（player1 网络）批量决策，返回 (N,2) 动作。恒 pid=0 + 通道重排。"""
    with torch.no_grad():
        o = _swap_player_channels(obs14) if pid == 1 else obs14
        a, _, _ = net.act(o, mm[:, 1].bool(), bm[:, 1].bool(), 0)
    return a


def _swap_player_channels(obs: torch.Tensor) -> torch.Tensor:
    """物理玩家 0/1 的 per-player 通道互换（play/duel.py 同款，内联免 pygame）。"""
    c = obs.shape[1]
    out = obs.clone()
    if c >= 2:
        out[:, [0, 1]] = obs[:, [1, 0]]          # 位置
    if c >= 4:
        out[:, [2, 3]] = obs[:, [3, 2]]          # 引信
    if c >= 9:
        out[:, [8, 9]] = obs[:, [9, 8]]          # 无敌
    if c >= 11:
        out[:, [10, 11]] = obs[:, [11, 10]]      # 可用泡
    if c >= 13:
        out[:, [12, 13]] = obs[:, [13, 12]]      # 泡上限
    return out


def obs7_batch(sim: BatchedSim, obs14: torch.Tensor) -> torch.Tensor:
    """从共享 obs14 + sim 状态批量组装 jax 7 通道双视角 (N,2,7,H,W)。

    ch5 = 泡威力图（bomb_blast/BLAST，不含危险图 —— 化繁为简的关键）。
    ch6 = 进度 t/MAX_STEPS（共享通道 6）。
    """
    n = sim.num_envs
    fuse = sim.fuse.float()                                   # (N,H,W)
    bomb_blast = sim.bomb_blast.float()
    bombed = (fuse > 0).float()
    blast_map = torch.where(bomb_blast > 0, bomb_blast, torch.full_like(
        bomb_blast, float(BLAST)))
    ch5 = bombed * (blast_map / float(BLAST))                 # (N,H,W)
    ch6 = obs14[:, 6:7]                                       # (N,1,H,W) 进度
    # pid0 视角：我=共享0/2、对手=共享1/3 → VIEW_MAP5；pid1 视角对调
    o7 = torch.stack([
        torch.cat([obs14[:, VIEW_MAP5], ch5[:, None], ch6], dim=1),
        torch.cat([obs14[:, VIEW1_MAP5], ch5[:, None], ch6], dim=1),
    ], dim=1)                                                 # (N,2,7,H,W)
    return o7


def run_batch(args, teacher: ActorCritic, opp_net, opp_bot, device,
              name: str, out_dir: str, n: int) -> int:
    """批量收集：teacher vs 一个对手，跑 --ticks 个并行 tick（每 tick N 局）。

    teacher 恒 player0；对手为 player1。auto_reset=True：终局 env 重置续跑，
    每 tick 只收集**该 tick 存活** env 的 (obs7, logits)（死亡帧是终局噪声，
    且 auto_reset 后同一 tick 的状态已是新局，排除更干净）。
    """
    sim = BatchedSim(make_cfg(), n, device=device, seed=args.seed)
    os.makedirs(out_dir, exist_ok=True)
    if opp_bot is not None:
        opp_bot = make_bot(sim, opp_bot)
    obs_acc, lg_acc, mm_acc, bm_acc = [], [], [], []
    t0 = time.time()
    for tick in range(args.ticks):
        obs14 = sim.observe()                                  # (N,14,H,W)
        mm, bm = sim.legal_mask()
        # teacher 双视角 logits（每局面 2 视角各一帧，同 student 2N 批量格式）
        ml0, bl0 = teacher_logits(teacher, obs14, mm, bm, 0)
        ml1, bl1 = teacher_logits(teacher, obs14, mm, bm, 1)
        a0 = torch.stack([ml0.argmax(-1), bl0.argmax(-1)], dim=-1)
        # 对手（player1）
        if opp_net is not None:
            a1 = net_actions(opp_net, obs14, mm, bm, 1, device)
        else:
            a1 = opp_bot.act(obs14, mm[:, 1], bm[:, 1], 1)
        # 双视角 obs7 + logits（mask 后）+ 各自合法掩码
        o7 = obs7_batch(sim, obs14)                            # (N,2,7,H,W)
        lp = torch.stack([torch.cat([ml0, bl0], dim=-1),
                          torch.cat([ml1, bl1], dim=-1)], dim=1)  # (N,2,7)
        alive = sim.alive.all(-1)                               # (N,) 还活着的 env
        obs_acc.append(o7[alive].cpu().numpy())                 # (K,2,7,H,W)
        lg_acc.append(lp[alive].float().cpu().numpy())          # (K,2,7)
        mm_acc.append(mm[alive].bool().cpu().numpy())           # (K,2,5)
        bm_acc.append(bm[alive].bool().cpu().numpy())           # (K,2,2)
        sim.step(torch.stack([a0.to(device), a1], dim=1),
                 auto_reset=True)
        if tick % 100 == 0 and tick:
            dt = time.time() - t0
            nf = sum(len(x) for x in obs_acc)
            print(f"[{name}] tick={tick}/{args.ticks} "
                  f"已收 {nf:,} 帧 {nf/dt:,.0f} 帧/s", flush=True)
    obs_arr = (np.concatenate(obs_acc) * 255.0).round().astype(np.uint8)  # (T,2,7,H,W)
    lg_arr = np.concatenate(lg_acc).astype(np.float32)          # (T,2,7) 存 fp32，防溢出
    mv_arr = np.concatenate(mm_acc).astype(np.bool_)            # (T,2,5)
    bm_arr = np.concatenate(bm_acc).astype(np.bool_)            # (T,2,2)
    T = obs_arr.shape[0]
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, obs7=obs_arr, logits=lg_arr,
                        move_mask=mv_arr, bomb_mask=bm_arr,
                        meta=np.asarray([f"teacher={os.path.basename(args.teacher)} "
                                         f"opp={name} ticks={args.ticks} envs={n}"]))
    dt = time.time() - t0
    print(f"[{name}] {T:,} 帧 -> {path} "
          f"({T*2*7*H*W*1.0/1e6:.0f} MB obs + {T*(7+7)/1e6:.0f} MB logits/mask) "
          f"{dt:.1f}s", flush=True)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default="distill_data")
    ap.add_argument("--ticks", type=int, default=500,
                    help="每对手跑多少个并行 tick（每 tick N 个 env 各产一帧）")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--opp-nets", nargs="*", default=None)
    ap.add_argument("--opp-bots", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} num_envs={args.num_envs}", flush=True)
    teacher = load_net(args.teacher, device)
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"teacher {os.path.basename(args.teacher)} params={n_t:,}", flush=True)

    opp_nets = args.opp_nets or ["duel_nobc_3B", "duel_cnn",
                                 "cnn_course_latest_min", "duel_5x3",
                                 "duel_course_453M"]
    opp_bots = args.opp_bots or ["random", "greedy", "astar"]

    t0 = time.time()
    grand = 0
    for name in opp_nets:
        path = os.path.join(args.ckpt_dir, name + ".pt")
        if not os.path.exists(path):
            print(f"[skip] {name} 无文件", flush=True)
            continue
        try:
            opp = load_net(path, device)
        except Exception as e:
            print(f"[skip] {name} 加载失败: {e}", flush=True)
            continue
        grand += run_batch(args, teacher, opp, None, device, name, args.out,
                           args.num_envs)
    for kind in opp_bots:
        grand += run_batch(args, teacher, None, kind, device, kind, args.out,
                           args.num_envs)
    dt = time.time() - t0
    print(f"总计 {grand:,} 帧，耗时 {dt:.1f}s = {grand/dt:,.0f} 帧/s "
          f"（每局~500 tick → {grand/max(1,dt):,.0f} 帧/s）", flush=True)


if __name__ == "__main__":
    main()
