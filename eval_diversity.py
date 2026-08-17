"""多样性评测：每个候选 ckpt / 规则 AI 对战参照对手（astar），
统计行为特征向量，用于筛选蒸馏/对打的对手池（选分布差异大的）。

每个候选打 N 局（--games 默认 5，用户说"测个几盘就够了"），
输出每候选的特征向量 + 与参照的对比，供人工挑选。

特征（每局统计后取平均）：
  win_rate      对参照对手的胜率（0/1/0.5）
  ticks         对局时长
  bomb_rate     放泡概率（bomb=1 的 tick 占比）
  idle_rate     MOVE_IDLE 占比
  move_rate     实际移动（非 idle）占比
  dir_entropy   四方向移动的分布熵（0 = 只会一个方向，1 = 四向均匀）
  speed         avg 位移/ tick（0.756 满速 ≈ 0.756）
  pos_entropy   位置分布熵（0 = 钉死一点，1 = 均匀覆盖全场）
  approach      avg(自己到对手距离) / 对角线长（<0.5 爱贴脸，>0.5 爱躲）

用法：
  .venv/bin/python eval_diversity.py --games 5
"""
import argparse
import math
import os

import numpy as np
import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from sim.bots import make_bot
from play.duel import ai_action, _swap_player_channels
from train.model import ActorCritic

DEVICE = "cpu"
H, W, P = 13, 13, 2
DIAG = math.hypot(H - 1, W - 1)


def make_cfg():
    return SimConfig(height=H, width=W, n_players=P, map_mode="open")


def load_net(path: str) -> ActorCritic:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(tuple(ck["obs_shape"]), arch=ck["arch"],
                      n_players=ck.get("n_players", P))
    net.load_state_dict(ck["model"] if isinstance(ck["model"], dict)
                        else ck["model"].state_dict())
    net.eval()
    return net


def net_action(net: ActorCritic, sim: BatchedSim, obs, mm, bm, pid: int,
               hidden_st=None):
    """网络决策：恒 pid=0 视角（训练时 learner 恒为 player0），物理 P1 靠重排。
    LSTM 用局部特征三元组（自带视角），不需要重排。"""
    if getattr(net, "arch", "cnn") == "lstm":
        from sim.obs import local_view_features
        lf = local_view_features(sim.cfg, obs, sim.pos, sim.alive,
                                 sim.t, sim.fuse, sim.hp, only_p0=False)
        feats = (lf[0][:, pid], lf[1][:, pid], lf[2][:, pid])
        with torch.no_grad():
            a, _, _, h = net.act(feats, mm[:, pid], bm[:, pid], 0, hidden_st[pid])
        hidden_st[pid] = h
        return a
    with torch.no_grad():
        if pid == 1:
            a = net.act(_swap_player_channels(obs), mm[:, 1], bm[:, 1], 0)[0]
        else:
            a = ai_action(net, obs, mm, bm, DEVICE, pid)
    return a


def one_game(sim: BatchedSim, cand, ref, seed: int) -> dict:
    """跑一局：cand 是 player0 的决策函数（或 bot），ref 是 player1。"""
    torch.manual_seed(seed)
    sim.reset_all()
    stats = {"bombs": 0, "moves": [0] * 5, "pos": [], "dist": [], "ticks": 0}
    done = False
    for t in range(int(sim.cfg.max_steps) + 1):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        a0 = cand(0, obs, mm, bm)
        a1 = ref(1, obs, mm, bm)
        p0 = sim.pos[0, 0].clone()
        rew, done_any, info = sim.step(torch.stack([a0.to(DEVICE), a1], dim=1),
                                       auto_reset=False)
        a0c = a0[0].tolist()
        stats["moves"][a0c[0]] += 1
        stats["bombs"] += a0c[1]
        p1 = sim.pos[0, 1]
        stats["pos"].append(sim.pos[0, 0].tolist())
        stats["dist"].append(float((p0 - p1).norm().item()))
        if done_any:
            done = True
            stats["ticks"] = t + 1
            # 胜负：player0 存活即胜（双亡/超时算平）
            alive0, alive1 = sim.alive[0, 0].item(), sim.alive[0, 1].item()
            stats["win"] = 1.0 if (alive0 and not alive1) else \
                           (0.0 if (not alive0 and alive1) else 0.5)
            break
    if not done:
        stats["ticks"] = int(sim.cfg.max_steps)
        stats["win"] = 0.5
    return stats


def featurize(stats_list: list[dict]) -> dict:
    n = len(stats_list)
    out = {}
    out["win_rate"] = np.mean([s["win"] for s in stats_list])
    out["ticks"] = np.mean([s["ticks"] for s in stats_list])
    out["bomb_rate"] = np.mean([s["bombs"] / max(1, s["ticks"]) for s in stats_list])
    mv = np.mean([np.array(s["moves"], float) for s in stats_list], axis=0)
    tot = mv.sum()
    out["idle_rate"] = mv[0] / max(1, tot)
    d = mv[1:] / max(1, mv[1:].sum())
    d = d[d > 0]
    out["dir_entropy"] = float(-(d * np.log(d)).sum() / math.log(4)) if len(d) else 0.0
    spd = []
    pos_all = []
    dist_all = []
    for s in stats_list:
        pos = np.array(s["pos"])
        if len(pos) > 1:
            spd.append(float(np.abs(np.diff(pos, axis=0)).sum(1).mean()))
        pos_all.append(pos)
        dist_all += s["dist"]
    out["speed"] = float(np.mean(spd)) if spd else 0.0
    pos = np.concatenate(pos_all) if pos_all else np.zeros((1, 2))
    h = np.histogram2d(pos[:, 0], pos[:, 1], bins=(H, W), range=[[0, H], [0, W]])[0]
    ph = h / max(1, h.sum())
    ph = ph[ph > 0]
    out["pos_entropy"] = float(-(ph * np.log(ph)).sum() / math.log(H * W))
    out["approach"] = float(np.mean(dist_all) / DIAG)
    return out


CAND_NETS = [
    "duel_nobc_3B", "duel_nobc_4B", "duel_nobc_5.95B", "duel_nobc_6.40B",
    "duel_nobc_6.46B", "duel_nobc_8b_live", "duel_nobc_11b_live",
    "duel_nobc_e3970", "duel_5x3", "duel_cnn", "duel_cnn_cpu",
    "duel_course_453M", "course_1023m", "course_bc_571m",
    "cnn_course_latest_min", "cnn_course_min",
]
CAND_BOTS = ["random", "greedy", "astar", "hunter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--nets", nargs="*", default=None,
                    help="只测指定 ckpt（名字，不含 .pt）")
    args = ap.parse_args()

    cfg = make_cfg()
    results = {}

    # 参照对手：固定 astar（玩家1）
    def make_ref():
        sim = BatchedSim(cfg, 1, device=DEVICE, seed=0)
        bot = make_bot(sim, "astar")
        return sim, lambda pid, obs, mm, bm: bot.act(obs, mm[:, pid], bm[:, pid], pid)

    for name in (args.nets or CAND_NETS):
        path = os.path.join(args.ckpt_dir, name + ".pt")
        if not os.path.exists(path):
            print(f"[skip] {name}: 无文件", flush=True)
            continue
        sim, ref = make_ref()
        try:
            net = load_net(path)
        except Exception as e:
            print(f"[skip] {name}: 加载失败 {type(e).__name__}: {e}", flush=True)
            continue
        hidden_st = {0: None, 1: None}
        cand = lambda pid, obs, mm, bm, _net=net: net_action(  # noqa
            _net, sim, obs, mm, bm, pid, hidden_st)
        st = []
        for g in range(args.games):
            st.append(one_game(sim, cand, ref, args.seed_base + g))
        results[name] = featurize(st)
        print(f"{name:22s} {results[name]}", flush=True)

    for kind in CAND_BOTS:
        sim, ref = make_ref()
        bot0 = make_bot(sim, kind)
        cand = lambda pid, obs, mm, bm, _b=bot0: _b.act(obs, mm[:, pid], bm[:, pid], pid)  # noqa
        st = []
        for g in range(args.games):
            st.append(one_game(sim, cand, ref, args.seed_base + g))
        results[kind] = featurize(st)
        print(f"{kind:22s} {results[kind]}", flush=True)

    print("\n===== 特征表（win, ticks, bomb, idle, dirH, speed, posH, approach）=====")
    names = list(results)
    keys = ["win_rate", "ticks", "bomb_rate", "idle_rate", "dir_entropy",
            "speed", "pos_entropy", "approach"]
    print("name".ljust(22), " ".join(f"{k[:6]:>8}" for k in keys))
    for n in names:
        r = results[n]
        row = " ".join(f"{r[k]:8.3f}" for k in keys)
        print(n.ljust(22), row)


if __name__ == "__main__":
    main()
