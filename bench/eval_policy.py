"""纯推理评估：随机 / 启发式 / 已训练 三种策略的移动质量对照。

不训练、不开 grad。目的：把"多少步才算有正常移动"从拍脑袋变成有锚点。

- 随机策略   = 0 步的地板
- 启发式     = 知道全局危险图的参照（几乎不死的上界）
- 已训练     = checkpoint 实测（当前是 16 万步的冒烟模型）

对照场景：player 0 是被测策略，其余玩家随机放泡；统计只算 player 0。
指标（按"存活角色"统计）：
    存活 tick      一局能活多久（上限 --ticks）
    IDLE%          按了发呆的比例，越低越"在动"
    撞墙按动%      按了被掩码挡住的方向，越低移动越聪明
    距离效率       实际位移 / step_len，越接近 1 越接近持续匀速前进
    放泡/100t      每 100 tick 放几个泡
    危险站桩       每 tick 平均站在爆炸范围内的 danger 值（与奖励同源）
    被动%          有泡泡预算却超 passivity_ticks 没放的比例（学没学会定期放泡）
    受伤/100t      每 100 tick 自己掉的血量
    伤敌/100t      每 100 tick 对对手造成的血量（1v1 = 我方输出）

注意：训练在空场（wall_density=0）上，所以"撞墙"指标天然偏低，
衡量移动质量主要看 IDLE%、距离效率和存活；**躲泡**看危险站桩、
受伤/100t 与存活三者的组合 —— 光看存活会被"龟缩角落"骗到。
"""

from __future__ import annotations

import argparse

import torch

from sim.config import MOVE_IDLE, N_BOMB, N_MOVES, SimConfig
from sim.torch_sim import BatchedSim
from train.model import ActorCritic

# (dy, dx)，与 config.DIRS 对齐：0上 1下 2左 3右
D = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def random_action(mm, bm, gen, device) -> torch.Tensor:
    """(N,P,2) 随机动作。multinomial 在 CPU 上生成（MPS 不一定支持），再搬回设备。"""
    n, p, _ = mm.shape
    mm_c, bm_c = mm.float().cpu(), bm.float().cpu()
    mv = torch.multinomial(mm_c.view(-1, N_MOVES), 1, generator=gen).view(n, p)
    bo = torch.multinomial(bm_c.view(-1, N_BOMB), 1, generator=gen).view(n, p)
    return torch.stack([mv, bo], dim=-1).to(device)


def heuristic_action(sim, obs, gen) -> torch.Tensor:
    """朝"非墙且危险最低"的相邻格走，不放泡。

    平局（空场/等危险）用随机破，否则永远选方向 0 会卡成原地踏步。
    """
    cfg = sim.cfg
    n = sim.num_envs
    danger = obs.float()[:, 2 * cfg.n_players + 1].view(n, -1)   # (N,H*W)
    dev = danger.device
    wall = sim.wall.view(n, -1)                                  # (N,H*W)
    acts = torch.zeros(n, cfg.n_players, 2, dtype=torch.long, device=dev)
    for pid in range(cfg.n_players):
        r = sim.pos[:, pid, 0].floor().long().clamp(0, cfg.height - 1)
        c = sim.pos[:, pid, 1].floor().long().clamp(0, cfg.width - 1)
        best = torch.zeros(n, dtype=torch.long, device=dev)
        best_score = torch.full((n,), float("inf"), device=dev)
        for d, (dr, dc) in enumerate(D):
            nr = (r + dr).clamp(0, cfg.height - 1)
            nc = (c + dc).clamp(0, cfg.width - 1)
            flat = nr * cfg.width + nc
            blocked = wall.gather(1, flat.unsqueeze(1)).squeeze(1).bool()
            score = danger.gather(1, flat.unsqueeze(1)).squeeze(1)
            score = torch.where(blocked, torch.full_like(score, 1e9), score)
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best = torch.where(better, torch.full_like(best, d), best)
            tie = score == best_score
            # 平局的候选里随机抽一个，避免永远偏心第一个方向
            coin = torch.randint(0, 2, (n,), generator=gen).to(dev)
            best = torch.where(tie & coin.bool(), torch.full_like(best, d), best)
        acts[:, pid, 0] = best
    return acts


def make_policy(sim, kind: str, net, device, opp_bomb_rate: float):
    gen = torch.Generator(device="cpu").manual_seed(0)

    def policy(obs, mm, bm) -> torch.Tensor:
        a = random_action(mm, bm, gen, device)            # 默认全随机（对手）
        if opp_bomb_rate < 1.0:
            # 只压制**对手**的放泡（索引 1..P-1）；player 0 是各策略的本体，
            # 必须保持原样，否则 random/heuristic 被变成"和平主义者"，
            # 和 trained 的对比就失真了（它靠活得久骗过高存活）。
            keep = (torch.rand(a[:, 1:, 1].shape) <= opp_bomb_rate).to(a.device)
            a[:, 1:, 1] = a[:, 1:, 1] * keep.long()
        if kind == "heuristic":
            a[:, 0, :] = heuristic_action(sim, obs, gen)[:, 0, :]
        elif kind == "trained":
            with torch.no_grad():
                a0, _, _ = net.act(obs, mm[:, 0], bm[:, 0], 0)
            a[:, 0, :] = a0
        return a

    return policy


def run(sim, policy, max_ticks: int) -> dict:
    step_len = sim.cfg.speed / sim.cfg.tick_hz
    n = sim.num_envs
    dev = sim.device
    cnt = {"idle": 0, "wallpress": 0, "bombs": 0, "apt": 0,
           "self_hit": 0, "self_hit_own": 0}
    cnt["dng"] = 0.0            # 每 tick 累计 danger 值（按存活角色）
    cnt["passive"] = 0          # 有预算却超阈值没放泡的 tick 数
    cnt["budget"] = 0           # 有泡泡预算的 tick 数（存活时）
    dmg_taken = torch.zeros(n, device=dev)
    dmg_dealt = torch.zeros(n, device=dev)
    dist = 0.0
    death = torch.zeros(n, device=dev)
    dead = torch.zeros(n, dtype=torch.bool, device=dev)
    cfg = sim.cfg

    def danger_at(pos: torch.Tensor) -> torch.Tensor:
        """角色中心格的 danger 值（与训练奖励同源）。"""
        from sim.blast import danger_map
        from sim.move import center_cell
        dmap = danger_map(sim.fuse, sim.wall, cfg.blast, cfg.fuse)
        cell = center_cell(pos)
        flat = cell[..., 0] * cfg.width + cell[..., 1]
        return dmap.view(n, -1).gather(1, flat.unsqueeze(1)).squeeze(1)

    for t in range(1, max_ticks + 1):
        obs = sim.observe()
        mm, bm = sim.legal_mask()
        pos0 = sim.pos[:, 0].clone()
        alive0 = sim.alive[:, 0]
        hp0 = sim.hp[:, 0].clone()
        hp_opp0 = sim.hp[:, 1:].clone()
        a = policy(obs, mm, bm)
        _, done, _ = sim.step(a, auto_reset=False)

        ap0 = a[:, 0, :]
        idle = ap0[:, 0] == MOVE_IDLE
        wallpress = (ap0[:, 0] < 4) & (~mm[:, 0].gather(1, ap0[:, 0:1]).squeeze(1))
        moved = (sim.pos[:, 0] - pos0).norm(dim=-1)
        bomb = ap0[:, 1] == 1
        m = alive0
        cnt["apt"] += int(m.sum())
        cnt["idle"] += int((idle & m).sum())
        cnt["wallpress"] += int((wallpress & m).sum())
        dist += float((moved * m.float()).sum())
        cnt["bombs"] += int((bomb & m).sum())
        # 危险区站桩：与奖励同源，中心格 gather 当前在场泡泡的爆炸范围
        dng = danger_at(sim.pos[:, 0])
        cnt["dng"] += float((dng * m.float()).sum())
        # 自伤率：掉血时自己名下泡泡在场（近似"被自己炸"）的比例 ——
        # 用户实测"AI 不会躲自己的炸弹"，用这个指标追踪每版是否改善
        dmg_now = (hp0 - sim.hp[:, 0]).clamp(min=0)
        own_bombs = ((sim.owner == 0) & (sim.fuse > 0)).flatten(1).sum(dim=1) > 0
        cnt["self_hit"] += int((dmg_now > 0).sum())
        cnt["self_hit_own"] += int(((dmg_now > 0) & own_bombs).sum())
        # 被动罚触发率：有预算（在场泡数 < max_bombs）却已超 passivity_ticks 没放
        live = ((sim.owner == 0) & (sim.fuse > 0)).flatten(1).sum(dim=1)
        budget = (live < cfg.max_bombs) & m
        cnt["budget"] += int(budget.sum())
        since = sim.since_bomb[:, 0]
        cnt["passive"] += int((budget & (since >= cfg.passivity_ticks)).sum())
        # 伤害：掉血 1 格 = 受 1 点伤害；1v1 里我方造成伤害 = 对手掉血
        dmg_taken += ((hp0 - sim.hp[:, 0]).clamp(min=0)).float()
        dmg_dealt += (hp_opp0 - sim.hp[:, 1:]).clamp(min=0).sum(dim=1).float()
        just = done & ~dead
        death[just] = t
        dead |= done
        if bool(dead.all()):
            break
    death[~dead] = max_ticks
    apt = max(1, cnt["apt"])
    return {
        "survival": float(death.mean()),
        "idle_rate": cnt["idle"] / apt,
        "wall_press_rate": cnt["wallpress"] / apt,
        "dist_eff": dist / apt / step_len,
        "bomb_per_100": cnt["bombs"] / max(1, max_ticks / 100.0),
        "dead_frac": float(dead.float().mean()),
        "danger_standing": cnt["dng"] / apt,
        "passive_rate": cnt["passive"] / max(1, cnt["budget"]),
        "dmg_taken": float(dmg_taken.sum()) / max(1, max_ticks / 100.0),
        "dmg_dealt": float(dmg_dealt.sum()) / max(1, max_ticks / 100.0),
        "self_hit_rate": cnt["self_hit_own"] / max(1, cnt["self_hit"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="all",
                    choices=["random", "heuristic", "trained", "all"])
    ap.add_argument("--ckpt", default="ckpt/eval_ckpt.pt")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--opp-bomb-rate", type=float, default=0.05,
                    help="对手的放泡概率；默认 0.05 ≈ 3 泡/100t。满放泡时满地雷，"
                         "谁都是十几 tick 死，移动质量测不出来")
    ap.add_argument("--map-mode", default="open", choices=["open", "corridor"],
                    help="评测地图：corridor 含可炸墙/宝箱/成长，与训练一致")
    args = ap.parse_args()

    net = None
    obs_shape = None
    if args.policy in ("trained", "all"):
        ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        obs_shape = tuple(ck["obs_shape"])
        arch = ck.get("arch", "cnn")          # 旧 ckpt 无 arch 字段 → 默认 cnn
        np_ck = ck.get("n_players")           # 含 obs_extra 的新 ckpt 存了 n_players
        net = ActorCritic(obs_shape, arch=arch, n_players=np_ck).to(args.device)
        net.load_state_dict(ck["model"])
        net.eval()
        c, h, w = obs_shape
        p = net.n_players
        extra_ck = obs_shape[0] > 2 * p + 3   # 新 14 通道 ckpt → 开扩展观测
    else:
        h = w = 13
        p = 2
        extra_ck = True
    if args.map_mode == "corridor":
        cfg = SimConfig(height=h, width=w, n_players=p, map_mode="corridor",
                        speed=3.0, max_steps=1800, obs_extra_enabled=extra_ck)
    else:
        cfg = SimConfig(height=h, width=w, n_players=p,
                        obs_extra_enabled=extra_ck)      # 与训练同地图（空场）

    kinds = [args.policy] if args.policy != "all" else ["random", "heuristic", "trained"]
    header = (f"{'策略':<10}{'存活tick':>9}{'IDLE%':>8}{'撞墙%':>8}{'距离效率':>8}"
              f"{'放泡/100t':>10}{'危险站桩':>9}{'被动%':>8}{'受伤/100t':>10}"
              f"{'伤敌/100t':>10}{'死亡率':>8}{'自伤率':>8}")
    print(header)
    for kind in kinds:
        sim = BatchedSim(cfg, args.envs, device=args.device, seed=0)
        r = run(sim, make_policy(sim, kind, net, args.device, args.opp_bomb_rate),
                args.ticks)
        print(f"{kind:<10}{r['survival']:>9.0f}{100 * r['idle_rate']:>8.1f}"
              f"{100 * r['wall_press_rate']:>8.1f}{r['dist_eff']:>8.2f}"
              f"{r['bomb_per_100']:>10.1f}{r['danger_standing']:>9.3f}"
              f"{100 * r['passive_rate']:>8.1f}{r['dmg_taken']:>10.2f}"
              f"{r['dmg_dealt']:>10.2f}{100 * r['dead_frac']:>8.1f}"
              f"{100 * r['self_hit_rate']:>8.1f}")


if __name__ == "__main__":
    main()
