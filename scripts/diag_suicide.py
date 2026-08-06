"""793M vs 597M 实测：大样本胜率 + 自杀率 + 死亡方式分布。

自杀判定（与 duel --die-log 同源）：死亡 tick 时自己名下有在场泡
（owner==自己 & fuse>0，step 前快照）→ 自爆嫌疑。

**P1 侧必须 swap_channels**（和 duel/launcher 真实对打一致）：模型一律用
pid=0 视角，物理 P1 的模型观测重排成"自己=通道0"。否则 P1 位直接用原始 obs
会把对手当自己（per-player 通道错位）→ 疯狂自爆（错误的评测会夸大自杀）。
"""
import sys, torch
sys.path.insert(0, "/Users/pippo/operater-dev/qqt-gpu-sim")
from sim.config import SimConfig
from sim.torch_sim import BatchedSim
from train.train import load_fixed_checkpoint

CFG = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.5, ring_fraction=0.0, open_crate_cross=True,
                hit_attr_penalty=2)


def swap_channels(obs, p=2):
    base = 2 * p + 3
    idx = list(range(obs.shape[1]))
    for seg in (range(0, p), range(p, 2 * p),
                range(base + 1, base + 1 + p),
                range(base + 1 + p, base + 1 + 2 * p),
                range(base + 1 + 2 * p, base + 1 + 3 * p)):
        seg = list(seg)
        idx[seg[0]], idx[seg[1]] = idx[seg[1]], idx[seg[0]]
    return obs[:, idx]


def duel_with_death(sim, pol0, pol1, episodes):
    """返回 (win0, draw, win1, P0自杀, P0他杀, 局数)。polX 已含 swap 视角。"""
    n = sim.num_envs; dev = sim.device
    sim.reset_all()
    w0 = w1 = dr = rounds = 0
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    suicide = torch.zeros(n, dtype=torch.long, device=dev)
    killed = torch.zeros(n, dtype=torch.long, device=dev)
    while rounds < episodes:
        obs = sim.observe(); mm, bm = sim.legal_mask()
        a0 = pol0(obs, mm[:, 0], bm[:, 0]); a1 = pol1(obs, mm[:, 1], bm[:, 1])
        owner_snap = sim.owner.clone()
        fuse_snap = sim.fuse.clone()
        _, d, info = sim.step(torch.stack([a0, a1], 1))
        died0 = info["died"][:, 0]
        own_live = (owner_snap == 0) & (fuse_snap > 0)
        own_cnt = own_live.flatten(1).sum(dim=1)
        suicide += (died0 & (own_cnt > 0)).long()
        killed += (died0 & (own_cnt == 0)).long()
        just = d & ~done
        w0 += int((just & info["winner"][:, 0]).sum())
        w1 += int((just & info["winner"][:, 1]).sum())
        dr += int((just & ~info["winner"][:, 0] & ~info["winner"][:, 1]).sum())
        done |= d; rounds += int(just.sum())
        if bool(done.all()):
            sim.reset_all(); done.zero_()
    return w0/max(1,rounds), dr/max(1,rounds), w1/max(1,rounds), \
        int(suicide.sum()), int(killed.sum()), rounds

def main():
    a, b = "ckpt/course_793m.pt", "ckpt/course_597m.pt"
    episodes = 256
    sim = BatchedSim(CFG, 128, device="cpu", seed=0)
    netA = load_fixed_checkpoint(a, CFG.obs_shape, "cpu"); netA.eval()
    netB = load_fixed_checkpoint(b, CFG.obs_shape, "cpu"); netB.eval()
    @torch.no_grad()
    def pa(o,m,bb): return netA.act(o,m,bb,0)[0]
    @torch.no_grad()
    def pa_swap(o,m,bb): return netA.act(swap_channels(o),m,bb,0)[0]
    @torch.no_grad()
    def pb(o,m,bb): return netB.act(o,m,bb,0)[0]
    @torch.no_grad()
    def pb_swap(o,m,bb): return netB.act(swap_channels(o),m,bb,0)[0]
    print(f"=== {a} vs {b}（P1 侧 swap 视角，{episodes} 局）===")
    w0, dr, w1, su, ki, rd = duel_with_death(sim, pa, pb_swap, episodes)
    print(f"793M(P0) win {w0:.1%} / draw {dr:.1%} / 597M(P1 swap) win {w1:.1%}  ({rd} 局)")
    print(f"793M 自杀 {su} / 他杀 {ki}")
    sim.reset_all()
    w0, dr, w1, su, ki, rd = duel_with_death(sim, pb, pa_swap, episodes)
    print(f"\n597M(P0) win {w0:.1%} / draw {dr:.1%} / 793M(P1 swap) win {w1:.1%}  ({rd} 局)")
    print(f"793M(P1) 自杀 {su} / 他杀 {ki}")

if __name__ == "__main__":
    main()
