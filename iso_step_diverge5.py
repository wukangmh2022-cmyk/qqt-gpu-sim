"""seed(0) 确定性：找第一个发散 tick，然后逐步重放该 tick 定位子步骤。"""
import sys, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim, center_cell
from sim.triton_sim import place_bombs_triton, move_players_triton, resolve_triton
from sim.blast import resolve_explosions
from sim.triton_step import triton_step_core
from sim.dev import pick_device

dev = pick_device()
N = 2048
cfg = SimConfig(map_mode="corridor", speed=3.0, max_steps=1800,
                open_fraction=0.0, timeout_draw=True, combo_reward=0.0,
                hit_attr_penalty=0, place_cover_reward=0.0,
                place_chain_reward=0.0, place_dist_reward=0.0,
                chain_blast_bonus=0.0, growth_crate_prob=0.0,
                brick_reward=0.0, win_bonus=0.0, recycle_crate_prob=0.0)
T = 40
acts_seq = []
for i in range(T):
    mv = torch.randint(0, 5, (N, 2), device=dev)
    bmb = (torch.rand(N, 2, device=dev) < 0.4).long()
    acts_seq.append(torch.stack([mv, bmb], dim=-1))

def sync():
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("npu"):
        torch.npu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()

sim_t = BatchedSim(cfg, N, device=dev, seed=0)
sim_k = BatchedSim(cfg, N, device=dev, seed=0)
sim_t.reset_all(); sim_k.reset_all()

FIELDS = ["fuse", "owner", "bomb_blast", "pos", "alive", "hp", "invuln",
          "t", "since_bomb", "bombs_cap", "blast_cap", "spd_g", "crate", "brick"]
div = None
for i in range(T):
    sim_t.step(acts_seq[i])
    pk, ck, tk, dk = triton_step_core(sim_k, acts_seq[i])
    if bool(dk.any()):
        sim_k.reset_(dk)
    sync()
    bad = {f: int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
           for f in FIELDS if f != "t"}
    bad = {f: v for f, v in bad.items() if v > 0}
    if bad:
        div = i
        print(f"首个发散 tick {i}: {bad}")
        # 直接在 live sims 上对比 resolve（不重建）
        cov_t, trig_t = resolve_explosions(sim_t.fuse, sim_t.owner, sim_t.wall,
                                           sim_t._blast_map(), cfg.max_chain,
                                           sim_t.brick)
        cov_k, trig_k = resolve_triton(sim_k.fuse, sim_k.owner, sim_k.wall,
                                       sim_k.bomb_blast, sim_k.brick,
                                       cfg.max_chain)
        sync()
        print("  live covered 差:", int((cov_t != cov_k).sum()),
              "| triggered 差:", int((trig_t != trig_k).sum()))
        for f in ("crate", "brick"):
            dd = int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
            if dd:
                print(f"  {f} 差 {dd}:")
                for it in (getattr(sim_t, f) != getattr(sim_k, f)).nonzero()[:6]:
                    tt = tuple(it.tolist())
                    env, r, c = tt
                    print(f"    @env {env} ({r},{c}) t={getattr(sim_t, f)[tt].item()} "
                          f"k={getattr(sim_k, f)[tt].item()} fuse_t={sim_t.fuse[env,r,c].item()} "
                          f"owner_t={sim_t.owner[env,r,c].item()} wall={sim_t.wall[env,r,c].item()}")
                    if f == "crate":
                        # dump env 泡布局（fuse>0 格）
                        bm_t = (sim_t.fuse[env] > 0)
                        for bi in bm_t.nonzero()[:14]:
                            br, bc = int(bi[0]), int(bi[1])
                            print(f"      bomb_t @({br},{bc}) fuse={sim_t.fuse[env,br,bc].item()} "
                                  f"owner={sim_t.owner[env,br,bc].item()} "
                                  f"blast={sim_t.bomb_blast[env,br,bc].item()} | "
                                  f"k: fuse={sim_k.fuse[env,br,bc].item()} "
                                  f"owner={sim_k.owner[env,br,bc].item()} "
                                  f"blast={sim_k.bomb_blast[env,br,bc].item()}")
        break
if div is None:
    print("40 tick 全部一致 ✓")
    sys.exit(0)

# 重放 div tick（从干净起点：已经跑到 div-1，pre-state 一致）
a = acts_seq[div]
move, bomb = a[..., 0], a[..., 1]
alive0 = sim_t.alive.clone()

def cmp(tag, *pairs):
    for name, x, y in pairs:
        dd = int((x != y).sum())
        print(f"  {tag} {name:<14}: 差 {dd} {'✓' if dd == 0 else '✗'}")
        if dd and dd < 8:
            for it in (x != y).nonzero()[:3]:
                tt = tuple(it.tolist())
                print(f"      t={x[tt].item()} k={y[tt].item()} @{tt}")

torch.where(sim_t.fuse > 0, sim_t.fuse - 1, sim_t.fuse, out=sim_t.fuse)
torch.where(sim_k.fuse > 0, sim_k.fuse - 1, sim_k.fuse, out=sim_k.fuse)
cmp("fuse_dec", ("fuse", sim_t.fuse, sim_k.fuse))

placed_t = sim_t._place_bombs(bomb, alive0, sim_t.bombs_cap, sim_t.blast_cap)
placed_k = place_bombs_triton(cfg, sim_k.fuse, sim_k.owner, sim_k.bomb_blast,
                              sim_k.pos, sim_k.alive, bomb,
                              sim_k.bombs_cap, sim_k.blast_cap, sim_k.brick,
                              sim_k.wall)
cmp("place", ("placed", placed_t, placed_k),
    ("fuse", sim_t.fuse, sim_k.fuse), ("owner", sim_t.owner, sim_k.owner),
    ("bomb_blast", sim_t.bomb_blast, sim_k.bomb_blast))
if int((placed_t != placed_k).sum()):
    for it in (placed_t != placed_k).nonzero()[:5]:
        tt = tuple(it.tolist())
        env, pl = tt
        cell = center_cell(sim_t.pos[env:env+1, pl:pl+1])
        r, c = int(cell[0, 0, 0]), int(cell[0, 0, 1])
        print(f"    env {env} pl {pl} t={placed_t[tt].item()} k={placed_k[tt].item()} "
              f"pos=({sim_t.pos[env,pl,0].item():.4f},{sim_t.pos[env,pl,1].item():.4f}) "
              f"cell=({r},{c}) alive={alive0[env,pl].item()} bomb_act={bomb[env,pl].item()} "
              f"bombs_cap={sim_t.bombs_cap[env,pl].item()} "
              f"cur_fuse_t={sim_t.fuse[env,r,c].item()} cur_fuse_k={sim_k.fuse[env,r,c].item()} "
              f"on_brick={sim_t.brick[env,r,c].item()} on_wall={sim_t.wall[env,r,c].item()}")

sim_t.since_bomb.add_(1); sim_t.since_bomb[placed_t] = 0
sim_k.since_bomb.add_(1); sim_k.since_bomb[placed_k] = 0
cmp("since", ("since_bomb", sim_t.since_bomb, sim_k.since_bomb))

blocked_t = sim_t.wall | sim_t.brick | (sim_t.fuse > 0)
blocked_k = sim_k.wall | sim_k.brick | (sim_k.fuse > 0)
p_t = move_players_triton(cfg, sim_t.pos, move, sim_t.alive, blocked_t, sim_t.spd_g)
p_k = move_players_triton(cfg, sim_k.pos, move, sim_k.alive, blocked_k, sim_k.spd_g)
cmp("move", ("pos_out", p_t, p_k), ("blocked", blocked_t, blocked_k))
