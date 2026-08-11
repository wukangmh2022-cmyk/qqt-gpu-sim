"""手动 step 流水线：两个 sim 用相同子步骤代码逐步执行，每子步骤对比。

消解"重放 vs 真实 step"的差异——全部走手动路径，找到第一个分叉子步骤。
"""
import sys, torch
torch.manual_seed(0)
sys.path.insert(0, ".")

from sim.config import SimConfig
from sim.torch_sim import BatchedSim, center_cell
from sim.triton_sim import place_bombs_triton, move_players_triton, resolve_triton
from sim.blast import resolve_explosions
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

def cmp(tag, *pairs):
    bad = []
    for name, x, y in pairs:
        dd = int((x != y).sum())
        if dd:
            bad.append((name, dd))
            print(f"  [{tag}] {name:<12}: 差 {dd} ✗")
            for it in (x != y).nonzero()[:4]:
                tt = tuple(it.tolist())
                print(f"      t={x[tt].item()} k={y[tt].item()} @{tt}")
    if not bad:
        print(f"  [{tag}] 一致 ✓")
    return bad

for i in range(T):
    a = acts_seq[i]
    move, bomb = a[..., 0], a[..., 1]
    alive0_t = sim_t.alive.clone(); alive0_k = sim_k.alive.clone()

    # 1 fuse_dec
    torch.where(sim_t.fuse > 0, sim_t.fuse - 1, sim_t.fuse, out=sim_t.fuse)
    torch.where(sim_k.fuse > 0, sim_k.fuse - 1, sim_k.fuse, out=sim_k.fuse)
    sync()

    # 2 place
    pt = sim_t._place_bombs(bomb, alive0_t, sim_t.bombs_cap, sim_t.blast_cap)
    pk = place_bombs_triton(cfg, sim_k.fuse, sim_k.owner, sim_k.bomb_blast,
                            sim_k.pos, sim_k.alive, bomb,
                            sim_k.bombs_cap, sim_k.blast_cap, sim_k.brick,
                            sim_k.wall)
    sync()
    b_place = cmp("place", ("placed", pt, pk), ("fuse", sim_t.fuse, sim_k.fuse),
                  ("owner", sim_t.owner, sim_k.owner),
                  ("bomb_blast", sim_t.bomb_blast, sim_k.bomb_blast))
    if b_place:
        print(f"→ tick {i} place 分叉"); sys.exit(1)

    # 3 since_bomb
    sim_t.since_bomb.add_(1); sim_t.since_bomb[pt] = 0
    sim_k.since_bomb.add_(1); sim_k.since_bomb[pk] = 0
    sync()

    # 4 move
    bt_ = sim_t.wall | sim_t.brick | (sim_t.fuse > 0)
    bk_ = sim_k.wall | sim_k.brick | (sim_k.fuse > 0)
    pt_ = move_players_triton(cfg, sim_t.pos, move, sim_t.alive, bt_, sim_t.spd_g)
    pk_ = move_players_triton(cfg, sim_k.pos, move, sim_k.alive, bk_, sim_k.spd_g)
    sync()
    b_move = cmp("move", ("pos", pt_, pk_), ("blocked", bt_, bk_))
    sim_t.pos.copy_(pt_); sim_k.pos.copy_(pk_)
    if b_move:
        print(f"→ tick {i} move 分叉"); sys.exit(1)

    # 5 resolve
    cov_t, trig_t = resolve_explosions(sim_t.fuse, sim_t.owner, sim_t.wall,
                                       sim_t._blast_map(), cfg.max_chain, sim_t.brick)
    cov_k, trig_k = resolve_triton(sim_k.fuse, sim_k.owner, sim_k.wall,
                                   sim_k.bomb_blast, sim_k.brick, cfg.max_chain)
    sync()
    b_res = cmp("resolve", ("covered", cov_t, cov_k), ("triggered", trig_t, trig_k))
    if b_res:
        print(f"→ tick {i} resolve 分叉")
        for f in ("fuse", "owner", "bomb_blast"):
            dd = int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
            print(f"   pre-resolve {f} 差 {dd}")
        sys.exit(1)

    # 5.5 crate/brick
    if cfg.map_mode == "corridor":
        sim_t.crate.bitwise_or_(sim_t.brick & cov_t)
        sim_k.crate.bitwise_or_(sim_k.brick & cov_k)
    sim_t.brick.bitwise_and_(~cov_t)
    sim_k.brick.bitwise_and_(~cov_k)
    sync()
    b_cr = cmp("crate", ("crate", sim_t.crate, sim_k.crate), ("brick", sim_t.brick, sim_k.brick))
    if b_cr:
        print(f"→ tick {i} crate/brick 分叉")
        for f in ("crate", "brick"):
            dd = int((getattr(sim_t, f) != getattr(sim_k, f)).sum())
            if dd:
                for it in (getattr(sim_t, f) != getattr(sim_k, f)).nonzero()[:4]:
                    tt = tuple(it.tolist())
                    env, r, c = tt
                    print(f"   {f} @env {env} ({r},{c}) t={getattr(sim_t,f)[tt].item()} "
                          f"k={getattr(sim_k,f)[tt].item()}")
                    if f == "crate" and env is not None:
                        # 该 env 的泡
                        for bi in (sim_t.fuse[env] > 0).nonzero()[:10]:
                            br, bc = int(bi[0]), int(bi[1])
                            print(f"      bomb @({br},{bc}) fuse={sim_t.fuse[env,br,bc].item()} "
                                  f"owner={sim_t.owner[env,br,bc].item()} "
                                  f"blast={sim_t.bomb_blast[env,br,bc].item()}")
        sys.exit(1)

    # 6 damage（简化：不比较，只同步执行）
    cell_t = center_cell(sim_t.pos)
    flat_t = (cell_t[..., 0] * cfg.width + cell_t[..., 1]).clamp(0, cfg.height * cfg.width - 1)
    hit_t = alive0_t & cov_t.view(N, -1).gather(1, flat_t)
    inv_ok_t = sim_t.invuln <= 0
    hit_eff_t = hit_t & inv_ok_t
    hp_new_t = (sim_t.hp.to(torch.int32) - hit_eff_t.to(torch.int32)).clamp(min=0)
    died_t = hit_eff_t & (hp_new_t == 0)
    sim_t.hp.copy_(hp_new_t.to(torch.uint8)); sim_t.alive.copy_(alive0_t & ~died_t)
    sim_t.invuln.sub_(1); sim_t.invuln.clamp_(min=0); sim_t.invuln[hit_eff_t] = cfg.invuln_ticks
    cell_k = center_cell(sim_k.pos)
    flat_k = (cell_k[..., 0] * cfg.width + cell_k[..., 1]).clamp(0, cfg.height * cfg.width - 1)
    hit_k = alive0_k & cov_k.view(N, -1).gather(1, flat_k)
    inv_ok_k = sim_k.invuln <= 0
    hit_eff_k = hit_k & inv_ok_k
    hp_new_k = (sim_k.hp.to(torch.int32) - hit_eff_k.to(torch.int32)).clamp(min=0)
    died_k = hit_eff_k & (hp_new_k == 0)
    sim_k.hp.copy_(hp_new_k.to(torch.uint8)); sim_k.alive.copy_(alive0_k & ~died_k)
    sim_k.invuln.sub_(1); sim_k.invuln.clamp_(min=0); sim_k.invuln[hit_eff_k] = cfg.invuln_ticks
    sync()

    # 7 clear
    torch.where(trig_t, torch.zeros_like(sim_t.fuse), sim_t.fuse, out=sim_t.fuse)
    torch.where(trig_t, torch.full_like(sim_t.owner, -1), sim_t.owner, out=sim_t.owner)
    torch.where(trig_t, torch.zeros_like(sim_t.bomb_blast), sim_t.bomb_blast, out=sim_t.bomb_blast)
    torch.where(trig_k, torch.zeros_like(sim_k.fuse), sim_k.fuse, out=sim_k.fuse)
    torch.where(trig_k, torch.full_like(sim_k.owner, -1), sim_k.owner, out=sim_k.owner)
    torch.where(trig_k, torch.zeros_like(sim_k.bomb_blast), sim_k.bomb_blast, out=sim_k.bomb_blast)
    sync()

    # 8 t++
    sim_t.t.add_(1); sim_k.t.add_(1)
    if i % 5 == 0 or i >= 30:
        b = cmp(f"tick {i} 终态", ("fuse", sim_t.fuse, sim_k.fuse),
                ("owner", sim_t.owner, sim_k.owner),
                ("pos", sim_t.pos, sim_k.pos), ("since_bomb", sim_t.since_bomb, sim_k.since_bomb),
                ("crate", sim_t.crate, sim_k.crate))
        if b:
            print(f"→ tick {i} 终态分叉"); sys.exit(1)
print("40 tick 全一致 ✓")
