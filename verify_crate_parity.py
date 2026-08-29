"""逐位一致性对照：同 seed 下 60 步 corridor（带宝箱/成长/回收）状态哈希。

用于验证宝箱段向量化（_grow_vec_all + scatter_ 标量）与旧逐 pl 版逐位一致。
用法：新版跑一次 > new.hash；git stash 后旧版跑一次 > old.hash；对比。
"""

import hashlib

import torch

from sim.config import SimConfig
from sim.torch_sim import BatchedSim


cfg = SimConfig(height=13, width=13, n_players=2, map_mode="corridor",
                speed=3.0, max_steps=1800, open_fraction=1.0, ring_fraction=0.0,
                hazard_fraction=0.0, crate_speed_only=False, timeout_draw=False,
                combo_reward=0.10, combo_gap_factor=0.9)
torch.manual_seed(0)
sim = BatchedSim(cfg, 64, seed=0)
gen = torch.Generator().manual_seed(5)
h = hashlib.sha256()
for t in range(60):
    mmask, bmask = sim.legal_mask()
    mv = torch.multinomial(mmask.float().view(-1, 5), 1, generator=gen).view(64, 2)
    bm = torch.multinomial(bmask.float().view(-1, 2), 1, generator=gen).view(64, 2)
    acts = torch.stack([mv, bm], dim=-1)
    rew, done, info = sim.step(acts)
    for name, tensor in (("bc", sim.bombs_cap), ("bz", sim.blast_cap),
                         ("sp", sim.spd_g), ("cr", sim.crate),
                         ("rc", sim._recycle_crate), ("rw", rew),
                         ("dn", done), ("pd", sim._pending_lost)):
        h.update(name.encode())
        h.update(tensor.cpu().numpy().tobytes())
print(h.hexdigest())
