"""Check the shared JAX/Torch hit, step, and death-kill reward primitives.

JAX intentionally has different reward-hacking protections for crate, novelty, and
annealed timeout HP lead. Those policies are covered by tests/test_jax_rewards.py;
this script excludes all-alive timeout terminals and verifies only the common core.
"""
import numpy as np
import torch

import jax
import jax.numpy as jnp

from deploy.collect_distill import make_cfg
from sim.factory import make_sim

# --- 奖励常量（与 jax_train 对齐）---
STEP_PENALTY = 0.001
HIT_REWARD = 1.5
WIN_BONUS = 10.0


def jax_reward(dmg, alive0, alive, hp, done):
    """The common hit/step/fixed-death portion of the JAX reward."""
    dmg = dmg.astype(jnp.float32)
    dealt = dmg.sum(axis=-1, keepdims=True) - dmg
    alive0_f = alive0.astype(jnp.float32)
    rew = (dealt - dmg) * HIT_REWARD - STEP_PENALTY * alive0_f
    n_alive = alive.sum(axis=-1)
    death_done = done & (n_alive == 1)
    win = death_done[:, None] & alive
    lose = death_done[:, None] & ~alive
    rew = rew + WIN_BONUS * (win.astype(jnp.float32) - lose.astype(jnp.float32))
    return rew


def main():
    import dataclasses
    cfg = make_cfg()
    cfg = dataclasses.replace(
        cfg,
        place_cover_reward=0.0, place_chain_reward=0.0, place_dist_reward=0.0,
        brick_reward=0.0, danger_penalty=0.0, combo_reward=0.0,
        passivity_penalty=0.0)

    n = 128
    torch.manual_seed(0)
    sim = make_sim(cfg, n, backend="torch", device="cpu", seed=0)
    sim.reset_all()

    # --- 主循环：torch sim 步进，两侧公式对同输入比较 ---
    max_diff = 0.0
    bad = None
    for tick in range(200):
        hp_before = sim.hp.clone()
        alive0 = sim.alive.clone()
        acts = torch.randint(0, 5, (n, 2, 2))
        with torch.no_grad():
            reward, done, info_t = sim.step(acts, auto_reset=False)
        dmg = (hp_before - sim.hp).clamp(min=0).float()
        alive = sim.alive
        hp = sim.hp
        ref = reward.numpy().copy()
        # Timeout HP-lead shaping is intentionally JAX-specific. Remove Torch's
        # timeout contribution before comparing the shared hit/step/death core.
        timeout = done & (alive.sum(dim=-1) == 2)
        hp_diff = (sim.hp[:, :1] - sim.hp[:, 1:]).float()
        timeout_delta = (cfg.win_bonus / cfg.max_hp) * torch.cat(
            [hp_diff, -hp_diff], dim=-1) * timeout[:, None].float() * sim._explore_coef
        ref -= timeout_delta.numpy()
        got = np.asarray(jax_reward(
            jnp.asarray(torch.as_tensor(dmg).numpy()),
            jnp.asarray(torch.as_tensor(alive0).numpy()),
            jnp.asarray(torch.as_tensor(alive).numpy()),
            jnp.asarray(torch.as_tensor(hp).numpy()),
            jnp.asarray(torch.as_tensor(done).numpy())))
        d = np.abs(got - ref).max()
        if d > max_diff:
            max_diff = d
            bad = tick
        if d > 1e-3:
            print(f"tick {tick} MISMATCH maxdiff={d:.4f}")
            idx = np.unravel_index(np.argmax(np.abs(got - ref)), got.shape)
            print(f"  at {idx}: ref={ref[idx]} jax={got[idx]}")
            print(f"  dmg={dmg[idx[0]].tolist()} alive={alive[idx[0]].tolist()} "
                  f"hp={hp[idx[0]].tolist()} done={done[idx[0]].item()}")
            if tick > 40:
                break
        # 推进：reset 已由 sim.step(auto_reset=False) 保持？不 —— 需要手动
        # 维持终局环境推进。这里用 reset_ 让所有 done 的 env 开新局
        if bool(done.any()):
            sim.reset_(done)
    print(f"done. max diff over 200 ticks = {max_diff:.6f} (tick {bad})")
    print("REWARD PARITY PASS" if max_diff <= 1e-3 else "REWARD PARITY FAIL")


if __name__ == "__main__":
    main()
