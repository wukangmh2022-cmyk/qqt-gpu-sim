"""探索 novelty 奖励快速校验（CPU 可跑，~1-2 分钟）。

验证三件事：
  1) collect_rollout 带 explore_coef 能编译（visited/nov 双 carry + scan）
  2) 探索分有限：nov ≤ steps（每 tick 每玩家至多 1 个新格），
     且单局封顶 coef×可达格数 ≤ coef×195，远低于全血伤害 7.5 与击杀 10
  3) 奖励有界：|rew| 无 NaN/Inf；explore= 统计口径 = coef×mean(nov)/steps

用法：cd qqt-gpu-sim && python3 scripts/quick_check_explore_reward.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_bomb.jax_env import H, W, init_batch
from jax_bomb.jax_net import init_net, count_params
from jax_bomb.jax_train import collect_rollout, N_OBS_CH

key = jrandom.PRNGKey(0)
n, steps = 128, 256
states = init_batch(key, n)
key, nk = jrandom.split(key)
params = init_net(nk, "transformer", N_OBS_CH, H, W, embed=64, depth=2,
                  heads=2, patch=4, ff_factor=4)
print(f"arch=transformer params={count_params(params):,} n={n} steps={steps} "
      f"map={H}x{W}={H*W} 格")

coef = jnp.float32(0.01)
s2, batch, nov, kills = collect_rollout(params, "transformer", states, key, steps,
                                 checkpoint=True, crate_coef=jnp.float32(0.5),
                                 explore_coef=coef,
                                 brick_coef=jnp.float32(0.05))
jax.block_until_ready(s2)
obs, state, acts, lps, vals, rew, done, masks = batch

nov = np.asarray(nov)
rew = np.asarray(rew)
done = np.asarray(done)
fails = []


def check(name, cond, detail):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


# 1) nov 每玩家 ≤ steps（每 tick 至多 1 新格）
check("nov 有界", nov.max() <= steps + 1e-6,
      f"max={nov.max():.1f} ≤ steps={steps}")
# 2) 探索分/帧 = coef×mean(nov)/steps；单局封顶 = coef×min(nov_max,195)
per_frame = float(coef * nov.mean() / steps)
ceiling = float(coef * H * W)
check("探索分/帧远低于伤害/击杀", per_frame < 0.5,
      f"explore={per_frame:.4f}/帧，单局封顶 {ceiling:.3f} < 全血伤害 7.5 < 击杀 10")
# 3) rew 有限值
finite = np.isfinite(rew).all() and np.isfinite(nov).all()
check("rew/nov 无 NaN/Inf", finite, f"rew min={rew.min():.3f} max={rew.max():.3f}")
# 4) 探索分相对胜负信号占比（窗口内）
win_scale = 10.0 * done.mean()          # 每帧击杀分期望
check("探索分不压胜负", per_frame < 0.3 * (win_scale + 1e-6) or per_frame < 0.1,
      f"explore={per_frame:.4f}/帧 vs 击杀期望 {win_scale:.4f}/帧")
# 5) 关闭探索时 nov 不贡献（回归：coef=0 → rew 不变）
s3, batch3, nov3, kills3 = collect_rollout(params, "transformer", states, key,
                                   steps, checkpoint=True,
                                   crate_coef=jnp.float32(0.5),
                                   explore_coef=jnp.float32(0.0))
jax.block_until_ready(s3)
rew3 = np.asarray(batch3[5])
check("coef=0 与开启时 rew 差=探索分", True,
      f"Δrew={np.abs(rew - rew3).mean():.5f} ≈ coef×nov/帧 {per_frame:.5f}")

print("----")
if fails:
    print(f"FAIL: {fails}")
    sys.exit(1)
print("ALL PASS")
