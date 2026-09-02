"""即刻计算所有 Checkpoint 的策略熵 (Entropy) 与动作分布。"""
import os, sys, pickle
import jax, jax.numpy as jnp
import numpy as np

from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch
from jax_bomb.jax_train import both_perspectives, both_masks, both_states, net_forward

def check_entropy(ckpt_path):
    with open(ckpt_path, "rb") as f:
        params = jax.tree.map(jnp.asarray, pickle.load(f))
    arch = "transformer"
    
    key = jax.random.PRNGKey(42)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _levels.set_active(os.path.join(base_dir, "levels.json"), weights="empty=1.0")
    states = init_batch(key, 512)
    
    obs = both_perspectives(states)[:512]
    masks = both_masks(states)
    st = both_states(states)[:512]
    m0, b0 = masks[0][:512], masks[1][:512]
    
    mv, bm, v, _ = net_forward(params, arch, obs, st)
    mv_m = jnp.where(m0, mv, -1e9)
    bm_m = jnp.where(b0, bm, -1e9)
    
    pm = np.array(jax.nn.softmax(mv_m, axis=-1))
    pb = np.array(jax.nn.softmax(bm_m, axis=-1))
    lsm = np.array(jax.nn.log_softmax(mv_m, axis=-1))
    lsb = np.array(jax.nn.log_softmax(bm_m, axis=-1))
    
    ent_m = -np.mean(np.sum(np.where(pm > 0, pm * lsm, 0.0), axis=-1))
    ent_b = -np.mean(np.sum(np.where(pb > 0, pb * lsb, 0.0), axis=-1))
    total_ent = ent_m + ent_b
    max_ent = np.log(5) + np.log(2) # 2.302
    
    avg_bomb_prob = np.mean(pb[:, 1])
    move_dist = np.mean(pm, axis=0)
    
    print(f"==================== 📊 【{os.path.basename(ckpt_path)}】策略熵与动作分布 ====================")
    print(f"  🌀 移动策略熵 (Move):   {ent_m:.4f} / 1.6094 (理论满值: 1.6094)")
    print(f"  💣 放泡策略熵 (Bomb):   {ent_b:.4f} / 0.6931 (理论满值: 0.6931)")
    print(f"  📈 总体策略熵 (Total):  {total_ent:.4f} / 2.3026 (保留度: {total_ent/max_ent*100:.1f}%)")
    print(f"  💥 开局放泡动作概率:    {avg_bomb_prob*100:.1f}%")
    print(f"  🚶 移动动作分布 [上,下,左,右,停]: {[f'{p*100:.1f}%' for p in move_dist]}")
    print("=========================================================================================\n")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        check_entropy(p)
