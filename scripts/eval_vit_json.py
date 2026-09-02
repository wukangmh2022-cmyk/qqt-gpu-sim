import sys, os, json, base64
import numpy as np, jax, jax.numpy as jnp, jax.random as jrandom

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step
from jax_bomb.jax_train import both_masks, both_states, sample_actions, net_forward

def load_json_model(json_path):
    with open(json_path) as f:
        d = json.load(f)
    flat_bytes = base64.b64decode(d["flat"])
    raw_f32 = np.frombuffer(flat_bytes, dtype=np.float32)
    
    tensors = {}
    for item in d["tensors"]:
        name = item["name"]
        shape = item["shape"]
        off, cnt = item["offset"], item["count"]
        arr = raw_f32[off : off + cnt].reshape(shape)
        tensors[name] = arr
    
    # Reconstruct JAX PyTree for Transformer
    # Keys: 'tok', 'pos', 'state_w', 'state_b', 'blocks', 'heads'
    meta = d["meta"]
    depth = meta.get("depth", 4)
    
    tok = (jnp.asarray(tensors["tok_w"]), jnp.asarray(tensors["tok_b"]))
    pos = jnp.asarray(tensors["pos"])
    state_w = jnp.asarray(tensors["state_w"])
    state_b = jnp.asarray(tensors["state_b"])
    
    blocks = []
    for l in range(depth):
        blk = {
            "ln1": (jnp.asarray(tensors[f"b{l}_ln1_g"]), jnp.asarray(tensors[f"b{l}_ln1_b"])),
            "qkv": (jnp.asarray(tensors[f"b{l}_qkv_w"]), jnp.asarray(tensors[f"b{l}_qkv_b"])),
            "proj": (jnp.asarray(tensors[f"b{l}_proj_w"]), jnp.asarray(tensors[f"b{l}_proj_b"])),
            "ln2": (jnp.asarray(tensors[f"b{l}_ln2_g"]), jnp.asarray(tensors[f"b{l}_ln2_b"])),
            "ff1": (jnp.asarray(tensors[f"b{l}_ff1_w"]), jnp.asarray(tensors[f"b{l}_ff1_b"])),
            "ff2": (jnp.asarray(tensors[f"b{l}_ff2_w"]), jnp.asarray(tensors[f"b{l}_ff2_b"])),
        }
        blocks.append(blk)
    
    heads = {
        "wm": (jnp.asarray(tensors["head_wm_w"]), jnp.asarray(tensors["head_wm_b"])),
        "wb": (jnp.asarray(tensors["head_wb_w"]), jnp.asarray(tensors["head_wb_b"])),
        "wv": (jnp.asarray(tensors["head_wv_w"]), jnp.asarray(tensors["head_wv_b"])),
    }
    
    params = {
        "tok": tok,
        "pos": pos,
        "state_w": state_w,
        "state_b": state_b,
        "blocks": blocks,
        "heads": heads,
    }
    return params, meta

def make_obs_13(state, pid, danger):
    """13通道 make_obs (ViTModel_68 初代格式)"""
    from jax_bomb.jax_env import _splat, H, W, MAX_STEPS, FUSE
    pos, fuse, owner, bomb_blast, wall, brick, pushable, push_t, bush, crate, rec_crate, alive, hp, invuln, bombs_cap, blast_cap, spd_g, buffs, debuffs, items, gametype, is_open, t, level_id = state
    me, opp = pid, 1 - pid
    fuse_norm = fuse.astype(jnp.float32) / float(FUSE)
    
    c_p1 = crate == 1
    c_p2 = crate == 2
    c_p3 = crate == 3
    c_p4 = (crate >= 4) & (crate <= 6)
    
    obs = jnp.stack([
        _splat(pos[me], alive[me], H, W),
        jnp.where(owner == me, fuse_norm, jnp.zeros_like(fuse_norm)),
        _splat(pos[opp], alive[opp], H, W),
        jnp.where(owner == opp, fuse_norm, jnp.zeros_like(fuse_norm)),
        (wall | brick).astype(jnp.float32),
        danger,
        jnp.full((H, W), t.astype(jnp.float32) / float(MAX_STEPS), jnp.float32),
        crate > 0,
        bush,
        c_p1.astype(jnp.float32),
        c_p2.astype(jnp.float32),
        c_p3.astype(jnp.float32),
        c_p4.astype(jnp.float32),
    ])
    return obs

def both_perspectives_13(states):
    from jax_bomb.jax_env import _danger_map
    danger = jax.vmap(lambda s: _danger_map(s.fuse, s.wall, s.bomb_blast, s.brick))(states)
    obs0 = jax.vmap(lambda s, d: make_obs_13(s, 0, d))(states, danger)
    obs1 = jax.vmap(lambda s, d: make_obs_13(s, 1, d))(states, danger)
    return jnp.concatenate([obs0, obs1], axis=0)

def eval_json_model(json_path, opponent_type="idle", n_games=128, max_ticks=800):
    params, meta = load_json_model(json_path)
    model_name = meta.get("name", os.path.basename(json_path))
    obs_channels = meta.get("obs_shape", [14])[0]
    
    print(f"\n==========================================================================")
    print(f"🥊 原版标杆对局评测: [{model_name}] (obs_ch={obs_channels}) vs [{opponent_type}]")
    print(f"   对局总数: {n_games} 局 | 动作策略: 随机采样 (Sample)")
    print(f"==========================================================================")
    
    _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights="empty=1.0")
    
    states = init_batch(jrandom.PRNGKey(42), n_games)
    key = jrandom.PRNGKey(101)
    
    carry_init = (
        states, key,
        jnp.ones(n_games, dtype=bool), jnp.ones(n_games, dtype=bool),
        jnp.zeros(n_games, dtype=bool), jnp.full(n_games, max_ticks, dtype=jnp.int32),
        jnp.zeros(n_games, dtype=jnp.int32), jnp.zeros(n_games, dtype=jnp.int32)
    )
    
    def step_fn(carry, tick_idx):
        states, key, a0_alive, a1_alive, ended, end_t, b0_cnt, b1_cnt = carry
        key, k0, k1, kstep = jrandom.split(key, 4)
        
        if obs_channels == 13:
            obs = both_perspectives_13(states)
        else:
            from jax_bomb.jax_train import both_perspectives
            obs = both_perspectives(states)
            
        masks = both_masks(states); gv = both_states(states)
        
        a0 = sample_actions(params, "transformer", obs[:n_games], (masks[0][:n_games], masks[1][:n_games]), k0, state=gv[:n_games])[0]
        a1 = jnp.full((n_games, 2), 4, jnp.int32).at[:, 1].set(0) # IDLE
        
        env_acts = jnp.stack([a0, a1], axis=1)
        keys = jrandom.split(kstep, n_games)
        new_states, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, auto_reset=False, return_info=True))(states, env_acts, keys)
        
        cur_a0 = info["alive"][:, 0]
        cur_a1 = info["alive"][:, 1]
        
        is_b0 = (env_acts[:, 0, 1] == 1) & ~ended
        is_b1 = (env_acts[:, 1, 1] == 1) & ~ended
        
        just_ended = (~cur_a0 | ~cur_a1) & ~ended
        new_ended = ended | just_ended
        new_end_t = jnp.where(just_ended, tick_idx + 1, end_t)
        
        new_a0_alive = jnp.where(just_ended, cur_a0, a0_alive)
        new_a1_alive = jnp.where(just_ended, cur_a1, a1_alive)
        
        return (new_states, key, new_a0_alive, new_a1_alive, new_ended, new_end_t,
                b0_cnt + is_b0.astype(jnp.int32), b1_cnt + is_b1.astype(jnp.int32)), None
    
    ticks = jnp.arange(max_ticks)
    (final_s, _, final_a0, final_a1, final_ended, final_end_t, final_b0, final_b1), _ = jax.lax.scan(
        step_fn, carry_init, ticks
    )
    
    win = final_a0 & ~final_a1
    loss = ~final_a0 & final_a1
    both = ~final_a0 & ~final_a1
    timeout = final_a0 & final_a1
    
    win_pct = float(jnp.mean(win.astype(jnp.float32))) * 100
    loss_pct = float(jnp.mean(loss.astype(jnp.float32))) * 100
    both_pct = float(jnp.mean(both.astype(jnp.float32))) * 100
    timeout_pct = float(jnp.mean(timeout.astype(jnp.float32))) * 100
    avg_b0 = float(jnp.mean(final_b0.astype(jnp.float32)))
    avg_surv = float(jnp.mean(final_end_t.astype(jnp.float32)))
    
    print(f"\n🗺️  【空旷场景道场】")
    print(f"   🏆 击败对手胜率: {win_pct:5.1f}% ({int(jnp.sum(win))}/{n_games})")
    print(f"   💀 败局/自杀率  : {loss_pct:5.1f}% ({int(jnp.sum(loss))}/{n_games})")
    print(f"   💥 同归于尽率  : {both_pct:5.1f}% ({int(jnp.sum(both))}/{n_games})")
    print(f"   ⏱️ 超时保命平局: {timeout_pct:5.1f}% ({int(jnp.sum(timeout))}/{n_games})")
    print(f"   💣 模型放泡量  : {avg_b0:5.1f} 颗/局")
    print(f"   ⏳ 平均存活时间: {avg_surv:5.1f} ticks")

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "web/models/ViTModel_68.json"
    eval_json_model(p)
