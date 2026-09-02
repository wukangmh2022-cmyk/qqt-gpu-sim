import sys, os, json, base64
import numpy as np, jax, jax.numpy as jnp, jax.random as jrandom

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from deploy.export_jax_ckpt import transformer_forward_js, _softmax
from jax_bomb import levels as _levels
from jax_bomb.jax_env import init_batch, step, _splat, _danger_map, H, W, MAX_STEPS, FUSE
from jax_bomb.jax_train import both_masks, both_states

def load_params_flat(json_path):
    with open(json_path) as f:
        d = json.load(f)
    flat_bytes = base64.b64decode(d["flat"])
    raw_f32 = np.frombuffer(flat_bytes, dtype=np.float32)
    
    meta = d["meta"]
    tensors_idx = d["tensors"]
    
    params_flat = {}
    for name, (off, cnt) in tensors_idx.items():
        params_flat[name] = raw_f32[off : off + cnt]
    
    E = meta.get("embed", 392)
    obs_c = meta.get("obs_shape", [13])[0]
    P = meta.get("patch", 4)
    depth = meta.get("depth", 4)
    
    params_flat["tok_w"] = params_flat["tok_w"].reshape(obs_c * P * P, E)
    params_flat["tok_b"] = params_flat["tok_b"].reshape(E)
    gp = -(-13 // P)
    params_flat["pos"] = params_flat["pos"].reshape(gp * gp + 1, E)
    params_flat["state_w"] = params_flat["state_w"].reshape(-1, E)
    params_flat["state_b"] = params_flat["state_b"].reshape(E)
    
    for i in range(depth):
        p = str(i)
        params_flat[f"b{p}_q_w"] = params_flat[f"b{p}_q_w"].reshape(E, E)
        params_flat[f"b{p}_q_b"] = params_flat[f"b{p}_q_b"].reshape(E)
        params_flat[f"b{p}_k_w"] = params_flat[f"b{p}_k_w"].reshape(E, E)
        params_flat[f"b{p}_k_b"] = params_flat[f"b{p}_k_b"].reshape(E)
        params_flat[f"b{p}_v_w"] = params_flat[f"b{p}_v_w"].reshape(E, E)
        params_flat[f"b{p}_v_b"] = params_flat[f"b{p}_v_b"].reshape(E)
        params_flat[f"b{p}_proj_w"] = params_flat[f"b{p}_proj_w"].reshape(E, E)
        params_flat[f"b{p}_proj_b"] = params_flat[f"b{p}_proj_b"].reshape(E)
        params_flat[f"b{p}_ff1_w"] = params_flat[f"b{p}_ff1_w"].reshape(E, E * 4)
        params_flat[f"b{p}_ff1_b"] = params_flat[f"b{p}_ff1_b"].reshape(E * 4)
        params_flat[f"b{p}_ff2_w"] = params_flat[f"b{p}_ff2_w"].reshape(E * 4, E)
        params_flat[f"b{p}_ff2_b"] = params_flat[f"b{p}_ff2_b"].reshape(E)
        
    params_flat["head_wm_w"] = params_flat["head_wm_w"].reshape(E, 5)
    params_flat["head_wm_b"] = params_flat["head_wm_b"].reshape(5)
    params_flat["head_wb_w"] = params_flat["head_wb_w"].reshape(E, 2)
    params_flat["head_wb_b"] = params_flat["head_wb_b"].reshape(2)
    params_flat["head_wv_w"] = params_flat["head_wv_w"].reshape(E, -1)
    params_flat["head_wv_b"] = params_flat["head_wv_b"].reshape(-1)
    return params_flat, meta

def make_obs_np(state, pid, is_14ch=False):
    """Numpy/JAX 视角生成"""
    pos, fuse, owner, bomb_blast, wall, brick, pushable, push_t, bush, crate, rec_crate, alive, hp, invuln, bombs_cap, blast_cap, spd_g, buffs, debuffs, items, gametype, is_open, t, level_id = state
    danger = _danger_map(fuse, wall, bomb_blast, brick)
    me, opp = pid, 1 - pid
    fuse_norm = fuse.astype(jnp.float32) / float(FUSE)
    
    c_p1 = crate == 1
    c_p2 = crate == 2
    c_p3 = crate == 3
    c_p4 = (crate >= 4) & (crate <= 6)
    
    chans = [
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
    ]
    if is_14ch:
        chans.append(rec_crate.astype(jnp.float32))
    return jnp.stack(chans)

def eval_json_model(json_path, opponent_type="idle", n_games=128, max_ticks=800):
    params_flat, meta = load_params_flat(json_path)
    model_name = meta.get("name", os.path.basename(json_path))
    obs_channels = meta.get("obs_shape", [13])[0]
    is_14ch = (obs_channels == 14)
    
    print(f"\n==========================================================================")
    print(f"🥊 原版标杆对局评测: [{model_name}] (obs_ch={obs_channels}) vs [{opponent_type}]")
    print(f"   对局总数: {n_games} 局 | 动作策略: 随机采样 (Sample)")
    print(f"==========================================================================")
    
    _levels.set_active("/Users/a1-6/Documents/llm-train/qqt-gpu-sim/levels.json", weights="empty=1.0")
    
    wins = 0
    losses = 0
    boths = 0
    timeouts = 0
    total_bombs = 0
    total_surv = 0
    
    for g in range(n_games):
        key = jrandom.PRNGKey(1000 + g)
        state = init_batch(key, 1)
        
        b0_cnt = 0
        end_tick = max_ticks
        
        for tick in range(max_ticks):
            obs_p0 = np.asarray(make_obs_np(jax.tree.map(lambda x: x[0], state), 0, is_14ch=is_14ch))
            gv = np.asarray(both_states(state))[0]
            masks = both_masks(state)
            m_mask = np.asarray(masks[0][0])
            b_mask = np.asarray(masks[1][0])
            
            mv, bm, v = transformer_forward_js(params_flat, obs_p0, gv)
            
            # Mask logits
            mv = np.where(m_mask, mv, -1e9)
            bm = np.where(b_mask, bm, -1e9)
            
            # Sample action
            pm = _softmax(mv)
            pb = _softmax(bm)
            a_mv = np.random.choice(5, p=pm)
            a_bm = np.random.choice(2, p=pb)
            
            if a_bm == 1:
                b0_cnt += 1
                
            a0 = jnp.array([a_mv, a_bm], jnp.int32)
            a1 = jnp.array([4, 0], jnp.int32) # IDLE
            
            env_acts = jnp.stack([a0, a1])[None, :]
            key, kstep = jrandom.split(key)
            new_state, done, info = jax.vmap(lambda s, a, kk: step(s, a, kk, auto_reset=False, return_info=True))(
                state, env_acts, jrandom.split(kstep, 1)
            )
            
            a0_alive = bool(info["alive"][0, 0])
            a1_alive = bool(info["alive"][0, 1])
            
            if not a0_alive or not a1_alive:
                end_tick = tick + 1
                if a0_alive and not a1_alive:
                    wins += 1
                elif not a0_alive and a1_alive:
                    losses += 1
                else:
                    boths += 1
                break
            state = new_state
            
        if end_tick == max_ticks:
            timeouts += 1
            
        total_bombs += b0_cnt
        total_surv += end_tick
        
        if (g + 1) % 32 == 0 or g == n_games - 1:
            print(f"[{g+1:3d}/{n_games}] Win: {wins} | Loss/Suicide: {losses} | Both: {boths} | Timeout: {timeouts} | Avg Bombs: {total_bombs/(g+1):.1f}")
            
    print(f"\n🗺️  【空旷场景道场】")
    print(f"   🏆 击败对手胜率: {wins/n_games*100:5.1f}% ({wins}/{n_games})")
    print(f"   💀 败局/自杀率  : {losses/n_games*100:5.1f}% ({losses}/{n_games})")
    print(f"   💥 同归于尽率  : {boths/n_games*100:5.1f}% ({boths}/{n_games})")
    print(f"   ⏱️ 超时保命平局: {timeouts/n_games*100:5.1f}% ({timeouts}/{n_games})")
    print(f"   💣 模型放泡量  : {total_bombs/n_games:5.1f} 颗/局")
    print(f"   ⏳ 平均存活时间: {total_surv/n_games:5.1f} ticks")

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "web/models/ViTModel_68.json"
    eval_json_model(p, n_games=64)
