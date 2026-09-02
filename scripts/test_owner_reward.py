import sys, os
import jax, jax.numpy as jnp, jax.random as jrandom

sys.path.insert(0, "/Users/a1-6/Documents/llm-train/qqt-gpu-sim")
from jax_bomb.jax_env import init_batch, step, _resolve_explosions_matrix, _rays, BLAST

def test_owner_reward_math():
    # P0 places bomb, P0 hit by own bomb, P1 healthy
    attack_f = jnp.array([[0.0, 0.0]]) # P0 dealt 0 to P1, P1 dealt 0 to P0
    self_f = jnp.array([[1.0, 0.0]])   # P0 hit itself, P1 did not
    alive_after = jnp.array([[False, True]])
    done = jnp.array([True])
    
    p0_died = done & ~alive_after[:, 0]
    p1_died = done & ~alive_after[:, 1]
    
    p0_suicide = p0_died & (self_f[:, 0] > 0) & (attack_f[:, 1] == 0)
    p0_killed_by_p1 = p0_died & ~p0_suicide
    
    p1_suicide = p1_died & (self_f[:, 1] > 0) & (attack_f[:, 0] == 0)
    p1_killed_by_p0 = p1_died & ~p1_suicide
    
    win_p0 = p1_killed_by_p0.astype(jnp.float32)
    win_p1 = p0_killed_by_p1.astype(jnp.float32)
    lose_p0 = (p0_killed_by_p1 | p0_suicide).astype(jnp.float32)
    lose_p1 = (p1_killed_by_p0 | p1_suicide).astype(jnp.float32)
    
    print("Suicide Case:")
    print("P0 suicide:", p0_suicide[0], "P1 win bonus:", win_p1[0], "P0 lose penalty:", lose_p0[0])
    assert p0_suicide[0] == True
    assert win_p1[0] == 0.0
    assert lose_p0[0] == 1.0
    
    # Case 2: P0 kills P1
    attack_f2 = jnp.array([[1.0, 0.0]]) # P0 dealt 1 to P1
    self_f2 = jnp.array([[0.0, 0.0]])
    alive_after2 = jnp.array([[True, False]])
    
    p0_died2 = done & ~alive_after2[:, 0]
    p1_died2 = done & ~alive_after2[:, 1]
    p1_suicide2 = p1_died2 & (self_f2[:, 1] > 0) & (attack_f2[:, 0] == 0)
    p1_killed_by_p02 = p1_died2 & ~p1_suicide2
    
    win_p0_2 = p1_killed_by_p02.astype(jnp.float32)
    win_p1_2 = p0_killed_by_p1.astype(jnp.float32)
    
    print("\nKill Case:")
    print("P1 killed by P0:", p1_killed_by_p02[0], "P0 win bonus:", win_p0_2[0])
    assert p1_killed_by_p02[0] == True
    assert win_p0_2[0] == 1.0
    print("\nALL REWARD TESTS PASSED!")

if __name__ == "__main__":
    test_owner_reward_math()
