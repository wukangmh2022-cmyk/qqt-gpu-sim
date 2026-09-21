import jax
import jax.numpy as jnp
from jax_bomb.jax_env import init_batch
from jax_bomb.jax_net import init_transformer
from jax_bomb.jax_train import collect_rollout

def test_rollout():
    key = jax.random.PRNGKey(42)
    k_net, k_env, k_roll = jax.random.split(key, 3)
    n_envs = 4
    num_steps = 8
    arch = "transformer"
    params = init_transformer(k_net, c=14, h=13, w=15, embed=64, depth=2, heads=2)
    states = init_batch(k_env, n_envs)

    print("Testing action_repeat = 1...")
    final_states1, batch1, nov1, kills1 = collect_rollout(
        params, arch, states, k_roll, num_steps=num_steps, action_repeat=1
    )
    obs1, st1, acts1, lps1, vals1, rew1, done1, masks1 = batch1
    print("  batch1 obs shape:", obs1.shape, "acts shape:", acts1.shape, "rew shape:", rew1.shape)
    assert obs1.shape[0] == num_steps
    assert rew1.shape[0] == num_steps

    print("Testing action_repeat = 2 (5Hz macro-step)...")
    final_states2, batch2, nov2, kills2 = collect_rollout(
        params, arch, states, k_roll, num_steps=num_steps, action_repeat=2
    )
    obs2, st2, acts2, lps2, vals2, rew2, done2, masks2 = batch2
    print("  batch2 obs shape:", obs2.shape, "acts shape:", acts2.shape, "rew shape:", rew2.shape)
    assert obs2.shape[0] == num_steps
    assert rew2.shape[0] == num_steps

    print("Action repeat 1 & 2 verification PASSED PERFECTLY!")

if __name__ == "__main__":
    test_rollout()
