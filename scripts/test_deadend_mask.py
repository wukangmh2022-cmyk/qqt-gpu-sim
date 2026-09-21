import jax
import jax.numpy as jnp
from jax_bomb.jax_env import init_batch, legal_mask

def test_mask():
    key = jax.random.PRNGKey(42)
    batch = init_batch(key, 1)
    state = jax.tree_util.tree_map(lambda x: x[0], batch)

    # Test 1: Normal open map
    mm, bm = legal_mask(state)
    assert bm[0, 1] == True, "Open map player 0 should be allowed to bomb"
    assert bm[1, 1] == True, "Open map player 1 should be allowed to bomb"
    print("Test 1 PASS: Normal map bombing allowed.")

    # Test 2: Trapped player in 4 walls
    p0_cell = state.pos[0].astype(jnp.int32)
    r, c = int(p0_cell[0]), int(p0_cell[1])
    wall = state.wall
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        wall = wall.at[r+dr, c+dc].set(True)
    trapped_state = state._replace(wall=wall)
    mm_t, bm_t = legal_mask(trapped_state)
    assert bm_t[0, 1] == False, "Trapped player in 4 walls must NOT bomb"
    print("Test 2 PASS: 4-wall trapped player bomb=1 masked.")

    # Test 3: Player facing a 1-tile dead-end pocket (clean map, no stray bricks)
    wall3 = jnp.zeros_like(state.wall)
    brick3 = jnp.zeros_like(state.brick)
    for dr, dc in [(-1,0), (1,0), (0,-1)]:
        wall3 = wall3.at[r+dr, c+dc].set(True)
    # Right neighbor (r, c+1) has no other exits
    wall3 = wall3.at[r-1, c+1].set(True)
    wall3 = wall3.at[r+1, c+1].set(True)
    wall3 = wall3.at[r, c+2].set(True)
    deadend_pocket_state = state._replace(wall=wall3, brick=brick3)
    mm_d, bm_d = legal_mask(deadend_pocket_state)
    assert bm_d[0, 1] == False, "Player facing 0-exit dead end must NOT bomb"
    print("Test 3 PASS: 1-tile dead-end pocket bomb=1 masked.")

    # Test 4: Player in a corner, but exit opens to free space
    wall4 = jnp.zeros_like(state.wall)
    brick4 = jnp.zeros_like(state.brick)
    for dr, dc in [(-1,0), (1,0), (0,-1)]:
        wall4 = wall4.at[r+dr, c+dc].set(True)
    corner_state = state._replace(wall=wall4, brick=brick4)
    mm_c, bm_c = legal_mask(corner_state)
    assert bm_c[0, 1] == True, "Corner player with open escape route MUST be allowed to bomb"
    print("Test 4 PASS: Corner player with safe escape route CAN bomb.")

    print("ALL 4 TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_mask()
