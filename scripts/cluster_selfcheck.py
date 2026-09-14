import jax
import optax
from jax_bomb.jax_train import ppo_update_gradsync, ppo_update_lsgd
from jax_bomb.jax_env import N_OBS_CH

ndev = jax.local_device_count()
print(f"SELFCHECK_OK jax={jax.__version__} optax={optax.__version__} devices={ndev} obs_ch={N_OBS_CH}")
