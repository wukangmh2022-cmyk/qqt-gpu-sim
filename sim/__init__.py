""""泡泡堂"基础关卡"的批量模拟器 + PPO 自我博弈训练。"""

from .config import N_BOMB, N_MOVES, SimConfig
from .factory import make_sim
from .torch_sim import BatchedSim

__all__ = ["SimConfig", "N_MOVES", "N_BOMB", "make_sim", "BatchedSim"]
