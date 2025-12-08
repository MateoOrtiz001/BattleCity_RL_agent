# Training module for BattleCity RL
# Export main training classes

from .environment import BattleCityEnvironment
from .trainer import QLearningTrainer

__all__ = [
    'BattleCityEnvironment',
    'QLearningTrainer',
]
