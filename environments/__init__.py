import gym
from gym.envs.registration import register

from .frozen_lake import *
from .millionaire import *

__all__ = ['RewardingFrozenLakeEnv', 'MillionaireEnv']

register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'is_slippery': 'False'}
)

register(
    id='Millionaire-v0',
    entry_point='environments:MillionaireEnv',
)

def get_large_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake-v0')


def get_millionaire_environment():
    return gym.make('Millionaire-v0')
