"""Transformer-PPO mutation-location framework for the CCP100 study.

The package deliberately imports the frozen NSGA-II implementation instead of
changing it.  ``PPOController`` implements the existing mutation logger hook,
so crossover, repair, evaluation and environmental selection keep their
audited order.
"""

from .archive import ObjectiveNormalizer, RewardArchive, RewardConfig
from .config import PPOConfig, SCENARIOS

__all__ = [
    "ObjectiveNormalizer",
    "PPOConfig",
    "RewardArchive",
    "RewardConfig",
    "SCENARIOS",
]
