"""RL trainers: training loop orchestration and weight sync."""

from vrl.trainers.base import Trainer
from vrl.trainers.data import DistributedKRepeatSampler, TextPromptDataset
from vrl.trainers.ema import EMAModuleWrapper
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import InMemoryWeightSyncer, WeightSyncer

__all__ = [
    "DistributedKRepeatSampler",
    "EMAModuleWrapper",
    "InMemoryWeightSyncer",
    "OnlineTrainer",
    "TextPromptDataset",
    "Trainer",
    "TrainerConfig",
    "TrainState",
    "WeightSyncer",
]
