"""Rollout collector construction for RL training."""

from vrl.rollouts.collector.base import Collector
from vrl.rollouts.collector.configs import (
    CosmosPredict2CollectorConfig,
    JanusProCollectorConfig,
    NextStep1CollectorConfig,
    SD3_5CollectorConfig,
    Wan_2_1CollectorConfig,
)
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.factory import (
    COLLECTOR_REGISTRY,
    LAST_COLLECT_PHASES,
    CollectorRegistryEntry,
    build_rollout_collector,
    collector_config_cls,
)

__all__ = [
    "COLLECTOR_REGISTRY",
    "LAST_COLLECT_PHASES",
    "Collector",
    "CollectorRegistryEntry",
    "CosmosPredict2CollectorConfig",
    "JanusProCollectorConfig",
    "NextStep1CollectorConfig",
    "RolloutCollector",
    "SD3_5CollectorConfig",
    "Wan_2_1CollectorConfig",
    "build_rollout_collector",
    "collector_config_cls",
]
