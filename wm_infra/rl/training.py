"""Compatibility shim for consumer-side RL experiment primitives."""

from wm_infra.consumers.rl.training import (
    CollectedBatch,
    Collector,
    Evaluator,
    ExperimentSpec,
    LearnerAdapter,
    LocalActorCriticLearner,
    SynchronousCollector,
    run_local_experiment,
)

__all__ = [
    "CollectedBatch",
    "Collector",
    "Evaluator",
    "ExperimentSpec",
    "LearnerAdapter",
    "LocalActorCriticLearner",
    "SynchronousCollector",
    "run_local_experiment",
]
