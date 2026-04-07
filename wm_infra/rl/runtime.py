"""Compatibility shim for the learned environment runtime.

Trainer-facing code historically imported ``RLEnvironmentManager`` from
``wm_infra.rl.runtime``. The execution logic now lives under
``wm_infra.runtime.env.manager`` so that learned-environment stepping is part of
the runtime substrate rather than the RL package.
"""

from wm_infra.runtime.env.manager import LearnedEnvRuntimeManager, RLEnvironmentManager

__all__ = ["LearnedEnvRuntimeManager", "RLEnvironmentManager"]
