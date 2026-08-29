"""Training: the one shared loop, and the role-specific tasks that feed it."""

from .loop import TrainConfig, TrainResult, finetune_steps, set_determinism, train
from .tasks import (
    TrainTask,
    base_task,
    oracle_task,
    run_task,
    shadow_indices,
    shadow_task,
)

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train",
    "finetune_steps",
    "set_determinism",
    "TrainTask",
    "base_task",
    "oracle_task",
    "shadow_task",
    "shadow_indices",
    "run_task",
]
