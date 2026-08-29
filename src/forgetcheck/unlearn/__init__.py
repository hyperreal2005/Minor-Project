"""Approximate unlearning methods, behind one interface."""

from .base import (
    REGISTRY,
    UnlearnContext,
    Unlearner,
    get_unlearner,
    method_names,
    register,
)
from .methods import SCRUB, SSD, FineTune, L1Sparse, NegGrad, NegGradPlus, SalUn
from .runner import base_run_id_for, run_unlearn

#: The six ranked methods, in the order they appear in the master reference. SSD is excluded --
#: it is a class-unlearning side condition, not a core method (see methods.SSD).
CORE_METHODS = ("finetune", "neggradplus", "neggrad", "scrub", "salun", "l1sparse")

__all__ = [
    "Unlearner", "UnlearnContext", "REGISTRY", "register", "get_unlearner", "method_names",
    "FineTune", "NegGrad", "NegGradPlus", "SCRUB", "SalUn", "L1Sparse", "SSD",
    "run_unlearn", "base_run_id_for", "CORE_METHODS",
]
