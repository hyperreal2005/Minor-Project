"""The Unlearner interface.

BINDING — implementation plan §4.5.

Every method is a drop-in behind one signature. Checkpointing, timing and record-writing live
*outside* the method, in :mod:`forgetcheck.unlearn.runner`, which is what makes the efficiency
column comparable across methods by construction rather than by six people each remembering to
start their timer in the same place.

Methods must not mutate the model they are handed. The caller owns the original checkpoint and
may reuse it for the next method in the queue; an in-place update would silently make the second
method operate on the first one's output.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

__all__ = ["Unlearner", "UnlearnContext", "REGISTRY", "register", "get_unlearner", "method_names"]


@dataclass(frozen=True, slots=True)
class UnlearnContext:
    """Everything a method may draw on, beyond the model itself.

    ``retain_loader`` is shuffled and augmented like a training loader; ``forget_loader`` is too.
    ``forget_eval_loader`` is the unshuffled, un-augmented view of the same forget set, for
    methods that need deterministic per-example quantities (SalUn's saliency mask).
    """

    forget_loader: Any
    retain_loader: Any
    forget_eval_loader: Any = None
    device: torch.device | str = "cpu"
    num_classes: int = 10
    seed: int = 0


class Unlearner(ABC):
    """Base class for approximate unlearning methods."""

    #: Registry key. Must be a valid run-id segment: lowercase, digits, single hyphens.
    name: str = ""

    #: Default hyperparameters, overridden per-run from ``configs/methods/<name>.yaml``.
    defaults: Mapping[str, Any] = {}

    #: Implementation version. **Bump this whenever the method's behaviour changes without its
    #: hyperparameters changing.** The stale-checkpoint guard compares configuration hashes, and
    #: a loop-structure fix with the same ``defaults`` is invisible to it -- exactly what happened
    #: with SalUn's retain-exposure fix in Stage 5, whose 16 pre-fix checkpoints on account 1 read
    #: as current because nothing in the hash had moved. Version 1 is "as first shipped" and is
    #: omitted from the hash, so the shas already stored for unchanged methods stay valid.
    version: int = 1

    def signature(self) -> dict[str, Any]:
        """Everything that determines what this method computes, for hashing.

        The single source of truth for the stale-checkpoint guard: ``run_unlearn`` stores a hash
        of this, and the CLI compares against it. Two call sites computing it independently is
        how the guard came to be unreachable once already.
        """
        sig: dict[str, Any] = {"method": self.name, **self.cfg}
        if self.version > 1:
            sig["impl_version"] = self.version
        return sig

    def __init__(self, **cfg: Any):
        merged = {**self.defaults, **cfg}
        unknown = set(cfg) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown hyperparameter(s) {sorted(unknown)}; "
                f"known: {sorted(self.defaults)}"
            )
        self.cfg = merged

    @abstractmethod
    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        """Return an unlearned copy of ``model``. Must not mutate the input."""

    # -- helpers available to every method -----------------------------------

    @staticmethod
    def clone(model: nn.Module) -> nn.Module:
        """A deep copy, so the caller's checkpoint survives untouched."""
        return copy.deepcopy(model)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.cfg})"


# --------------------------------------------------------------------------- registry

REGISTRY: dict[str, type[Unlearner]] = {}


def register(cls: type[Unlearner]) -> type[Unlearner]:
    """Class decorator adding a method to the registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a `name`")
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate unlearner name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def get_unlearner(name: str, **cfg: Any) -> Unlearner:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown unlearning method {name!r}; registered: {sorted(REGISTRY)}"
        ) from None
    return cls(**cfg)


def method_names() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))
