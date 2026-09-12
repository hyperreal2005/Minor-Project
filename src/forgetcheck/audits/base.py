"""The Audit interface.

BINDING — implementation plan §4 and configs/audits.yaml.

An audit answers one question about one unlearned model: *does this model still carry the
influence of the forget set?* Six of them do so by different mechanisms, and the project's
contribution is the pattern of where they agree and where they do not. So the interface has to
make the audits genuinely comparable — same probes, same reference models, same record shape —
while leaving each free in how it decides.

Three design commitments, each of which the earlier stages made necessary:

**Audits are shard-local.** A Stage 5 shard produces ~3.6 GB of checkpoints on the Kaggle account
that ran it. An audit turns a 44.7 MB checkpoint into a few KB of Parquet, so audits run *on the
account that holds the checkpoints* and only records cross machines. Nothing in this module may
require every checkpoint to be present at once; :func:`available_targets` exists to enumerate
what this machine actually has.

**Audits must survive degenerate models.** Stage 5's `neggrad` control collapses to a constant
predictor (macro-F1 0.0187 at accuracy 0.1031). Constant outputs give an undefined ROC-AUC, a
zero-variance activation matrix for CKA, and a relearning curve indistinguishable from training
from scratch. Each of those is a **result to record**, not a run to skip — and each is a division
by zero waiting to happen. :func:`describe_degeneracy` characterises the model once, up front, so
every audit can branch on it explicitly rather than discovering it as a NaN three modules later.

**Audits never decide their own direction.** Whether "higher is better" for a metric is declared
in ``configs/metrics.yaml``, before any result exists, and ``records.validate()`` enforces it.
An audit emits a value and a name; what the value *means* is not its call to make.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "Audit",
    "AuditContext",
    "Degeneracy",
    "describe_degeneracy",
    "REGISTRY",
    "register",
    "get_audit",
    "audit_names",
    "UNDEFINED",
]

#: Value recorded when a statistic is genuinely undefined for this model — an AUC over
#: zero-variance scores, a correlation with a constant vector. NaN rather than a plausible
#: stand-in: a NaN cannot be silently averaged into a result, and `notes` always says why.
UNDEFINED = float("nan")


# --------------------------------------------------------------------------- degeneracy


@dataclass(frozen=True, slots=True)
class Degeneracy:
    """What is pathological about a model's outputs, measured once and passed around.

    Not a verdict on whether the model is *bad* — a collapsed model is a legitimate experimental
    condition here, since the destructive control is supposed to collapse. This only records the
    facts an audit needs in order to avoid computing something meaningless.
    """

    #: Number of distinct classes the model actually predicts across the probe set.
    n_predicted_classes: int
    #: True if every input gets the same predicted class.
    is_constant: bool
    #: Standard deviation of the max-softmax confidence. Zero means no per-example signal at
    #: all, which is what makes membership inference undefined rather than merely weak.
    confidence_std: float
    #: Standard deviation across the whole logit matrix.
    logit_std: float
    #: Fraction of probes whose logits contain a non-finite value.
    nonfinite_frac: float

    @property
    def has_per_example_signal(self) -> bool:
        """False when every example looks identical to any score-based attack.

        The threshold is not zero: float32 softmax over a saturated network produces
        confidences that differ in the last bit or two without carrying information. 1e-9 is
        comfortably below any real signal and comfortably above numerical dust.
        """
        return self.confidence_std > 1e-9 and self.nonfinite_frac == 0.0

    @property
    def summary(self) -> str:
        """A short phrase for the ``notes`` column, or '' when the model is unremarkable."""
        if self.nonfinite_frac > 0:
            return f"non-finite logits on {self.nonfinite_frac:.1%} of probes"
        if self.is_constant:
            return f"constant predictor (1 class, confidence sd {self.confidence_std:.2e})"
        if not self.has_per_example_signal:
            return f"no per-example signal (confidence sd {self.confidence_std:.2e})"
        return ""


def describe_degeneracy(logits: np.ndarray) -> Degeneracy:
    """Characterise a model's outputs from its logits on a probe set.

    Args:
        logits: ``(n_probes, n_classes)``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"expected (n_probes, n_classes), got shape {logits.shape}")

    finite_rows = np.isfinite(logits).all(axis=1)
    nonfinite_frac = float(1.0 - finite_rows.mean()) if logits.size else 0.0

    usable = logits[finite_rows]
    if usable.size == 0:
        return Degeneracy(0, True, 0.0, 0.0, nonfinite_frac)

    preds = usable.argmax(axis=1)
    # Softmax in a numerically stable way; these logits can be enormous after gradient ascent
    # (Stage 5 saw cross-entropy of 82 on the collapsed control).
    shifted = usable - usable.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)

    n_classes = int(np.unique(preds).size)
    return Degeneracy(
        n_predicted_classes=n_classes,
        is_constant=n_classes <= 1,
        confidence_std=float(probs.max(axis=1).std()),
        logit_std=float(usable.std()),
        nonfinite_frac=nonfinite_frac,
    )


# --------------------------------------------------------------------------- context


@dataclass
class AuditContext:
    """Everything an audit may draw on for one target model.

    The reference models are plural on purpose. A single retrained oracle is one sample from the
    retraining distribution, and comparing against one sample cannot distinguish "this model
    differs from a retrain" from "two retrains differ from each other" — the review's finding B4.
    ``oracle_logits`` therefore carries the *ensemble*, and every oracle-referenced metric is
    expressed against its spread.
    """

    #: The unlearned model under audit.
    run_id: str
    #: Its logits on the shared probe sets: ``{probe_set: (n, n_classes)}``.
    logits: Mapping[str, np.ndarray]

    #: The original model M0, same shape. The upper anchor.
    original_logits: Mapping[str, np.ndarray] = field(default_factory=dict)
    #: The retrained oracle ensemble: ``{probe_set: (n_oracles, n, n_classes)}``. The reference.
    oracle_logits: Mapping[str, np.ndarray] = field(default_factory=dict)

    #: True labels per probe set, for accuracy-like and attack-like statistics.
    labels: Mapping[str, np.ndarray] = field(default_factory=dict)

    #: Experimental coordinates, needed for the record and not derivable from the run_id.
    forget_kind: str = "random"
    forget_size: int = 0
    forget_stratum: str | None = None

    #: Protocol settings, straight from ``configs/audits.yaml``. Never defaulted in an audit:
    #: a value that differs between two models makes their comparison a protocol comparison.
    config: Mapping[str, Any] = field(default_factory=dict)

    #: Seeds any stochastic audit must use. An audit that draws its own randomness without
    #: recording the seed cannot be reproduced.
    audit_seed: int = 0

    device: str = "cpu"
    #: Lazily supplied, and only by audits that need to run the model rather than its outputs
    #: (relearning). Kept out of the common path so most audits never load 44.7 MB of weights.
    load_model: Any = None
    store: Any = None

    _degeneracy: Degeneracy | None = field(default=None, repr=False)

    @property
    def degeneracy(self) -> Degeneracy:
        """Computed once, from the forget-set probe, and cached."""
        if self._degeneracy is None:
            probe = self.logits.get("forget")
            if probe is None:  # a model with no forget probe cannot be audited meaningfully
                probe = next(iter(self.logits.values()))
            self._degeneracy = describe_degeneracy(probe)
        return self._degeneracy

    def has_oracles(self, probe_set: str) -> bool:
        arr = self.oracle_logits.get(probe_set)
        return arr is not None and len(arr) > 0


# --------------------------------------------------------------------------- interface


class Audit(ABC):
    """Base class for audits.

    Subclasses implement :meth:`measure`, returning ``{(metric, probe_set): value}``. Record
    construction, provenance and validation happen in :mod:`forgetcheck.audits.runner`, for the
    same reason timing lives outside the unlearning methods: so that six audits written by
    different people cannot disagree about how a record is built.
    """

    #: Registry key, and the value written to the record's ``audit`` column.
    name: str = ""

    #: The metric names this audit may emit. Every one must exist in configs/metrics.yaml —
    #: :func:`forgetcheck.audits.runner.check_registry` asserts it at import time, so an
    #: unregistered metric fails on a laptop rather than after six GPU-hours.
    metrics: tuple[str, ...] = ()

    #: Audits that need the oracle ensemble declare it, so the runner can skip them with a clear
    #: message rather than letting them produce silently meaningless numbers.
    needs_oracles: bool = True

    #: Audits that need model weights rather than cached logits declare it too — these are the
    #: expensive ones, and the runner reports them separately.
    needs_weights: bool = False

    def __init__(self, **cfg: Any):
        self.cfg = dict(cfg)

    @abstractmethod
    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        """Return ``{(metric_name, probe_set): value}`` for one model.

        Implementations must not raise on a degenerate model. Consult ``ctx.degeneracy`` and
        record :data:`UNDEFINED` for anything genuinely undefined; the runner copies
        ``ctx.degeneracy.summary`` into the record's ``notes`` so the reason travels with the
        number.
        """

    def notes_for(self, ctx: AuditContext) -> str:
        """Anything this audit wants recorded alongside its values. Override as needed."""
        return ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


# --------------------------------------------------------------------------- registry

REGISTRY: dict[str, type[Audit]] = {}


def register(cls: type[Audit]) -> type[Audit]:
    """Class decorator. Refuses duplicates — two audits under one name would make the ``audit``
    column ambiguous, and the whole analysis groups on it."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"audit name {cls.name!r} is already registered to {REGISTRY[cls.name]}")
    REGISTRY[cls.name] = cls
    return cls


def get_audit(name: str, **cfg: Any) -> Audit:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown audit {name!r}; known: {sorted(REGISTRY)}") from None
    return cls(**cfg)


def audit_names() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


# --------------------------------------------------------------------------- shared helpers


def softmax(logits: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Row-wise softmax, stable against the very large logits gradient ascent produces.

    ``floor`` clamps zero probabilities away from zero so that a log-based divergence cannot
    return infinity — configs/audits.yaml sets it for the behavioural audit.
    """
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    if floor > 0:
        p = np.clip(p, floor, None)
        p = p / p.sum(axis=-1, keepdims=True)
    return p


def oracle_mean(values: Sequence[float]) -> float:
    """Mean over the oracle ensemble, ignoring undefined members."""
    arr = np.asarray(values, dtype=np.float64)
    good = arr[np.isfinite(arr)]
    return float(good.mean()) if good.size else UNDEFINED


def iter_oracles(ctx: AuditContext, probe_set: str) -> Iterator[np.ndarray]:
    """Yield each oracle's logits for a probe set, or nothing if the ensemble is absent."""
    arr = ctx.oracle_logits.get(probe_set)
    if arr is None:
        return
    for i in range(len(arr)):
        yield np.asarray(arr[i])
