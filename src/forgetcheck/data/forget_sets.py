"""ForgetSpec: declarative forget-set definitions that resolve deterministically.

BINDING — implementation plan §4.2. Stage 2's gate is that every spec regenerates a
byte-identical index array on two different machines.

The design separates two orthogonal factors that v1.0 of the master reference had collapsed into
one list of conditions:

**Size axis** — ``rand-500``, ``rand-2500``, ``rand-5000``. Demoted to a scaling check. A random
CIFAR-10 subset is overwhelmingly low-memorization, and for low-memorization examples the ideal
untraining solution is close to no change from the original model at all, so these conditions are
expected to carry little signal (master reference §13, and [19]).

**Difficulty axis** — ``mem-low-3000``, ``mem-med-3000``, ``mem-high-3000`` at fixed size, so
strata are compared like with like. ``mem-high-3000`` is the primary condition; ``mem-low-3000``
is a *designed negative control* whose job is to show no signal.

Plus ``canary-500``, where residual influence is known by construction.

Determinism comes from a seeded ``numpy.random.Generator`` with an explicit ``PCG64`` bit
generator rather than the legacy global RNG, because the legacy one's stream is not guaranteed
stable across numpy versions and Stage 2's gate is exactly that stability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "ForgetSpec",
    "ForgetKind",
    "Stratum",
    "STANDARD_CONDITIONS",
    "PRIMARY_CONDITION",
    "spec_by_id",
    "all_specs",
]

ForgetKind = Literal["random", "memstratum", "canary"]
Stratum = Literal["low", "medium", "high"]

_STRATUM_SLUG: Final[dict[str, str]] = {"low": "low", "medium": "med", "high": "high"}

#: Independent random streams per condition kind. Fixed values, never reordered — changing one
#: changes which examples every condition of that kind selects, invalidating anything already
#: trained against it.
_KIND_STREAM: Final[dict[str, int]] = {"random": 0, "canary": 1, "memstratum": 2}

#: Default seed for forget-set selection. Fixed across the project so that every method and every
#: training seed attacks an identical forget set — the comparison is between unlearning
#: algorithms, not between the draws they happened to get.
DEFAULT_SELECTION_SEED: Final = 100


@dataclass(frozen=True, slots=True)
class ForgetSpec:
    """A declarative forget-set definition.

    Resolving a spec is a pure function of its fields plus the dataset labels, so two machines
    that agree on the dataset hash necessarily agree on the indices.
    """

    kind: ForgetKind
    size: int
    stratum: Stratum | None = None
    selection_seed: int = DEFAULT_SELECTION_SEED
    proxy: str = "rum"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.kind == "memstratum":
            if self.stratum is None:
                raise ValueError("kind='memstratum' requires a stratum")
            if self.stratum not in _STRATUM_SLUG:
                raise ValueError(
                    f"stratum must be one of {sorted(_STRATUM_SLUG)}, got {self.stratum!r}"
                )
        elif self.stratum is not None:
            raise ValueError(f"kind={self.kind!r} does not take a stratum")

    # -- identity ------------------------------------------------------------

    @property
    def forget_id(self) -> str:
        """The human-readable, stable id used in run identifiers and record rows."""
        if self.kind == "random":
            return f"rand-{self.size}"
        if self.kind == "memstratum":
            return f"mem-{_STRATUM_SLUG[self.stratum]}-{self.size}"
        return f"canary-{self.size}"

    @property
    def forget_kind(self) -> str:
        """The value written to ``RunRecord.forget_kind``."""
        return self.kind

    def as_record_fields(self) -> dict[str, Any]:
        """The identity fields a record needs, so audits do not re-derive them by hand."""
        return {
            "forget_id": self.forget_id,
            "forget_kind": self.kind,
            "forget_size": self.size,
            "forget_stratum": self.stratum,
            "selection_seed": self.selection_seed,
        }

    def with_seed(self, seed: int) -> "ForgetSpec":
        return replace(self, selection_seed=seed)

    # -- resolution ----------------------------------------------------------

    def _rng(self) -> np.random.Generator:
        """A generator on a stream private to this kind of condition.

        Each ``kind`` gets an independent stream derived from the same selection seed. Without
        this, ``rand-500`` and ``canary-500`` — same seed, same size — draw *identical*
        examples, so the canary condition would corrupt exactly the examples the random-500
        condition forgets. Two conditions meant to be independent would then share every
        per-example idiosyncrasy, and any difference between them would be confounded.

        Streams are separated by kind rather than by full spec identity so that nesting survives
        *within* the size axis, which is what makes size comparisons clean.
        """
        stream = _KIND_STREAM[self.kind]
        return np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([self.selection_seed, stream]))
        )

    def indices(
        self,
        n_train: int,
        *,
        memorization: np.ndarray | None = None,
    ) -> np.ndarray:
        """Resolve to a sorted, unique int64 index array.

        Args:
            n_train: size of the training set the indices point into.
            memorization: per-example memorization scores, required for ``memstratum``.

        Returns:
            Sorted ascending, so the array — and therefore its hash — is canonical. The store
            enforces the same invariant on write.
        """
        if self.size > n_train:
            raise ValueError(f"forget size {self.size} exceeds training set size {n_train}")

        rng = self._rng()

        if self.kind in ("random", "canary"):
            # permutation()[:size] rather than choice(size, replace=False), for two reasons.
            #
            # Nesting: the size axis becomes rand-500 subset rand-2500 subset rand-3000 subset
            # rand-5000, so a difference between size conditions is attributable to size alone
            # and not to which examples happened to be drawn.
            #
            # Determinism: numpy's choice() switches between Floyd's algorithm and a full
            # permutation depending on the size-to-population ratio, which produced inconsistent
            # nesting here (rand-2500 was nested in rand-3000 but rand-500 was not nested in
            # rand-2500) and makes the result depend on an implementation detail that could
            # change between numpy versions. permutation() has one code path.
            return np.sort(rng.permutation(n_train)[: self.size]).astype(np.int64)

        # memstratum
        if memorization is None:
            raise ValueError(
                f"{self.forget_id}: memorization scores are required for a stratified forget "
                "set. Load them with forgetcheck.data.memorization.load_scores()."
            )
        scores = np.asarray(memorization, dtype=np.float64)
        if scores.shape != (n_train,):
            raise ValueError(
                f"memorization scores have shape {scores.shape}, expected ({n_train},)"
            )
        return _stratified_indices(scores, self.size, self.stratum, rng)


#: The midpoint of the memorization range, which the medium stratum is centred on. RUM defines
#: medium-mem as the N examples "nearest to 0.5, i.e. the midpoint of the range of memorization
#: scores" — a point on the *score* scale, not the rank scale.
_MEM_MIDPOINT: Final = 0.5


def _stratified_indices(
    scores: np.ndarray, size: int, stratum: str, rng: np.random.Generator
) -> np.ndarray:
    """Select a memorization stratum, following RUM's definitions exactly.

    From the RUM paper (Zhao et al., NeurIPS 2024, §3): they sort all examples by score and take
    "the lowest N scores ('low-mem'), the highest N ('high-mem'), and the N that are nearest to
    0.5, i.e. the midpoint of the range of memorization scores ('medium-mem'), where N = 3000."

    Note that **medium is a point on the score scale, not the rank scale**. This matters a great
    deal on a memorization distribution, which is heavily bottom-weighted: the middle of the
    *rank* ordering still sits near zero, so a rank-defined medium stratum collapses toward the
    low one and stops interpolating anything. Selecting by distance to 0.5 picks examples with
    genuinely intermediate memorization instead.

    Ties are broken by index rather than by numpy's internal sort order. CIFAR-10 memorization
    contains large ties near zero — many examples are simply not memorized at all — and an
    unstable tiebreak there would make the low stratum differ between numpy versions, breaking
    Stage 2's determinism gate in the one place it is most likely to matter.
    """
    n = len(scores)

    if stratum == "medium":
        # Rank by |score - 0.5|, ties by index.
        distance = np.abs(scores - _MEM_MIDPOINT)
        order = np.lexsort((np.arange(n), distance))
        chosen = order[:size]
    else:
        order = np.lexsort((np.arange(n), scores))  # ascending score, then ascending index
        chosen = order[:size] if stratum == "low" else order[-size:]

    return np.sort(chosen).astype(np.int64)


# --------------------------------------------------------------------------- the eight

#: The eight conditions of implementation plan §3.1, in the order they appear there.
STANDARD_CONDITIONS: Final[tuple[ForgetSpec, ...]] = (
    ForgetSpec(kind="random", size=500),
    ForgetSpec(kind="random", size=2500),
    ForgetSpec(kind="random", size=5000),
    ForgetSpec(kind="random", size=3000),
    ForgetSpec(kind="memstratum", size=3000, stratum="low"),
    ForgetSpec(kind="memstratum", size=3000, stratum="medium"),
    ForgetSpec(kind="memstratum", size=3000, stratum="high"),
    ForgetSpec(kind="canary", size=500),
)

#: Where the oracle ensemble and the validity analysis are anchored.
PRIMARY_CONDITION: Final[ForgetSpec] = ForgetSpec(
    kind="memstratum", size=3000, stratum="high"
)


def all_specs() -> dict[str, ForgetSpec]:
    """``{forget_id: spec}`` for the eight standard conditions."""
    out: dict[str, ForgetSpec] = {}
    for spec in STANDARD_CONDITIONS:
        fid = spec.forget_id
        if fid in out:
            raise ValueError(f"two standard conditions share the id {fid!r}")
        out[fid] = spec
    return out


def spec_by_id(forget_id: str) -> ForgetSpec:
    """Look up a standard condition by its id."""
    specs = all_specs()
    try:
        return specs[forget_id]
    except KeyError:
        raise KeyError(
            f"unknown forget condition {forget_id!r}; the eight standard conditions are "
            f"{sorted(specs)}"
        ) from None


def stratum_summary(memorization: np.ndarray, size: int = 3000) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of the memorization score within each stratum.

    Stage 2's gate is that these match RUM's published CIFAR-10 values — low 0.084 +/- 0.203,
    medium 0.134 +/- 0.235, high 0.390 +/- 0.326. Run this the moment the real scores are
    downloaded, before anything is trained against them.

    Watch the low/medium gap specifically. A large fraction of CIFAR-10 has a memorization score
    at or near zero, so a rank-defined medium stratum can collapse onto the low one. RUM's
    published means are close but distinct; if ours are not, the medium stratum is not
    interpolating anything and the difficulty axis effectively has two points, not three. That
    is a finding about the data, not a bug — but it must be noticed before week 13, not after.
    """
    scores = np.asarray(memorization, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for stratum in ("low", "medium", "high"):
        spec = ForgetSpec(kind="memstratum", size=size, stratum=stratum)
        sel = scores[spec.indices(len(scores), memorization=scores)]
        out[stratum] = {
            "mean": float(sel.mean()),
            "std": float(sel.std()),
            "min": float(sel.min()),
            "max": float(sel.max()),
        }
    return out


def materialise(
    spec: ForgetSpec,
    n_train: int,
    *,
    memorization: np.ndarray | None = None,
    store=None,
) -> np.ndarray:
    """Resolve a spec, optionally caching the result in an artifact store.

    The cache is not a performance optimisation — resolution is milliseconds. It exists so that
    every run can *prove* it used the same index array rather than assuming it, and so that a
    mismatch is caught at load time rather than inferred from strange results later.
    """
    idx = spec.indices(n_train, memorization=memorization)

    if store is not None:
        fid = spec.forget_id
        if store.has_forget_set(fid):
            cached, _ = store.load_forget_set(fid)
            if not np.array_equal(cached, idx):
                raise RuntimeError(
                    f"{fid}: freshly resolved indices differ from the cached ones. Either the "
                    "spec, the memorization scores, or the numpy version changed. Do not "
                    "proceed — results computed against the two arrays are not comparable."
                )
        else:
            store.save_forget_set(fid, idx, meta=spec.as_record_fields())

    return idx
