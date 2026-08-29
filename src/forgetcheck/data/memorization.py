"""Per-example memorization scores, and the cheap proxies that stand in for them.

The difficulty axis of the experiment is built on these. Exact Feldman-Zhang memorization scores
require training thousands of models; the project uses the CIFAR-10 scores released with the RUM
codebase instead. Note that Feldman & Zhang's own release covers CIFAR-100 and ImageNet, *not*
CIFAR-10 — the RUM repository is the only source, so there is no canonical fallback if it
disappears, and a proxy is the contingency (research log §2.5, m3).

Proxies, with Spearman correlations against exact memorization reported by Zhao et al. [31]:

===================  ===========  =========================================================
proxy                rho          cost
===================  ===========  =========================================================
confidence           -0.80..-0.91  ~0.002% of exact
binary accuracy      -0.71..-0.89  comparable
holdout retraining    0.62..0.67   ~0.001% of exact
loss curvature        0.69..0.70   ~0.074% of exact
===================  ===========  =========================================================

**The sign is the trap.** Confidence and accuracy correlate *negatively* with memorization: an
example the model is confident about is one it generalised to, not one it memorised. A proxy fed
in without inversion silently swaps the high and low strata — which would turn the primary
condition into the negative control and the negative control into the primary condition, while
producing results that look entirely plausible. Every proxy in this module therefore returns a
value that is already oriented as *memorization* (higher = more memorised), and the conversion
is done in one place, named, and tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

import numpy as np

__all__ = [
    "load_scores",
    "confidence_proxy",
    "accuracy_proxy",
    "ProxyKind",
    "RUM_ES_PARTITION_MEMORIZATION",
    "check_scores",
]

ProxyKind = Literal["rum", "confidence", "accuracy"]

#: RUM's Figure 8 statistics, recorded here **only to document what they are not**.
#:
#: These are the mean memorization scores within RUM's low/medium/high **embedding-space
#: entanglement (ES)** partitions — their *other* difficulty factor — not within their
#: memorization partitions. An earlier version of this module compared our memorization strata
#: against them, which was meaningless: ES partitions are not sorted by memorization, which is
#: exactly why their standard deviations (+/- 0.203 and up) are so large.
#:
#: RUM does not publish summary statistics for its memorization strata, so there is no external
#: number to check ours against. The checks in this module are therefore self-contained.
RUM_ES_PARTITION_MEMORIZATION: Final[dict[str, tuple[float, float]]] = {
    "low": (0.084, 0.203),
    "medium": (0.134, 0.235),
    "high": (0.390, 0.326),
}


def load_scores(
    path: str | Path,
    *,
    n_train: int = 50_000,
    key: str | None = None,
) -> np.ndarray:
    """Load memorization scores from ``.npy``, ``.npz``, ``.csv`` or ``.json``.

    Args:
        path: the scores file, as distributed with the RUM codebase.
        n_train: expected length. A mismatch is fatal: scores that do not line up with the
            training set index-for-index would silently select the wrong examples.
        key: array name, for ``.npz`` files with more than one array.

    Returns:
        float64, shape ``(n_train,)``, oriented as memorization (higher = more memorised).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"memorization scores not found at {p}. Download them from the RUM repository "
            "(github.com/kairanzhao/RUM), or fall back to a proxy — see "
            "forgetcheck.data.memorization.confidence_proxy."
        )

    suffix = p.suffix.lower()
    if suffix == ".npy":
        scores = np.load(p)
    elif suffix == ".npz":
        with np.load(p) as z:
            names = list(z.files)
            if key is not None:
                scores = z[key]
            elif len(names) == 1:
                scores = z[names[0]]
            else:
                raise ValueError(
                    f"{p} holds several arrays {names}; pass key= to say which one carries "
                    "the scores"
                )
    elif suffix == ".csv":
        scores = np.loadtxt(p, delimiter=",")
    elif suffix == ".json":
        scores = np.asarray(json.loads(p.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"unsupported score file type {suffix!r}")

    return _validate(np.asarray(scores, dtype=np.float64).ravel(), n_train, source=str(p))


def _validate(scores: np.ndarray, n_train: int, *, source: str) -> np.ndarray:
    if scores.shape != (n_train,):
        raise ValueError(
            f"{source}: expected {n_train} scores, got {scores.shape}. Scores must align with "
            "the training set index-for-index."
        )
    if not np.all(np.isfinite(scores)):
        n_bad = int((~np.isfinite(scores)).sum())
        raise ValueError(f"{source}: {n_bad} non-finite score(s)")
    lo, hi = float(scores.min()), float(scores.max())
    if lo < -0.01 or hi > 1.01:
        raise ValueError(
            f"{source}: scores span [{lo:.3f}, {hi:.3f}], outside the expected [0, 1] range "
            "for memorization estimates. Check whether this file holds something else."
        )
    return scores


# --------------------------------------------------------------------------- proxies


def _as_memorization(signal: np.ndarray, *, higher_means_generalised: bool) -> np.ndarray:
    """Orient a proxy signal so that higher always means *more memorised*.

    The single place the sign flip happens. Confidence and accuracy are 'how well did the model
    generalise to this example', which is the opposite of memorization, so they are inverted
    here rather than at each call site — a flip that is done in four places is a flip that will
    eventually be forgotten in one of them.
    """
    s = np.asarray(signal, dtype=np.float64)
    return (1.0 - s) if higher_means_generalised else s


@np.errstate(all="ignore")
def confidence_proxy(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    n_train: int | None = None,
) -> np.ndarray:
    """Memorization proxy from held-out-ish model confidence on the true label.

    Highest-correlating proxy in Zhao et al. (Spearman -0.80 to -0.91 against exact
    memorization, before the inversion applied here), at roughly 0.002% of the cost.

    Args:
        logits: ``(n, n_classes)`` model outputs over the training set.
        labels: ``(n,)`` true labels.

    Returns:
        ``1 - p(true label)``, in [0, 1], oriented as memorization.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2:
        raise ValueError(f"logits must be 2-D, got shape {z.shape}")
    if len(z) != len(y):
        raise ValueError("logits and labels disagree in length")

    z = z - z.max(axis=1, keepdims=True)  # stabilise before exponentiating
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    p_true = probs[np.arange(len(y)), y]

    scores = _as_memorization(p_true, higher_means_generalised=True)
    return _validate(scores, n_train or len(scores), source="confidence_proxy")


def accuracy_proxy(
    predictions: np.ndarray, labels: np.ndarray, *, n_train: int | None = None
) -> np.ndarray:
    """Binary memorization proxy: 1 where the model gets the example wrong.

    Coarser than :func:`confidence_proxy` (Spearman -0.71 to -0.89) and it produces only two
    distinct values, so the stratified selection falls back almost entirely on the index
    tiebreak. Usable as a sanity check, a poor choice for the difficulty axis itself.
    """
    pred = np.asarray(predictions, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if pred.shape != y.shape:
        raise ValueError("predictions and labels disagree in shape")
    correct = (pred == y).astype(np.float64)
    scores = _as_memorization(correct, higher_means_generalised=True)
    return _validate(scores, n_train or len(scores), source="accuracy_proxy")


# --------------------------------------------------------------------------- Stage 2 gate

#: Median memorization above this means most of the training set is "highly memorised", which no
#: real CIFAR-10 score distribution is. The strongest available signal that a proxy was fed in
#: without inversion.
_MAX_PLAUSIBLE_MEDIAN: Final = 0.5

#: Minimum spread between the low and high stratum means for the difficulty axis to be worth
#: running. Below this the axis has no dynamic range and the primary condition is not meaningfully
#: harder than the negative control.
_MIN_STRATUM_SPREAD: Final = 0.05

#: Overlap between two strata beyond this fraction makes them a confounded pair rather than two
#: conditions.
_MAX_STRATUM_OVERLAP: Final = 0.05


def check_scores(
    scores: np.ndarray,
    *,
    size: int = 3000,
    strict: bool = True,
) -> dict[str, dict[str, float]]:
    """Sanity-check a memorization score array before anything is trained against it.

    Stage 2's gate, made runnable. Call it the moment the real scores are downloaded.

    **There is no external number to check against.** RUM publishes memorization statistics only
    for its embedding-space entanglement partitions (see
    :data:`RUM_ES_PARTITION_MEMORIZATION`), not for its memorization partitions, so a comparison
    against those figures would be comparing two different quantities. The checks here are
    self-contained instead, and they test the two things that can actually go wrong.

    **Orientation.** Note that stratum *ordering* carries no information: the strata are defined
    by rank on the array passed in, so ``low <= medium <= high`` holds for any input including a
    fully inverted one. What discriminates is magnitude — real memorization is heavily
    bottom-weighted, so the median sits near zero, while an un-inverted proxy is top-weighted.

    **Dynamic range.** If the extreme strata barely differ, the difficulty axis has no range:
    the primary condition would not be meaningfully harder than the negative control, and
    hypothesis H4 could not be tested at all.

    Args:
        strict: raise on failure rather than only reporting.

    Returns:
        ``{stratum: {mean, std, min, max}}`` plus a ``"summary"`` entry carrying the median and
        the low-to-high spread.
    """
    from .forget_sets import stratum_summary

    s = np.asarray(scores, dtype=np.float64)
    report = dict(stratum_summary(s, size=size))

    from .forget_sets import ForgetSpec

    median = float(np.median(s))
    spread = report["high"]["mean"] - report["low"]["mean"]

    # Disjointness is not guaranteed by RUM's definitions. low and high are rank extremes, but
    # medium is "nearest to 0.5" on the score scale, so on a distribution with a thin tail the
    # medium selection can reach up into the top of the ranking and overlap with high. Two
    # conditions sharing examples would be confounded, so this is checked rather than assumed.
    picks = {
        st: set(
            ForgetSpec(kind="memstratum", size=size, stratum=st)
            .indices(len(s), memorization=s)
            .tolist()
        )
        for st in ("low", "medium", "high")
    }
    overlaps = {
        f"{a}_{b}": len(picks[a] & picks[b])
        for a, b in (("low", "medium"), ("medium", "high"), ("low", "high"))
    }
    worst = max(overlaps.values())

    report["summary"] = {
        "median": round(median, 4),
        "low_to_high_spread": round(spread, 4),
        "fraction_near_zero": round(float((s < 0.01).mean()), 4),
        "fraction_above_half": round(float((s > 0.5).mean()), 4),
        "max_stratum_overlap": worst,
        **{f"overlap_{k}": v for k, v in overlaps.items()},
    }

    if not strict:
        return report

    if median > _MAX_PLAUSIBLE_MEDIAN:
        raise ValueError(
            f"median memorization is {median:.3f}, above {_MAX_PLAUSIBLE_MEDIAN}. That would "
            "mean most of CIFAR-10 is highly memorised, which no real score distribution shows. "
            "The scores look inverted: if a proxy was used, confidence and accuracy measure "
            "generalisation and must be passed through the inversion in this module rather "
            "than used raw. Left uncorrected this swaps the high and low strata, turning the "
            "primary condition into the negative control while producing plausible-looking "
            "results."
        )

    if spread < _MIN_STRATUM_SPREAD:
        raise ValueError(
            f"the high and low strata differ by only {spread:.4f} in mean memorization "
            f"(below {_MIN_STRATUM_SPREAD}). The difficulty axis has no dynamic range: the "
            "primary condition would not be meaningfully harder than the negative control, and "
            "hypothesis H4 could not be tested. Either the scores are near-constant or they are "
            "not memorization estimates."
        )

    if worst > _MAX_STRATUM_OVERLAP * size:
        raise ValueError(
            f"strata overlap: {overlaps}. medium-mem is defined as the examples nearest to 0.5 "
            "on the score scale, so on a distribution with too thin a tail above 0.5 it reaches "
            "into the top of the ranking and collides with high-mem. Two conditions sharing "
            "examples are confounded, not independent. Either the scores are not what they "
            "should be, or the stratum size is too large for this distribution -- reduce it, or "
            "define medium by a percentile instead and record the departure."
        )

    return report
