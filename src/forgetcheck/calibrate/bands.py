"""Oracle null bands: what a genuine retrain scores on each metric.

A band here is a **prediction interval for one new retrain**, not a confidence interval for the
retrains' mean and not a fixed ``mean ± 2·sd``. The distinction decides whether Stage 7 measures
the audits or the band.

``configs/audits.yaml`` specifies ``band_width_sd: 2.0``. With twelve retrains that is roughly
right; with the four a leave-one-out band has at seven of the eight conditions it is badly wrong.
The sample sd from four values is so noisy that a new draw from the *same* distribution falls
outside ``mean ± 2·s`` far more than 5% of the time -- the correct multiplier at n = 4 is
``t(3, 0.977) · sqrt(1 + 1/4) ≈ 3.6``, not 2. A fixed 2-sd band would therefore report inflated
false-positive rates at the small-n conditions *because of the band*, and they would read as
audit invalidity. The prediction interval keeps the nominal rate equal at every n, so a rate that
differs from nominal is about the audit.

``band_width_sd`` is honoured as the nominal coverage: 2.0 sd ↔ P(|Z| < 2) = 0.9545, applied
through the t-quantile. Documented as a deviation from the literal config reading.

The direction of "flagged" comes from the metric registry, never from here:

* ``lower_better``      (js_to_oracle, logit_l2) -- flagged above the band only;
* ``higher_better``     (pred_agreement)          -- flagged below the band only;
* ``closer_to_oracle``  (MIA AUCs, CKA, ...)      -- flagged outside it on either side.

A model *more* similar to the retrains than retrains are to one another is not evidence of
retained influence under a one-sided metric, and is a mismatch under a two-sided one. The
registry already made that call for every metric, before any result existed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "Band",
    "coverage_for",
    "prediction_band",
    "flag",
    "loo_flags",
    "nominal_fpr",
    "clopper_pearson",
]

#: Minimum retrains a band may be formed from. Two values give an sd with one degree of freedom
#: and a t-multiplier near 16 at 95% -- a band too wide to flag anything, which would report a
#: perfect false-positive rate for the wrong reason.
MIN_BAND_N = 3


def coverage_for(width_sd: float) -> float:
    """Two-sided normal coverage of ``± width_sd``: 2.0 → 0.9545."""
    return float(math.erf(width_sd / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class Band:
    n: int
    mean: float
    sd: float
    lo: float
    hi: float

    @property
    def defined(self) -> bool:
        return self.n >= MIN_BAND_N and math.isfinite(self.lo) and math.isfinite(self.hi)


def prediction_band(values, *, coverage: float) -> Band:
    """Prediction interval for one new draw, from ``values`` (NaNs ignored).

    ``mean ± t(n-1, (1+coverage)/2) · s · sqrt(1 + 1/n)``. Undefined below ``MIN_BAND_N``.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < MIN_BAND_N:
        mean = float(v.mean()) if n else float("nan")
        return Band(n, mean, float("nan"), float("nan"), float("nan"))
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + coverage / 2.0, df=n - 1)) * sd * math.sqrt(1.0 + 1.0 / n)
    return Band(n, mean, sd, mean - half, mean + half)


def flag(value: float, band: Band, direction: str) -> bool | None:
    """Does ``value`` fall on the "differs from a retrain" side of ``band``? ``None`` if unknowable."""
    if not (band.defined and math.isfinite(value)):
        return None
    if direction == "lower_better":
        return bool(value > band.hi)
    if direction == "higher_better":
        return bool(value < band.lo)
    if direction == "closer_to_oracle":
        return bool(value < band.lo or value > band.hi)
    raise ValueError(f"unknown direction {direction!r}")


def nominal_fpr(direction: str, coverage: float) -> float:
    """The false-positive rate a perfectly calibrated band should show on held-out retrains."""
    tail = (1.0 - coverage) / 2.0
    return 2 * tail if direction == "closer_to_oracle" else tail


def loo_flags(values, *, direction: str, coverage: float) -> list[bool | None]:
    """Flag each retrain against a band formed from the *other* retrains.

    Leave-one-out, so every retrain is tested out of sample and every one of them counts -- at
    seven conditions there are only five, and holding three out for a separate test (the
    pre-registered 9/3 split, which exists only at the primary condition) would leave two to
    form a band from. The 9/3 split is still reported at the primary condition as a check.
    """
    v = np.asarray(values, dtype=np.float64)
    out: list[bool | None] = []
    for i in range(v.size):
        others = np.delete(v, i)
        out.append(flag(float(v[i]), prediction_band(others, coverage=coverage), direction))
    return out


def clopper_pearson(k: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for a rate of ``k`` in ``n``. ``(nan, nan)`` when ``n`` is 0.

    Exact rather than normal-approximate because the rates here are measured on 3 to 17 models,
    where a Wald interval would run below 0 and report 0/5 as certainly zero.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi
