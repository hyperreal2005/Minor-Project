"""Native decision rules: each retrain-free audit used the way practitioners use it.

An oracle-calibrated band flags about 5% of held-out retrains *by construction* -- it asks
whether a retrain looks like the other retrains. That verifies the calibration and says nothing
about the audit. The validity question is what the audit's **own** rule does to a model that
provably never saw the forget set. A retrain an audit calls "not forgotten" is a false positive
of that audit as commonly used, and a rate far above the rule's nominal alpha means the rule is
broken -- not strict (plan §15.5).

Only the retrain-free audits have native rules. Behavioural divergence, CKA and relearning are
*defined* relative to a retrain; there is no standalone version of them to test, and their
calibration is the band.

Each rule is a one-sided significance test at ``alpha``, so a valid rule should flag about
``alpha`` of genuine retrains:

``auc_above_chance``
    ROC-AUC exceeds 0.5 -- the reading every MIA evaluation reports. The null standard error is
    Hanley & McNeil's ``sqrt((n1 + n2 + 1) / (12 n1 n2))`` with ``n1 = n2 = n_probe``: both
    attacks score members against the same number of non-members (population: size-matched;
    RMIA: ``min(n_forget, n_test/2)``, equal to ``n_forget`` at every condition here).

``tpr_above_fpr``
    TPR at the nominal operating point exceeds that point -- the low-FPR reading the plan names
    as RMIA's headline. Under the null, members exceed a threshold set at FPR = f on non-members
    with probability ≈ f, so the count is ≈ Binomial(n1, f). Approximate: it ignores the noise
    in the threshold itself, which makes the test slightly liberal.

``verdict_in_training``
    SDE's own decision: margin ≤ 0, i.e. the forget set looks like training data. Not a test, a
    rule; its "alpha" is whatever SDE's authors' rule implies, which is exactly what is measured.
"""

from __future__ import annotations

import math

from scipy import stats

__all__ = ["NATIVE_RULES", "native_flag", "native_rule_for"]

#: (audit, metric) -> rule. Only metrics listed here have a native false-positive rate.
NATIVE_RULES: dict[tuple[str, str], str] = {
    ("privacy_population", "mia_auc_pop"): "auc_above_chance",
    ("privacy_population", "mia_tpr_at_fpr_pop"): "tpr_above_fpr",
    ("privacy_rmia", "mia_auc_rmia"): "auc_above_chance",
    ("privacy_rmia", "mia_tpr_at_fpr_rmia"): "tpr_above_fpr",
    ("sde", "sde_verdict"): "verdict_in_training",
}


def native_rule_for(audit: str, metric: str) -> str | None:
    return NATIVE_RULES.get((audit, metric))


def native_flag(
    rule: str, value: float, *, n_probe: int, alpha: float = 0.05, fpr: float = 0.01
) -> bool | None:
    """Does the audit's own rule call this model "not forgotten"? ``None`` if it cannot say."""
    if not math.isfinite(value) or n_probe <= 0:
        return None
    if rule == "auc_above_chance":
        n = float(n_probe)
        se = math.sqrt((2 * n + 1) / (12 * n * n))
        return bool((value - 0.5) / se > stats.norm.ppf(1 - alpha))
    if rule == "tpr_above_fpr":
        k = int(round(value * n_probe))
        p = float(stats.binom.sf(k - 1, n_probe, fpr)) if k > 0 else 1.0
        return bool(p < alpha)
    if rule == "verdict_in_training":
        return bool(value < 0.5)  # sde_verdict is 1.0 for "out of training", 0.0 for "in training"
    raise ValueError(f"unknown native rule {rule!r}")
