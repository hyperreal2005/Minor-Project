"""Audit Layer 2 — the population membership-inference attack (family: privacy_weak).

One attacker for every example. Train a classifier to tell forget-set examples from never-seen
test examples using only what the model outputs about them — loss, confidence, entropy — and
report how well it does out of fold. If unlearning worked, forget examples should look like test
examples and the attacker should be at chance.

This is the attack the NeurIPS unlearning starter kit ships and the one most papers report. It
is included *because* of that, not despite it: population attacks are known to overstate privacy
protection [21], since a single decision rule across all examples cannot see the per-example
leakage that RMIA (Layer 3) exists to expose. Where the two disagree, the disagreement is the
within-family result the project is looking for.

**Why this audit does not need the oracle ensemble.** AUC is a property of one model. The
oracle's AUC is obtained by running this same audit on oracle checkpoints, which the Stage 6
runner does — the held-out oracles pass through every audit as candidates, and the band-forming
ones supply the null distribution. Comparing the two happens at analysis time, under the
``closer_to_oracle`` direction the registry declares. An audit that needed to see the oracle in
order to score a candidate would be an audit that could shape its answer to the reference.

**On the constant predictor.** Stage 5's `neggrad` control collapses to one class for every
input. Confidence and entropy are then identical across examples; loss is not — it is low for
examples whose true label happens to be the predicted class and high otherwise — but that
variation tracks the *label*, which members and non-members share, and carries no membership
signal. The attacker therefore lands at chance up to sampling noise, and that value is recorded,
not NaN, because it is *true*: the model leaks nothing about the forget set because it retains
nothing about anything. A model that wins the naive forgetting metric and passes the privacy
audit while being useless is the audit-validity finding in one row. The degeneracy note travels
with the record so nobody reads the ~0.5 as a success.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .base import UNDEFINED, Audit, AuditContext, register, softmax

__all__ = ["PopulationMIA", "attack_features", "roc_auc", "tpr_at_fpr", "balanced_accuracy"]


# --------------------------------------------------------------------------- statistics


def roc_auc(scores: np.ndarray, is_member: np.ndarray) -> float:
    """ROC-AUC as the Mann–Whitney U statistic: P(score(member) > score(non-member)), ties
    counted as one half. Exact, dependency-free, and well-defined when every score is identical
    (it returns 0.5, which is the truth about an attacker with no signal)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_member = np.asarray(is_member, dtype=bool)
    pos, neg = scores[is_member], scores[~is_member]
    if pos.size == 0 or neg.size == 0:
        return UNDEFINED
    # Average ranks handle ties correctly; argsort twice is the standard trick.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Replace ranks within tie groups by their mean.
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    u = ranks[is_member].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def tpr_at_fpr(scores: np.ndarray, is_member: np.ndarray, fpr: float) -> float:
    """True-positive rate at the largest threshold whose false-positive rate is <= ``fpr``.

    The headline privacy number. Aggregate AUC averages over every operating point, including
    the useless high-FPR ones; an attacker who can name a few members with near certainty is a
    privacy failure that AUC 0.55 hides completely.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_member = np.asarray(is_member, dtype=bool)
    neg = np.sort(scores[~is_member])[::-1]  # descending
    pos = scores[is_member]
    if pos.size == 0 or neg.size == 0:
        return UNDEFINED
    # Threshold: the score above which at most fpr * n_neg non-members fall.
    k = int(np.floor(fpr * neg.size))
    if k <= 0:
        thresh = neg[0]  # must exceed the top non-member
        return float((pos > thresh).mean())
    thresh = neg[k - 1]
    return float((pos > thresh).mean())


def balanced_accuracy(scores: np.ndarray, is_member: np.ndarray, threshold: float = 0.5) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    is_member = np.asarray(is_member, dtype=bool)
    pred = scores >= threshold
    tpr = pred[is_member].mean() if is_member.any() else UNDEFINED
    tnr = (~pred[~is_member]).mean() if (~is_member).any() else UNDEFINED
    return float(0.5 * (tpr + tnr))


# --------------------------------------------------------------------------- features


def attack_features(logits: np.ndarray, labels: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """``(n, 3)``: per-example loss, max-softmax confidence, and predictive entropy."""
    p = softmax(logits, floor=floor)
    n = p.shape[0]
    loss = -np.log(p[np.arange(n), np.asarray(labels, dtype=int)])
    confidence = p.max(axis=1)
    entropy = -(p * np.log(p)).sum(axis=1)
    return np.stack([loss, confidence, entropy], axis=1)


# --------------------------------------------------------------------------- classifier


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by L-BFGS. Tiny problem, no need for a framework."""
    n, d = x.shape
    yy = np.where(y, 1.0, -1.0)

    def f(wb):
        w, b = wb[:d], wb[d]
        z = yy * (x @ w + b)
        # log(1 + exp(-z)) computed stably
        loss = np.logaddexp(0.0, -z).mean() + 0.5 * l2 * (w @ w)
        s = -yy / (1.0 + np.exp(z))  # d/dz of the mean term, sign folded in
        gw = (x * s[:, None]).mean(axis=0) + l2 * w
        gb = s.mean()
        return loss, np.append(gw, gb)

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d], float(res.x[d])


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    # A zero-variance feature (the constant predictor gives two: confidence and entropy) must not
    # become a division by zero. Leave it centred at zero; it carries no information either way.
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def cross_validated_scores(
    features: np.ndarray, is_member: np.ndarray, *, n_splits: int, seed: int
) -> np.ndarray:
    """Out-of-fold membership probabilities from a stratified ``n_splits``-fold fit.

    Every example is scored by a classifier that never saw it, so the reported AUC is an honest
    estimate of what an attacker could achieve, not a fit statistic.
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    scores = np.full(n, np.nan)

    folds = np.empty(n, dtype=int)
    for cls in (True, False):
        idx = np.flatnonzero(is_member == cls)
        rng.shuffle(idx)
        folds[idx] = np.arange(idx.size) % n_splits

    for k in range(n_splits):
        test = folds == k
        train = ~test
        if is_member[train].all() or not is_member[train].any():
            scores[test] = 0.5  # a fold with one class cannot be fit; chance is the honest score
            continue
        xtr, xte = _standardise(features[train], features[test])
        w, b = _fit_logistic(xtr, is_member[train])
        scores[test] = 1.0 / (1.0 + np.exp(-(xte @ w + b)))
    return scores


# --------------------------------------------------------------------------- audit


@register
class PopulationMIA(Audit):
    """Population membership inference: forget set (members) vs test set (non-members)."""

    name = "privacy_population"
    metrics = ("mia_auc_pop", "mia_acc_pop", "mia_tpr_at_fpr_pop")
    needs_oracles = False

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        cfg = ctx.config
        floor = float(cfg.get("prob_floor", 1e-12))
        n_splits = int(cfg.get("n_splits", 5))
        fpr = float(cfg.get("tpr_at_fpr", 0.01))
        nonmember_source = str(cfg.get("nonmember_source", "test"))

        forget = ctx.logits.get("forget")
        nonmem = ctx.logits.get(nonmember_source)
        y_f = ctx.labels.get("forget")
        y_n = ctx.labels.get(nonmember_source)
        if forget is None or nonmem is None or y_f is None or y_n is None:
            return {}

        forget = np.asarray(forget, dtype=np.float64)
        nonmem = np.asarray(nonmem, dtype=np.float64)
        undefined = {(m, "forget"): UNDEFINED for m in self.metrics}
        if not (np.isfinite(forget).all() and np.isfinite(nonmem).all()):
            return undefined

        # Non-members matched in size to the forget set, drawn with the audit seed so the same
        # candidate audited twice sees the same attack. Size-matching keeps balanced accuracy
        # and the fold stratification meaningful.
        rng = np.random.default_rng(ctx.audit_seed)
        n = min(len(forget), len(nonmem))
        f_idx = rng.choice(len(forget), n, replace=False) if len(forget) > n else np.arange(n)
        n_idx = rng.choice(len(nonmem), n, replace=False) if len(nonmem) > n else np.arange(n)

        x = np.concatenate([
            attack_features(forget[f_idx], np.asarray(y_f)[f_idx], floor=floor),
            attack_features(nonmem[n_idx], np.asarray(y_n)[n_idx], floor=floor),
        ])
        is_member = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])

        scores = cross_validated_scores(x, is_member, n_splits=n_splits, seed=ctx.audit_seed)

        return {
            ("mia_auc_pop", "forget"): roc_auc(scores, is_member),
            ("mia_acc_pop", "forget"): balanced_accuracy(scores, is_member),
            ("mia_tpr_at_fpr_pop", "forget"): tpr_at_fpr(scores, is_member, fpr),
        }

    def notes_for(self, ctx: AuditContext) -> str:
        d = ctx.degeneracy
        if not d.has_per_example_signal:
            return f"{d.summary}; attack has no per-example signal, AUC 0.5 is the true value"
        return d.summary
