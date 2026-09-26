"""Canary ground truth: residual influence measured where it is known by construction.

The canary condition mislabels 500 training examples deterministically. The original model
trained on the wrong labels; the retrained oracle never saw them. So for this one condition,
whether a model still carries the association is *measurable directly*, independently of every
audit -- the ground truth the six audits are scored against.

**Which statistic, and why not the registered ones alone.** The registry pre-declares
`canary_acc` (argmax equals the assigned wrong label) and `canary_prob` (mean probability of it).
Both have a flaw that the collapsed `neggrad` control exposes: a model that knows *nothing*
predicts the assigned label about one time in ten by chance, while the oracle -- which confidently
predicts the *true* label -- almost never does. So on `canary_prob` a destroyed model sits well
above the oracle and reads as "still remembers", which is the wrong answer for the wrong reason.

`canary_top_wrong` removes the confound: **among the nine wrong labels only**, is the assigned
one the model's favourite? The true label is excluded, so confidence in it no longer matters. A
model with no memory of the canary picks the assigned label at the natural-confusion rate of the
oracle (≈ 1/9, since the assigned label is spread over wrong classes by example index); a model
that remembers picks it far more often; a destroyed model sits at chance, i.e. at or below the
oracle. Ground truth for an unlearned model is `canary_top_wrong` above the oracle band.

All three are recorded. The registered two are reported as declared, with the flaw stated.
"""

from __future__ import annotations

import numpy as np

__all__ = ["canary_scores"]


def canary_scores(logits: np.ndarray, y_true: np.ndarray, y_canary: np.ndarray) -> dict[str, float]:
    """``{canary_acc, canary_prob, canary_top_wrong}`` for one model's forget-set logits.

    Rows of ``logits`` must be in the order of ``y_true`` / ``y_canary``: the forget probe is
    sorted by example index in every model, and the labels are indexed the same way.
    """
    x = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_can = np.asarray(y_canary, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y_true) or len(x) != len(y_can):
        raise ValueError(f"shape mismatch: logits {x.shape}, labels {y_true.shape}/{y_can.shape}")
    if not np.isfinite(x).all():
        return {"canary_acc": float("nan"), "canary_prob": float("nan"),
                "canary_top_wrong": float("nan")}

    rows = np.arange(len(x))
    z = x - x.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)

    wrong = x.copy()
    wrong[rows, y_true] = -np.inf  # exclude the true label; rank only among the wrong ones
    return {
        "canary_acc": float((x.argmax(axis=1) == y_can).mean()),
        "canary_prob": float(p[rows, y_can].mean()),
        "canary_top_wrong": float((wrong.argmax(axis=1) == y_can).mean()),
    }
