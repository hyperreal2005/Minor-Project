"""Shared prediction and metrics.

Used by training, by the behavioural audit, and by the relearning audit. Deliberately one
implementation: if training reported accuracy one way and the audit computed it another, every
oracle-referenced metric would inherit the discrepancy and nobody would notice until the numbers
stopped adding up.

:func:`predict` returns logits, labels and original example indices together. The indices travel
with the predictions so that downstream code can align rows to examples without trusting loader
order — and so that a probe set can be re-ordered safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Predictions", "predict", "classification_metrics", "js_divergence",
           "prediction_agreement", "logit_distance"]


@dataclass(frozen=True, slots=True)
class Predictions:
    """Model outputs over a probe set, with the example indices that produced them."""

    logits: np.ndarray  # (n, n_classes) float32
    labels: np.ndarray  # (n,) int64
    indices: np.ndarray  # (n,) int64 - original dataset indices

    def __post_init__(self) -> None:
        n = len(self.labels)
        if self.logits.shape[0] != n or len(self.indices) != n:
            raise ValueError("logits, labels and indices disagree in length")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def probs(self) -> np.ndarray:
        """Softmax probabilities, computed stably."""
        z = self.logits - self.logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @property
    def predicted(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    def reorder_like(self, other: "Predictions") -> "Predictions":
        """Reorder to match another set's example order.

        Two models' outputs are only comparable row-for-row if they refer to the same examples
        in the same order. Rather than assume that, align explicitly.
        """
        if set(self.indices) != set(other.indices):
            raise ValueError("cannot align: the two prediction sets cover different examples")
        order = np.argsort(self.indices)[np.argsort(np.argsort(other.indices))]
        return Predictions(self.logits[order], self.labels[order], self.indices[order])


@torch.no_grad()
def predict(model, loader, *, device: str | torch.device = "cpu") -> Predictions:
    """Run a loader through a model and collect logits, labels and indices."""
    model = model.to(device).eval()
    logits, labels, indices = [], [], []
    for batch in loader:
        x, y, idx = batch
        out = model(x.to(device, non_blocking=True))
        logits.append(out.detach().float().cpu())
        labels.append(y)
        indices.append(idx)
    if not logits:
        raise ValueError("loader yielded no batches")
    return Predictions(
        logits=torch.cat(logits).numpy(),
        labels=torch.cat(labels).numpy().astype(np.int64),
        indices=torch.cat(indices).numpy().astype(np.int64),
    )


def classification_metrics(p: Predictions, *, n_classes: int = 10) -> dict[str, float]:
    """Accuracy, macro-F1 and mean cross-entropy over a probe set.

    Macro-F1 is computed over the classes actually present in ``labels``. Averaging over absent
    classes would silently drag the score toward zero on a probe set that does not span all ten
    — which forget sets frequently do not.
    """
    if len(p) == 0:
        raise ValueError("cannot compute metrics over an empty probe set")

    pred, y = p.predicted, p.labels
    acc = float((pred == y).mean())

    probs = p.probs
    eps = np.finfo(probs.dtype).tiny
    ce = float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, None)).mean())

    f1s = []
    for c in np.unique(y):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)

    return {
        "acc": acc,
        "macro_f1": float(np.mean(f1s)),
        "ce_loss": ce,
    }


# ---------------------------------------------------------------- oracle-referenced comparisons


def js_divergence(a: Predictions, b: Predictions, *, floor: float = 1e-12) -> float:
    """Mean Jensen-Shannon divergence between two models' output distributions.

    Symmetric and bounded in [0, ln 2] with natural log, so unlike KL it cannot blow up when one
    model assigns a class zero probability. The floor guards the logs regardless.
    """
    b = b.reorder_like(a)
    p, q = np.clip(a.probs, floor, None), np.clip(b.probs, floor, None)
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=1)
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=1)
    return float((0.5 * (kl_pm + kl_qm)).mean())


def prediction_agreement(a: Predictions, b: Predictions) -> float:
    """Fraction of examples where two models predict the same class."""
    b = b.reorder_like(a)
    return float((a.predicted == b.predicted).mean())


def logit_distance(a: Predictions, b: Predictions) -> float:
    """Mean L2 distance between logit vectors."""
    b = b.reorder_like(a)
    return float(np.linalg.norm(a.logits - b.logits, axis=1).mean())
