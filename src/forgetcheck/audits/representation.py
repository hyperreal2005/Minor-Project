"""Audit Layer 4 — representation similarity to the retrained-oracle ensemble.

Two models can agree on every prediction and still have arrived there differently. This layer
asks whether the unlearned model's internal representations look like a retrain's, per layer, on
GAP-pooled activations over a fixed mixed probe set.

**Read the caveat before reading the numbers.** Davari et al. show CKA values move without any
corresponding change in functional behaviour — a layer can register 0.6 against a model it is
behaviourally indistinguishable from. Three consequences are built into this module rather than
left to the reader:

1. **Nothing is interpretable without the oracle-vs-oracle baseline.** Two independent retrains
   of the *same* data do not score 1.0. Whatever they do score is the ceiling, and a candidate's
   distance is only meaningful against it. That baseline comes free: Stage 7 passes the held-out
   oracles through this audit as candidates, so their `cka_linear` against the ensemble *is* the
   oracle-vs-oracle value.
2. **Two measures, and where they disagree the disagreement is the result.** `cka_linear` is the
   declared primary and `cka_rbf` the second; a representation claim is made only where the two
   agree in rank. This is `configs/audits.yaml`'s open decision 2, settled by measurement.
3. **A representation claim never stands alone.** It is reported alongside Layer 1, and the
   project's own thesis is that cross-family disagreement is the finding — so a CKA result that
   contradicts behaviour is to be reported, not reconciled.

**GAP pooling is a stated limitation.** Raw layer1 output for 3000 probes is ~786 MB per model,
untenable across ~300 models, so activations are global-average-pooled over spatial dimensions
before storage. This discards spatial structure — an accepted convention for CKA on convnets, and
one the write-up must name rather than assume.

**Degenerate models.** A collapsed network produces near-constant activations. Centred CKA
divides by the Frobenius norms of the centred Gram matrices, and a constant feature matrix
centres to exactly zero — so CKA is genuinely undefined there, not merely small. It is recorded
as such, with the reason, rather than as a 0.0 that would read as "maximally dissimilar" and be
averaged into a family mean.
"""

from __future__ import annotations

import numpy as np

from .base import UNDEFINED, Audit, AuditContext, register

__all__ = ["Representation", "linear_cka", "rbf_cka", "gram_linear", "gram_rbf", "center_gram"]

#: Below this, a centred Gram matrix is numerically zero and CKA has no denominator.
_ZERO = 1e-10


def center_gram(k: np.ndarray) -> np.ndarray:
    """Double-centre a Gram matrix: ``HKH`` with ``H = I - 11ᵀ/n``."""
    n = k.shape[0]
    means = k.mean(axis=0, keepdims=True)
    return k - means - means.T + k.mean()


def gram_linear(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(x, dtype=np.float64).T


def gram_rbf(x: np.ndarray, *, sigma: str | float = "median") -> np.ndarray:
    """RBF Gram matrix. ``sigma='median'`` uses the median pairwise distance — the standard
    data-dependent bandwidth, and the one ``configs/audits.yaml`` declares."""
    x = np.asarray(x, dtype=np.float64)
    sq = (x**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (x @ x.T), 0.0)
    if sigma == "median":
        off = d2[~np.eye(len(x), dtype=bool)]
        med = np.median(off) if off.size else 0.0
        # All-identical rows give a median distance of zero, which would divide by zero. The
        # resulting Gram matrix is all-ones, which centres to zero, which CKA reports as
        # undefined -- the correct outcome, reached without a warning.
        s2 = med if med > _ZERO else 1.0
    else:
        s2 = 2.0 * float(sigma) ** 2
    return np.exp(-d2 / s2)


def _cka_from_grams(ka: np.ndarray, kb: np.ndarray) -> float:
    a, b = center_gram(ka), center_gram(kb)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _ZERO or nb < _ZERO:
        return UNDEFINED  # a constant representation has no direction to align with
    return float((a * b).sum() / (na * nb))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centred Kernel Alignment with a linear kernel. 1.0 = identical up to rotation and scale."""
    return _cka_from_grams(gram_linear(x), gram_linear(y))


def rbf_cka(x: np.ndarray, y: np.ndarray, *, sigma: str | float = "median") -> float:
    return _cka_from_grams(gram_rbf(x, sigma=sigma), gram_rbf(y, sigma=sigma))


@register
class Representation(Audit):
    """Per-layer CKA against the oracle ensemble, plus raw activation distance."""

    name = "representation"
    metrics = ("cka_linear", "cka_rbf", "activation_l2")
    needs_oracles = True

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        layers = tuple(ctx.config.get("layers", ("layer1", "layer2", "layer3", "layer4")))
        sigma = ctx.config.get("rbf_sigma", "median")
        out: dict[tuple[str, str], float] = {}

        for layer in layers:
            target = ctx.activations.get(layer)
            oracles = ctx.oracle_activations.get(layer)
            if target is None or oracles is None or len(oracles) == 0:
                continue

            target = np.asarray(target, dtype=np.float64)
            if not np.isfinite(target).all():
                for m in self.metrics:
                    out[(m, layer)] = UNDEFINED
                continue

            lin, rbf, l2 = [], [], []
            for oracle in np.asarray(oracles, dtype=np.float64):
                if not np.isfinite(oracle).all():
                    continue
                lin.append(linear_cka(target, oracle))
                rbf.append(rbf_cka(target, oracle, sigma=sigma))
                l2.append(float(np.linalg.norm(target - oracle, axis=1).mean()))

            # Averaged per-oracle, like Layer 1: the mean of several activation matrices is not
            # itself a model's representation, and the spread across oracles is the reference
            # distribution the calibration rests on.
            out[("cka_linear", layer)] = _mean_defined(lin)
            out[("cka_rbf", layer)] = _mean_defined(rbf)
            out[("activation_l2", layer)] = _mean_defined(l2)

        return out

    def notes_for(self, ctx: AuditContext) -> str:
        d = ctx.degeneracy
        if d.is_constant:
            return f"{d.summary}; near-constant activations make centred CKA undefined"
        return d.summary


def _mean_defined(values: list[float]) -> float:
    """Mean over the oracles that produced a defined value; UNDEFINED if none did.

    Not ``np.mean``: one undefined member must not turn the whole layer into NaN, and equally an
    all-undefined layer must not quietly become 0.0.
    """
    good = [v for v in values if np.isfinite(v)]
    return float(np.mean(good)) if good else UNDEFINED
