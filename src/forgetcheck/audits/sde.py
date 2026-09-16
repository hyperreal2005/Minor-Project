"""Audit Layer 6 — SDE, split-half statistical independence (family: independence).

Zarifzadeh-style reference models are not needed here, and neither is a retrained oracle. SDE
(Wang et al., *Unlearning Evaluation through Subset Statistical Independence*, ICLR 2026 =
arXiv 2603.00587) asks a different question from every other layer: **do the model's outputs on a
subset still show the statistical dependence that co-training induces?**

Split a subset in half, compute the Hilbert–Schmidt Independence Criterion between the model's
outputs on the two halves, repeat over many random splits to get a distribution, and compare that
distribution against two references — a subset known to be in training, and one known to be out.
Whichever it resembles is the verdict. No retraining, no attack, no auxiliary classifier.

**Why it is in this project at all.** Its central claim is aimed at ForgetCheck's foundation:
retrained-oracle evaluation "defeats the purpose of developing a standalone, verifiably unlearned
model". That critique lands on *deployment-time verification* and not on *research-time audit
validation*, which needs a ground truth by definition — and SDE cannot escape that either, since
it validates itself against subsets whose membership is known by construction. See
`RESEARCH_LOG.md` §7.1. Including it does two things: it makes the audit set current with ICLR
2026, and it lets ForgetCheck do to SDE what SDE cannot do for itself — check it against a
retrained oracle, across memorization strata, on a method set that includes a known-destroyed
control.

**The prediction worth testing, stated as a prediction.** A constant predictor has no per-example
output variation, so its split-half HSIC should be degenerate. Whether SDE then reads the forget
set as *out* of training (a false pass on a destroyed model) or as *in* (a false alarm) is not
obvious from the paper and is left for the measurement to settle — `sde_margin` is signed so the
answer is legible either way. Do not assume the sign in advance.

**Deviation: the kernel bandwidth is the median heuristic, not the paper's ``sqrt(dim)``.**
Measured on CIFAR-10 softmax outputs, the pairwise squared distances have a median of 0.15 while
``sqrt(10)`` sets the kernel width to 20 — so ``exp(-d²/2σ²) ≈ 1`` for every pair, the Gram matrix
centres to numerical dust, and the resulting HSIC values are ~1e-7. They still *ordered* correctly
in testing, but only by comparing float noise, which is not a measurement anyone should build a
claim on. With the median bandwidth the same subsets give HSIC ~1e-2, a usable dynamic range.

This is not a disagreement with the paper: ``sqrt(dim)`` is sensible for the raw feature vectors
it was proposed for, and the paper itself names bandwidth as a known limitation — "the heuristic
σ=√dim ... fails to achieve the best results" outside its tested setting. Model **outputs** live
on a 10-simplex where no two points can be further apart than √2, and a bandwidth of 20 cannot
resolve anything there. Set ``sigma`` to a number in `configs/audits.yaml` to override.

**Its other named sensitivity** — "the choice of reference sets affects the performance" — is a
declared protocol setting here, identical across every model compared.
"""

from __future__ import annotations

import numpy as np

from .base import UNDEFINED, Audit, AuditContext, register, softmax
from .behavioral import js_divergence
from .representation import center_gram, gram_rbf

__all__ = ["SDE", "hsic", "split_half_distribution", "distribution_jsd"]

_ZERO = 1e-12


def hsic(x: np.ndarray, y: np.ndarray, *, sigma: str | float = "median") -> float:
    """Biased HSIC estimator, ``tr(KHLH)/(n-1)²``, with Gaussian RBF kernels.

    Zero means the two halves' outputs are independent. Exactly zero — not merely small — when
    either side is constant, because a constant kernel matrix double-centres to zero. That is a
    real and informative outcome for a collapsed model, so it is returned as 0.0 rather than
    being special-cased into UNDEFINED.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2 or len(y) != n:
        return UNDEFINED
    kc, lc = center_gram(gram_rbf(x, sigma=sigma)), center_gram(gram_rbf(y, sigma=sigma))
    return float((kc * lc).sum() / (n - 1) ** 2)


def split_half_distribution(
    features: np.ndarray, *, n_draws: int, sigma: str | float = "median",
    rng: np.random.Generator
) -> np.ndarray:
    """``n_draws`` split-half HSIC values, each from an independent random halving.

    Re-drawing the split (rather than permuting one fixed half) is what makes this a distribution
    over the subset rather than over one arbitrary partition of it.
    """
    features = np.asarray(features, dtype=np.float64)
    n = len(features)
    if n < 4:
        return np.array([], dtype=np.float64)
    half = n // 2
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        perm = rng.permutation(n)
        out[i] = hsic(features[perm[:half]], features[perm[half : 2 * half]], sigma=sigma)
    return out


def distribution_jsd(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    """Jensen–Shannon divergence between two samples, via a shared histogram.

    The shared support matters: histogramming each sample on its own edges would compare
    differently-scaled quantities and report a divergence for two identical distributions.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return UNDEFINED
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi - lo < _ZERO:
        return 0.0  # both degenerate at the same point: identical, not incomparable
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, bins=edges)[0].astype(np.float64)
    pb = np.histogram(b, bins=edges)[0].astype(np.float64)
    pa /= pa.sum()
    pb /= pb.sum()
    return float(js_divergence(pa[None, :], pb[None, :])[0])


@register
class SDE(Audit):
    """Split-half dependence against in-training and out-of-training reference subsets."""

    name = "sde"
    metrics = ("sde_hsic", "sde_margin", "sde_verdict")
    needs_oracles = False

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        cfg = ctx.config
        n_draws = int(cfg.get("n_draws", 200))
        bins = int(cfg.get("jsd_bins", 32))
        in_source = str(cfg.get("in_training_source", "retain"))
        out_source = str(cfg.get("out_of_training_source", "test"))

        target = ctx.logits.get("forget")
        ref_in = ctx.logits.get(in_source)
        ref_out = ctx.logits.get(out_source)
        undefined = {(m, "forget"): UNDEFINED for m in self.metrics}
        if target is None or ref_in is None or ref_out is None:
            return {}

        target = np.asarray(target, dtype=np.float64)
        if not np.isfinite(target).all():
            return undefined

        # Softmax outputs, per the paper's "model outputs on a given subset". Probabilities
        # rather than raw logits: gradient ascent leaves logits on wildly different scales
        # (Stage 5 saw |logit| in the tens), and an RBF kernel on unnormalised scales would
        # measure the scale, not the dependence.
        rng = np.random.default_rng(ctx.audit_seed)
        p_t = softmax(target)
        sigma = cfg.get("sigma", "median")

        # Reference subsets matched in size to the target, so the split-half sample size -- which
        # HSIC's estimator depends on -- is the same for all three.
        n = len(p_t)
        def _match(arr):
            arr = softmax(np.asarray(arr, dtype=np.float64))
            if len(arr) <= n:
                return arr
            return arr[rng.choice(len(arr), n, replace=False)]

        d_t = split_half_distribution(p_t, n_draws=n_draws, sigma=sigma, rng=rng)
        d_in = split_half_distribution(_match(ref_in), n_draws=n_draws, sigma=sigma, rng=rng)
        d_out = split_half_distribution(_match(ref_out), n_draws=n_draws, sigma=sigma, rng=rng)
        if d_t.size == 0 or d_in.size == 0 or d_out.size == 0:
            return undefined

        jsd_in = distribution_jsd(d_t, d_in, bins=bins)
        jsd_out = distribution_jsd(d_t, d_out, bins=bins)
        margin = jsd_in - jsd_out  # > 0: closer to out-of-training, i.e. SDE says "unlearned"

        return {
            ("sde_hsic", "forget"): float(np.mean(d_t)),
            ("sde_margin", "forget"): float(margin) if np.isfinite(margin) else UNDEFINED,
            ("sde_verdict", "forget"): float(margin > 0) if np.isfinite(margin) else UNDEFINED,
        }

    def notes_for(self, ctx: AuditContext) -> str:
        d = ctx.degeneracy
        if not d.has_per_example_signal:
            return (
                f"{d.summary}; split-half dependence is degenerate -- record the verdict, it is "
                "the audit-validity test, not a failure to measure"
            )
        return d.summary
