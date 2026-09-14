"""Audit Layer 3 — RMIA, the per-example membership attack (family: privacy_strong).

Zarifzadeh, Liu and Shokri (2024). Where the population attack (Layer 2) learns one decision rule
for every example, RMIA asks a separate question per example: *is this example more surprising to
the target model than a random population example is?* An example that is easy for everyone is
not evidence of membership; an example the target is unusually confident about, relative to how
confident models in general are about it, is.

Its score for a target example ``x`` under model ``θ`` is

    score(x) = Pr_{z ~ population} [ ratio(x) / ratio(z) >= gamma ]
    ratio(u) = p_θ(u) / p̃(u)

where ``p_θ(u)`` is the model's probability of ``u``'s true label and ``p̃(u)`` is how probable
that label is *in general*, estimated from the reference models. Dividing by ``ratio(z)`` is what
makes the test per-example: it cancels whatever the target model happens to be globally, which is
precisely what a single population threshold cannot do.

**This layer exists to disagree with Layer 2.** Population attacks are known to overstate privacy
protection [21] — they average over examples and so miss a model that leaks a handful of records
badly while looking safe in aggregate. If the two layers agreed everywhere, one of them would be
redundant; the project's within-family result is where and how they part.

**The offline estimator, and why.** ``p̃(u)`` is ideally ``½(E_IN[p] + E_OUT[p])`` over reference
models trained with and without ``u``. Our shadows each see a random half of the *training* set,
so a forget-set example has roughly 16 IN and 16 OUT models — but a test-set example is OUT for
all 32, and there is no IN average to be had. Using the online estimator for members and the
offline one for non-members would build the member/non-member distinction into the estimator
itself, which is exactly the bias the attack is supposed to measure. So the **offline** form is
used for everything:

    p̃(u) = ½ ((1 + a) * E_OUT[p_ref(u)] + (1 - a))

with ``a`` interpolating between "trust the OUT average" (a = 1) and "assume an uninformative
prior" (a = 0). This is the paper's own offline variant, and it is the symmetric choice.

**Degenerate models.** A constant predictor gives every example the same ``p_θ``, so every
``ratio`` differs only through ``p̃`` — which is a property of the *example*, identical for
members and non-members drawn from the same distribution. The attack lands at chance, which is
the true answer and the audit-validity finding: the destroyed model leaks nothing because it
retains nothing. Recorded, not skipped, with the reason in ``notes``.
"""

from __future__ import annotations

import numpy as np

from .base import UNDEFINED, Audit, AuditContext, register, softmax
from .privacy_population import balanced_accuracy, roc_auc, tpr_at_fpr

__all__ = ["RMIA", "true_label_prob", "offline_prior", "rmia_scores"]


def true_label_prob(logits: np.ndarray, labels: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """``p(true label)`` per example. Accepts ``(n, k)`` or ``(m, n, k)``."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    p = softmax(logits, floor=floor)
    if p.ndim == 2:
        return p[np.arange(p.shape[0]), labels]
    return p[:, np.arange(p.shape[1]), labels]  # (m, n)


def offline_prior(
    ref_probs: np.ndarray, in_mask: np.ndarray | None, *, a: float
) -> np.ndarray:
    """``p̃(u)`` from the reference models that did **not** train on ``u``.

    Args:
        ref_probs: ``(n_refs, n)`` true-label probabilities from each reference model.
        in_mask: ``(n_refs, n)`` bool, True where that reference trained on that example.
            ``None`` means no reference trained on any of them (the test set), so all are OUT.
        a: offline interpolation, 0 = uninformative prior, 1 = the OUT average alone.
    """
    ref_probs = np.asarray(ref_probs, dtype=np.float64)
    out = np.ones(ref_probs.shape, dtype=bool) if in_mask is None else ~np.asarray(in_mask, bool)

    counts = out.sum(axis=0)
    totals = np.where(out, ref_probs, 0.0).sum(axis=0)
    # An example that every reference happened to train on has no OUT estimate. Fall back to the
    # uninformative prior for it rather than dividing by zero; with 32 shadows at half coverage
    # this is ~1 example in 4 billion, but a silent inf here would poison the whole ROC curve.
    mean_out = np.where(counts > 0, totals / np.maximum(counts, 1), 0.5)
    return 0.5 * ((1.0 + a) * mean_out + (1.0 - a))


def rmia_scores(
    target: np.ndarray, prior: np.ndarray, z_target: np.ndarray, z_prior: np.ndarray, *, gamma: float
) -> np.ndarray:
    """``Pr_z[ ratio(x)/ratio(z) >= gamma ]`` for each x, vectorised over the population.

    Args:
        target: ``(n,)`` p_θ for the examples being scored.
        prior: ``(n,)`` p̃ for those examples.
        z_target: ``(m,)`` p_θ for the population examples.
        z_prior: ``(m,)`` p̃ for the population.
    """
    ratio_x = np.asarray(target, dtype=np.float64) / np.maximum(prior, 1e-30)
    ratio_z = np.asarray(z_target, dtype=np.float64) / np.maximum(z_prior, 1e-30)
    if ratio_z.size == 0:
        return np.full(ratio_x.shape, UNDEFINED)
    # (n, m) comparison. n and m are a few thousand at most, so the dense form is fine and far
    # clearer than a loop.
    return (ratio_x[:, None] >= gamma * ratio_z[None, :]).mean(axis=1)


@register
class RMIA(Audit):
    """Per-example membership inference against the reference-model ensemble."""

    name = "privacy_rmia"
    metrics = ("mia_auc_rmia", "mia_acc_rmia", "mia_tpr_at_fpr_rmia")
    needs_oracles = False
    needs_references = True

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        cfg = ctx.config
        floor = float(cfg.get("prob_floor", 1e-12))
        gamma = float(cfg.get("gamma", 2.0))
        fpr = float(cfg.get("tpr_at_fpr", 0.01))
        a = float(cfg.get("offline_a", 0.3))
        n_refs = cfg.get("n_references")

        undefined = {(m, "forget"): UNDEFINED for m in self.metrics}

        f_log, t_log = ctx.logits.get("forget"), ctx.logits.get("test")
        y_f, y_t = ctx.labels.get("forget"), ctx.labels.get("test")
        if f_log is None or t_log is None or y_f is None or y_t is None:
            return {}
        if not (ctx.has_references("forget") and ctx.has_references("test")):
            return undefined
        f_log, t_log = np.asarray(f_log, float), np.asarray(t_log, float)
        if not (np.isfinite(f_log).all() and np.isfinite(t_log).all()):
            return undefined

        ref_f = np.asarray(ctx.reference_logits["forget"], dtype=np.float64)
        ref_t = np.asarray(ctx.reference_logits["test"], dtype=np.float64)
        if n_refs is not None:  # the halving stability check runs the same code with 16
            ref_f, ref_t = ref_f[: int(n_refs)], ref_t[: int(n_refs)]
        if len(ref_f) == 0:
            return undefined

        in_f = ctx.reference_in_mask.get("forget")
        if in_f is not None:
            in_f = np.asarray(in_f, bool)[: len(ref_f)]

        p_f = true_label_prob(f_log, y_f, floor=floor)
        p_t = true_label_prob(t_log, y_t, floor=floor)
        prior_f = offline_prior(true_label_prob(ref_f, y_f, floor=floor), in_f, a=a)
        # Test examples are outside every shadow's training pool by construction, so no mask.
        prior_t = offline_prior(true_label_prob(ref_t, y_t, floor=floor), None, a=a)

        # Three disjoint roles: members, non-members, and the population z the ratio is compared
        # against. Reusing one slice for two roles would let an example be its own reference.
        rng = np.random.default_rng(ctx.audit_seed)
        n_mem = len(p_f)
        order = rng.permutation(len(p_t))
        n_non = min(n_mem, len(order) // 2)
        if n_non == 0:
            return undefined
        non_idx, pop_idx = order[:n_non], order[n_non:]

        members = rmia_scores(p_f, prior_f, p_t[pop_idx], prior_t[pop_idx], gamma=gamma)
        nonmembers = rmia_scores(
            p_t[non_idx], prior_t[non_idx], p_t[pop_idx], prior_t[pop_idx], gamma=gamma
        )

        scores = np.concatenate([members, nonmembers])
        is_member = np.concatenate([np.ones(len(members), bool), np.zeros(len(nonmembers), bool)])

        return {
            ("mia_auc_rmia", "forget"): roc_auc(scores, is_member),
            # RMIA scores are fractions in [0, 1] and cluster low; the median is the
            # distribution-free split point, where a fixed 0.5 would call almost everything a
            # non-member and report a "balanced accuracy" of 0.5 for every model alike.
            ("mia_acc_rmia", "forget"): balanced_accuracy(
                scores, is_member, threshold=float(np.median(scores))
            ),
            ("mia_tpr_at_fpr_rmia", "forget"): tpr_at_fpr(scores, is_member, fpr),
        }

    def notes_for(self, ctx: AuditContext) -> str:
        d = ctx.degeneracy
        n = len(ctx.reference_logits.get("forget", ()))
        base = f"{n} reference models"
        if not d.has_per_example_signal:
            return f"{d.summary}; per-example attack has no signal, chance AUC is the true value"
        return f"{d.summary}; {base}" if d.summary else base
