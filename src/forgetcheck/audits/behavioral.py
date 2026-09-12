"""Audit Layer 1 — behavioural comparison against the retrained-oracle ensemble.

Answers: *does this model behave like one that never saw the forget set?* Not "has forget
accuracy dropped" — a retrained oracle still classifies most forgotten examples correctly,
because it learned the concept from the examples it kept. Stage 3 measured exactly that: the
`mem-high-3000` oracle scores 0.5575 on its own forget set, and the `mem-low-3000` oracle scores
0.998. Distance from the oracle is the question; distance from zero is not.

**What this audit deliberately does not emit.** Stage 5 already records `retain_acc`, `test_acc`,
`forget_acc`, `forget_loss`, `ce_loss` and `macro_f1` under ``audit="meta"``, computed from the
same forward passes. Recomputing them here under ``audit="behavior"`` would put identical numbers
in the table twice under different labels, and the instance-level agreement analysis groups on
``(metric, audit)`` — double-counting them would inflate the behavioural family's weight against
the other five for no added information. This audit therefore emits only the four quantities that
*need* a reference model and so could not be computed at unlearning time.

**The oracle-vs-oracle baseline is not special-cased.** The gate for this module is that
oracle-vs-oracle JS divergence is near zero while M0-vs-oracle is measurably larger. That falls
out of the ordinary path: Stage 7 passes the three held-out oracles (seeds 209–211) through every
audit as candidates, so `js_to_oracle` computed on a held-out oracle *is* the oracle-vs-oracle
baseline. An audit that needed to know whether its input was "really" an oracle would be an audit
that could cheat.
"""

from __future__ import annotations

import numpy as np

from .base import UNDEFINED, Audit, AuditContext, register, softmax

__all__ = ["Behavioral", "js_divergence"]


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row Jensen–Shannon divergence between two probability matrices, in nats.

    Bounded above by ln 2, which is what makes it readable next to a cross-entropy that gradient
    ascent can push past 80 — an unbounded divergence would make the collapsed control's row
    dominate every average it appears in.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> np.ndarray:
        # 0 log 0 = 0. Masking beats adding an epsilon: an epsilon shifts every value slightly,
        # and this quantity is compared against an oracle band a few thousandths wide.
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(a > 0, a * (np.log(a) - np.log(m)), 0.0)
        return terms.sum(axis=-1)

    return 0.5 * _kl(p) + 0.5 * _kl(q)


@register
class Behavioral(Audit):
    """Output-space distance from the oracle ensemble, and direction of travel from M0."""

    name = "behavior"
    metrics = ("js_to_oracle", "js_to_original", "pred_agreement", "logit_l2")
    needs_oracles = True

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        floor = float(ctx.config.get("prob_floor", 1e-12))
        probe_sets = tuple(ctx.config.get("probe_sets", ("forget", "retain", "test")))
        out: dict[tuple[str, str], float] = {}

        for probe in probe_sets:
            target = ctx.logits.get(probe)
            if target is None:
                continue

            target = np.asarray(target, dtype=np.float64)
            # A single non-finite row would poison every mean below. Gradient ascent produced
            # cross-entropy of 82 in Stage 5; overflow to inf is a live possibility, not a
            # hypothetical, so it is checked rather than assumed away.
            if not np.isfinite(target).all():
                if ctx.has_oracles(probe):
                    for metric in ("js_to_oracle", "pred_agreement", "logit_l2"):
                        out[(metric, probe)] = UNDEFINED
                if ctx.original_logits.get(probe) is not None:
                    out[("js_to_original", probe)] = UNDEFINED
                continue

            p_target = softmax(target, floor=floor)
            preds = target.argmax(axis=1)

            # --- against the oracle ensemble ---------------------------------------------
            if ctx.has_oracles(probe):
                js, agree, l2 = [], [], []
                for oracle in np.asarray(ctx.oracle_logits[probe], dtype=np.float64):
                    if not np.isfinite(oracle).all():
                        continue
                    js.append(float(js_divergence(p_target, softmax(oracle, floor=floor)).mean()))
                    agree.append(float((preds == oracle.argmax(axis=1)).mean()))
                    l2.append(float(np.linalg.norm(target - oracle, axis=1).mean()))

                # Averaging the per-oracle distances, rather than measuring distance to the
                # ensemble's mean output, is deliberate: the mean of several softmax vectors is
                # not itself a model's output, and the spread across oracles is the reference
                # distribution the whole calibration rests on (review finding B4).
                if js:
                    out[("js_to_oracle", probe)] = float(np.mean(js))
                    out[("pred_agreement", probe)] = float(np.mean(agree))
                    out[("logit_l2", probe)] = float(np.mean(l2))
                else:
                    out[("js_to_oracle", probe)] = UNDEFINED
                    out[("pred_agreement", probe)] = UNDEFINED
                    out[("logit_l2", probe)] = UNDEFINED

            # --- against the original model ----------------------------------------------
            original = ctx.original_logits.get(probe)
            if original is not None:
                original = np.asarray(original, dtype=np.float64)
                if np.isfinite(original).all():
                    out[("js_to_original", probe)] = float(
                        js_divergence(p_target, softmax(original, floor=floor)).mean()
                    )
                else:
                    out[("js_to_original", probe)] = UNDEFINED

        return out

    def notes_for(self, ctx: AuditContext) -> str:
        return ctx.degeneracy.summary
