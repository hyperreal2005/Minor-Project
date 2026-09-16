"""Audit Layer 5 — reversibility, measured by how fast the forget set can be relearned.

The intuition the other four layers cannot reach: a model that has genuinely lost information has
to *learn* it again, whereas a model that merely hid it snaps back. Reintroduce a fixed subset of
the forget set, take a fixed number of optimiser steps, and watch forget accuracy recover. A
primed model recovers faster than a retrain does.

**This is the project's least-scooped angle.** Relearning appears in the literature as an
occasional diagnostic; ranking a family of methods by it, against a calibrated retrain anchor, is
one of the four openings the review left standing (`RESEARCH_LOG.md`). It is worth getting right.

**The anchors are the self-test.** `relearn_norm` normalises against two arms:

    relearn_norm = (AUC(Mu) − AUC(Mr)) / (AUC(M0) − AUC(Mr))

* **0** — recovers like a genuine retrain. Nothing was left behind.
* **1** — recovers like the model that never forgot. The forgetting was cosmetic.
* **>1** — recovers *faster* than the original. The unlearning left a primed state: the model was
  pushed somewhere that makes the forgotten data unusually easy to pick back up.

If the `original` arm does not normalise to ≈1 and the `oracle` arm to ≈0, **the protocol is
wrong, not the finding** — the plan's Stage 6 gate says exactly this. The `randinit` arm is the
floor: a freshly initialised network relearning the same subset must land well below 0, which is
what proves the reintroduction data alone cannot manufacture recovery.

**The utility guard is not optional.** A recovery curve obtained by wrecking the model is not
evidence of retained structure — a collapsed network relearning 500 examples is just training.
`relearn_utility_drop` records the worst retain/test drop seen *during* relearning, and
`configs/audits.yaml` sets the threshold at which a recovery claim is void.

**Why the trainer lives here but is called by the runner.** Optimiser, learning rate, batch order
and step schedule must be identical across all four arms. One implementation, called four times
with the same seed, is the only way to guarantee that; four call sites that each "use the same
settings" is how protocol drift happens. So :func:`relearn_curve` is the single trainer and the
audit below only interprets what it produced.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .base import UNDEFINED, Audit, AuditContext, register

__all__ = ["Relearning", "relearn_curve", "curve_auc", "normalized_recovery", "steps_to_fraction"]


# --------------------------------------------------------------------------- curve maths


def curve_auc(steps: Sequence[float], values: Sequence[float]) -> float:
    """Area under a recovery curve, normalised by the step range so it reads as a mean accuracy.

    Trapezoidal over **log₁₀(1 + step)**, not raw steps. The schedule in `configs/audits.yaml`
    is [0, 1, 2, 5, 10, 25, 50, 100] — geometric, because recovery is fast early and flat late.
    On a linear axis the single 50→100 gap would carry as much weight as everything before step
    25 combined, and the measure would be dominated by the part of the curve where every arm has
    already converged. The log axis weights the region that actually separates the arms.
    """
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[good], values[good]
    if steps.size < 2:
        return UNDEFINED
    x = np.log10(1.0 + steps)
    span = x[-1] - x[0]
    if span < 1e-12:
        return UNDEFINED
    return float(np.trapezoid(values, x) / span)


def normalized_recovery(auc_method: float, auc_oracle: float, auc_original: float) -> float:
    """``(AUC(Mu) − AUC(Mr)) / (AUC(M0) − AUC(Mr))``.

    Undefined when the two anchors coincide: if a retrain recovers as fast as the model that
    never forgot, the condition cannot distinguish reversibility at all, and any ratio would be
    noise divided by noise. That is a real situation here — Stage 3 measured the `mem-low-3000`
    oracle at 0.998 forget accuracy, which leaves almost no gap between the anchors — so it is
    reported as undefined rather than as a large number.
    """
    if not all(np.isfinite([auc_method, auc_oracle, auc_original])):
        return UNDEFINED
    denom = auc_original - auc_oracle
    if abs(denom) < 1e-6:
        return UNDEFINED
    return float((auc_method - auc_oracle) / denom)


def steps_to_fraction(
    steps: Sequence[float], values: Sequence[float], *, target: float
) -> float:
    """First step at which the curve reaches ``target``, linearly interpolated.

    Returns ``inf`` when the curve never gets there, which is meaningful (the model did not
    recover within the budget) and must not be silently clipped to the last step.
    """
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    hit = np.flatnonzero(values >= target)
    if hit.size == 0:
        return float("inf")
    i = int(hit[0])
    if i == 0:
        return float(steps[0])
    v0, v1 = values[i - 1], values[i]
    if v1 == v0:
        return float(steps[i])
    frac = (target - v0) / (v1 - v0)
    return float(steps[i - 1] + frac * (steps[i] - steps[i - 1]))


# --------------------------------------------------------------------------- the trainer


def relearn_curve(
    model: Any,
    batches: Sequence[tuple],
    evaluate: Callable[[Any], Mapping[str, float]],
    *,
    eval_steps: Sequence[int],
    lr: float = 0.01,
    momentum: float = 0.9,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Relearn ``batches`` for ``max(eval_steps)`` steps, evaluating at each checkpoint.

    ``batches`` is a *materialised* list, not a loader: every arm must see the same examples in
    the same order, and a shuffling loader re-drawn per arm would not guarantee that. The caller
    builds it once and passes the same object to all four arms.

    Returns arrays keyed ``steps`` plus whatever ``evaluate`` reports.
    """
    import torch

    from ..train.loop import set_determinism

    set_determinism(seed)
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    crit = torch.nn.CrossEntropyLoss()

    schedule = sorted(set(int(s) for s in eval_steps))
    rows: dict[str, list[float]] = {"steps": []}

    def record(step: int) -> None:
        was_training = model.training
        model.eval()
        metrics = evaluate(model)
        model.train(was_training)
        rows["steps"].append(float(step))
        for k, v in metrics.items():
            rows.setdefault(k, []).append(float(v))

    step = 0
    if schedule and schedule[0] == 0:
        record(0)  # the pre-relearning point: where unlearning actually left the model

    if batches:
        i = 0
        while step < (schedule[-1] if schedule else 0):
            x, y = batches[i % len(batches)][:2]
            i += 1
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
            step += 1
            if step in schedule:
                record(step)

    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


# --------------------------------------------------------------------------- the audit


@register
class Relearning(Audit):
    """Reversibility: how fast the forget set comes back, against the retrain anchor."""

    name = "relearning"
    metrics = ("relearn_auc", "relearn_norm", "relearn_t80", "relearn_utility_drop")
    needs_oracles = True
    needs_weights = True

    def measure(self, ctx: AuditContext) -> dict[tuple[str, str], float]:
        curves = ctx.relearn_curves
        method = curves.get("method")
        if not method or "forget_acc" not in method:
            return {}

        threshold = float(ctx.config.get("t80_threshold", 0.8))
        out: dict[tuple[str, str], float] = {}

        steps = np.asarray(method["steps"], dtype=np.float64)
        forget = np.asarray(method["forget_acc"], dtype=np.float64)
        auc = curve_auc(steps, forget)
        out[("relearn_auc", "forget")] = auc

        # --- normalised against the two anchors -------------------------------------------
        def anchor_auc(arm: str) -> float:
            c = curves.get(arm)
            if not c or "forget_acc" not in c:
                return UNDEFINED
            return curve_auc(c["steps"], c["forget_acc"])

        out[("relearn_norm", "forget")] = normalized_recovery(
            auc, anchor_auc("oracle"), anchor_auc("original")
        )

        # --- steps to recover 80% of M0's step-0 forget performance ------------------------
        original = curves.get("original")
        if original and "forget_acc" in original and len(original["forget_acc"]):
            target = threshold * float(np.asarray(original["forget_acc"])[0])
            out[("relearn_t80", "forget")] = steps_to_fraction(steps, forget, target=target)
        else:
            out[("relearn_t80", "forget")] = UNDEFINED

        # --- the utility guard ------------------------------------------------------------
        # Worst drop from the curve's own starting point, over both probes: a recovery bought by
        # destroying the model is not evidence of retained structure.
        for probe in ("retain", "test"):
            key = f"{probe}_acc"
            if key not in method:
                continue
            vals = np.asarray(method[key], dtype=np.float64)
            if vals.size and np.isfinite(vals).any():
                out[("relearn_utility_drop", probe)] = float(
                    np.nanmax(vals[0] - vals) * 100.0  # percentage points
                )
        return out

    def notes_for(self, ctx: AuditContext) -> str:
        bits = []
        if ctx.degeneracy.summary:
            bits.append(ctx.degeneracy.summary)
        missing = [a for a in ("oracle", "original") if a not in ctx.relearn_curves]
        if missing:
            bits.append(f"anchors missing: {', '.join(missing)}; relearn_norm undefined")
        drop = ctx.config.get("max_utility_drop_pp")
        m = ctx.relearn_curves.get("method") or {}
        if drop is not None and "test_acc" in m:
            v = np.asarray(m["test_acc"], dtype=np.float64)
            if v.size and float(np.nanmax(v[0] - v)) * 100.0 > float(drop):
                bits.append(f"utility fell more than {drop} pp during relearning; recovery void")
        return "; ".join(bits)
