"""The six core unlearning methods, plus SSD for the class-unlearning side condition.

All six operate at instance level, which is the setting this project studies (Untraining, in the
sense of [19]). Mechanistic diversity is the point: if all six worked the same way, agreement
between audits would say nothing about audits and everything about the methods being identical.

============  ==================================================================
Fine-tune     keep training on the retain set only
NegGrad       gradient *ascent* on the forget set alone -- destructive control
NegGrad+      ascent on forget, descent on retain, balanced by alpha
SCRUB         teacher/student: match the teacher on retain, diverge on forget
SalUn         gradient-saliency mask, then random-label fine-tuning through it
L1-sparse     retain fine-tuning under an L1 penalty
SSD           Fisher-importance dampening -- class unlearning only, see below
============  ==================================================================

**Why plain NegGrad is here at all.** It is a labelled *control*, not a competitor. It is
expected to wreck the model while driving forget accuracy to zero, which makes it the cleanest
demonstration that "looks forgotten" can mean "damaged" — the failure mode Audit Layer 1 exists
to detect. A method set without such a case would let a reader assume every low forget accuracy
is a success.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train.loop import set_determinism
from .base import Unlearner, UnlearnContext, register

__all__ = [
    "FineTune",
    "NegGrad",
    "NegGradPlus",
    "SCRUB",
    "SalUn",
    "L1Sparse",
    "SSD",
]


def _sgd(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 5e-4),
    )


def _cycle(loader):
    """Endlessly repeat a loader, for pairing two loaders of different length."""
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- 1. fine-tune


@register
class FineTune(Unlearner):
    """Continue training on the retain set only.

    The simplest practical baseline, and a meaningful one: it tests whether exposure to the
    retained data alone is enough to move the model toward what retraining would have produced.
    It never sees the forget set, so anything it removes is removed by catastrophic forgetting
    rather than by targeting.
    """

    name = "finetune"
    defaults = {"epochs": 5, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        m = self.clone(model).to(ctx.device).train()
        opt = _sgd(m, self.cfg)
        crit = nn.CrossEntropyLoss()

        for _ in range(self.cfg["epochs"]):
            for x, y, _ in ctx.retain_loader:
                x, y = x.to(ctx.device), y.to(ctx.device)
                opt.zero_grad(set_to_none=True)
                crit(m(x), y).backward()
                opt.step()
        return m


# --------------------------------------------------------------------------- 2. NegGrad


@register
class NegGrad(Unlearner):
    """Gradient ascent on the forget set, with no retain-set counterweight.

    The destructive control. Reported as such, never as a competitive method: it is expected to
    destroy retained utility, and that is the point.
    """

    name = "neggrad"
    defaults = {"epochs": 1, "lr": 0.001, "momentum": 0.9, "weight_decay": 0.0}

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        m = self.clone(model).to(ctx.device).train()
        opt = _sgd(m, self.cfg)
        crit = nn.CrossEntropyLoss()

        for _ in range(self.cfg["epochs"]):
            for x, y, _ in ctx.forget_loader:
                x, y = x.to(ctx.device), y.to(ctx.device)
                opt.zero_grad(set_to_none=True)
                (-crit(m(x), y)).backward()  # ascend
                opt.step()
        return m


# --------------------------------------------------------------------------- 3. NegGrad+


@register
class NegGradPlus(Unlearner):
    """Ascent on the forget set, descent on the retain set, balanced by ``alpha``.

    The standard strong gradient-ascent baseline in current vision unlearning work, and a
    consistent top performer on CIFAR-10 / ResNet-18. The retain term is what stops the ascent
    from collapsing the model, which is the whole difference from plain NegGrad.

    Loss = alpha * CE(retain) - (1 - alpha) * CE(forget)
    """

    name = "neggradplus"
    defaults = {
        "epochs": 5,
        "lr": 0.01,
        "alpha": 0.95,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        m = self.clone(model).to(ctx.device).train()
        opt = _sgd(m, self.cfg)
        crit = nn.CrossEntropyLoss()
        alpha = self.cfg["alpha"]
        forget_iter = _cycle(ctx.forget_loader)

        for _ in range(self.cfg["epochs"]):
            for xr, yr, _ in ctx.retain_loader:
                xf, yf, _ = next(forget_iter)
                xr, yr = xr.to(ctx.device), yr.to(ctx.device)
                xf, yf = xf.to(ctx.device), yf.to(ctx.device)

                opt.zero_grad(set_to_none=True)
                loss = alpha * crit(m(xr), yr) - (1.0 - alpha) * crit(m(xf), yf)
                loss.backward()
                opt.step()
        return m


# --------------------------------------------------------------------------- 4. SCRUB


@register
class SCRUB(Unlearner):
    """Teacher/student unlearning (Kurmanji et al., NeurIPS 2023).

    The student starts as a copy of the original model and the teacher *is* the original, frozen.
    Two alternating objectives:

    * **max-steps** on the forget set: maximise KL(student || teacher), pushing the student's
      predictions away from what the original believed about the forgotten data.
    * **min-steps** on the retain set: minimise KL(student || teacher) plus the ordinary
      cross-entropy, holding everything else in place.

    Max-steps run only for the first ``msteps`` epochs. Running them throughout drives the
    student arbitrarily far from the teacher on the forget set, which over-forgets: the goal is
    to resemble a model retrained without the data, and that model does *not* have pathological
    outputs on it.
    """

    name = "scrub"
    defaults = {
        "epochs": 5,
        "msteps": 2,
        "lr": 0.005,
        "alpha": 0.5,  # weight on the retain KL term
        "gamma": 1.0,  # weight on the retain CE term
        "temperature": 4.0,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        teacher = self.clone(model).to(ctx.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student = self.clone(model).to(ctx.device)
        opt = _sgd(student, self.cfg)
        T = self.cfg["temperature"]

        def kl(student_logits, teacher_logits):
            # Scaled by T^2 so the gradient magnitude is comparable across temperatures.
            return (
                F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                )
                * T
                * T
            )

        for epoch in range(self.cfg["epochs"]):
            student.train()

            if epoch < self.cfg["msteps"]:
                for x, _y, _ in ctx.forget_loader:
                    x = x.to(ctx.device)
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        t_out = teacher(x)
                    (-kl(student(x), t_out)).backward()  # diverge from the teacher
                    opt.step()

            for x, y, _ in ctx.retain_loader:
                x, y = x.to(ctx.device), y.to(ctx.device)
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    t_out = teacher(x)
                s_out = student(x)
                loss = self.cfg["alpha"] * kl(s_out, t_out) + self.cfg["gamma"] * F.cross_entropy(
                    s_out, y
                )
                loss.backward()
                opt.step()

        return student


# --------------------------------------------------------------------------- 5. SalUn


@register
class SalUn(Unlearner):
    """Saliency-based unlearning (Fan et al., ICLR 2024).

    Two stages. First, a **weight-saliency mask**: accumulate the gradient of the forget-set loss
    over the whole forget set and keep the top ``sparsity`` fraction of weights by gradient
    magnitude. Second, **random-label fine-tuning** on the forget set, with updates applied only
    through the mask, plus retain-set training to hold utility.

    Restricting updates to the salient weights is what makes it targeted rather than global: an
    unmasked random-label pass is simply label noise injection.
    """

    name = "salun"
    defaults = {
        "epochs": 5,
        "lr": 0.01,
        "sparsity": 0.5,  # fraction of weights left unmasked
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }

    def _saliency_mask(self, model: nn.Module, ctx: UnlearnContext) -> dict[str, torch.Tensor]:
        model = model.to(ctx.device).eval()
        model.zero_grad(set_to_none=True)
        crit = nn.CrossEntropyLoss(reduction="sum")

        loader = ctx.forget_eval_loader or ctx.forget_loader
        total = 0
        for x, y, _ in loader:
            x, y = x.to(ctx.device), y.to(ctx.device)
            # Ascent direction, matching the sign SalUn uses to locate forget-relevant weights.
            (-crit(model(x), y)).backward()
            total += y.size(0)

        grads = {
            n: (p.grad.detach().abs() / max(1, total))
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        model.zero_grad(set_to_none=True)

        flat = torch.cat([g.flatten() for g in grads.values()])
        k = int(self.cfg["sparsity"] * flat.numel())
        if k <= 0:
            return {n: torch.zeros_like(g) for n, g in grads.items()}
        threshold = torch.topk(flat, k, largest=True).values.min()
        return {n: (g >= threshold).float() for n, g in grads.items()}

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        mask = self._saliency_mask(self.clone(model), ctx)

        m = self.clone(model).to(ctx.device)
        opt = _sgd(m, self.cfg)
        crit = nn.CrossEntropyLoss()
        gen = torch.Generator(device="cpu").manual_seed(ctx.seed)
        retain_iter = _cycle(ctx.retain_loader)

        for _ in range(self.cfg["epochs"]):
            m.train()
            for xf, yf, _ in ctx.forget_loader:
                xf = xf.to(ctx.device)
                # Random labels, drawn to differ from the true one so the update is informative.
                offset = torch.randint(
                    1, ctx.num_classes, yf.shape, generator=gen
                )
                y_rand = ((yf + offset) % ctx.num_classes).to(ctx.device)

                opt.zero_grad(set_to_none=True)
                crit(m(xf), y_rand).backward()
                self._masked_step(opt, m, mask)

                xr, yr, _ = next(retain_iter)
                xr, yr = xr.to(ctx.device), yr.to(ctx.device)
                opt.zero_grad(set_to_none=True)
                crit(m(xr), yr).backward()
                self._masked_step(opt, m, mask)
        return m

    @staticmethod
    def _masked_step(
        opt: torch.optim.Optimizer, model: nn.Module, mask: dict[str, torch.Tensor]
    ) -> None:
        """Take an optimiser step that moves only the salient weights.

        Zeroing the gradient outside the mask is **not sufficient**, which is easy to miss.
        SGD's momentum carries velocity accumulated on earlier steps, and weight decay adds
        ``wd * p`` to the update *inside* the optimiser, after any masking we do to ``p.grad``.
        Both move weights whose gradient is exactly zero. Left uncorrected, SalUn drifts toward
        unmasked random-label fine-tuning — that is, plain label-noise injection — while still
        appearing to be masked.

        So: zero the gradients outside the mask, snapshot, step, then restore the masked-out
        weights exactly. Correct regardless of what the optimiser does internally.
        """
        for n, p in model.named_parameters():
            if p.grad is not None and n in mask:
                p.grad.mul_(mask[n])

        snapshot = {
            n: p.detach().clone() for n, p in model.named_parameters() if n in mask
        }
        opt.step()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in mask:
                    frozen = mask[n] == 0
                    p[frozen] = snapshot[n][frozen]


# --------------------------------------------------------------------------- 6. L1-sparse


@register
class L1Sparse(Unlearner):
    """Retain-set fine-tuning under an L1 penalty (Jia et al., NeurIPS 2023).

    The sparsity route to unlearning: shrinking small weights toward zero during retain-set
    fine-tuning removes the low-magnitude structure that carries memorised detail, while the
    retain loss preserves what generalises.
    """

    name = "l1sparse"
    defaults = {
        "epochs": 5,
        "lr": 0.01,
        "l1_lambda": 5e-5,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        m = self.clone(model).to(ctx.device).train()
        opt = _sgd(m, self.cfg)
        crit = nn.CrossEntropyLoss()
        lam = self.cfg["l1_lambda"]

        for _ in range(self.cfg["epochs"]):
            for x, y, _ in ctx.retain_loader:
                x, y = x.to(ctx.device), y.to(ctx.device)
                opt.zero_grad(set_to_none=True)
                # Penalise weight matrices only. Including BatchNorm scales and biases would
                # shrink the normalisation itself, which damages utility without removing
                # memorised structure.
                l1 = sum(
                    p.abs().sum()
                    for n, p in m.named_parameters()
                    if p.dim() > 1
                )
                (crit(m(x), y) + lam * l1).backward()
                opt.step()
        return m


# --------------------------------------------------------------------------- SSD (side only)


@register
class SSD(Unlearner):
    """Selective Synaptic Dampening (Foster et al., AAAI 2024).

    **Not a core method in this project.** SSD selects parameters whose Fisher importance is
    disproportionately high for the forget set relative to the full training set, then dampens
    them. A randomly drawn instance-level forget set is distributionally identical to the retain
    set, so that importance ratio is approximately uniform, nothing exceeds the selection
    threshold, and the dampening step approximates a no-op. This is a mismatch between mechanism
    and task, not a tuning problem, and it is reported empirically: SSD fails to forget at all
    tested forget fractions under random forgetting while performing at or above the state of the
    art on class and sub-class unlearning [24].

    It is retained only for the optional class-unlearning side condition, where it is the
    appropriate tool. Using it in an instance-level condition would contribute a constant row to
    every ranking.
    """

    name = "ssd"
    defaults = {"dampening_constant": 1.0, "selection_weighting": 10.0}

    def unlearn(self, model: nn.Module, ctx: UnlearnContext) -> nn.Module:
        set_determinism(ctx.seed)
        m = self.clone(model).to(ctx.device)

        forget_imp = self._fisher(m, ctx.forget_loader, ctx.device)
        full_imp = self._fisher(m, ctx.retain_loader, ctx.device)

        alpha = self.cfg["dampening_constant"]
        weighting = self.cfg["selection_weighting"]

        with torch.no_grad():
            for (_n, p), f_imp, d_imp in zip(
                m.named_parameters(), forget_imp.values(), full_imp.values()
            ):
                selected = f_imp > d_imp * weighting
                if not selected.any():
                    continue
                factor = ((d_imp * alpha) / (f_imp + 1e-12)).clamp(max=1.0)
                p[selected] = p[selected] * factor[selected]
        return m

    @staticmethod
    def _fisher(model: nn.Module, loader, device) -> dict[str, torch.Tensor]:
        """Diagonal empirical Fisher: mean squared gradient over a loader."""
        imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        crit = nn.CrossEntropyLoss()
        batches = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    imp[n] += p.grad.detach().pow(2)
            batches += 1
        model.zero_grad(set_to_none=True)
        return {n: v / max(1, batches) for n, v in imp.items()}
