"""The one training loop.

M0, the oracles and the shadow models all go through this function, unmodified. That is not
tidiness — it is the precondition for every oracle-referenced metric in the project. If the
oracle were trained under even a slightly different schedule than M0, the gap between them would
contain a schedule difference as well as a forget-set difference, and no audit could separate
the two.

The same applies to the relearning arms: M0, M_u and M_r are relearned by :func:`finetune_steps`
under identical settings, so a difference in recovery speed is attributable to the model rather
than to the protocol.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..evaluation import Predictions, classification_metrics, predict

__all__ = ["TrainConfig", "TrainResult", "train", "finetune_steps", "set_determinism"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything that can change a trained model. Hashed into ``hparams_sha``."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    label_smoothing: float = 0.0
    amp: bool = True
    channels_last: bool = True
    grad_clip: float | None = None

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__slots__}


@dataclass(slots=True)
class TrainResult:
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    final: dict[str, float] = field(default_factory=dict)


def set_determinism(seed: int, *, strict: bool = False) -> None:
    """Seed every RNG this project touches.

    ``strict`` additionally forces deterministic cuDNN kernels. It is off by default because it
    can cost 20-30% throughput and because GPU nondeterminism is *already* part of what the
    oracle ensemble measures — the ensemble's spread is the reference distribution, so
    suppressing run-to-run variation would understate the very noise the calibration depends on.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _build_scheduler(opt, cfg: TrainConfig, steps_per_epoch: int):
    """Cosine decay with linear warmup, stepped per batch."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(0, cfg.warmup_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if cfg.scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def _amp_enabled(cfg: TrainConfig, device: torch.device) -> bool:
    return bool(cfg.amp) and device.type == "cuda"


def train(
    model: nn.Module,
    train_loader,
    cfg: TrainConfig,
    *,
    device: str | torch.device = "cpu",
    seed: int = 0,
    eval_loaders: Mapping[str, object] | None = None,
    eval_every: int = 0,
    on_epoch: Callable[[int, dict], None] | None = None,
) -> TrainResult:
    """Train ``model`` and return it along with a per-epoch history.

    Args:
        eval_loaders: named evaluation loaders (``{"test": ..., "retain": ...}``) run every
            ``eval_every`` epochs and at the end. Evaluation loaders must be unshuffled.
        eval_every: 0 evaluates only at the end, which is what the matrix runs use — intermediate
            evaluation on CPU is a large fraction of total time for no scientific gain.

    The optimiser is SGD throughout: it is what the CIFAR unlearning literature uses, and
    switching would put our accuracies on a different scale from the published baselines.
    """
    device = torch.device(device)
    set_determinism(seed)

    model = model.to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    sched = _build_scheduler(opt, cfg, max(1, len(train_loader)))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    use_amp = _amp_enabled(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult(model=model)
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen, correct = 0.0, 0, 0

        for x, y, _idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if cfg.channels_last and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            running += loss.item() * y.size(0)
            correct += int((out.argmax(1) == y).sum())
            seen += y.size(0)

        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "train_acc": correct / max(1, seen),
            "lr": opt.param_groups[0]["lr"],
        }

        last = epoch == cfg.epochs - 1
        if eval_loaders and (last or (eval_every and (epoch + 1) % eval_every == 0)):
            for name, loader in eval_loaders.items():
                m = classification_metrics(predict(model, loader, device=device))
                row.update({f"{name}_{k}": v for k, v in m.items()})

        result.history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, row)

    result.seconds = time.perf_counter() - t0
    result.final = dict(result.history[-1]) if result.history else {}
    return result


def finetune_steps(
    model: nn.Module,
    loader,
    *,
    steps: int,
    lr: float,
    momentum: float = 0.9,
    device: str | torch.device = "cpu",
    seed: int = 0,
    eval_at: Sequence[int] = (),
    evaluate: Callable[[nn.Module, int], dict] | None = None,
    weight_decay: float = 0.0,
) -> list[dict]:
    """Fine-tune for a fixed number of optimiser *steps*, evaluating at given step counts.

    This is the relearning protocol. Steps rather than epochs, because the arms may have forget
    sets of different sizes and "one epoch" would then mean different amounts of optimisation —
    which would make recovery speed a function of forget-set size rather than of retained
    structure.

    No scheduler: a decaying learning rate would confound recovery speed with schedule position.

    Returns:
        One row per entry in ``eval_at``, including step 0 (the pre-relearning baseline) when
        requested.
    """
    device = torch.device(device)
    set_determinism(seed)

    model = model.to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    want = sorted(set(eval_at))
    rows: list[dict] = []

    def record(step: int) -> None:
        if evaluate is not None:
            rows.append({"step": step, **evaluate(model, step)})

    if want and want[0] == 0:
        record(0)

    step = 0
    done = False
    while not done:
        for x, y, _idx in loader:
            if step >= steps:
                done = True
                break
            model.train()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            step += 1
            if step in want:
                record(step)
        if len(loader) == 0:
            raise ValueError("relearning loader is empty")

    return rows
