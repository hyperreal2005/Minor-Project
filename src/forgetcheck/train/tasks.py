"""Role-specific training tasks: M0, oracles, and RMIA shadow models.

Each task decides *what data* a model sees and *what identity* it carries, then hands off to the
one shared loop in :mod:`forgetcheck.train.loop`. Keeping the data selection here and the
optimisation there is what guarantees that an oracle differs from its paired M0 only in the data
it saw.

Three data selections, and the differences between them are easy to get wrong:

============  ==========================================================================
M0 (full)     all 50,000, clean labels
M0 (canary)   all 50,000, with the canary subset's labels corrupted
oracle        the retain set only -- the forget examples are *removed*, not relabelled.
              For the canary condition this means clean labels throughout, because the
              canaries carried the only corrupted labels and they are gone.
shadow        a random 50% subset of the clean training set
============  ==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..data.cifar import DataBundle, apply_canaries, make_loader
from ..data.forget_sets import ForgetSpec, spec_by_id
from ..evaluation import classification_metrics, predict
from ..models.resnet import make_resnet18
from ..registry import (
    ArtifactStore,
    RunRecord,
    config_sha,
    make_record,
    run_id,
    write_records,
)
from .loop import TrainConfig, TrainResult, train

__all__ = ["TrainTask", "base_task", "oracle_task", "shadow_task", "run_task", "shadow_indices"]

FULL = "full"


@dataclass(frozen=True, slots=True)
class TrainTask:
    """A single training run: its identity, its data, and its record fields."""

    run_id: str
    role: str
    indices: np.ndarray  # training-set indices this model sees
    bundle: DataBundle  # the (possibly canary-corrupted) bundle to draw labels from
    seed: int  # seeds both initialisation and data order
    record_fields: dict[str, Any]

    @property
    def n_examples(self) -> int:
        return len(self.indices)


# --------------------------------------------------------------------------- task builders


def base_task(bundle: DataBundle, *, seed: int, variant: str = FULL) -> TrainTask:
    """M0: trained on everything.

    ``variant`` names the *dataset variant*, not a forget set. ``full`` is the clean training
    set; ``canary-500`` is the corrupted one. The canary condition needs its own original model
    because canaries must be present during training for there to be anything to forget.
    """
    if variant == FULL:
        data = bundle
        fields = {"forget_kind": "none", "forget_size": 0, "forget_id": FULL}
    else:
        spec = spec_by_id(variant)
        if spec.kind != "canary":
            raise ValueError(
                f"base variant must be 'full' or a canary condition, got {variant!r}"
            )
        idx = spec.indices(bundle.n_train)
        data, _wrong = apply_canaries(bundle, idx)
        fields = {
            "forget_kind": "none",  # M0 forgets nothing; the canaries are simply present
            "forget_size": 0,
            "forget_id": variant,
            "selection_seed": spec.selection_seed,
        }

    return TrainTask(
        run_id=run_id(role="base", forget=variant, seed=seed),
        role="base",
        indices=bundle.all_indices(),
        bundle=data,
        seed=seed,
        record_fields=fields,
    )


def oracle_task(
    bundle: DataBundle,
    spec: ForgetSpec,
    forget_indices: np.ndarray,
    *,
    seed: int,
    ensemble: bool = False,
) -> TrainTask:
    """A retrained oracle: trained from scratch on the retain set only.

    Args:
        seed: for a **paired** oracle this is the ``train_seed`` of the M0 it is matched to, so
            the pair shares an initialisation and differs only in the data. For an **ensemble**
            member it is an ``oracle_seed``, independent of any M0 — the ensemble's job is to
            span training randomness, so its members must not be tied to particular M0s.
        ensemble: selects which of those two roles this oracle plays.

    The bundle passed in is always the **clean** one. For the canary condition the canaries are
    removed rather than relabelled, and they carried the only corrupted labels, so what remains
    is clean by construction.
    """
    retain = bundle.retain_indices(forget_indices)
    kind = "oracle" if ensemble else "train"
    return TrainTask(
        run_id=run_id(
            role="oracle", forget=spec.forget_id, seed=seed, seed_kind=kind
        ),
        role="oracle",
        indices=retain,
        bundle=bundle,
        seed=seed,
        record_fields=spec.as_record_fields(),
    )


def shadow_indices(n_train: int, idx: int, *, audit_seed: int, fraction: float = 0.5) -> np.ndarray:
    """The training subset for RMIA shadow model ``idx``.

    Each shadow sees an independent random half of the training set, so across 32 shadows every
    example is OUT for roughly 16 of them. Those OUT models are what the per-example attack
    compares a target against.

    The draw is seeded by ``(audit_seed, idx)`` so any shadow can be regenerated on its own,
    without replaying the ones before it — which matters when 32 trainings are spread across
    several accounts.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([audit_seed, idx])))
    k = int(round(fraction * n_train))
    return np.sort(rng.permutation(n_train)[:k]).astype(np.int64)


def shadow_task(bundle: DataBundle, idx: int, *, audit_seed: int, fraction: float = 0.5) -> TrainTask:
    """One RMIA reference model.

    Shadows are condition-independent: they model the data distribution, not any forget set, so
    the same 32 serve all eight conditions. 32 trainings, not 32 x 8.
    """
    return TrainTask(
        run_id=run_id(role="shadow", forget=FULL, seed=idx, seed_kind="shadow"),
        role="shadow",
        indices=shadow_indices(bundle.n_train, idx, audit_seed=audit_seed, fraction=fraction),
        # Seeded by index so two shadows are independent models, not two draws of one model.
        bundle=bundle,
        seed=audit_seed * 1000 + idx,
        record_fields={
            "forget_kind": "none",
            "forget_size": 0,
            "forget_id": FULL,
            "audit_seed": audit_seed,
        },
    )


# --------------------------------------------------------------------------- execution


def run_task(
    task: TrainTask,
    cfg: TrainConfig,
    *,
    store: ArtifactStore,
    records_dir,
    device: str = "cpu",
    eval_bundle: DataBundle | None = None,
    forget_indices: np.ndarray | None = None,
    num_workers: int = 0,
    skip_existing: bool = True,
    notes: str = "",
) -> str | None:
    """Train one task, checkpoint it, and write its records.

    Returns:
        The ``run_id``, or ``None`` if the run already existed and was skipped.

    Skipping is what makes a Kaggle queue resumable: a session that dies mid-run costs only its
    in-flight model, and re-running the same command picks up where it stopped.
    """
    if skip_existing and store.has_checkpoint(task.run_id):
        return None

    eval_bundle = eval_bundle if eval_bundle is not None else task.bundle
    hp = config_sha({**cfg.as_dict(), "n_train": task.n_examples})

    train_loader = make_loader(
        task.bundle,
        task.indices,
        train=True,
        batch_size=cfg.batch_size,
        seed=task.seed,
        num_workers=num_workers,
    )

    eval_loaders = {
        "test": make_loader(
            eval_bundle, train=False, split="test", batch_size=512, num_workers=num_workers
        )
    }
    if forget_indices is not None and len(forget_indices):
        eval_loaders["forget"] = make_loader(
            eval_bundle, forget_indices, train=False, batch_size=512, num_workers=num_workers
        )
        eval_loaders["retain"] = make_loader(
            eval_bundle,
            eval_bundle.retain_indices(forget_indices),
            train=False,
            batch_size=512,
            num_workers=num_workers,
        )

    model = make_resnet18(num_classes=task.bundle.num_classes, seed=task.seed)
    result: TrainResult = train(
        model, train_loader, cfg, device=device, seed=task.seed, eval_loaders=eval_loaders
    )

    store.save_checkpoint(
        task.run_id,
        result.model.state_dict(),
        epochs=cfg.epochs,
        train_seed=task.seed,
        final_metrics={k: v for k, v in result.final.items() if isinstance(v, float)},
        hparams_sha=hp,
        notes=notes,
    )

    rows = _records_for(task, result, hp, notes=notes)
    if rows:
        write_records(rows, records_dir, suffix="train")
    return task.run_id


def _records_for(
    task: TrainTask, result: TrainResult, hparams_sha: str, *, notes: str
) -> list[RunRecord]:
    """Translate the final epoch's metrics into registered record rows."""
    # Which registered metric each (probe set, statistic) pair maps to. The forget set gets
    # forget_acc/forget_loss -- both closer_to_oracle -- rather than the plain utility metrics,
    # because on the forget set "lower loss" and "higher accuracy" are not better, they are just
    # different from the oracle. macro_f1 is a utility guard and is not defined on the forget set.
    metric_map = {
        ("test", "acc"): "test_acc",
        ("test", "macro_f1"): "macro_f1",
        ("test", "ce_loss"): "ce_loss",
        ("retain", "acc"): "retain_acc",
        ("retain", "macro_f1"): "macro_f1",
        ("retain", "ce_loss"): "ce_loss",
        ("forget", "acc"): "forget_acc",
        ("forget", "ce_loss"): "forget_loss",
    }

    rows: list[RunRecord] = []
    for (probe, stat), name in metric_map.items():
            key = f"{probe}_{stat}"
            if key not in result.final:
                continue
            probe_set = probe
            rows.append(
                make_record(
                    run_id=task.run_id,
                    audit="meta",
                    metric=name,
                    probe_set=probe_set,
                    value=result.final[key],
                    n_probe=task.n_examples,
                    hparams=hparams_sha,
                    runtime_s=result.seconds,
                    notes=notes,
                    **task.record_fields,
                )
            )

    rows.append(
        make_record(
            run_id=task.run_id,
            audit="meta",
            metric="runtime_s",
            probe_set="all",
            value=result.seconds,
            n_probe=task.n_examples,
            hparams=hparams_sha,
            runtime_s=result.seconds,
            notes=notes,
            **task.record_fields,
        )
    )
    return rows
