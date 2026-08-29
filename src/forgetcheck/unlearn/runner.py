"""Running one unlearning method against one M0, and recording the result.

Everything that is not the method itself lives here: loading the original checkpoint, building
the loaders, timing, checkpointing and record-writing. That is deliberate — it is what makes
``runtime_s`` a fair comparison across six methods rather than six people's timers.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..data.cifar import DataBundle, apply_canaries, make_loader
from ..data.forget_sets import ForgetSpec
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
from .base import UnlearnContext, get_unlearner

__all__ = ["run_unlearn", "base_run_id_for"]


def base_run_id_for(spec: ForgetSpec, seed: int) -> str:
    """The M0 an unlearning run should start from.

    The canary condition has its own original model, trained on the corrupted labels — the
    canaries must have been present during training for there to be anything to forget. Every
    other condition starts from the clean M0.
    """
    variant = spec.forget_id if spec.kind == "canary" else "full"
    return run_id(role="base", forget=variant, seed=seed)


def run_unlearn(
    *,
    method: str,
    spec: ForgetSpec,
    forget_indices: np.ndarray,
    bundle: DataBundle,
    seed: int,
    store: ArtifactStore,
    records_dir,
    hparams: dict[str, Any] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    skip_existing: bool = True,
    notes: str = "",
) -> str | None:
    """Apply one unlearning method to one original model.

    Returns the new ``run_id``, or ``None`` if it already existed and was skipped.

    Raises:
        FileNotFoundError: if the required M0 has not been trained yet. Unlearning cannot be
            queued ahead of the model it modifies, and failing here is clearer than silently
            unlearning from a fresh initialisation.
    """
    rid = run_id(role="unlearn", forget=spec.forget_id, method=method, seed=seed)
    if skip_existing and store.has_checkpoint(rid):
        return None

    base_rid = base_run_id_for(spec, seed)
    if not store.has_checkpoint(base_rid):
        raise FileNotFoundError(
            f"{rid}: requires the original model {base_rid}, which is not in the store. "
            "Train stage 3 before stage 5."
        )

    # The canary M0 was trained on corrupted labels, so unlearning must see the same labels it
    # was trained on -- otherwise the "forget set" it is asked to remove is not what the model
    # actually learned.
    train_bundle = bundle
    if spec.kind == "canary":
        train_bundle, _ = apply_canaries(bundle, forget_indices)

    retain_indices = bundle.retain_indices(forget_indices)

    forget_loader = make_loader(
        train_bundle, forget_indices, train=True, batch_size=batch_size,
        seed=seed, num_workers=num_workers,
    )
    retain_loader = make_loader(
        train_bundle, retain_indices, train=True, batch_size=batch_size,
        seed=seed, num_workers=num_workers,
    )
    forget_eval_loader = make_loader(
        train_bundle, forget_indices, train=False, batch_size=batch_size,
        num_workers=num_workers,
    )

    state, base_meta = store.load_checkpoint(base_rid)
    model = make_resnet18(num_classes=bundle.num_classes)
    model.load_state_dict(state)

    unlearner = get_unlearner(method, **(hparams or {}))
    ctx = UnlearnContext(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        forget_eval_loader=forget_eval_loader,
        device=device,
        num_classes=bundle.num_classes,
        seed=seed,
    )

    t0 = time.perf_counter()
    unlearned = unlearner.unlearn(model, ctx)
    elapsed = time.perf_counter() - t0

    hp = config_sha({"method": method, **unlearner.cfg})

    evals = {
        "test": make_loader(bundle, train=False, split="test", batch_size=512,
                            num_workers=num_workers),
        "retain": make_loader(train_bundle, retain_indices, train=False, batch_size=512,
                              num_workers=num_workers),
        "forget": make_loader(train_bundle, forget_indices, train=False, batch_size=512,
                              num_workers=num_workers),
    }
    final = {}
    for name, loader in evals.items():
        for k, v in classification_metrics(predict(unlearned, loader, device=device)).items():
            final[f"{name}_{k}"] = v

    store.save_checkpoint(
        rid,
        unlearned.state_dict(),
        train_seed=seed,
        final_metrics=final,
        hparams_sha=hp,
        notes=f"from {base_rid}" + (f"; {notes}" if notes else ""),
    )

    rows = _records(rid, spec, final, hp, elapsed=elapsed, n=len(forget_indices), notes=notes)
    write_records(rows, records_dir, suffix="unlearn")
    return rid


def _records(
    rid: str,
    spec: ForgetSpec,
    final: dict[str, float],
    hparams_sha: str,
    *,
    elapsed: float,
    n: int,
    notes: str,
) -> list[RunRecord]:
    # Same mapping as training: the forget set gets the oracle-referenced metrics, never the
    # plain utility ones. See configs/metrics.yaml on why ce_loss is not defined on forget.
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
    fields = spec.as_record_fields()

    rows = [
        make_record(
            run_id=rid,
            audit="meta",
            metric=name,
            probe_set=probe,
            value=final[f"{probe}_{stat}"],
            n_probe=n,
            hparams=hparams_sha,
            runtime_s=elapsed,
            notes=notes,
            **fields,
        )
        for (probe, stat), name in metric_map.items()
        if f"{probe}_{stat}" in final
    ]

    rows.append(
        make_record(
            run_id=rid,
            audit="meta",
            metric="runtime_s",
            probe_set="all",
            value=elapsed,
            n_probe=n,
            hparams=hparams_sha,
            runtime_s=elapsed,
            notes=notes,
            **fields,
        )
    )
    return rows
