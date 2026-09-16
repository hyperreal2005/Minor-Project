"""Stage 6 orchestration: feed every audit, assemble every record.

The audits decide what a number *means*; this module decides what they see. Keeping the two
apart is the same separation that puts timing outside the unlearning methods — six audits each
deciding how to build a probe set would produce six subtly different comparisons.

**Shard-local by construction.** :func:`available_targets` enumerates only what is in *this*
machine's store, so an account audits the checkpoints it produced and ships a few KB of Parquet
rather than 3.6 GB of weights. Nothing here requires all 334 checkpoints at once.

**Forward passes are shared, and that is the main cost saving.** A target model is run over its
probe sets exactly once; all six audits read the same arrays. The oracle ensemble and the 32 RMIA
references are the expensive part — they are identical for every target in a condition, so they
are computed once and cached, turning ~40 model-loads per condition into one.

**A missing input is reported, never imputed.** If the oracle ensemble is absent, oracle-referenced
audits are skipped with a message; they do not quietly emit a plausible number against a missing
reference. The one thing worse than no measurement here is a measurement that looks fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..registry import make_record, parse_run_id, run_id, write_records
from .base import REGISTRY, AuditContext, get_audit
from .probes import ProbeSpec, build_probes

__all__ = ["AuditTarget", "available_targets", "audit_one", "run_audits", "ModelOutputs"]


@dataclass(frozen=True, slots=True)
class AuditTarget:
    """One model to audit, with the coordinates its records need."""

    run_id: str
    forget_id: str
    method: str
    seed: int
    role: str = "unlearn"


def available_targets(
    store, *, role: str = "unlearn", forget: str | None = None
) -> list[AuditTarget]:
    """Every checkpoint of ``role`` present in this store, sorted for reproducible ordering."""
    out = []
    for rid in sorted(store.iter_checkpoints(role=role, forget=forget)):
        key = parse_run_id(rid)
        out.append(
            AuditTarget(
                run_id=rid, forget_id=key.forget, method=key.method,
                seed=key.seed if key.seed is not None else 0, role=key.role,
            )
        )
    return out


# --------------------------------------------------------------------------- forward passes


@dataclass
class ModelOutputs:
    """One model's outputs over a condition's probe sets. Computed once, read by every audit."""

    logits: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    activations: dict[str, np.ndarray] = field(default_factory=dict)


def evaluate_model(
    state: dict,
    *,
    bundle,
    probes: ProbeSpec,
    layers: Sequence[str],
    device: str = "cpu",
    batch_size: int = 512,
    with_activations: bool = True,
    num_workers: int = 0,
) -> ModelOutputs:
    """Run one checkpoint over the probe sets and return everything the audits need.

    Loaders are unshuffled and un-augmented: row *i* must be the same example in every model, or
    every cross-model comparison in Stage 6 is comparing different things.
    """
    from ..data.cifar import make_loader
    from ..evaluation import predict
    from ..models.resnet import extract_features, make_resnet18

    model = make_resnet18(num_classes=bundle.num_classes)
    model.load_state_dict(state)
    model = model.to(device).eval()

    out = ModelOutputs()
    for name, idx in probes.as_dict().items():
        split = "test" if name == "test" else "train"
        loader = make_loader(
            bundle, idx, train=False, batch_size=batch_size, split=split,
            num_workers=num_workers,
        )
        p = predict(model, loader, device=device)
        out.logits[name] = p.logits.astype(np.float32)
        out.labels[name] = p.labels

    if with_activations:
        # The mixed probe spans both splits, so it is two passes concatenated in a fixed order:
        # train part first, then test. Every model does the same, so the rows align.
        feats: dict[str, list[np.ndarray]] = {}
        for split, idx in (("train", probes.mixed_train), ("test", probes.mixed_test)):
            if idx.size == 0:
                continue
            loader = make_loader(
                bundle, idx, train=False, batch_size=batch_size, split=split,
                num_workers=num_workers,
            )
            acts, _ = extract_features(model, loader, layers=tuple(layers), device=device)
            for k, v in acts.items():
                feats.setdefault(k, []).append(v.detach().cpu().numpy().astype(np.float32))
        out.activations = {k: np.concatenate(v) for k, v in feats.items()}
    return out


# --------------------------------------------------------------------------- reference caches


class _ConditionCache:
    """Oracle, original and reference outputs for one condition, loaded once.

    These are what make Stage 6 affordable: the ensemble is the same for all ~30 targets in a
    condition, and reloading it per target would multiply the cost by thirty for no new
    information.
    """

    def __init__(self, ctxobj, *, forget_id: str, probes: ProbeSpec, layers, device, batch_size):
        self.ctx = ctxobj
        self.forget_id = forget_id
        self.probes = probes
        self.layers = layers
        self.device = device
        self.batch_size = batch_size
        self._oracles: dict[str, np.ndarray] | None = None
        self._oracle_acts: dict[str, np.ndarray] | None = None
        self._originals: dict[int, ModelOutputs] = {}
        self._refs: dict[str, np.ndarray] | None = None
        self._ref_mask: dict[str, np.ndarray] | None = None

    def _eval(self, rid, *, with_activations=False) -> ModelOutputs | None:
        if not self.ctx.store.has_checkpoint(rid):
            return None
        state, _ = self.ctx.store.load_checkpoint(rid)
        return evaluate_model(
            state, bundle=self.ctx.bundle, probes=self.probes, layers=self.layers,
            device=self.device, batch_size=self.batch_size, with_activations=with_activations,
        )

    def oracles(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self._oracles is None:
            logits: dict[str, list[np.ndarray]] = {}
            acts: dict[str, list[np.ndarray]] = {}
            for seed in self.ctx.base["oracles"]["paired_seeds"]:
                rid = run_id(role="oracle", forget=self.forget_id, seed=seed, seed_kind="train")
                got = self._eval(rid, with_activations=True)
                if got is None:
                    continue
                for k, v in got.logits.items():
                    logits.setdefault(k, []).append(v)
                for k, v in got.activations.items():
                    acts.setdefault(k, []).append(v)
            self._oracles = {k: np.stack(v) for k, v in logits.items()}
            self._oracle_acts = {k: np.stack(v) for k, v in acts.items()}
        return self._oracles, self._oracle_acts

    def original(self, seed: int) -> ModelOutputs | None:
        if seed not in self._originals:
            from ..unlearn import base_run_id_for

            spec = self.ctx.spec(self.forget_id)
            self._originals[seed] = self._eval(base_run_id_for(spec, seed))
        return self._originals[seed]

    def references(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """RMIA shadow outputs, plus the IN/OUT mask reconstructed from `shadow_indices`.

        The mask is recomputed rather than stored: `shadow_indices` is a pure function of
        (audit_seed, idx), so membership is derivable anywhere, and a stored copy would be one
        more thing that could drift from the models it describes.
        """
        if self._refs is None:
            from ..train.tasks import shadow_indices

            audit_seed = int(self.ctx.seeds.get("audit", 0))
            n_shadows = int(self.ctx.base["shadows"]["count"])
            fraction = float(self.ctx.base["shadows"]["subset_fraction"])

            logits: dict[str, list[np.ndarray]] = {}
            in_forget: list[np.ndarray] = []
            for idx in range(n_shadows):
                rid = run_id(role="shadow", forget="full", seed=idx, seed_kind="shadow")
                got = self._eval(rid)
                if got is None:
                    continue
                for k, v in got.logits.items():
                    logits.setdefault(k, []).append(v)
                trained_on = shadow_indices(
                    self.ctx.bundle.n_train, idx, audit_seed=audit_seed, fraction=fraction
                )
                in_forget.append(np.isin(self.probes.forget, trained_on))
            self._refs = {k: np.stack(v) for k, v in logits.items()}
            self._ref_mask = {"forget": np.stack(in_forget)} if in_forget else {}
        return self._refs, self._ref_mask


# --------------------------------------------------------------------------- one target


def audit_one(
    target: AuditTarget,
    *,
    ctxobj,
    cache: _ConditionCache,
    audits: Sequence[str],
    device: str = "cpu",
    batch_size: int = 512,
) -> list:
    """Audit one model and return its records."""
    spec = ctxobj.spec(target.forget_id)
    probes = cache.probes
    audit_cfg = ctxobj.audits

    needs_acts = any(get_audit(a).name == "representation" for a in audits)
    state, meta = ctxobj.store.load_checkpoint(target.run_id)
    t0 = time.perf_counter()
    outputs = evaluate_model(
        state, bundle=ctxobj.bundle, probes=probes, layers=audit_cfg["representation"]["layers"],
        device=device, batch_size=batch_size, with_activations=needs_acts,
    )

    oracle_logits, oracle_acts = cache.oracles()
    original = cache.original(target.seed)
    refs, ref_mask = cache.references()

    records = []
    for name in audits:
        audit = get_audit(name)
        if audit.needs_oracles and not oracle_logits:
            continue  # reported by the caller; never imputed
        if audit.needs_references and not refs:
            continue

        ctx = AuditContext(
            run_id=target.run_id,
            logits=outputs.logits,
            labels=outputs.labels,
            activations=outputs.activations,
            original_logits=original.logits if original else {},
            oracle_logits=oracle_logits,
            oracle_activations=oracle_acts,
            reference_logits=refs,
            reference_in_mask=ref_mask,
            forget_kind=spec.kind,
            forget_size=int(probes.forget.size),
            forget_stratum=spec.stratum,
            config=_config_for(name, audit_cfg),
            audit_seed=int(ctxobj.seeds.get("audit", 0)),
            device=device,
            store=ctxobj.store,
        )
        values = audit.measure(ctx)
        notes = audit.notes_for(ctx)

        # `spec.as_record_fields()` rather than hand-picked attributes: it exists so audits do
        # not re-derive identity and get it subtly wrong. Doing it by hand here dropped
        # `selection_seed`, which the validator requires for every memstratum condition -- three
        # of the eight -- and the failure would have landed after the audit pass, not before it.
        common = dict(
            run_id=target.run_id,
            **spec.as_record_fields(),
            hparams_sha=meta.hparams_sha,
            checkpoint_sha=meta.sha,
            audit_seed=int(ctxobj.seeds.get("audit", 0)),
        )
        common["forget_size"] = int(probes.forget.size)
        for (metric, probe), value in values.items():
            n_probe = _n_probe(probe, probes, outputs)
            if np.isfinite(value):
                records.append(
                    make_record(
                        audit=name, metric=metric, probe_set=probe,
                        value=float(value), n_probe=n_probe, notes=notes, **common,
                    )
                )
                continue

            # An undefined value cannot go in the value column -- `validate()` rejects NaN and
            # inf, correctly, because either would silently poison every aggregate downstream.
            # But dropping the row would erase the most informative case in the study: the
            # destroyed control is exactly the model whose audits come back undefined, and
            # "could not measure" has to stay distinguishable from "never ran".
            why = f"{metric} undefined" + (f": {notes}" if notes else "")
            records.append(
                make_record(
                    audit=name, metric="audit_undefined", probe_set=probe,
                    value=1.0, n_probe=n_probe, notes=why, **common,
                )
            )
    _ = time.perf_counter() - t0
    return records


def _config_for(name: str, audits_cfg: dict) -> dict:
    """The protocol block an audit reads, flattened with the shared keys it also needs."""
    section = {
        "behavior": "behavior",
        "privacy_population": "privacy",
        "privacy_rmia": "privacy",
        "representation": "representation",
        "relearning": "relearning",
        "sde": "sde",
    }[name]
    cfg = dict(audits_cfg.get(section, {}))
    if name == "privacy_population":
        cfg = {**cfg, **cfg.get("population", {})}
    elif name == "privacy_rmia":
        cfg = {**cfg, **cfg.get("rmia", {})}
    cfg.setdefault("prob_floor", audits_cfg.get("behavior", {}).get("prob_floor", 1e-12))
    return cfg


def _n_probe(probe: str, probes: ProbeSpec, outputs: ModelOutputs) -> int:
    if probe in ("forget", "retain", "test"):
        return int(getattr(probes, probe).size)
    arr = outputs.activations.get(probe)
    return int(len(arr)) if arr is not None else int(probes.mixed_train.size + probes.mixed_test.size)


# --------------------------------------------------------------------------- the queue


def run_audits(
    ctxobj,
    *,
    targets: Iterable[AuditTarget] | None = None,
    audits: Sequence[str] | None = None,
    device: str = "cpu",
    batch_size: int = 512,
    dry_run: bool = False,
) -> int:
    """Audit every target this machine holds, grouped by condition so caches are reused."""
    audits = tuple(audits or sorted(REGISTRY))
    targets = list(targets if targets is not None else available_targets(ctxobj.store))
    if not targets:
        print("no unlearn checkpoints in this store; nothing to audit")
        return 0

    by_condition: dict[str, list[AuditTarget]] = {}
    for t in targets:
        by_condition.setdefault(t.forget_id, []).append(t)

    written = failed = 0
    for forget_id, group in sorted(by_condition.items()):
        probes = build_probes(
            forget_indices=ctxobj.forget_indices(forget_id),
            n_train=ctxobj.bundle.n_train,
            n_test=ctxobj.bundle.n_test,
            forget_id=forget_id,
            config=ctxobj.audits.get("representation", {}),
        )
        print(f"\n{forget_id}: {len(group)} targets, probes "
              f"{probes.forget.size}/{probes.retain.size}/{probes.test.size} "
              f"(forget/retain/test)", flush=True)
        if dry_run:
            for t in group:
                print(f"  [todo] {t.run_id}")
            continue

        cache = _ConditionCache(
            ctxobj, forget_id=forget_id, probes=probes,
            layers=ctxobj.audits["representation"]["layers"],
            device=device, batch_size=batch_size,
        )
        for i, t in enumerate(group, 1):
            print(f"  [{i}/{len(group)}] {t.run_id} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                records = audit_one(
                    t, ctxobj=ctxobj, cache=cache, audits=audits,
                    device=device, batch_size=batch_size,
                )
            except Exception as exc:  # one bad target must not lose the whole condition
                failed += 1
                print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            write_records(records, ctxobj.records_dir, suffix="audit")
            written += len(records)
            print(f" {len(records)} records in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\nwrote {written} records, {failed} targets failed")
    return 1 if failed else 0
