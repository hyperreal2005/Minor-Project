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

__all__ = [
    "AuditTarget", "available_targets", "audit_one", "run_audits", "ModelOutputs",
    "shard_conditions", "already_audited", "audited_audits",
]


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
        # The canary condition's models were trained and unlearned on corrupted labels
        # (Stage 5's `apply_canaries`), and the audits must see the same labels: a membership
        # attack asks whether an example looks like it did *in training*, and relearning must
        # reintroduce the association that was forgotten, not the true label the oracle already
        # knows. Evaluating on the clean bundle -- what the first Stage 6 run did -- inverted RMIA
        # (0.14-0.30, "below chance") and made the relearning anchors point the wrong way.
        spec = ctxobj.spec(forget_id)
        self.spec = spec
        if spec.kind == "canary":
            from ..data.cifar import apply_canaries

            self.bundle, _ = apply_canaries(ctxobj.bundle, probes.forget)
        else:
            self.bundle = ctxobj.bundle
        self.layers = layers
        self.device = device
        self.batch_size = batch_size
        self._oracles: dict[str, np.ndarray] | None = None
        self._oracle_acts: dict[str, np.ndarray] | None = None
        self._originals: dict[int, ModelOutputs] = {}
        self._refs: dict[str, np.ndarray] | None = None
        self._ref_mask: dict[str, np.ndarray] | None = None
        self._relearn_anchors: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        self._randinit_curve: dict[str, np.ndarray] | None = None
        self.missing: list[str] = []  # reference checkpoints this store does not hold

    def _eval(self, rid, *, with_activations=False, cache: bool = False) -> ModelOutputs | None:
        """Evaluate a reference model, reading the on-disk cache when ``cache`` is set.

        Only models that belong to exactly one condition are cacheable by run_id -- unlearn
        models and paired oracles. Base models and shadows serve every condition with different
        forget/retain probes, so their outputs are recomputed per condition (a few minutes) rather
        than stored under a key that could not say which condition they were for.
        """
        if cache and self.ctx.store.has_outputs(rid):
            got = self._from_cache(rid, with_activations=with_activations)
            if got is not None:
                return got
        if not self.ctx.store.has_checkpoint(rid):
            self.missing.append(rid)
            return None
        state, _ = self.ctx.store.load_checkpoint(rid)
        out = evaluate_model(
            state, bundle=self.bundle, probes=self.probes, layers=self.layers,
            device=self.device, batch_size=self.batch_size, with_activations=with_activations,
        )
        if cache:
            self._to_cache(rid, out)
        return out

    def _from_cache(self, rid, *, with_activations) -> ModelOutputs | None:
        from ..registry.store import StoreError

        try:
            logits, labels = self.ctx.store.load_outputs(rid, forget_id=self.forget_id)
            acts = {}
            if with_activations:
                if not self.ctx.store.has_activations(rid):
                    return None
                acts, _ = self.ctx.store.load_activations(rid)
            return ModelOutputs(logits=logits, labels=labels, activations=acts)
        except StoreError:
            return None  # wrong condition or corrupt file: recompute rather than trust

    def _to_cache(self, rid, out: ModelOutputs) -> None:
        self.ctx.store.save_outputs(rid, out.logits, out.labels, forget_id=self.forget_id)
        if out.activations:
            probe_ids = np.concatenate([self.probes.mixed_train, self.probes.mixed_test])
            self.ctx.store.save_activations(rid, out.activations, probe_ids=probe_ids)

    # -- relearning --------------------------------------------------------------------------

    def relearn_arm(self, state, *, seed: int) -> dict[str, np.ndarray]:
        """One relearning curve. Every arm goes through here with the same batches and seed --
        that shared path is the guarantee that the protocol is identical across arms."""
        from .relearning import relearn_curve

        cfg = self.ctx.audits["relearning"]
        return relearn_curve(
            self._fresh_model(state), self._relearn_batches(), self._relearn_eval(),
            eval_steps=cfg["eval_steps"], lr=float(cfg["lr"]), momentum=float(cfg["momentum"]),
            device=self.device, seed=seed,
        )

    def _fresh_model(self, state):
        from ..models.resnet import make_resnet18

        m = make_resnet18(num_classes=self.bundle.num_classes)
        if state is not None:
            m.load_state_dict(state)
        return m

    def _relearn_batches(self):
        """The reintroduction subset, materialised once so every arm sees identical batches."""
        if not hasattr(self, "_batches"):
            from ..data.cifar import make_loader
            from .probes import probe_seed

            cfg = self.ctx.audits["relearning"]
            rng = np.random.default_rng(probe_seed(self.forget_id, base=1))
            k = min(int(cfg["reintroduction_size"]), self.probes.forget.size)
            subset = np.sort(rng.choice(self.probes.forget, k, replace=False))
            # Unaugmented and unshuffled: the batch *order* is part of the protocol.
            loader = make_loader(
                self.bundle, subset, train=False, batch_size=int(cfg["batch_size"]),
                split="train",
            )
            self._batches = [(x, y) for x, y, _ in loader]
        return self._batches

    def _relearn_eval(self):
        """Forget / retain / test accuracy on fixed probes, for the curve checkpoints.

        Retain and test are capped at 1000 here -- eight evaluations per arm, four arms per
        target -- while the forget set is always evaluated in full, because forget accuracy is
        the curve.
        """
        if not hasattr(self, "_eval_loaders"):
            from ..data.cifar import make_loader

            b = self.bundle
            self._eval_loaders = {
                "forget": make_loader(b, self.probes.forget, train=False, batch_size=512),
                "retain": make_loader(b, self.probes.retain[:1000], train=False, batch_size=512),
                "test": make_loader(b, self.probes.test[:1000], train=False, batch_size=512,
                                    split="test"),
            }

        def evaluate(model) -> dict[str, float]:
            from ..evaluation import predict

            out = {}
            for name, loader in self._eval_loaders.items():
                p = predict(model, loader, device=self.device)
                out[f"{name}_acc"] = float((p.predicted == p.labels).mean())
            return out

        return evaluate

    def anchor_records(self, seed: int) -> list:
        """The oracle and original arms' own curve AUCs, as records under their own run_ids.

        Stage 6's gate for relearning is that the original arm normalises to ~1 and the oracle
        arm to ~0. That check needs the anchors' AUCs, and a runner that only recorded the method
        arm would leave Stage 7 unable to verify the protocol it depends on. Written once per
        (condition, seed) -- the first target with that seed carries them, later ones skip.
        """
        from ..unlearn import base_run_id_for
        from .relearning import curve_auc

        if getattr(self, "_anchor_written", None) is None:
            self._anchor_written = set()
        if seed in self._anchor_written:
            return []
        self._anchor_written.add(seed)

        spec = self.ctx.spec(self.forget_id)
        arms = self.relearn_anchors(seed)
        ids = {
            "oracle": run_id(role="oracle", forget=self.forget_id, seed=seed, seed_kind="train"),
            "original": base_run_id_for(spec, seed),
        }
        out = []
        for arm, rid in ids.items():
            curve = arms.get(arm)
            if not curve:
                continue
            auc = curve_auc(curve["steps"], curve["forget_acc"])
            if not np.isfinite(auc):
                continue
            fields = spec.as_record_fields()
            fields["forget_size"] = int(self.probes.forget.size)
            out.append(make_record(
                run_id=rid, audit="relearning", metric="relearn_auc", probe_set="forget",
                value=float(auc), n_probe=int(self.probes.forget.size),
                audit_seed=int(self.ctx.seeds.get("audit", 0)),
                notes=f"relearning anchor arm '{arm}' for condition {self.forget_id}",
                **fields,
            ))
        return out

    def relearn_anchors(self, seed: int) -> dict[str, dict[str, np.ndarray]]:
        """The oracle, original and random-init arms for one (condition, train seed).

        Cached per seed, not per target: a condition's ~30 targets share five seeds, so the
        anchors are computed five times rather than thirty. The random-init arm is a per-condition
        protocol check (configs/audits.yaml `randinit_sanity_check`) and is computed once.
        """
        if seed not in self._relearn_anchors:
            from ..unlearn import base_run_id_for

            arms: dict[str, dict[str, np.ndarray]] = {}
            spec = self.ctx.spec(self.forget_id)
            pairs = (
                ("oracle", run_id(role="oracle", forget=self.forget_id, seed=seed, seed_kind="train")),
                ("original", base_run_id_for(spec, seed)),
            )
            for arm, rid in pairs:
                if self.ctx.store.has_checkpoint(rid):
                    state, _ = self.ctx.store.load_checkpoint(rid)
                    arms[arm] = self.relearn_arm(state, seed=seed)
                else:
                    self.missing.append(rid)
            if self.ctx.audits["relearning"].get("randinit_sanity_check", True):
                if self._randinit_curve is None:
                    self._randinit_curve = self.relearn_arm(None, seed=0)
                arms["randinit"] = self._randinit_curve
            self._relearn_anchors[seed] = arms
        return self._relearn_anchors[seed]

    def oracles(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if self._oracles is None:
            logits: dict[str, list[np.ndarray]] = {}
            acts: dict[str, list[np.ndarray]] = {}
            for seed in self.ctx.base["oracles"]["paired_seeds"]:
                rid = run_id(role="oracle", forget=self.forget_id, seed=seed, seed_kind="train")
                got = self._eval(rid, with_activations=True, cache=True)
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
                if self.spec.kind == "canary":
                    # Shadows trained on clean labels never saw the canary association, whether
                    # or not the example's *image* was in their half. For the corrupted label
                    # every shadow is OUT, and all 32 may serve as the reference.
                    in_forget.append(np.zeros(self.probes.forget.size, dtype=bool))
                else:
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
    outputs = cache._eval(target.run_id, with_activations=needs_acts, cache=True)

    oracle_logits, oracle_acts = cache.oracles()
    original = cache.original(target.seed)
    refs, ref_mask = cache.references()

    relearn_curves: dict[str, dict[str, np.ndarray]] = {}
    if "relearning" in audits:
        relearn_curves = dict(cache.relearn_anchors(target.seed))
        relearn_curves["method"] = cache.relearn_arm(state, seed=target.seed)

    records = []
    if "relearning" in audits:
        records += cache.anchor_records(target.seed)
    skipped: list[str] = []
    for name in audits:
        audit = get_audit(name)
        # A missing reference is reported, never imputed: an oracle-referenced audit run against
        # no oracle would emit a plausible number that means nothing.
        if audit.needs_oracles and not oracle_logits:
            skipped.append(f"{name} (no oracle ensemble in this store)")
            continue
        if audit.needs_references and not refs:
            skipped.append(f"{name} (no shadow models in this store)")
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
            relearn_curves=relearn_curves,
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
            hparams=meta.hparams_sha,  # make_record derives hparams_sha from this
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
    if skipped:
        print("\n      skipped: " + "; ".join(skipped), end="", flush=True)
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


def shard_conditions(conditions: Sequence[str], *, account: int, of: int) -> list[str]:
    """This account's share of the conditions.

    Sharded by *condition*, not by model, deliberately. Every condition carries a fixed setup
    cost -- five oracles, five originals, 32 shadows and eleven relearning anchors evaluated
    before the first target -- and a stripe over models would make every account pay that for
    every condition. Splitting whole conditions means each cache is built exactly once
    project-wide. With eight conditions and three accounts the split is 3/3/2, i.e. 90/90/60
    models; uneven, and cheaper than the alternative.
    """
    if not 1 <= account <= of:
        raise ValueError(f"account must be in 1..{of}, got {account}")
    ordered = sorted(conditions)
    return ordered[account - 1 :: of]


def _audit_suffix(name: str) -> str:
    return f"audit-{name}"


def audited_audits(ctxobj, run_id_: str) -> set[str]:
    """Which audits already have records for this model.

    One shard **per audit**, `<run_id>--audit-<name>.parquet`, so re-running one audit rewrites
    one file. The first Stage 6 runs wrote a single `<run_id>--audit.parquet` holding all six;
    with that layout `--audits sde --force` would have replaced a model's six-audit shard with
    an SDE-only one and silently destroyed the other five. Legacy combined shards are still
    recognised here and are split on the next write to that model (`_migrate_legacy_shard`).
    """
    from ..registry.records import shard_path

    done: set[str] = set()
    legacy = shard_path(ctxobj.records_dir, run_id_, suffix="audit")
    if legacy.is_file():
        import pyarrow.parquet as pq

        done |= set(pq.read_table(legacy, columns=["audit"])["audit"].to_pylist())
    for name in REGISTRY:
        if shard_path(ctxobj.records_dir, run_id_, suffix=_audit_suffix(name)).is_file():
            done.add(name)
    return done


def already_audited(ctxobj, run_id_: str, audits: Sequence[str] | None = None) -> bool:
    """True if every requested audit (default: all registered) has records for this model.

    The audit stage is resumable the same way the training stages are: a session that dies
    on model 40 of 90 costs the one in flight, and re-running the same command picks up there.
    """
    wanted = set(audits) if audits is not None else set(REGISTRY)
    return wanted <= audited_audits(ctxobj, run_id_)


def _migrate_legacy_shard(ctxobj, run_id_: str, *, keep_out: set[str]) -> None:
    """Split a combined `<run_id>--audit.parquet` into per-audit shards, then remove it.

    ``keep_out`` names the audits about to be rewritten; their legacy rows are dropped rather
    than copied, so the new shard is the only one. Done with pyarrow directly -- the rows were
    validated when first written, and rebuilding RunRecords through pandas would turn every
    nullable integer into a float NaN on the way.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from ..registry.records import shard_path

    legacy = shard_path(ctxobj.records_dir, run_id_, suffix="audit")
    if not legacy.is_file():
        return
    table = pq.read_table(legacy)
    for name in set(table["audit"].to_pylist()):
        if name in keep_out:
            continue
        target = shard_path(ctxobj.records_dir, run_id_, suffix=_audit_suffix(name))
        if target.is_file():
            continue  # a per-audit shard already exists and is newer by construction
        sub = table.filter(pc.equal(table["audit"], name))
        tmp = target.with_suffix(".parquet.tmp")
        pq.write_table(sub, tmp, compression="zstd")
        tmp.replace(target)
    legacy.unlink()


def run_audits(
    ctxobj,
    *,
    targets: Iterable[AuditTarget] | None = None,
    audits: Sequence[str] | None = None,
    device: str = "cpu",
    batch_size: int = 512,
    dry_run: bool = False,
    account: int = 1,
    of: int = 1,
    force: bool = False,
) -> int:
    """Audit this account's share of the conditions, grouped so caches are reused."""
    audits = tuple(audits or sorted(REGISTRY))
    targets = list(targets if targets is not None else available_targets(ctxobj.store))
    if not targets:
        print("no unlearn checkpoints in this store; nothing to audit")
        return 0

    by_condition: dict[str, list[AuditTarget]] = {}
    for t in targets:
        by_condition.setdefault(t.forget_id, []).append(t)
    mine = shard_conditions(list(by_condition), account=account, of=of)
    print(f"account {account} of {of}: conditions {mine}")

    written = failed = skipped_done = 0
    for forget_id in mine:
        group = by_condition[forget_id]
        # Per target, only the audits that are missing (or all of them under --force). A model
        # with five of six done gets the sixth, not a full redo.
        plan: list[tuple[AuditTarget, tuple[str, ...]]] = []
        for t in group:
            todo = audits if force else tuple(a for a in audits if a not in audited_audits(ctxobj, t.run_id))
            if todo:
                plan.append((t, todo))
            else:
                skipped_done += 1
        if not plan:
            print(f"\n{forget_id}: all targets already audited", flush=True)
            continue
        group = [t for t, _ in plan]
        todo_by_target = {t.run_id: a for t, a in plan}
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
            todo = todo_by_target[t.run_id]
            tag = "" if len(todo) == len(audits) else f" [{', '.join(todo)}]"
            print(f"  [{i}/{len(group)}] {t.run_id}{tag} ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                records = audit_one(
                    t, ctxobj=ctxobj, cache=cache, audits=todo,
                    device=device, batch_size=batch_size,
                )
            except Exception as exc:  # one bad target must not lose the whole condition
                failed += 1
                print(f" FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            # One shard per (run_id, audit) for the target, so a partial re-run touches only
            # its own files. Relearning anchors go under the oracle's and original's own ids
            # with the condition in the suffix: the five clean base models serve seven
            # conditions each, and a suffix without it made every condition overwrite the
            # previous one's (the first full run kept 40 anchor rows where there should have
            # been 60). Stage 7 passing an oracle *through* the audits writes `--audit-*`
            # shards and cannot collide with its anchor shard.
            _migrate_legacy_shard(ctxobj, t.run_id, keep_out=set(todo))
            by_key: dict[tuple[str, str], list] = {}
            for r in records:
                by_key.setdefault((r.run_id, r.audit), []).append(r)
            for (rid, audit_name), rows in by_key.items():
                if rid == t.run_id:
                    suffix = _audit_suffix(audit_name)
                else:
                    suffix = f"relearn-anchor-{forget_id}"
                write_records(rows, ctxobj.records_dir, suffix=suffix)
            written += len(records)
            print(f" {len(records)} records in {time.perf_counter() - t0:.1f}s", flush=True)

        if cache.missing:
            uniq = sorted(set(cache.missing))
            print(f"  {forget_id}: {len(uniq)} reference checkpoint(s) not in this store, "
                  f"e.g. {uniq[0]}. Oracle/shadow-dependent audits were skipped above; attach "
                  f"the Stage 3 and Stage 4 artifact datasets and re-run.", flush=True)

    done_note = f", {skipped_done} already audited" if skipped_done else ""
    print(f"\nwrote {written} records, {failed} targets failed{done_note}")
    return 1 if failed else 0
