"""Stage 6 end to end: real checkpoints, a real (synthetic) bundle, the real runner.

Every other audit test hands the audits a hand-built AuditContext. That verified the audits and
missed the glue: `run_audits` reached Kaggle and died on `bundle.n_test`, an attribute that had
never existed, with 480 tests green. This test drives the same entry point the CLI does, over a
store holding actual ResNet-18 checkpoints, so the next mismatch between runner and data layer
fails on a laptop in a minute rather than on a GPU after the setup cells.

Sizes are the minimum that exercise every branch: one target, one paired oracle, one original,
two shadows, a 3-step relearning schedule.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


from stage6_fixtures import K, N_TEST, N_TRAIN, _Ctx, _bundle, _seed_store  # noqa: F401


def test_run_audits_end_to_end(tmp_path, capsys):
    from forgetcheck.audits import audit_names
    from forgetcheck.audits.runner import run_audits
    from forgetcheck.registry import read_records

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)

    rc = run_audits(ctx, device="cpu", batch_size=64)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "skipped:" not in out, f"a reference was missing:\n{out}"

    df = read_records(ctx.records_dir)
    got = df[df["run_id"] == target]
    assert set(got["audit"]) == set(audit_names()), (
        f"audits with no rows for the target: {set(audit_names()) - set(got['audit'])}"
    )
    # Relearning anchors are recorded under their own run_ids, not the target's, and the
    # random-init floor rides with the seed-0 oracle's anchors.
    anchors = df[(df["audit"] == "relearning") & (df["run_id"] != target)]
    assert set(anchors["role"]) == {"oracle", "base"}, anchors[["run_id", "metric"]]
    floor = anchors[anchors["metric"] == "relearn_randinit_auc"]
    assert len(floor) == 1 and floor["role"].iloc[0] == "oracle"

    # The caches Stage 6 is supposed to leave behind.
    assert ctx.store.has_outputs(target)
    assert ctx.store.has_activations(target)


def test_a_second_run_skips_audited_models_and_reads_the_cache(tmp_path, capsys):
    from forgetcheck.audits.runner import run_audits

    ctx = _Ctx(tmp_path)
    _seed_store(ctx)
    run_audits(ctx, device="cpu", batch_size=64)
    capsys.readouterr()

    rc = run_audits(ctx, device="cpu", batch_size=64)
    out = capsys.readouterr().out
    assert rc == 0
    assert "already audited" in out

    rc = run_audits(ctx, device="cpu", batch_size=64, force=True)
    out = capsys.readouterr().out
    assert rc == 0 and "records in" in out


def test_a_partial_rerun_touches_only_its_own_audit(tmp_path, capsys):
    """`--audits sde --force` must rewrite SDE's rows and nothing else. With one combined shard
    per model -- the first Stage 6 layout -- it replaced the six-audit shard with an SDE-only one
    and destroyed the other five audits' rows."""
    from forgetcheck.audits import audit_names
    from forgetcheck.audits.runner import audited_audits, run_audits
    from forgetcheck.registry import read_records

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)
    run_audits(ctx, device="cpu", batch_size=64)
    before = read_records(ctx.records_dir)
    before = before[before["run_id"] == target]
    n_other = len(before[before["audit"] != "sde"])
    capsys.readouterr()

    rc = run_audits(ctx, device="cpu", batch_size=64, audits=["sde"], force=True)
    assert rc == 0
    after = read_records(ctx.records_dir)
    after = after[after["run_id"] == target]
    assert set(after["audit"]) == set(audit_names()), "other audits' rows were lost"
    assert len(after[after["audit"] != "sde"]) == n_other
    assert audited_audits(ctx, target) == set(audit_names())


def test_legacy_combined_shards_are_split_not_duplicated(tmp_path, capsys):
    """Accounts 1 and 2 wrote `<run_id>--audit.parquet` before per-audit shards existed. On the
    next write to that model the combined shard is split; its rows must appear exactly once."""
    from forgetcheck.audits.runner import audited_audits, run_audits
    from forgetcheck.registry import read_records, write_records
    from forgetcheck.registry.records import shard_path

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)
    run_audits(ctx, device="cpu", batch_size=64)
    df = read_records(ctx.records_dir)
    rows_before = len(df[df["run_id"] == target])

    # Reconstruct the legacy layout from the per-audit shards, then delete them.
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = []
    for f in sorted(ctx.records_dir.glob(f"{target}--audit-*.parquet")):
        tables.append(pq.read_table(f))
        f.unlink()
    pq.write_table(pa.concat_tables(tables), shard_path(ctx.records_dir, target, suffix="audit"))
    assert audited_audits(ctx, target), "the legacy shard must still count as audited"
    capsys.readouterr()

    run_audits(ctx, device="cpu", batch_size=64, audits=["behavior"], force=True)
    df = read_records(ctx.records_dir)
    got = df[df["run_id"] == target]
    assert len(got) == rows_before, "rows were duplicated or lost in migration"
    assert not shard_path(ctx.records_dir, target, suffix="audit").exists()


def test_an_unreadable_cache_file_is_recomputed_not_fatal(tmp_path, capsys):
    """An empty `.npz` in the outputs cache -- what a dataset upload of symlinks produces --
    failed all 30 canary models with EOFError. A cache must never be able to fail a model."""
    from forgetcheck.audits.runner import run_audits

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)
    run_audits(ctx, device="cpu", batch_size=64)
    capsys.readouterr()

    # Truncate the cache to zero bytes, as a zipped symlink comes back.
    ctx.store.outputs_path(target).write_bytes(b"")
    assert ctx.store.has_outputs(target)

    rc = run_audits(ctx, device="cpu", batch_size=64, audits=["behavior"], force=True)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 targets failed" in out
    assert "unreadable" in out and "EOFError" in out
    assert ctx.store.outputs_path(target).stat().st_size > 0, "the bad file must be overwritten"
    ctx.store.load_outputs(target, forget_id="rand-500")  # and readable again


def test_migrate_then_delete_one_audit_is_a_resumable_redo(tmp_path, capsys):
    """The resumable alternative to --force: migrate combined shards, delete one audit's shards,
    run without force. A restart must skip models already redone."""
    from forgetcheck.audits import audit_names
    from forgetcheck.audits.runner import audited_audits, run_audits
    from forgetcheck.registry.records import shard_path
    import pyarrow as pa
    import pyarrow.parquet as pq

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)
    run_audits(ctx, device="cpu", batch_size=64)

    # Rebuild the first-pass layout: one combined shard.
    tables = []
    for f in sorted(ctx.records_dir.glob(f"{target}--audit-*.parquet")):
        tables.append(pq.read_table(f)); f.unlink()
    pq.write_table(pa.concat_tables(tables), shard_path(ctx.records_dir, target, suffix="audit"))
    capsys.readouterr()

    assert run_audits(ctx, migrate_only=True) == 0
    assert "migrated 1 combined" in capsys.readouterr().out
    assert not shard_path(ctx.records_dir, target, suffix="audit").exists()
    assert audited_audits(ctx, target) == set(audit_names())

    shard_path(ctx.records_dir, target, suffix="audit-relearning").unlink()
    assert "relearning" not in audited_audits(ctx, target)

    rc = run_audits(ctx, device="cpu", batch_size=64, audits=["relearning"])  # no force
    out = capsys.readouterr().out
    assert rc == 0 and "records in" in out
    assert "relearning" in audited_audits(ctx, target)

    rc = run_audits(ctx, device="cpu", batch_size=64, audits=["relearning"])  # the "restart"
    assert "already audited" in capsys.readouterr().out


def test_an_oracle_audited_as_a_candidate_is_excluded_from_its_own_reference(tmp_path, capsys):
    """Stage 7's null band comes from auditing genuine retrains. The reference ensemble at every
    condition IS the paired oracles, so a paired oracle scored against all of them is compared
    with itself -- one JS divergence of exactly 0, one CKA of exactly 1 -- and the band comes out
    tighter than the truth, making every audit look more valid than it is."""
    import torch

    from forgetcheck.audits.probes import build_probes
    from forgetcheck.audits.runner import _ConditionCache
    from forgetcheck.models.resnet import make_resnet18
    from forgetcheck.registry import run_id

    ctx = _Ctx(tmp_path)
    ctx.base["oracles"]["paired_seeds"] = [0, 1]
    torch.manual_seed(0)
    ids = [run_id(role="oracle", forget="rand-500", seed=s, seed_kind="train") for s in (0, 1)]
    for rid in ids:
        ctx.store.save_checkpoint(rid, make_resnet18(num_classes=K).state_dict(),
                                  train_seed=0, hparams_sha="e2e")

    probes = build_probes(forget_indices=ctx.forget_indices("rand-500"),
                          n_train=ctx.bundle.n_train, n_test=ctx.bundle.n_test,
                          forget_id="rand-500", config=ctx.audits["representation"])
    cache = _ConditionCache(ctx, forget_id="rand-500", probes=probes, layers=("layer4",),
                            device="cpu", batch_size=64)

    full, _ = cache.oracles()
    assert len(full["forget"]) == 2, "both oracles form the reference for an unlearned model"

    loo, loo_acts = cache.oracles(exclude=ids[0])
    assert len(loo["forget"]) == 1, "a candidate must not be in its own reference"
    assert len(loo_acts["layer4"]) == 1
    # And the forward passes are shared, not repeated, between the two views.
    assert len(cache._oracle_by_rid) == 2

    # A candidate that is not in the ensemble keeps the whole reference.
    other, _ = cache.oracles(exclude="c10r18__oracle__rand-500__none__oracle209")
    assert len(other["forget"]) == 2


def test_an_ensemble_oracle_has_no_paired_original_and_borrows_anchors(tmp_path):
    """Seeds 200-211 are independent retrains that pair with no base model. Asking for one would
    invent a checkpoint; `js_to_original` is simply undefined for them."""
    from forgetcheck.audits.probes import build_probes
    from forgetcheck.audits.runner import _ConditionCache

    ctx = _Ctx(tmp_path)
    probes = build_probes(forget_indices=ctx.forget_indices("rand-500"),
                          n_train=ctx.bundle.n_train, n_test=ctx.bundle.n_test,
                          forget_id="rand-500", config=ctx.audits["representation"])
    cache = _ConditionCache(ctx, forget_id="rand-500", probes=probes, layers=("layer4",),
                            device="cpu", batch_size=64)
    ctx.seeds = {"audit": 0, "train": [0]}

    assert cache.original(209) is None
    assert not cache.missing, "a phantom checkpoint must not be reported as missing"


def _seed_canary_store(ctx):
    """Canary condition: two oracles, one M0, one unlearned model, two shadows."""
    import torch

    from forgetcheck.models.resnet import make_resnet18
    from forgetcheck.registry import run_id
    from forgetcheck.unlearn import base_run_id_for

    ctx.base["oracles"]["paired_seeds"] = [0, 1]
    ids = {
        "unlearn": "c10r18__unlearn__canary-500__finetune__train0",
        "oracle0": run_id(role="oracle", forget="canary-500", seed=0, seed_kind="train"),
        "oracle1": run_id(role="oracle", forget="canary-500", seed=1, seed_kind="train"),
        "base": base_run_id_for(ctx.spec("canary-500"), 0),
        "shadow0": run_id(role="shadow", forget="full", seed=0, seed_kind="shadow"),
        "shadow1": run_id(role="shadow", forget="full", seed=1, seed_kind="shadow"),
    }
    for i, rid in enumerate(ids.values()):
        torch.manual_seed(i)
        ctx.store.save_checkpoint(rid, make_resnet18(num_classes=K).state_dict(),
                                  train_seed=0, hparams_sha="e2e")
    return ids


def test_stage7_oracles_m0_and_canary_ground_truth_end_to_end(tmp_path, capsys):
    """The whole Stage 7 path through the real runner: unlearned models (already audited in
    Stage 6), then oracles as candidates, then M0 under its condition -- with canary ground truth
    emitted on the way through -- then calibration over the records they wrote."""
    from forgetcheck.audits.runner import (
        audited_audits, available_targets, base_targets, run_audits,
    )
    from forgetcheck.calibrate import CalibrationConfig, calibrate, load_audit_records
    from forgetcheck.registry.metrics import default_registry

    ctx = _Ctx(tmp_path, forget_id="canary-500", n_forget=40)
    ids = _seed_canary_store(ctx)

    # Stage 6 without ground truth, as it actually ran.
    assert run_audits(ctx, device="cpu", batch_size=64, ground_truth=False) == 0
    capsys.readouterr()

    # A plain re-run now picks up ground truth for the already-audited model, auditing nothing.
    assert run_audits(ctx, device="cpu", batch_size=64) == 0
    out = capsys.readouterr().out
    line = out.split(ids["unlearn"])[1].splitlines()[0]
    assert "[ground truth only]" in line and " 3 records" in line, line

    assert run_audits(ctx, targets=available_targets(ctx.store, role="oracle"),
                      device="cpu", batch_size=64) == 0
    m0 = base_targets(ctx)
    assert [t.forget_id for t in m0] == ["canary-500"] and m0[0].role == "base"
    assert run_audits(ctx, targets=m0, device="cpu", batch_size=64) == 0
    assert audited_audits(ctx, ids["base"], condition="canary-500")

    df = load_audit_records(ctx.records_dir)
    gt = df[df["metric"] == "canary_top_wrong"]
    assert set(gt["role"]) == {"unlearn", "oracle", "base"}, gt[["run_id", "role"]]

    tables = calibrate(df, registry=default_registry(),
                       config=CalibrationConfig(primary_condition="mem-high-3000"))
    assert set(tables["canary"]["role"]) == {"unlearn", "oracle", "base"}
    truth = tables["canary"].set_index("role")["ground_truth"]
    assert truth["oracle"].tolist() == [False, False] and bool(truth["base"]) is True
    v = tables["validity"]
    assert {"native_fpr", "native_tpr"} <= set(v.columns)
    # Two oracles are too few for a band (MIN_BAND_N = 3): calibrated rates must be absent, not
    # zero -- a rate computed from nothing would read as a perfect audit.
    assert "calibrated_fpr" not in v.columns or v["calibrated_fpr"].isna().all()


def test_an_ensemble_oracle_writes_anchors_only_under_real_run_ids(tmp_path, capsys):
    """Ensemble oracles (seeds 200-211) have no paired M0 and borrow seed 0's anchors. The curves
    were borrowed correctly, but the anchor rows were written under run_ids built from the raw
    seed -- `...__train200` -- which do not exist: phantom 'oracles' and 'M0s' that calibration
    would have counted, collapsing the relearn_auc band. Every anchor row must name a real
    checkpoint, and calibration must read the ensemble oracle's null oracle_seed correctly."""
    import torch

    from forgetcheck.audits.runner import available_targets, run_audits
    from forgetcheck.calibrate import CalibrationConfig, calibrate, load_audit_records
    from forgetcheck.models.resnet import make_resnet18
    from forgetcheck.registry import run_id
    from forgetcheck.registry.metrics import default_registry

    ctx = _Ctx(tmp_path)
    target = _seed_store(ctx)                       # unlearn target, oracle0, base0, 2 shadows
    ens = run_id(role="oracle", forget="rand-500", seed=200, seed_kind="oracle")
    torch.manual_seed(7)
    ctx.store.save_checkpoint(ens, make_resnet18(num_classes=K).state_dict(),
                              train_seed=0, hparams_sha="e2e")

    oracles = available_targets(ctx.store, role="oracle")
    assert [t.run_id for t in oracles][0] == ens, "the ensemble oracle must come first to test this"
    assert run_audits(ctx, targets=oracles, device="cpu", batch_size=64) == 0, capsys.readouterr().out

    df = load_audit_records(ctx.records_dir)
    phantom = df[df["run_id"].str.contains("train200")]
    assert phantom.empty, phantom[["run_id", "metric", "notes"]]
    for rid in df.loc[df["audit"] == "relearning", "run_id"].unique():
        assert ctx.store.has_checkpoint(rid), f"relearning row under a non-existent model: {rid}"

    tables = calibrate(df, registry=default_registry(),
                       config=CalibrationConfig(primary_condition="rand-500"))
    assert len(tables["validity"]) > 0
