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
