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
import yaml

from forgetcheck.config import find_configs

pytestmark = pytest.mark.slow


N_TRAIN, N_TEST, K = 240, 120, 10


def _bundle():
    from forgetcheck.data.cifar import DataBundle

    rng = np.random.default_rng(0)
    return DataBundle(
        name="cifar10",
        train_x=rng.integers(0, 256, size=(N_TRAIN, 32, 32, 3), dtype=np.uint8),
        train_y=(np.arange(N_TRAIN) % K).astype(np.int64),
        test_x=rng.integers(0, 256, size=(N_TEST, 32, 32, 3), dtype=np.uint8),
        test_y=(np.arange(N_TEST) % K).astype(np.int64),
        sha="synthetic",
        num_classes=K,
    )


class _Ctx:
    """The slice of `forgetcheck.config.Context` the runner touches, over synthetic data."""

    def __init__(self, tmp_path, forget_id="rand-500", n_forget=40):
        from forgetcheck.data.forget_sets import spec_by_id
        from forgetcheck.registry import ArtifactStore

        self.root = tmp_path
        self.store = ArtifactStore(tmp_path / "artifacts")
        self.records_dir = tmp_path / "results" / "records"
        self.records_dir.mkdir(parents=True)
        self.bundle = _bundle()
        self._spec = spec_by_id(forget_id)
        self._fidx = np.arange(n_forget, dtype=np.int64)
        self.base = {
            "oracles": {"paired_seeds": [0]},
            "shadows": {"count": 2, "subset_fraction": 0.5},
        }
        self.seeds = {"audit": 0}
        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        cfg["relearning"]["eval_steps"] = [0, 1, 2]
        cfg["relearning"]["reintroduction_size"] = 16
        cfg["relearning"]["batch_size"] = 8
        cfg["representation"]["probe_size"] = 60
        cfg["representation"]["retain_probe_size"] = 60
        cfg["sde"]["n_draws"] = 10
        cfg["privacy"]["population"]["n_splits"] = 2
        self.audits = cfg

    def spec(self, forget_id):
        return self._spec

    def forget_indices(self, forget_id):
        return self._fidx


def _seed_store(ctx):
    """Write every checkpoint the runner will look for, from one random ResNet-18."""
    import torch

    from forgetcheck.models.resnet import make_resnet18
    from forgetcheck.registry import run_id
    from forgetcheck.unlearn import base_run_id_for

    torch.manual_seed(0)
    state = make_resnet18(num_classes=K).state_dict()
    ids = [
        "c10r18__unlearn__rand-500__finetune__train0",
        run_id(role="oracle", forget="rand-500", seed=0, seed_kind="train"),
        base_run_id_for(ctx.spec("rand-500"), 0),
        run_id(role="shadow", forget="full", seed=0, seed_kind="shadow"),
        run_id(role="shadow", forget="full", seed=1, seed_kind="shadow"),
    ]
    for rid in ids:
        ctx.store.save_checkpoint(rid, state, train_seed=0, hparams_sha="e2e")
    return ids[0]


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
    # Relearning anchors are recorded under their own run_ids, not the target's.
    anchors = df[(df["audit"] == "relearning") & (df["run_id"] != target)]
    assert set(anchors["role"]) == {"oracle", "base"}, anchors[["run_id", "metric"]]

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
