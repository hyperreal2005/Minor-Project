"""Stage 1 gate: the record contract holds.

The gate in the implementation plan reads: "a dummy run writes a valid record shard;
validate() rejects an unregistered metric name and a null required field."

These tests encode the failure modes the contract exists to prevent — mainly the ones that would
otherwise surface in week 14, when four people's shards refuse to join.
"""

import math

import pyarrow.parquet as pq
import pytest

from forgetcheck.registry import (
    SCHEMA,
    RecordError,
    RunRecord,
    default_registry,
    make_record,
    read_records,
    run_id,
    validate,
    write_records,
)


UNLEARN_ID = run_id(role="unlearn", forget="mem-high-3000", method="scrub", seed=2)


def a_record(**over) -> RunRecord:
    """A valid baseline record; override one field per test to isolate a failure mode."""
    base = dict(
        run_id=UNLEARN_ID,
        role="unlearn",
        dataset="cifar10",
        arch="resnet18",
        forget_id="mem-high-3000",
        forget_kind="memstratum",
        forget_size=3000,
        forget_stratum="high",
        method="scrub",
        audit="behavior",
        metric="retain_acc",
        probe_set="retain",
        value=0.934,
        n_probe=47000,
        train_seed=2,
        selection_seed=100,
        hparams_sha="deadbeefdeadbeef",
    )
    base.update(over)
    return RunRecord(**base)


class TestValidation:
    def test_baseline_is_valid(self):
        validate(a_record())

    def test_unregistered_metric_rejected(self):
        # The plan's gate, literally.
        with pytest.raises(KeyError, match="not registered"):
            validate(a_record(metric="retain_accuracy"))

    def test_probe_set_must_be_declared_by_the_metric(self):
        # retain_acc declares probe_sets: [retain]. Writing it against 'forget' is a real bug
        # that would otherwise produce a plausible-looking column.
        with pytest.raises(RecordError, match="does not declare probe_set"):
            validate(a_record(probe_set="forget"))

    def test_nan_rejected(self):
        with pytest.raises(RecordError, match="silently poisons"):
            validate(a_record(value=math.nan))

    def test_inf_rejected(self):
        with pytest.raises(RecordError):
            validate(a_record(value=math.inf))

    def test_zero_probe_rejected(self):
        with pytest.raises(RecordError, match="not a measurement"):
            validate(a_record(n_probe=0))

    def test_unknown_audit_rejected(self):
        with pytest.raises(RecordError, match="is not one of"):
            validate(a_record(audit="representations"))

    def test_role_must_match_run_id(self):
        with pytest.raises(RecordError, match="contradicts the run_id"):
            validate(a_record(role="oracle"))

    def test_method_must_match_run_id(self):
        with pytest.raises(RecordError, match="contradicts the run_id"):
            validate(a_record(method="salun"))

    def test_seed_must_match_run_id(self):
        with pytest.raises(RecordError, match="contradicts the run_id"):
            validate(a_record(train_seed=3))

    def test_missing_train_seed_rejected(self):
        with pytest.raises(RecordError, match="train_seed is None"):
            validate(a_record(train_seed=None))

    def test_memstratum_requires_stratum(self):
        with pytest.raises(RecordError, match="requires forget_stratum"):
            validate(a_record(forget_stratum=None))

    def test_bad_stratum_rejected(self):
        with pytest.raises(RecordError, match="must be low, medium or high"):
            validate(a_record(forget_stratum="highest"))

    def test_forget_set_requires_selection_seed(self):
        # Without it the forget set is not reproducible, so neither is the result.
        with pytest.raises(RecordError, match="requires selection_seed"):
            validate(a_record(selection_seed=None))

    def test_oracle_ensemble_seed_checked(self):
        rid = run_id(role="oracle", forget="mem-high-3000", seed=205, seed_kind="oracle")
        ok = a_record(
            run_id=rid, role="oracle", method="none", train_seed=None, oracle_seed=205
        )
        validate(ok)
        with pytest.raises(RecordError, match="oracle_seed is None"):
            validate(
                a_record(run_id=rid, role="oracle", method="none", train_seed=None,
                         oracle_seed=None)
            )

    def test_shadow_index_checked(self):
        rid = run_id(role="shadow", forget="full", seed=7, seed_kind="shadow")
        rec = a_record(
            run_id=rid,
            role="shadow",
            method="none",
            forget_id="full",
            forget_kind="none",
            forget_size=0,
            forget_stratum=None,
            train_seed=None,
            selection_seed=None,
            shadow_idx=7,
        )
        validate(rec)
        with pytest.raises(RecordError, match="shadow_idx is None"):
            validate(
                a_record(
                    run_id=rid, role="shadow", method="none", forget_id="full",
                    forget_kind="none", forget_size=0, forget_stratum=None,
                    train_seed=None, selection_seed=None, shadow_idx=None,
                )
            )


class TestWriting:
    def test_shard_round_trips(self, tmp_path):
        rows = [
            a_record(),
            a_record(metric="test_acc", probe_set="test", value=0.921, n_probe=10000),
            a_record(metric="forget_acc", probe_set="forget", value=0.88, n_probe=3000),
        ]
        path = write_records(rows, tmp_path)
        assert path.is_file()

        df = read_records(tmp_path)
        assert len(df) == 3
        assert set(df["metric"]) == {"retain_acc", "test_acc", "forget_acc"}
        assert df["run_id"].nunique() == 1

    def test_written_schema_is_exact(self, tmp_path):
        # Shards from four people must be concatenable, which requires identical types. Letting
        # pyarrow infer produces int64 in one shard and null in another the moment a column
        # happens to be all-None.
        write_records([a_record()], tmp_path)
        written = pq.read_table(next(tmp_path.glob("*.parquet"))).schema
        assert written.names == SCHEMA.names
        for name in SCHEMA.names:
            assert written.field(name).type == SCHEMA.field(name).type, name

    def test_all_none_column_keeps_its_type(self, tmp_path):
        # The specific inference failure the explicit schema prevents.
        write_records([a_record(oracle_seed=None, shadow_idx=None)], tmp_path)
        t = pq.read_table(next(tmp_path.glob("*.parquet")))
        assert t.schema.field("oracle_seed").type == SCHEMA.field("oracle_seed").type

    def test_shards_from_different_runs_concatenate(self, tmp_path):
        write_records([a_record()], tmp_path)
        other = run_id(role="unlearn", forget="rand-500", method="salun", seed=0)
        write_records(
            [a_record(run_id=other, method="salun", train_seed=0,
                      forget_id="rand-500", forget_kind="random",
                      forget_size=500, forget_stratum=None)],
            tmp_path,
        )
        df = read_records(tmp_path)
        assert len(df) == 2 and df["run_id"].nunique() == 2

    def test_mixed_run_ids_in_one_shard_rejected(self, tmp_path):
        other = run_id(role="unlearn", forget="rand-500", method="salun", seed=0)
        with pytest.raises(RecordError, match="a shard holds one run"):
            write_records([a_record(), a_record(run_id=other, method="salun")], tmp_path)

    def test_duplicate_measurement_rejected(self, tmp_path):
        with pytest.raises(RecordError, match="duplicate"):
            write_records([a_record(), a_record(value=0.7)], tmp_path)

    def test_empty_batch_rejected(self, tmp_path):
        with pytest.raises(RecordError, match="empty shard"):
            write_records([], tmp_path)

    def test_nothing_written_when_a_row_is_invalid(self, tmp_path):
        # A shard is never half-valid: validation happens before any bytes are written.
        with pytest.raises(KeyError):
            write_records([a_record(), a_record(metric="bogus_metric")], tmp_path)
        assert list(tmp_path.glob("*.parquet")) == []

    def test_suffix_separates_audit_shards(self, tmp_path):
        # Two audits writing records for the same run must not race for one filename.
        write_records([a_record()], tmp_path, suffix="behavior")
        write_records(
            [a_record(audit="representation", metric="cka_linear", probe_set="layer3",
                      value=0.81, n_probe=3000)],
            tmp_path,
            suffix="representation",
        )
        assert len(list(tmp_path.glob("*.parquet"))) == 2
        assert len(read_records(tmp_path)) == 2

    def test_no_leftover_temp_files(self, tmp_path):
        write_records([a_record()], tmp_path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_empty_directory_raises_rather_than_returning_empty(self, tmp_path):
        # An empty analysis that looks like a real one is worse than a crash.
        with pytest.raises(FileNotFoundError, match="no .parquet shards"):
            read_records(tmp_path)


class TestMakeRecord:
    def test_derives_identity_from_run_id(self):
        rec = make_record(
            run_id=UNLEARN_ID,
            audit="behavior",
            metric="retain_acc",
            probe_set="retain",
            value=0.93,
            n_probe=47000,
            forget_kind="memstratum",
            forget_size=3000,
            forget_stratum="high",
            selection_seed=100,
            hparams={"lr": 0.01, "epochs": 5},
        )
        assert rec.role == "unlearn"
        assert rec.method == "scrub"
        assert rec.train_seed == 2
        assert rec.forget_id == "mem-high-3000"
        assert len(rec.hparams_sha) == 16
        validate(rec)

    def test_hparams_hash_is_key_order_independent(self):
        kw = dict(
            run_id=UNLEARN_ID, audit="behavior", metric="retain_acc", probe_set="retain",
            value=0.9, n_probe=100, forget_kind="memstratum", forget_size=3000,
            forget_stratum="high", selection_seed=100,
        )
        a = make_record(**kw, hparams={"lr": 0.01, "epochs": 5})
        b = make_record(**kw, hparams={"epochs": 5, "lr": 0.01})
        assert a.hparams_sha == b.hparams_sha

    def test_provenance_is_auto_filled(self):
        rec = a_record()
        assert rec.git_commit  # a sha or the 'nogit' sentinel, never empty
        assert len(rec.env_sha) == 16
        assert rec.timestamp.endswith("+00:00")


def test_registry_directions_are_sane():
    """closer_to_oracle metrics must be oracle-referenced, and forget_acc must be one."""
    reg = default_registry()
    assert reg["forget_acc"].direction == "closer_to_oracle", (
        "forget_acc is not lower_better: a retrained model still classifies most forgotten "
        "examples correctly (master reference §3.2)."
    )
    for spec in reg:
        if spec.direction == "closer_to_oracle":
            assert spec.oracle_ref, f"{spec.name} is closer_to_oracle but not oracle_ref"
