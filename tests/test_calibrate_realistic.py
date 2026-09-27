"""Stage 7 calibration over a record set shaped like the real one.

`test_calibrate.py` checks the statistics on hand-built DataFrames. The first Kaggle calibration
still died, on `oracles["value"]`: real records come back through Parquet with pandas-chosen
dtypes and in combinations no hand-built frame had -- here, a `relearn_norm` group with unlearned
rows but *no* oracle rows, because every oracle's value was undefined at that condition. This
test builds the record set with the real structure, writes it through the real writer, reads it
back through the real loader, and runs the real `calibrate` command and the notebook's results
cell, so the next mismatch between the calibration and the data fails here, on CPU, in seconds.

The structure mirrored, all of it observed in the real run:

* eight conditions; per condition 30 unlearned models, 5 paired oracles, 5 M0 audits;
* 12 ensemble oracles (seeds 200-211) at the primary condition only;
* the five clean base models sharing one run_id across seven conditions;
* every audit metric from the registry, on its real probe sets;
* `relearn_norm` undefined for everyone at rand-500 and mem-low, and for every *oracle* but not
  every unlearned model at rand-2500 -- the case that crashed;
* `sde_verdict` binary; canary ground truth for all three roles; the random-init floor;
* relearning-anchor rows duplicated by later candidate rows, and Stage 5 `meta` rows that must be
  ignored.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from forgetcheck.data.forget_sets import spec_by_id
from forgetcheck.registry import make_record, run_id, write_records

CONDITIONS = ["canary-500", "mem-high-3000", "mem-low-3000", "mem-med-3000",
              "rand-2500", "rand-3000", "rand-500", "rand-5000"]
PRIMARY = "mem-high-3000"
METHODS = ["finetune", "l1sparse", "neggrad", "neggradplus", "salun", "scrub"]
SEEDS = range(5)

# (audit, metric, probe_sets) -- the rows each audited model carries.
AUDIT_ROWS = [
    ("behavior", m, ("forget", "retain", "test"))
    for m in ("js_to_oracle", "js_to_original", "pred_agreement", "logit_l2")
] + [
    ("privacy_population", m, ("forget",)) for m in ("mia_auc_pop", "mia_acc_pop", "mia_tpr_at_fpr_pop")
] + [
    ("privacy_rmia", m, ("forget",)) for m in ("mia_auc_rmia", "mia_acc_rmia", "mia_tpr_at_fpr_rmia")
] + [
    ("representation", m, ("layer1", "layer2", "layer3", "layer4"))
    for m in ("cka_linear", "cka_rbf", "activation_l2")
] + [
    ("relearning", "relearn_auc", ("forget",)),
    ("relearning", "relearn_norm", ("forget",)),
    ("relearning", "relearn_t80", ("forget",)),
    ("relearning", "relearn_utility_drop", ("retain", "test")),
] + [
    ("sde", m, ("forget",)) for m in ("sde_hsic", "sde_margin", "sde_verdict")
]


def _value(metric, role, rng):
    """Plausible values; the test is about structure and dtypes, not about the numbers."""
    shift = {"oracle": 0.0, "unlearn": 0.05, "base": 0.3}[role]
    if metric == "sde_verdict":
        return float(rng.random() < 0.8)
    if metric.startswith("mia_"):
        return float(np.clip(0.55 + shift / 3 + rng.normal(scale=0.02), 0, 1))
    if metric == "relearn_norm":
        return {"oracle": 0.0, "unlearn": 0.6, "base": 1.0}[role] + float(rng.normal(scale=0.05))
    if metric == "relearn_t80":
        return float(abs(rng.normal(20, 5)))
    if metric == "relearn_auc":  # accuracy-like, above the simulated 0.2 random-init floor
        return {"oracle": 0.70, "unlearn": 0.80, "base": 0.95}[role] + float(rng.normal(scale=0.02))
    return float(abs(0.1 + shift + rng.normal(scale=0.02)))


def _relearn_norm_undefined(cond, role):
    if cond in ("rand-500", "mem-low-3000"):
        return True                      # anchor gap below min_anchor_gap for every model
    return cond == "rand-2500" and role == "oracle"   # the combination that crashed


def _model_rows(rid, role, cond, rng, *, ts):
    fields = spec_by_id(cond).as_record_fields()
    rows = []
    for audit, metric, probes in AUDIT_ROWS:
        for probe in probes:
            if metric == "relearn_norm" and _relearn_norm_undefined(cond, role):
                rows.append(make_record(run_id=rid, audit=audit, metric="audit_undefined",
                                        probe_set=probe, value=1.0, n_probe=500,
                                        notes="undefined: relearn_norm", timestamp=ts, **fields))
                continue
            rows.append(make_record(run_id=rid, audit=audit, metric=metric, probe_set=probe,
                                    value=_value(metric, role, rng), n_probe=500,
                                    timestamp=ts, **fields))
    if cond == "canary-500":
        top = {"oracle": 0.11, "base": 0.97, "unlearn": 0.25}[role] + float(rng.normal(scale=0.01))
        for metric, v in (("canary_acc", top / 2), ("canary_prob", top / 3), ("canary_top_wrong", top)):
            rows.append(make_record(run_id=rid, audit="meta", metric=metric, probe_set="canary",
                                    value=float(v), n_probe=500, timestamp=ts, **fields))
    return rows


@pytest.fixture(scope="module")
def records_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage7")
    rec = root / "results" / "records"
    rec.mkdir(parents=True)
    rng = np.random.default_rng(0)
    early, late = "2026-09-20T00:00:00", "2026-09-26T00:00:00"

    for cond in CONDITIONS:
        spec = spec_by_id(cond)
        fields = spec.as_record_fields()
        base_forget = cond if spec.kind == "canary" else "full"
        for m in METHODS:
            for s in SEEDS:
                rid = f"c10r18__unlearn__{cond}__{m}__train{s}"
                write_records(_model_rows(rid, "unlearn", cond, rng, ts=early), rec, suffix="audit")
        for s in SEEDS:
            orc = run_id(role="oracle", forget=cond, seed=s, seed_kind="train")
            base = run_id(role="base", forget=base_forget, seed=s)
            write_records(_model_rows(orc, "oracle", cond, rng, ts=late), rec, suffix="audit-sim")
            write_records(_model_rows(base, "base", cond, rng, ts=late), rec,
                          suffix=f"audit-sim-{cond}")
            # Stage 6 anchor rows, superseded by the candidate rows written later.
            for arid in (orc, base):
                write_records([make_record(run_id=arid, audit="relearning", metric="relearn_auc",
                                           probe_set="forget", value=0.5, n_probe=500,
                                           notes="relearning anchor arm", timestamp=early,
                                           **fields)], rec, suffix=f"relearn-anchor-{cond}")
        write_records([make_record(run_id=run_id(role="oracle", forget=cond, seed=0, seed_kind="train"),
                                   audit="relearning", metric="relearn_randinit_auc",
                                   probe_set="forget", value=0.2, n_probe=500, timestamp=early,
                                   **fields)], rec, suffix=f"floor-{cond}")
    for s in range(200, 212):
        rid = run_id(role="oracle", forget=PRIMARY, seed=s, seed_kind="oracle")
        write_records(_model_rows(rid, "oracle", PRIMARY, rng, ts=late), rec, suffix="audit-sim")
    # A Stage 5 training-time row: audit="meta", must not enter calibration.
    write_records([make_record(run_id="c10r18__unlearn__rand-500__salun__train0", audit="meta",
                               metric="forget_acc", probe_set="forget", value=0.9, n_probe=500,
                               **spec_by_id("rand-500").as_record_fields())], rec, suffix="unlearn")
    return rec


@pytest.fixture(scope="module")
def tables(records_dir):
    from forgetcheck.calibrate import CalibrationConfig, calibrate, load_audit_records
    from forgetcheck.registry.metrics import default_registry

    df = load_audit_records(records_dir)
    cfg = CalibrationConfig(primary_condition=PRIMARY, band_oracle_seeds=tuple(range(200, 209)),
                            holdout_oracle_seeds=(209, 210, 211))
    return calibrate(df, registry=default_registry(), config=cfg)


class TestRealisticRecords:
    def test_calibration_completes_and_covers_every_group(self, tables):
        v = tables["validity"]
        assert set(v["forget_id"]) == set(CONDITIONS)
        assert set(v["audit"]) == {a for a, _, _ in AUDIT_ROWS}

    def test_an_empty_oracle_subset_gives_an_undefined_rate_not_a_crash(self, tables):
        """rand-2500, relearn_norm: unlearned rows, no oracle rows. The case that crashed."""
        v = tables["validity"].set_index(["forget_id", "metric", "probe_set"])
        row = v.loc[("rand-2500", "relearn_norm", "forget")]
        assert "calibrated_fpr" not in row or np.isnan(row["calibrated_fpr"])
        b = tables["bands"].set_index(["forget_id", "metric", "probe_set"])
        assert b.loc[("rand-2500", "relearn_norm", "forget"), "n"] == 0

    def test_self_anchored_metrics_use_only_the_ensemble_at_the_primary_condition(self, tables):
        b = tables["bands"].set_index(["forget_id", "metric", "probe_set"])
        assert b.loc[(PRIMARY, "relearn_norm", "forget"), "n"] == 12
        # Everywhere else, paired oracles are their own anchors: no band for relearn_norm.
        assert b.loc[("rand-3000", "relearn_norm", "forget"), "n"] == 0
        # Ordinary metrics keep all oracles: 17 at the primary condition, 5 elsewhere.
        assert b.loc[(PRIMARY, "js_to_oracle", "forget"), "n"] == 17
        assert b.loc[("rand-3000", "js_to_oracle", "forget"), "n"] == 5

    def test_superseded_anchor_rows_do_not_add_oracles(self, tables):
        b = tables["bands"].set_index(["forget_id", "metric", "probe_set"])
        assert b.loc[("rand-3000", "relearn_auc", "forget"), "n"] == 5

    def test_m0_is_counted_once_per_condition_despite_a_shared_run_id(self, tables):
        v = tables["validity"].set_index(["forget_id", "metric", "probe_set"])
        for cond in ("rand-500", "rand-5000", "mem-low-3000"):
            assert v.loc[(cond, "js_to_oracle", "forget"), "m0_tpr_n"] == 5

    def test_the_holdout_split_is_reported_at_the_primary_condition(self, tables):
        v = tables["validity"].set_index(["forget_id", "metric", "probe_set"])
        assert v.loc[(PRIMARY, "js_to_oracle", "forget"), "holdout_fpr_n"] == 3

    def test_native_rates_exist_only_for_native_metrics(self, tables):
        v = tables["validity"]
        assert set(v.loc[v["native_rule"].notna(), "metric"]) == {
            "mia_auc_pop", "mia_tpr_at_fpr_pop", "mia_auc_rmia", "mia_tpr_at_fpr_rmia", "sde_verdict"}

    def test_stage5_meta_rows_are_ignored(self, tables):
        assert "forget_acc" not in set(tables["validity"]["metric"])

    def test_flags_cover_every_unlearned_model(self, tables):
        f = tables["flags"]
        assert f["run_id"].nunique() == len(CONDITIONS) * len(METHODS) * len(SEEDS)

    def test_canary_ground_truth_by_role(self, tables):
        c = tables["canary"].set_index("role")["ground_truth"]
        assert not c.loc["oracle"].any() and c.loc["base"].all()
        assert len(tables["canary_validity"]) > 0

    def test_every_table_writes_to_parquet(self, tables, tmp_path):
        from forgetcheck.calibrate import write_tables

        paths = write_tables(tables, tmp_path)
        assert sorted(p.stem for p in paths) == sorted(tables)


class TestTheCommandAndTheNotebookCell:
    def test_cmd_calibrate_runs_and_the_results_cell_reads_its_output(
        self, records_dir, monkeypatch, capsys
    ):
        from forgetcheck import cli
        from forgetcheck.config import find_configs

        class Ctx:
            pass

        ctx = Ctx()
        ctx.records_dir = records_dir
        ctx.audits = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        ctx.primary_condition = PRIMARY
        monkeypatch.setattr(cli, "_ctx", lambda args: ctx)

        args = cli.build_parser().parse_args(["calibrate"])
        assert cli.cmd_calibrate(args) == 0
        out = capsys.readouterr().out
        assert "validity" in out and "models by role" in out

        # The notebook's results cell, verbatim, against what the command just wrote.
        nb = json.loads((Path(__file__).parents[1] / "notebooks/kaggle/05_calibrate.ipynb")
                        .read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in nb["cells"]
                    if "results/calibration/validity.parquet" in "".join(c["source"]))
        monkeypatch.chdir(records_dir.parents[1])
        exec(compile(cell, "05_calibrate results cell", "exec"), {})
        shown = capsys.readouterr().out
        assert "NATIVE false-positive rate" in shown and "POWER" in shown


class TestProtocolAndSelfReference:
    def test_the_random_init_floor_is_not_banded(self, tables):
        assert "relearn_randinit_auc" not in set(tables["validity"]["metric"])
        assert "relearn_randinit_auc" not in set(tables["bands"]["metric"])

    def test_the_relearning_protocol_check_is_computed_per_condition(self, tables):
        p = tables["relearning_protocol"].set_index("forget_id")
        assert set(p.index) == set(CONDITIONS)
        assert (p["oracle_n"] >= 5).all() and (p["m0_n"] == 5).all()
        assert p.loc[PRIMARY, "oracle_n"] == 17
        assert p["floor_below_oracle"].all()  # the simulated floor, 0.2, sits below every oracle

    def test_power_is_not_reported_for_a_metric_defined_against_m0_itself(self, tables):
        v = tables["validity"]
        js_orig = v[v["metric"] == "js_to_original"]
        assert len(js_orig) > 0
        assert "m0_tpr" not in js_orig or js_orig["m0_tpr"].isna().all()
        # ...while its calibration on retrains is still reported.
        assert js_orig["calibrated_fpr_n"].notna().all()
