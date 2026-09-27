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


def _value(metric, role, rng, *, cond=None, method=None):
    """Plausible values; the test is about structure and dtypes, not about the numbers -- except
    two real patterns reproduced on purpose: the destroyed control (neggrad) lands far from the
    retrains on every audit, and at mem-low the retrains relearn at a ceiling (0.998, M0 1.000)."""
    shift = {"oracle": 0.0, "unlearn": 0.05, "base": 0.3}[role]
    if method == "neggrad":
        shift = 1.0
    if metric == "sde_verdict":
        return float(rng.random() < 0.8)
    if metric.startswith("mia_"):
        return float(np.clip(0.55 + shift / 3 + rng.normal(scale=0.02), 0, 1))
    if metric == "relearn_norm":
        return {"oracle": 0.0, "unlearn": 0.6, "base": 1.0}[role] + float(rng.normal(scale=0.05))
    if metric == "relearn_t80":
        return float(abs(rng.normal(20, 5)))
    if metric == "relearn_auc" and cond == "mem-low-3000":
        return float(min(1.0, {"oracle": 0.998, "unlearn": 0.999, "base": 1.0}[role]
                         + rng.normal(scale=0.0005)))
    if metric == "relearn_auc":  # accuracy-like, above the simulated 0.2 random-init floor
        return {"oracle": 0.70, "unlearn": 0.80, "base": 0.95}[role] + float(rng.normal(scale=0.02))
    return float(abs(0.1 + shift + rng.normal(scale=0.02)))


def _meta_rows(rid, role, cond, rng, *, relearn, method=None, fields=None):
    """Stage 3/5 training-time rows: test accuracy for the utility guard, and forget accuracy --
    the relearning curve's starting point. Two patterns built in: at the primary condition the
    unlearned models *start* like retrains but relearn fast (hidden knowledge, which only
    relearning can see); everywhere else the start tracks the relearning score exactly (so
    relearning adds nothing). The canary retrains' forget accuracy is on the clean labels, as
    Stage 3 records it -- 0.9, which must never be used as their start."""
    fields = fields or spec_by_id(cond).as_record_fields()
    test = {"oracle": 0.93, "base": 0.935, "unlearn": 0.92}[role]
    if method == "neggrad":
        test = 0.10
    rows = [make_record(run_id=rid, audit="meta", metric="test_acc", probe_set="test",
                        value=test + float(rng.normal(scale=0.002)), n_probe=10000, **fields)]
    if role == "base":
        return rows
    if cond == "canary-500" and role == "oracle":
        start = 0.9
    elif cond == PRIMARY and role == "unlearn":
        start = 0.69 + float(rng.normal(scale=0.01))
    else:
        start = relearn - 0.01
    rows.append(make_record(run_id=rid, audit="meta", metric="forget_acc", probe_set="forget",
                            value=start, n_probe=500, **fields))
    return rows


def _relearn(rows):
    return next(r.value for r in rows if r.metric == "relearn_auc")


def _relearn_norm_undefined(cond, role):
    if cond in ("rand-500", "mem-low-3000"):
        return True                      # anchor gap below min_anchor_gap for every model
    return cond == "rand-2500" and role == "oracle"   # the combination that crashed


def _model_rows(rid, role, cond, rng, *, ts, method=None):
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
                                    value=_value(metric, role, rng, cond=cond, method=method),
                                    n_probe=500, timestamp=ts, **fields))
    if cond == "canary-500":
        # The destroyed control keeps no canary association: it sits at the retrains' level.
        top = ({"oracle": 0.11, "base": 0.97, "unlearn": 0.25}[role] if method != "neggrad"
               else 0.11) + float(rng.normal(scale=0.01))
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

    full = {"forget_id": "full", "forget_kind": "none", "forget_size": 0}
    for s in SEEDS:  # the clean M0s' Stage 3 rows, recorded once under forget_id "full"
        base = run_id(role="base", forget="full", seed=s)
        write_records(_meta_rows(base, "base", "full", rng, relearn=None, fields=full), rec,
                      suffix="train")
    for cond in CONDITIONS:
        spec = spec_by_id(cond)
        fields = spec.as_record_fields()
        base_forget = cond if spec.kind == "canary" else "full"
        for m in METHODS:
            for s in SEEDS:
                rid = f"c10r18__unlearn__{cond}__{m}__train{s}"
                rows = _model_rows(rid, "unlearn", cond, rng, ts=early, method=m)
                write_records(rows, rec, suffix="audit")
                write_records(_meta_rows(rid, "unlearn", cond, rng, relearn=_relearn(rows),
                                         method=m), rec, suffix="unlearn")
        for s in SEEDS:
            orc = run_id(role="oracle", forget=cond, seed=s, seed_kind="train")
            base = run_id(role="base", forget=base_forget, seed=s)
            rows = _model_rows(orc, "oracle", cond, rng, ts=late)
            write_records(rows, rec, suffix="audit-sim")
            write_records(_meta_rows(orc, "oracle", cond, rng, relearn=_relearn(rows)), rec,
                          suffix="train")
            write_records(_model_rows(base, "base", cond, rng, ts=late), rec,
                          suffix=f"audit-sim-{cond}")
            if spec.kind == "canary":
                write_records(_meta_rows(base, "base", cond, rng, relearn=None), rec,
                              suffix="train")
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
        rows = _model_rows(rid, "oracle", PRIMARY, rng, ts=late)
        write_records(rows, rec, suffix="audit-sim")
        write_records(_meta_rows(rid, "oracle", PRIMARY, rng, relearn=_relearn(rows)), rec,
                      suffix="train")
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
        # The command prints the whole report itself, so a stale notebook cell cannot hide it.
        for section in ("NATIVE false-positive rate", "NATIVE POWER", "POWER --",
                        "CALIBRATED false-positive rate", "EFFECT SIZE",
                        "in retrain standard deviations", "RELEARNING PROTOCOL",
                        "RELEARNING vs STARTING ACCURACY", "UTILITY", "CANARY GROUND TRUTH",
                        "CANARY VALIDITY"):
            assert section in out, section
        assert "ba_guarded" in out and "relearn_auc @ mem-low-3000" in out

        # The notebook's results cell, verbatim, against what the command just wrote.
        nb = json.loads((Path(__file__).parents[1] / "notebooks/kaggle/05_calibrate.ipynb")
                        .read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in nb["cells"]
                    if "print_report" in "".join(c["source"]))
        monkeypatch.chdir(records_dir.parents[1])
        exec(compile(cell, "05_calibrate results cell", "exec"), {})
        shown = capsys.readouterr().out
        assert "NATIVE false-positive rate" in shown and "CANARY VALIDITY" in shown


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


class TestCanaryValiditySplit:
    def test_false_positives_are_split_by_who_they_are(self, tables):
        cv = tables["canary_validity"]
        assert {"fp_retrain", "fp_other"} <= set(cv.columns)
        assert (cv["fp_retrain"] + cv["fp_other"] == cv["fp"]).all()

    def test_m0_is_not_counted_for_a_metric_defined_against_it(self, tables):
        cv = tables["canary_validity"].set_index(["metric", "probe_set", "kind"])
        full = cv.loc[("js_to_oracle", "forget", "calibrated"), "n"]
        self_ref = cv.loc[("js_to_original", "forget", "calibrated"), "n"]
        assert self_ref == full - 5  # the five M0 audits excluded

    def test_the_canary_protocol_row_explains_its_floor(self, tables):
        p = tables["relearning_protocol"].set_index("forget_id")
        assert "floor ~ retrain expected" in p.loc["canary-500", "note"]
        assert p.loc["rand-500", "note"] == ""


def test_a_guard_metric_is_banded_but_never_scored_as_an_audit(tables):
    """relearn_utility_drop says whether a recovery claim is void, not whether anything was
    forgotten. On the real run it scored balanced accuracy 0.50 -- an 'audit at chance' that is
    not an audit."""
    assert "relearn_utility_drop" in set(tables["bands"]["metric"])
    assert "relearn_utility_drop" not in set(tables["canary_validity"]["metric"])
    v = tables["validity"]
    g = v[v["metric"] == "relearn_utility_drop"]
    assert len(g) and g["m0_tpr"].isna().all()


class TestUtilityGuard:
    def test_only_the_destroyed_control_is_damaged(self, tables):
        u = tables["utility"]
        damaged = u[u["damaged"]]
        assert set(damaged["method"]) == {"neggrad"}
        assert len(damaged) == len(CONDITIONS) * len(SEEDS)
        assert not u.loc[u["role"].isin(["oracle", "base"]), "damaged"].any()

    def test_m0_is_found_under_its_own_forget_id(self, tables):
        """Stage 3 records the clean M0 under forget_id 'full'; it must still be guarded at all
        seven conditions it serves, and the canary M0 at its own."""
        u = tables["utility"]
        m0 = u[u["role"] == "base"]
        assert set(m0["forget_id"]) == set(CONDITIONS)
        assert (m0.groupby("forget_id").size() == len(SEEDS)).all()

    def test_flags_carry_the_guard_for_stage8(self, tables):
        f = tables["flags"]
        assert f.loc[f["method"] == "neggrad", "damaged"].all()
        assert not f.loc[f["method"] == "scrub", "damaged"].any()
        assert (f.loc[f["method"] == "neggrad", "utility_drop_pp"] > 80).all()

    def test_the_guard_reads_the_destroyed_controls_flags_as_damage(self, tables):
        """The real canary pattern: every calibrated audit flagged all five destroyed models.
        Guarded, those five become true negatives, and nothing else changes."""
        cv = tables["canary_validity"].set_index(["metric", "probe_set", "kind"])
        row = cv.loc[("js_to_oracle", "forget", "calibrated")]
        assert row["n_damaged"] == 5 and row["fp_other"] == 5 and row["fp_other_guarded"] == 0
        # TNR goes from 5/10 to 10/10; the positives are untouched.
        assert row["ba_guarded"] - row["balanced_accuracy"] == pytest.approx(0.25)


class TestEffectSize:
    def test_a_ceiling_gap_is_marked_although_the_sd_rule_passes_it(self, tables):
        """relearn_auc at mem-low on the real run: 1.000 against 0.998 -- '4.2 sd', 0.002 raw."""
        v = tables["validity"].set_index(["forget_id", "metric", "probe_set"])
        cell = v.loc[("mem-low-3000", "relearn_auc", "forget")]
        assert abs(cell["m0_gap_raw"]) < 0.02 and cell["below_min_effect"]
        assert not cell["low_discriminability"]
        assert not v.loc[("rand-3000", "relearn_auc", "forget"), "below_min_effect"]

    def test_only_proportions_get_the_raw_floor(self, tables):
        v = tables["validity"]
        js = v[v["metric"] == "js_to_oracle"]
        assert js["below_min_effect"].isna().all()

    def test_the_oracle_gap_is_the_registered_formula(self, tables):
        """G = |m(Mu) - mean m(Mr)| / |m(M0) - mean m(Mr)|, implementation plan §4.4."""
        b = tables["bands"].set_index(["forget_id", "metric", "probe_set"])
        v = tables["validity"].set_index(["forget_id", "metric", "probe_set"])
        f = tables["flags"]
        key = ("rand-3000", "js_to_oracle", "forget")
        cell = f[(f["forget_id"] == key[0]) & (f["metric"] == key[1]) & (f["probe_set"] == key[2])]
        expect = (cell["value"] - b.loc[key, "mean"]).abs() / abs(v.loc[key, "m0_gap_raw"])
        np.testing.assert_allclose(cell["oracle_gap"], expect)
        assert cell.loc[cell["method"] != "neggrad", "oracle_gap"].between(0, 1).mean() > 0.5

    def test_the_oracle_gap_is_undefined_in_a_marked_cell(self, tables):
        f = tables["flags"]
        cell = f[(f["forget_id"] == "mem-low-3000") & (f["metric"] == "relearn_auc")]
        assert len(cell) and cell["oracle_gap"].isna().all()


class TestRelearningVsAccuracy:
    def test_hidden_knowledge_is_flagged_by_relearning_alone(self, tables):
        """At the primary condition the fixture's unlearned models start like retrains but
        relearn fast: exactly what relearning exists to see and accuracy cannot."""
        r = tables["relearning_vs_accuracy"].set_index("forget_id")
        assert r.loc[PRIMARY, "flagged_by_start"] <= 2
        assert r.loc[PRIMARY, "flagged_by_relearn_only"] >= 20
        assert r.loc[PRIMARY, "n_retrain"] == 17

    def test_where_relearning_is_the_starting_accuracy_it_adds_nothing(self, tables):
        r = tables["relearning_vs_accuracy"].set_index("forget_id")
        assert r.loc["rand-3000", "flagged_by_relearn_only"] == 0
        assert r.loc["rand-3000", "r_relearn_vs_start"] > 0.99

    def test_the_canary_start_is_on_the_canary_labels(self, tables):
        """Stage 3 scored the canary retrains on clean labels (0.9 here); the relearning curve
        runs on the assigned wrong labels, which canary_acc measures."""
        r = tables["relearning_vs_accuracy"].set_index("forget_id")
        assert r.loc["canary-500", "start_metric"] == "canary_acc"
        assert r.loc["canary-500", "retrain_start"] < 0.2

    def test_damaged_models_are_left_out(self, tables):
        r = tables["relearning_vs_accuracy"]
        assert set(r["forget_id"]) == set(CONDITIONS)
        assert (r["n_unlearned"] == (len(METHODS) - 1) * len(SEEDS)).all()  # all but neggrad


def test_missing_stage3_to_5_records_are_named_not_left_as_zero_rows(
    records_dir, tmp_path, monkeypatch, capsys
):
    """The real run with only the Stage 6 dataset attached printed 'utility 0 rows' and a
    one-row relearning check. Without the training/unlearning shards the command must say which
    dataset is missing, and the report which conditions it could not compute."""
    import shutil

    from forgetcheck import cli
    from forgetcheck.config import find_configs

    rec = tmp_path / "results" / "records"
    rec.mkdir(parents=True)
    for p in records_dir.glob("*.parquet"):
        if not p.name.endswith(("--train.parquet", "--unlearn.parquet")):
            shutil.copy2(p, rec / p.name)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.records_dir = rec
    ctx.audits = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
    ctx.primary_condition = PRIMARY
    monkeypatch.setattr(cli, "_ctx", lambda args: ctx)
    assert cli.cmd_calibrate(cli.build_parser().parse_args(["calibrate"])) == 0
    out = capsys.readouterr().out
    assert "records: 0 training shards (stages 3-4), 0 unlearning shards" in out
    assert "attach the forgetcheck-artifacts dataset" in out
    assert "not computed for mem-high-3000" in out  # canary alone survives, as on Kaggle
