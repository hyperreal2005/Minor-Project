"""Stage 8 on constructed calibration tables, where every answer is planted.

A latent "retention" per model drives two audits (they must agree), a third is independent noise
(it must not), the population attack is built 0.2 more lenient than RMIA (H2b), one method
retains more than the reference (the mixed model must find it), and at mem-high the methods are
spread three times wider (the interaction test must find that, and must not where it is absent).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from forgetcheck.analysis.agreement import (PRIMARY, AnalysisConfig, _spearman, correlations,
                                            difficulty_spread, disagreement, instance_tables,
                                            kendall_tau_b, patterns, privacy_contrast)
from forgetcheck.analysis.mixed_effects import fit_main, interaction_test

CONDS = ["canary-500", "mem-high-3000", "mem-low-3000", "mem-med-3000", "rand-3000"]
METHODS = ["finetune", "l1sparse", "neggrad", "neggradplus", "salun", "scrub"]
AUDIT = {"js_to_oracle": "behavior", "mia_auc_pop": "privacy_population",
         "mia_auc_rmia": "privacy_rmia", "cka_linear": "representation",
         "relearn_auc": "relearning", "activation_l2": "representation"}
CFG = AnalysisConfig(resamples=400, seed=0)


def _tables(*, interaction=True, simpson=False, seed=0):
    """flags + validity shaped like calibrate's. scrub retains 0.3 more than finetune."""
    rng = np.random.default_rng(seed)
    effect = {"finetune": 0.0, "l1sparse": 0.1, "neggradplus": 0.15, "salun": 0.05,
              "scrub": 0.3, "neggrad": 0.0}
    rows = []
    for cond in CONDS:
        spread = 3.0 if (interaction and cond == "mem-high-3000") else 1.0
        shift = {"canary-500": 0.0, "mem-high-3000": 0.2}.get(cond, 0.1)
        if simpson:  # conditions move every metric together; within a condition, nothing
            shift = 2.0 * CONDS.index(cond)
        for m in METHODS:
            for s in range(5):
                r = shift + spread * effect[m] + rng.normal(scale=0.03)
                g = {"js_to_oracle": r + rng.normal(scale=0.02),
                     "cka_linear": r + rng.normal(scale=0.02),
                     "mia_auc_rmia": r + rng.normal(scale=0.02),
                     "mia_auc_pop": r - 0.2 + rng.normal(scale=0.02),
                     "relearn_auc": rng.normal(loc=0.5, scale=0.2),
                     "activation_l2": r}
                if simpson:
                    g = {k: (shift + rng.normal(scale=0.3)) for k in g}
                for metric, gv in g.items():
                    marked = metric == "relearn_auc" and cond == "mem-low-3000"
                    rows.append({
                        "run_id": f"c10r18__unlearn__{cond}__{m}__train{s}", "forget_id": cond,
                        "method": m, "train_seed": s, "audit": AUDIT[metric], "metric": metric,
                        "probe_set": "layer4" if AUDIT[metric] == "representation" else "forget",
                        "value": float(gv), "calibrated_flag": bool(gv > 0.25),
                        "native_flag": None, "oracle_gap": float("nan") if marked else float(gv),
                        "utility_drop_pp": 84.0 if m == "neggrad" else 0.5,
                        "damaged": m == "neggrad", "retired": metric == "activation_l2"})
    flags = pd.DataFrame(rows)
    validity = (flags[["forget_id", "audit", "metric", "probe_set"]].drop_duplicates()
                .assign(low_discriminability=False, below_min_effect=False))
    validity.loc[(validity["metric"] == "relearn_auc")
                 & (validity["forget_id"] == "mem-low-3000"), "below_min_effect"] = True
    return flags, validity


@pytest.fixture(scope="module")
def planted():
    flags, validity = _tables()
    G, V, excluded = instance_tables(flags, validity)
    return flags, G, V, excluded


class TestRankStatistics:
    def test_tau_b_equals_scipys_with_ties(self):
        rng = np.random.default_rng(1)
        x, y = rng.integers(0, 5, 60).astype(float), rng.integers(0, 4, 60).astype(float)
        assert kendall_tau_b(x, y) == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-12)

    def test_spearman_equals_scipys(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=50), rng.integers(0, 6, 50).astype(float)
        assert float(_spearman(x[None], y[None])[0]) == pytest.approx(
            stats.spearmanr(x, y).statistic, abs=1e-12)


class TestInstances:
    def test_damaged_and_retired_are_out(self, planted):
        _, G, _, excluded = planted
        assert "neggrad" not in set(G.index.get_level_values("method"))
        assert "activation_l2" not in G.columns
        assert excluded["damaged models"] == len(CONDS) * 5
        assert len(G) == len(CONDS) * 5 * 5

    def test_a_marked_cell_has_no_g_and_no_verdict(self, planted):
        _, G, V, _ = planted
        low = G.index.get_level_values("forget_id") == "mem-low-3000"
        assert G.loc[low, "relearn_auc"].isna().all()
        assert V.loc[low, "relearn_auc"].isna().all()
        assert V.loc[~low, "relearn_auc"].notna().all()


class TestCorrelations:
    def test_shared_retention_agrees_and_noise_does_not(self, planted):
        _, G, _, _ = planted
        c = correlations(G, CFG).set_index(["pair", "scope"])
        agree = c.loc[("behavior vs representation", "pooled")]
        assert agree["tau"] > 0.6 and agree["tau_lo"] <= agree["tau"] <= agree["tau_hi"]
        noise = c.loc[("representation vs reversibility", "pooled")]
        assert noise["tau_lo"] < 0 < noise["tau_hi"]
        # relearn_auc is marked at mem-low: those instances drop out of its pairs.
        assert noise["n"] == len(G) - 25

    def test_the_bootstrap_is_reproducible(self, planted):
        _, G, _, _ = planted
        a, b = correlations(G, CFG), correlations(G, CFG)
        pd.testing.assert_frame_equal(a, b)

    def test_within_condition_strips_agreement_made_by_conditions(self):
        """Every metric shifts with the condition and nothing else: pooled tau is high, and it
        is the conditions talking -- the within-condition tau is near zero."""
        flags, validity = _tables(simpson=True)
        G, _, _ = instance_tables(flags, validity)
        c = correlations(G, CFG).set_index(["pair", "scope"])
        assert c.loc[("behavior vs representation", "pooled"), "tau"] > 0.5
        assert abs(c.loc[("behavior vs representation", "within"), "tau"]) < 0.15


class TestVerdictsAndPatterns:
    def test_disagreement_is_the_share_of_contradicting_verdicts(self):
        idx = pd.MultiIndex.from_tuples(
            [(f"r{i}", "rand-3000", "finetune", i) for i in range(4)],
            names=["run_id", "forget_id", "method", "train_seed"])
        V = pd.DataFrame({m: [None] * 4 for m in PRIMARY.values()}, index=idx, dtype=object)
        V[PRIMARY["behavior"]] = [True, True, False, False]
        V[PRIMARY["privacy_weak"]] = [True, False, False, None]
        d = disagreement(V).set_index(["pair", "scope"])
        row = d.loc[("behavior vs privacy_weak", "pooled")]
        assert row["n"] == 3 and row["disagree"] == pytest.approx(1 / 3)
        assert row["flag_rate_a"] == pytest.approx(2 / 3)

    def test_patterns_count_what_the_hypotheses_name(self):
        idx = pd.MultiIndex.from_tuples(
            [(f"r{i}", "rand-3000", "finetune", i) for i in range(3)],
            names=["run_id", "forget_id", "method", "train_seed"])
        G = pd.DataFrame({m: [0.2, 0.2, 0.8] for m in PRIMARY.values()}, index=idx)
        G[PRIMARY["representation"]] = [0.9, 0.1, 0.9]
        V = pd.DataFrame({m: [False, False, True] for m in PRIMARY.values()}, index=idx,
                         dtype=object)
        V[PRIMARY["representation"]] = [True, False, True]
        V[PRIMARY["reversibility"]] = [True, False, False]
        p = patterns(G, V).set_index("scope").loc["pooled"]
        assert p["rq3_behaviour_retrain_like"] == 2
        assert p["rq3_of_which_representation_M0_like"] == 1
        assert p["h1_pass_behaviour"] == 2 and p["h1_of_which_flagged_by_representation"] == 1
        assert p["rq4_pass_behaviour_and_privacy"] == 2
        assert p["rq4_of_which_flagged_by_relearning"] == 1

    def test_the_population_attack_is_found_more_lenient(self, planted):
        _, G, V, _ = planted
        pc = privacy_contrast(G, V, CFG).set_index("scope").loc["pooled"]
        assert pc["mean_diff"] == pytest.approx(-0.2, abs=0.02)
        assert pc["hi"] < 0 and pc["share_pop_more_lenient"] > 0.95 and pc["wilcoxon_p"] < 1e-6


class TestDifficultyAndMixedModels:
    def test_methods_spread_most_where_planted(self, planted):
        _, G, _, _ = planted
        d = difficulty_spread(G)
        beh = d[d["family"] == "behavior"].set_index("forget_id")["method_sd"]
        assert beh["mem-high-3000"] > 2 * beh["rand-3000"]

    def test_the_mixed_model_recovers_the_planted_method_effect(self, planted):
        _, G, _, _ = planted
        me = fit_main(G, "js_to_oracle").set_index("term")
        # Averaged over conditions, scrub's planted effect is 0.3 x (1 + 1 + 1 + 1 + 3) / 5.
        expected = 0.3 * 7 / 5
        assert me.loc["method: scrub", "lo"] < expected < me.loc["method: scrub", "hi"]
        assert me.loc["method: scrub", "p"] < 1e-6
        assert {"condition: mem-high-3000", "condition: mem-low-3000"} <= set(me.index)
        assert (me["method_ref"] == "finetune").all() and (me["condition_ref"] == "rand-3000").all()

    def test_the_interaction_is_found_when_planted_and_not_otherwise(self, planted):
        _, G, _, _ = planted
        assert interaction_test(G, "js_to_oracle")["p"] < 1e-6
        flags, validity = _tables(interaction=False, seed=3)
        G0, _, _ = instance_tables(flags, validity)
        assert interaction_test(G0, "js_to_oracle")["p"] > 0.01


def test_analyse_end_to_end_writes_and_prints(tmp_path, capsys):
    from forgetcheck.analysis import analyse, print_report, write_tables

    flags, validity = _tables()
    cal = tmp_path / "calibration"
    cal.mkdir()
    flags.to_parquet(cal / "flags.parquet", index=False)
    validity.to_parquet(cal / "validity.parquet", index=False)
    tables = analyse(cal, CFG)
    paths = write_tables(tables, tmp_path / "analysis")
    assert {p.stem for p in paths} == set(tables)
    print_report(tables)
    out = capsys.readouterr().out
    for section in ("RQ1", "registered comparisons", "H2b", "Pass/fail disagreement",
                    "RQ3, H1, RQ4", "H4", "CONFIRMATORY", "DESCRIPTIVE ONLY", "FULL MATRIX"):
        assert section in out, section
    inv = tables["inventory"].iloc[0]
    assert inv["instances"] == len(CONDS) * 25 and inv["retired"] == "activation_l2"
