"""Stage 7 statistics.

The band construction is tested by simulation, because its whole justification is a claim about
coverage: that a prediction interval keeps the false-positive rate on genuine retrains at the
nominal level at n = 4, where a fixed ``mean ± 2·sd`` does not. If that claim were false, every
small-n condition's "audit invalidity" would be the band's doing.
"""

import math

import numpy as np
import pandas as pd
import pytest

from forgetcheck.calibrate import (
    CalibrationConfig,
    calibrate,
    canary_scores,
    clopper_pearson,
    coverage_for,
    flag,
    loo_flags,
    native_flag,
    nominal_fpr,
    prediction_band,
)

COV = coverage_for(2.0)


class TestPredictionBand:
    @pytest.mark.parametrize("n", [4, 5, 12, 16])
    def test_holds_nominal_coverage_at_every_n(self, n):
        """A new draw from the same normal falls outside the band at ≈ 4.55%, whatever n."""
        rng = np.random.default_rng(n)
        trials, misses = 20000, 0
        for _ in range(trials):
            ref = rng.normal(size=n)
            b = prediction_band(ref, coverage=COV)
            x = rng.normal()
            misses += (x < b.lo) or (x > b.hi)
        assert abs(misses / trials - (1 - COV)) < 0.006, misses / trials

    def test_a_fixed_two_sd_band_would_not(self):
        """The reason for the deviation: at n = 4, `mean ± 2·s` misses genuine retrains at
        several times the nominal rate. Reported as audit invalidity, that would be the band's
        fault."""
        rng = np.random.default_rng(0)
        trials, misses = 20000, 0
        for _ in range(trials):
            ref = rng.normal(size=4)
            m, s = ref.mean(), ref.std(ddof=1)
            x = rng.normal()
            misses += abs(x - m) > 2 * s
        assert misses / trials > 2.5 * (1 - COV)

    def test_undefined_below_three(self):
        assert not prediction_band([1.0, 2.0], coverage=COV).defined
        assert prediction_band([1.0, 2.0, 3.0], coverage=COV).defined

    def test_ignores_nan(self):
        b = prediction_band([1.0, np.nan, 2.0, 3.0], coverage=COV)
        assert b.n == 3

    def test_coverage_for_two_sd(self):
        assert coverage_for(2.0) == pytest.approx(0.9545, abs=1e-4)


class TestFlagDirections:
    band = prediction_band([0.0, 1.0, 2.0, 1.0, 1.0], coverage=COV)

    def test_lower_better_flags_only_above(self):
        assert flag(self.band.hi + 1, self.band, "lower_better") is True
        assert flag(self.band.lo - 1, self.band, "lower_better") is False

    def test_higher_better_flags_only_below(self):
        assert flag(self.band.lo - 1, self.band, "higher_better") is True
        assert flag(self.band.hi + 1, self.band, "higher_better") is False

    def test_closer_to_oracle_flags_both_sides(self):
        assert flag(self.band.lo - 1, self.band, "closer_to_oracle") is True
        assert flag(self.band.hi + 1, self.band, "closer_to_oracle") is True
        assert flag(self.band.mean, self.band, "closer_to_oracle") is False

    def test_nominal_rates(self):
        assert nominal_fpr("closer_to_oracle", COV) == pytest.approx(0.0455, abs=1e-4)
        assert nominal_fpr("lower_better", COV) == pytest.approx(0.0228, abs=1e-4)

    def test_unknowable_is_none_not_false(self):
        assert flag(float("nan"), self.band, "lower_better") is None
        tiny = prediction_band([1.0, 2.0], coverage=COV)
        assert flag(1.5, tiny, "lower_better") is None


class TestLeaveOneOut:
    def test_each_value_is_tested_against_the_others(self):
        # An outlier among five is flagged against the other four; the rest are not.
        f = loo_flags(np.array([0.50, 0.51, 0.49, 0.50, 0.95]),
                      direction="closer_to_oracle", coverage=COV)
        assert f[-1] is True and not any(f[:-1])

    def test_loo_fpr_is_nominal_on_exchangeable_draws(self):
        rng = np.random.default_rng(3)
        flags = []
        for _ in range(4000):
            flags += loo_flags(rng.normal(size=5), direction="closer_to_oracle", coverage=COV)
        assert abs(np.mean(flags) - (1 - COV)) < 0.01


class TestClopperPearson:
    def test_zero_of_five_is_not_certainly_zero(self):
        lo, hi = clopper_pearson(0, 5)
        assert lo == 0.0 and hi > 0.4

    def test_brackets_the_point_estimate(self):
        lo, hi = clopper_pearson(3, 17)
        assert lo < 3 / 17 < hi

    def test_empty(self):
        assert all(math.isnan(x) for x in clopper_pearson(0, 0))


class TestNativeRules:
    def test_auc_at_chance_is_not_flagged(self):
        assert native_flag("auc_above_chance", 0.50, n_probe=3000) is False

    def test_a_small_auc_excess_is_flagged_at_large_n(self):
        # At n = 3000 the null SE is 0.0075; an AUC of 0.55 is 6.7 SE above chance. This is why
        # a population attack can call a genuine retrain "leaking" on high-memorization data.
        assert native_flag("auc_above_chance", 0.55, n_probe=3000) is True
        assert native_flag("auc_above_chance", 0.51, n_probe=500) is False

    def test_below_chance_is_not_leakage(self):
        assert native_flag("auc_above_chance", 0.20, n_probe=500) is False

    def test_tpr_at_the_nominal_fpr_is_not_flagged(self):
        assert native_flag("tpr_above_fpr", 0.01, n_probe=3000) is False
        assert native_flag("tpr_above_fpr", 0.05, n_probe=3000) is True

    def test_sde_verdict(self):
        assert native_flag("verdict_in_training", 0.0, n_probe=500) is True
        assert native_flag("verdict_in_training", 1.0, n_probe=500) is False

    def test_undefined_input(self):
        assert native_flag("auc_above_chance", float("nan"), n_probe=500) is None


class TestCanaryScores:
    y_true = np.array([0, 1, 2, 3])
    y_can = np.array([5, 6, 7, 8])

    def _logits(self, favour):
        x = np.zeros((4, 10))
        x[np.arange(4), favour] = 10.0
        return x

    def test_a_model_that_remembers_scores_one_everywhere(self):
        s = canary_scores(self._logits(self.y_can), self.y_true, self.y_can)
        assert s["canary_acc"] == 1.0 and s["canary_top_wrong"] == 1.0

    def test_a_correct_model_ignores_the_true_label_when_ranking_wrong_ones(self):
        # Confident in the true label, second choice the canary: remembers, and top_wrong says so
        # while canary_acc cannot see it.
        x = self._logits(self.y_true)
        x[np.arange(4), self.y_can] = 5.0
        s = canary_scores(x, self.y_true, self.y_can)
        assert s["canary_acc"] == 0.0 and s["canary_top_wrong"] == 1.0

    def test_the_constant_predictor_sits_at_chance_not_above_the_oracle(self):
        """The flaw in the registered metrics: a destroyed model predicts the assigned label
        about one time in ten by chance. canary_top_wrong ranks among wrong labels only, where a
        model with no memory is at chance -- the same place a retrain with no preference sits."""
        rng = np.random.default_rng(0)
        n = 9000
        y = rng.integers(0, 10, n)
        idx = np.arange(n)
        can = (y + 1 + idx % 9) % 10
        const = np.zeros((n, 10)); const[:, 3] = 20.0
        const += rng.normal(scale=1e-3, size=const.shape)  # break exact ties
        s = canary_scores(const, y, can)
        assert abs(s["canary_top_wrong"] - 1 / 9) < 0.02
        assert s["canary_prob"] > 0.05  # what makes canary_prob misleading here

    def test_rejects_misaligned_inputs(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            canary_scores(np.zeros((3, 10)), self.y_true, self.y_can)


# --------------------------------------------------------------------------- end to end


def _records():
    """A miniature Stage 7 record set: one condition, three roles, two metrics."""
    rows = []
    rng = np.random.default_rng(0)

    def add(run_id, role, method, value, metric, audit, oracle_seed=None, train_seed=None):
        rows.append(dict(run_id=run_id, role=role, method=method, forget_id="mem-high-3000",
                         audit=audit, metric=metric, probe_set="forget", value=value,
                         n_probe=3000, oracle_seed=oracle_seed, train_seed=train_seed,
                         timestamp="2026-09-25T00:00:00"))

    for s in range(200, 212):  # ensemble oracles: JS small, pop-MIA AUC ≈ 0.72 (atypical data)
        rid = f"c10r18__oracle__mem-high-3000__none__oracle{s}"
        add(rid, "oracle", "none", 0.03 + rng.normal(scale=0.003), "js_to_oracle", "behavior", oracle_seed=s)
        add(rid, "oracle", "none", 0.72 + rng.normal(scale=0.01), "mia_auc_pop", "privacy_population", oracle_seed=s)
    for s in range(5):  # M0: far from the oracles
        rid = f"c10r18__base__full__none__train{s}"
        add(rid, "base", "none", 0.20 + rng.normal(scale=0.01), "js_to_oracle", "behavior", train_seed=s)
        add(rid, "base", "none", 0.95, "mia_auc_pop", "privacy_population", train_seed=s)
    for s in range(5):  # an unlearned method in between
        rid = f"c10r18__unlearn__mem-high-3000__salun__train{s}"
        add(rid, "unlearn", "salun", 0.15, "js_to_oracle", "behavior", train_seed=s)
        add(rid, "unlearn", "salun", 0.84, "mia_auc_pop", "privacy_population", train_seed=s)
    return pd.DataFrame(rows)


class TestCalibrateEndToEnd:
    @pytest.fixture
    def tables(self):
        from forgetcheck.registry.metrics import default_registry

        cfg = CalibrationConfig(band_oracle_seeds=tuple(range(200, 209)),
                                holdout_oracle_seeds=(209, 210, 211))
        return calibrate(_records(), registry=default_registry(), config=cfg)

    def test_the_population_attack_flags_every_genuine_retrain_by_its_own_rule(self, tables):
        # The headline this stage exists to measure: AUC 0.72 on retrains is "leakage" to the
        # native rule, so its false-positive rate is 1 -- while the calibrated band's is nominal.
        v = tables["validity"].set_index("metric").loc["mia_auc_pop"]
        assert v["native_fpr"] == 1.0 and v["native_fpr_n"] == 12
        assert v["calibrated_fpr"] <= 0.2

    def test_m0_is_detected_by_a_discriminating_metric(self, tables):
        v = tables["validity"].set_index("metric").loc["js_to_oracle"]
        assert v["m0_tpr"] == 1.0 and v["low_discriminability"] == False  # noqa: E712

    def test_the_preregistered_holdout_is_reported_at_the_primary_condition(self, tables):
        v = tables["validity"].set_index("metric").loc["js_to_oracle"]
        assert v["holdout_fpr_n"] == 3

    def test_unlearned_models_get_per_model_verdicts_for_stage_8(self, tables):
        f = tables["flags"]
        assert set(f["method"]) == {"salun"} and len(f) == 10
        assert f.loc[f["metric"] == "js_to_oracle", "calibrated_flag"].all()

    def test_bands_are_written_for_every_metric(self, tables):
        assert set(tables["bands"]["metric"]) == {"js_to_oracle", "mia_auc_pop"}
