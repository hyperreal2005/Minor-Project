"""Memorization scores and proxies.

The sign of a proxy is the thing that matters most here. Confidence correlates *negatively*
with memorization, so a proxy fed in without inversion swaps the high and low strata — turning
the primary condition into the negative control while producing results that look plausible.
"""

import json

import numpy as np
import pytest

from forgetcheck.data.memorization import (
    accuracy_proxy,
    check_scores,
    confidence_proxy,
    load_scores,
)

N = 1000


class TestLoading:
    @pytest.mark.parametrize("fmt", ["npy", "npz", "csv", "json"])
    def test_round_trip(self, tmp_path, fmt):
        scores = np.random.default_rng(0).random(N)
        p = tmp_path / f"m.{fmt}"
        if fmt == "npy":
            np.save(p, scores)
        elif fmt == "npz":
            np.savez(p, memorization=scores)
        elif fmt == "csv":
            np.savetxt(p, scores, delimiter=",")
        else:
            p.write_text(json.dumps(scores.tolist()))
        np.testing.assert_allclose(load_scores(p, n_train=N), scores, atol=1e-6)

    def test_missing_file_names_the_source(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="RUM repository"):
            load_scores(tmp_path / "nope.npy", n_train=N)

    def test_wrong_length_is_fatal(self, tmp_path):
        p = tmp_path / "m.npy"
        np.save(p, np.random.random(50))
        with pytest.raises(ValueError, match="index-for-index"):
            load_scores(p, n_train=N)

    def test_out_of_range_rejected(self, tmp_path):
        p = tmp_path / "m.npy"
        np.save(p, np.linspace(0, 5, N))
        with pytest.raises(ValueError, match="outside the expected"):
            load_scores(p, n_train=N)

    def test_nan_rejected(self, tmp_path):
        p = tmp_path / "m.npy"
        s = np.random.random(N); s[3] = np.nan
        np.save(p, s)
        with pytest.raises(ValueError, match="non-finite"):
            load_scores(p, n_train=N)

    def test_ambiguous_npz_asks_which_array(self, tmp_path):
        p = tmp_path / "m.npz"
        np.savez(p, a=np.random.random(N), b=np.random.random(N))
        with pytest.raises(ValueError, match="pass key="):
            load_scores(p, n_train=N)
        load_scores(p, n_train=N, key="a")


class TestProxies:
    def test_confidence_is_oriented_as_memorization(self):
        # A confidently-correct example generalised; it must score LOW on memorization.
        logits = np.array([[10.0, 0.0], [0.0, 0.1]])
        labels = np.array([0, 0])
        s = confidence_proxy(logits, labels)
        assert s[0] < 0.01, "confidently correct -> low memorization"
        assert s[1] > 0.4, "barely correct -> higher memorization"
        assert s[0] < s[1]

    def test_confidence_in_unit_range(self):
        rng = np.random.default_rng(0)
        s = confidence_proxy(rng.normal(size=(N, 10)) * 5, rng.integers(0, 10, N))
        assert 0.0 <= s.min() and s.max() <= 1.0

    def test_confidence_is_numerically_stable(self):
        # Large logits must not overflow to nan before the softmax.
        s = confidence_proxy(np.array([[1e4, -1e4]]), np.array([0]))
        assert np.isfinite(s).all() and s[0] < 1e-6

    def test_accuracy_proxy_flags_errors_as_memorised(self):
        s = accuracy_proxy(np.array([0, 1, 2]), np.array([0, 9, 2]))
        np.testing.assert_array_equal(s, [0.0, 1.0, 0.0])

    def test_shape_mismatches_rejected(self):
        with pytest.raises(ValueError):
            confidence_proxy(np.zeros((5, 3)), np.zeros(4, dtype=int))
        with pytest.raises(ValueError):
            accuracy_proxy(np.zeros(5, dtype=int), np.zeros(4, dtype=int))


class TestScoreChecks:
    def test_reports_all_three_strata_plus_summary(self):
        rng = np.random.default_rng(0)
        report = check_scores(np.where(rng.random(50_000) < 0.65, rng.beta(0.5, 12, 50_000), rng.beta(1.5, 1.5, 50_000)))
        assert {"low", "medium", "high", "summary"} == set(report)
        assert set(report["low"]) == {"mean", "std", "min", "max"}
        summary = report["summary"]
        assert {"median", "low_to_high_spread", "fraction_near_zero",
                "fraction_above_half", "max_stratum_overlap"} <= set(summary)
        assert summary["max_stratum_overlap"] == 0, "strata must not share examples"

    def test_inverted_scores_are_caught_loudly(self):
        # The exact failure a forgotten sign flip produces. It must not pass silently.
        rng = np.random.default_rng(0)
        scores = np.where(rng.random(50_000) < 0.65, rng.beta(0.5, 12, 50_000), rng.beta(1.5, 1.5, 50_000))
        with pytest.raises(ValueError, match="inverted"):
            check_scores(1.0 - scores)

    def test_ordering_alone_cannot_detect_inversion(self):
        # Documents why the magnitude checks exist: strata are defined by rank on the array
        # passed in, so low <= medium <= high holds even for a fully inverted one.
        rng = np.random.default_rng(0)
        scores = np.where(rng.random(50_000) < 0.65, rng.beta(0.5, 12, 50_000), rng.beta(1.5, 1.5, 50_000))
        upright = check_scores(scores, strict=False)
        inverted = check_scores(1.0 - scores, strict=False)

        assert upright["low"]["mean"] < upright["high"]["mean"]
        assert inverted["low"]["mean"] < inverted["high"]["mean"]

        # Magnitude is what separates them: a real low stratum sits near zero.
        assert upright["low"]["mean"] < 0.01
        assert inverted["summary"]["median"] > 0.5

    def test_constant_scores_caught_as_no_dynamic_range(self):
        # A near-constant array cannot support the difficulty axis: the primary condition would
        # not be harder than the negative control, so H4 could not be tested at all.
        with pytest.raises(ValueError, match="no dynamic range"):
            check_scores(np.full(50_000, 0.2))

    def test_overlapping_strata_are_caught(self):
        # medium-mem is "nearest to 0.5" on the score scale, so a distribution with too thin a
        # tail above 0.5 makes it reach into the top of the ranking and collide with high-mem.
        # Two conditions sharing examples are confounded, not independent.
        rng = np.random.default_rng(0)
        thin_tail = rng.beta(0.35, 2.0, size=50_000)
        with pytest.raises(ValueError, match="strata overlap"):
            check_scores(thin_tail)

    def test_a_plausible_distribution_passes(self):
        rng = np.random.default_rng(1)
        report = check_scores(np.where(rng.random(50_000) < 0.65, rng.beta(0.5, 12, 50_000), rng.beta(1.5, 1.5, 50_000)))
        assert report["high"]["mean"] > report["low"]["mean"]

    def test_strict_can_be_disabled_for_diagnosis(self):
        report = check_scores(np.full(50_000, 0.9), strict=False)
        assert report["summary"]["low_to_high_spread"] == 0.0
