"""Stage 2 gate: forget sets resolve deterministically and the strata are what they claim.

The gate reads: "every ForgetSpec regenerates a byte-identical index array across two machines;
strata means match RUM's published values."

Cross-machine identity cannot be tested here directly, so these tests check the two properties
that make it true: resolution is a pure function of (spec, labels), and the ranking has no
implementation-defined tiebreak.
"""

import numpy as np
import pytest

from forgetcheck.data.forget_sets import (
    PRIMARY_CONDITION,
    STANDARD_CONDITIONS,
    ForgetSpec,
    all_specs,
    materialise,
    spec_by_id,
)

N = 50_000


@pytest.fixture
def scores():
    """A memorization-score distribution shaped like the real one.

    Feldman-Zhang memorization on CIFAR-10 is heavily bottom-weighted -- most examples are not
    memorised at all -- with a genuine spread above. This mixture reproduces that: ~65% drawn
    near zero, the rest spread across the range, giving real mass around 0.5 for the medium
    stratum to select from.

    An earlier fixture used beta(0.35, 2), whose tail above 0.5 is too thin: "nearest to 0.5"
    then reaches up into the top of the ranking and collides with the high stratum. That is a
    real failure mode of RUM's medium-mem definition, and check_scores() now detects it -- but
    it is not what CIFAR-10 looks like.
    """
    rng = np.random.default_rng(0)
    return np.where(
        rng.random(N) < 0.65, rng.beta(0.5, 12, N), rng.beta(1.5, 1.5, N)
    )


class TestIdentity:
    def test_ids_match_the_plan(self):
        assert [s.forget_id for s in STANDARD_CONDITIONS] == [
            "rand-500", "rand-2500", "rand-5000", "rand-3000",
            "mem-low-3000", "mem-med-3000", "mem-high-3000", "canary-500",
        ]

    def test_eight_conditions_all_distinct(self):
        assert len(all_specs()) == 8

    def test_primary_is_high_memorization(self):
        assert PRIMARY_CONDITION.forget_id == "mem-high-3000"

    def test_lookup(self):
        assert spec_by_id("mem-high-3000") == PRIMARY_CONDITION
        with pytest.raises(KeyError, match="unknown forget condition"):
            spec_by_id("mem-highest-3000")


class TestValidation:
    def test_memstratum_requires_a_stratum(self):
        with pytest.raises(ValueError, match="requires a stratum"):
            ForgetSpec(kind="memstratum", size=3000)

    def test_random_rejects_a_stratum(self):
        with pytest.raises(ValueError, match="does not take a stratum"):
            ForgetSpec(kind="random", size=500, stratum="high")

    def test_bad_stratum(self):
        with pytest.raises(ValueError, match="stratum must be one of"):
            ForgetSpec(kind="memstratum", size=3000, stratum="highest")

    def test_size_must_be_positive(self):
        with pytest.raises(ValueError, match="size must be positive"):
            ForgetSpec(kind="random", size=0)

    def test_oversized_forget_set_rejected(self):
        with pytest.raises(ValueError, match="exceeds training set size"):
            ForgetSpec(kind="random", size=100).indices(50)

    def test_memstratum_without_scores_explains_itself(self):
        with pytest.raises(ValueError, match="memorization scores are required"):
            PRIMARY_CONDITION.indices(N)

    def test_wrong_score_length_rejected(self, scores):
        with pytest.raises(ValueError, match="expected"):
            PRIMARY_CONDITION.indices(N, memorization=scores[:100])


class TestDeterminism:
    @pytest.mark.parametrize("spec", STANDARD_CONDITIONS)
    def test_resolution_is_pure(self, spec, scores):
        kw = {"memorization": scores} if spec.kind == "memstratum" else {}
        a = spec.indices(N, **kw)
        b = spec.indices(N, **kw)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("spec", STANDARD_CONDITIONS)
    def test_canonical_form(self, spec, scores):
        kw = {"memorization": scores} if spec.kind == "memstratum" else {}
        idx = spec.indices(N, **kw)
        assert idx.dtype == np.int64
        assert len(idx) == spec.size
        assert len(np.unique(idx)) == spec.size
        assert np.all(np.diff(idx) > 0), "must be sorted ascending to be canonical"
        assert idx.min() >= 0 and idx.max() < N

    def test_size_axis_is_nested(self):
        # rand-500 subset rand-2500 subset rand-3000 subset rand-5000, so a difference between
        # size conditions is attributable to size alone rather than to which examples were drawn.
        sizes = [500, 2500, 3000, 5000]
        sets = [set(ForgetSpec(kind="random", size=k).indices(N).tolist()) for k in sizes]
        for smaller, larger, a, b in zip(sets, sets[1:], sizes, sizes[1:]):
            assert smaller <= larger, f"rand-{a} is not nested in rand-{b}"

    def test_canary_and_random_draw_independently(self):
        # Same seed and same size, but they must not select the same examples: the canary
        # condition would otherwise corrupt exactly what rand-500 forgets, confounding two
        # conditions that are meant to be independent.
        r = ForgetSpec(kind="random", size=500).indices(N)
        c = ForgetSpec(kind="canary", size=500).indices(N)
        assert not np.array_equal(r, c)
        overlap = len(set(r.tolist()) & set(c.tolist()))
        assert overlap < 50, f"{overlap}/500 shared -- streams are not independent"

    def test_different_seeds_give_different_random_sets(self):
        a = ForgetSpec(kind="random", size=500).indices(N)
        b = ForgetSpec(kind="random", size=500, selection_seed=101).indices(N)
        assert not np.array_equal(a, b)

    def test_ties_are_broken_by_index_not_sort_order(self):
        # CIFAR-10 memorization has a large mass of exactly-zero scores. If the tiebreak were
        # implementation-defined, the low stratum would differ across numpy versions — breaking
        # the cross-machine gate in the one place it is most likely to bite.
        flat = np.zeros(1000)
        spec = ForgetSpec(kind="memstratum", size=100, stratum="low")
        np.testing.assert_array_equal(spec.indices(1000, memorization=flat), np.arange(100))

    def test_seed_does_not_affect_stratified_selection(self, scores):
        # Stratified selection is deterministic given the scores; the seed is irrelevant here,
        # and a spec that quietly depended on it would be a reproducibility trap.
        a = PRIMARY_CONDITION.indices(N, memorization=scores)
        b = PRIMARY_CONDITION.with_seed(999).indices(N, memorization=scores)
        np.testing.assert_array_equal(a, b)


class TestStrata:
    def test_strata_are_ordered_by_mean_score(self, scores):
        from forgetcheck.data.forget_sets import stratum_summary

        s = stratum_summary(scores)
        assert s["low"]["mean"] < s["medium"]["mean"] < s["high"]["mean"], s

    def test_stratum_summary_shape(self, scores):
        from forgetcheck.data.forget_sets import stratum_summary

        s = stratum_summary(scores)
        assert set(s) == {"low", "medium", "high"}
        assert set(s["low"]) == {"mean", "std", "min", "max"}

    def test_strata_are_disjoint(self, scores):
        sets = [
            set(ForgetSpec(kind="memstratum", size=3000, stratum=s).indices(N, memorization=scores).tolist())
            for s in ("low", "medium", "high")
        ]
        assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])

    def test_high_stratum_takes_the_top_of_the_distribution(self, scores):
        idx = PRIMARY_CONDITION.indices(N, memorization=scores)
        threshold = np.sort(scores)[-3000]
        assert scores[idx].min() >= threshold - 1e-12

    def test_medium_is_nearest_to_half_on_the_score_scale(self, scores):
        # RUM defines medium-mem as the N examples "nearest to 0.5, i.e. the midpoint of the
        # range of memorization scores" -- a point on the SCORE scale, not the rank scale.
        idx = ForgetSpec(kind="memstratum", size=3000, stratum="medium").indices(
            N, memorization=scores
        )
        selected = np.abs(scores[idx] - 0.5)
        rest = np.abs(np.delete(scores, idx) - 0.5)
        assert selected.max() <= rest.min() + 1e-12

    def test_medium_does_not_collapse_toward_low(self, scores):
        # The failure a rank-defined medium produces on a bottom-weighted distribution: the
        # middle of the rank ordering still sits near zero, so medium stops interpolating.
        from forgetcheck.data.forget_sets import stratum_summary

        s = stratum_summary(scores)
        low_to_med = s["medium"]["mean"] - s["low"]["mean"]
        med_to_high = s["high"]["mean"] - s["medium"]["mean"]
        assert low_to_med > 0.1, "medium has collapsed onto low"
        assert med_to_high > 0.0


class TestMaterialise:
    def test_caches_on_first_call(self, tmp_path, scores):
        from forgetcheck.registry import ArtifactStore

        store = ArtifactStore(tmp_path)
        idx = materialise(PRIMARY_CONDITION, N, memorization=scores, store=store)
        assert store.has_forget_set("mem-high-3000")
        cached, meta = store.load_forget_set("mem-high-3000")
        np.testing.assert_array_equal(cached, idx)
        assert meta["forget_stratum"] == "high"

    def test_second_call_agrees_with_the_cache(self, tmp_path, scores):
        from forgetcheck.registry import ArtifactStore

        store = ArtifactStore(tmp_path)
        materialise(PRIMARY_CONDITION, N, memorization=scores, store=store)
        materialise(PRIMARY_CONDITION, N, memorization=scores, store=store)  # no raise

    def test_drift_from_the_cache_is_fatal(self, tmp_path, scores):
        # This is the check that catches a changed numpy, changed scores, or changed spec
        # before any result is computed against the wrong index array.
        from forgetcheck.registry import ArtifactStore

        store = ArtifactStore(tmp_path)
        materialise(PRIMARY_CONDITION, N, memorization=scores, store=store)
        shifted = scores + np.linspace(0, 1, N)
        with pytest.raises(RuntimeError, match="differ from the cached"):
            materialise(PRIMARY_CONDITION, N, memorization=shifted, store=store)


def test_record_fields_round_trip():
    fields = PRIMARY_CONDITION.as_record_fields()
    assert fields == {
        "forget_id": "mem-high-3000",
        "forget_kind": "memstratum",
        "forget_size": 3000,
        "forget_stratum": "high",
        "selection_seed": 100,
    }
