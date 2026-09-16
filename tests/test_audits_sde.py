"""Layer 6 — SDE (split-half statistical independence).

The HSIC properties are tested directly because they are what the audit rests on. The
constant-predictor case is tested for *behaviour*, not for a predicted sign: which way SDE calls
a destroyed model is an open question this layer exists to answer, and a test that asserted the
answer in advance would be assuming the result.
"""

import numpy as np
import pytest

from forgetcheck.audits import AuditContext, get_audit
from forgetcheck.audits.sde import distribution_jsd, hsic, split_half_distribution

K = 10
CFG = {"n_draws": 40, "jsd_bins": 16, "in_training_source": "retain",
       "out_of_training_source": "test"}


def indep(n=64, d=K, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def cotrained_pool(n=128, d=K, seed=0, strength=4.0):
    """One pool of rows sharing a common factor -- the shape co-training induces.

    Returned as a pool and split by the caller, because "resembles the in-training reference"
    only means something if the target and that reference come from the *same* process. An
    earlier version of these tests drew two pools with independent common factors and called
    them both "dependent"; they were no more alike than either was to noise, and the test
    failed for a reason that had nothing to do with the audit.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(1, d))
    return rng.normal(size=(n, d)) + strength * common


def ctx_for(forget, retain, test, seed=0, cfg=None):
    return AuditContext(
        run_id="c10r18__unlearn__rand-2500__finetune__train0",
        logits={"forget": forget, "retain": retain, "test": test},
        labels={"forget": np.arange(len(forget)) % K},
        forget_kind="random",
        forget_size=len(forget),
        config=cfg or CFG,
        audit_seed=seed,
    )


class TestHSIC:
    def test_independent_data_gives_a_small_value(self):
        x, y = indep(seed=1), indep(seed=2)
        assert abs(hsic(x, y)) < 0.05

    def test_identical_data_gives_a_larger_value_than_independent(self):
        x = indep(seed=3)
        assert hsic(x, x) > abs(hsic(x, indep(seed=4)))

    def test_symmetric(self):
        x, y = indep(seed=5), indep(seed=6)
        assert hsic(x, y) == pytest.approx(hsic(y, x))

    def test_a_constant_side_gives_exactly_zero(self):
        # A constant kernel matrix double-centres to zero. Informative, not undefined.
        assert hsic(np.ones((20, K)), indep(n=20, seed=7)) == 0.0

    def test_mismatched_lengths_are_undefined(self):
        assert np.isnan(hsic(indep(n=10), indep(n=12), sigma=1.0))

    def test_too_few_rows_is_undefined(self):
        assert np.isnan(hsic(indep(n=1), indep(n=1), sigma=1.0))


class TestSplitHalfDistribution:
    def test_returns_one_value_per_draw(self):
        d = split_half_distribution(indep(), n_draws=25, rng=np.random.default_rng(0))
        assert d.shape == (25,) and np.isfinite(d).all()

    def test_the_bandwidth_gives_a_usable_dynamic_range(self):
        """The paper's sqrt(dim) heuristic on 10-class softmax outputs makes exp(-d2/2s2) ~ 1
        for every pair, so HSIC collapses to ~1e-7 of float noise. The median bandwidth must
        keep it several orders of magnitude above that."""
        from forgetcheck.audits.base import softmax

        feats = softmax(cotrained_pool(seed=3)[:64])
        wide = split_half_distribution(feats, n_draws=40, sigma=float(np.sqrt(K)),
                                       rng=np.random.default_rng(0))
        med = split_half_distribution(feats, n_draws=40, sigma="median",
                                      rng=np.random.default_rng(0))
        assert abs(wide).mean() < 1e-5, "sqrt(dim) should be degenerate here"
        assert abs(med).mean() > 1e-3, "median bandwidth must resolve real structure"

    def test_a_tiny_subset_yields_nothing_rather_than_garbage(self):
        d = split_half_distribution(indep(n=3), n_draws=10, rng=np.random.default_rng(0))
        assert d.size == 0

    def test_deterministic_given_the_generator(self):
        a = split_half_distribution(indep(), n_draws=10, sigma=1.0, rng=np.random.default_rng(4))
        b = split_half_distribution(indep(), n_draws=10, sigma=1.0, rng=np.random.default_rng(4))
        np.testing.assert_allclose(a, b)


class TestDistributionJSD:
    def test_identical_samples_score_zero(self):
        a = np.linspace(0, 1, 50)
        assert distribution_jsd(a, a.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_disjoint_samples_score_higher_than_overlapping_ones(self):
        a = np.random.default_rng(0).normal(0, 1, 200)
        near = np.random.default_rng(1).normal(0.1, 1, 200)
        far = np.random.default_rng(2).normal(8, 1, 200)
        assert distribution_jsd(a, far) > distribution_jsd(a, near)

    def test_shares_a_support_so_identical_distributions_do_not_diverge(self):
        # Histogramming each sample on its own edges would report a divergence here.
        a = np.array([1.0, 2.0, 3.0] * 20)
        assert distribution_jsd(a, a.copy() + 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_two_degenerate_samples_at_the_same_point_are_identical_not_undefined(self):
        assert distribution_jsd(np.zeros(20), np.zeros(20)) == 0.0

    def test_empty_is_undefined(self):
        assert np.isnan(distribution_jsd(np.array([]), np.ones(5)))


class TestSDEAudit:
    def test_emits_all_three_metrics(self):
        out = get_audit("sde").measure(ctx_for(indep(seed=1), indep(seed=2), indep(seed=3)))
        assert {m for m, _ in out} == {"sde_hsic", "sde_margin", "sde_verdict"}

    def test_a_forget_set_from_the_training_process_is_called_in_training(self):
        # target and the in-training reference come from ONE co-trained pool; test does not.
        pool = cotrained_pool(seed=1)
        out = get_audit("sde").measure(ctx_for(pool[:64], pool[64:], indep(seed=3)))
        assert out[("sde_margin", "forget")] < 0
        assert out[("sde_verdict", "forget")] == 0.0

    def test_a_forget_set_from_the_held_out_process_is_called_unlearned(self):
        # target now comes from the same process as the out-of-training reference.
        held = indep(n=128, seed=5)
        out = get_audit("sde").measure(ctx_for(held[:64], cotrained_pool(seed=1)[:64], held[64:]))
        assert out[("sde_margin", "forget")] > 0
        assert out[("sde_verdict", "forget")] == 1.0

    def test_the_verdict_is_consistent_with_the_sign_of_the_margin(self):
        for s in range(4):
            held = indep(n=128, seed=s)
            out = get_audit("sde").measure(
                ctx_for(held[:64], cotrained_pool(seed=s + 10)[:64], held[64:])
            )
            assert out[("sde_verdict", "forget")] == float(out[("sde_margin", "forget")] > 0)

    def test_the_constant_predictor_produces_a_recordable_verdict(self):
        """Stage 5's neggrad endpoint. Whether SDE calls it a pass or an alarm is the open
        question; what is required here is that it produces a finite, recordable answer rather
        than raising or returning NaN, because that answer IS the audit-validity measurement."""
        const = np.full((64, K), -20.0)
        const[:, 3] = 20.0
        audit = get_audit("sde")
        out = audit.measure(ctx_for(const, const.copy(), const.copy()))
        assert out[("sde_hsic", "forget")] == 0.0, "constant outputs have zero dependence"
        assert np.isfinite(out[("sde_margin", "forget")])
        assert out[("sde_verdict", "forget")] in (0.0, 1.0)
        ctx = ctx_for(const, indep(seed=1), indep(seed=2))
        assert "audit-validity test" in audit.notes_for(ctx)

    def test_non_finite_logits_are_undefined(self):
        bad = indep(seed=0).copy()
        bad[0, 0] = np.inf
        out = get_audit("sde").measure(ctx_for(bad, indep(seed=1), indep(seed=2)))
        assert all(np.isnan(v) for v in out.values())

    def test_deterministic_for_the_same_audit_seed(self):
        f, r, t = indep(seed=1), indep(seed=2), indep(seed=3)
        a = get_audit("sde").measure(ctx_for(f, r, t, seed=9))
        b = get_audit("sde").measure(ctx_for(f, r, t, seed=9))
        assert a == b

    def test_needs_neither_oracles_nor_references(self):
        # The whole point of the method: retrain-free.
        a = get_audit("sde")
        assert a.needs_oracles is False and a.needs_references is False
