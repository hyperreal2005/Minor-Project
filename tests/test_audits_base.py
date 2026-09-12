"""The audit interface, degeneracy detection, and Layer 1.

The properties asserted here are the ones that would quietly corrupt the contribution rather
than crash: an audit that raises on the destructive control would drop it from the comparison —
and the destructive control is the case that makes the audit-validity argument. So "does not
crash" is tested as carefully as "computes the right number".
"""

import numpy as np
import pytest

from forgetcheck.audits.base import (
    UNDEFINED,
    Audit,
    AuditContext,
    audit_names,
    describe_degeneracy,
    get_audit,
    register,
    softmax,
)
from forgetcheck.audits.behavioral import js_divergence

N, K = 40, 10


def logits(seed=0, scale=3.0, n=N):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, K))


def constant_logits(n=N, cls=3):
    """What Stage 5's neggrad control actually produced: one class for every input."""
    x = np.full((n, K), -20.0)
    x[:, cls] = 20.0
    return x


def ctx_for(target, *, oracles=2, original=True, seed=0):
    return AuditContext(
        run_id="c10r18__unlearn__mem-high-3000__neggrad__train0",
        logits={"forget": target, "retain": logits(9), "test": logits(8)},
        original_logits=(
            {"forget": logits(7), "retain": logits(6), "test": logits(5)} if original else {}
        ),
        oracle_logits={
            p: np.stack([logits(100 + i + j * 17) for i in range(oracles)])
            for j, p in enumerate(("forget", "retain", "test"))
        }
        if oracles
        else {},
        labels={"forget": np.arange(N) % K},
        forget_kind="memstratum",
        forget_size=3000,
        config={"prob_floor": 1e-12, "probe_sets": ["forget", "retain", "test"]},
        audit_seed=seed,
    )


class TestDegeneracy:
    def test_detects_a_constant_predictor(self):
        d = describe_degeneracy(constant_logits())
        assert d.is_constant
        assert d.n_predicted_classes == 1
        assert not d.has_per_example_signal
        assert "constant predictor" in d.summary

    def test_a_healthy_model_is_not_flagged(self):
        d = describe_degeneracy(logits())
        assert not d.is_constant
        assert d.n_predicted_classes > 1
        assert d.has_per_example_signal
        assert d.summary == ""

    def test_survives_the_enormous_logits_gradient_ascent_produces(self):
        # Stage 5's collapsed control reached cross-entropy 82. A naive exp() overflows here.
        huge = logits(1) * 1e3
        d = describe_degeneracy(huge)
        assert np.isfinite(d.confidence_std)
        assert d.nonfinite_frac == 0.0

    def test_reports_non_finite_logits_rather_than_propagating_them(self):
        x = logits(2)
        x[0, 0] = np.inf
        x[1, 3] = np.nan
        d = describe_degeneracy(x)
        assert d.nonfinite_frac == pytest.approx(2 / N)
        assert "non-finite" in d.summary

    def test_all_non_finite_does_not_raise(self):
        d = describe_degeneracy(np.full((5, K), np.nan))
        assert d.nonfinite_frac == 1.0
        assert not d.has_per_example_signal

    def test_rejects_a_wrong_shape_loudly(self):
        with pytest.raises(ValueError, match="n_probes, n_classes"):
            describe_degeneracy(np.zeros(10))


class TestSoftmax:
    def test_rows_sum_to_one(self):
        p = softmax(logits())
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_stable_against_huge_logits(self):
        p = softmax(logits(3) * 1e4)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_floor_removes_exact_zeros(self):
        p = softmax(constant_logits() * 100, floor=1e-12)
        assert (p > 0).all(), "a zero probability makes the log-based divergence infinite"
        np.testing.assert_allclose(p.sum(axis=1), 1.0)


class TestJSDivergence:
    def test_zero_against_itself(self):
        p = softmax(logits())
        np.testing.assert_allclose(js_divergence(p, p), 0.0, atol=1e-12)

    def test_symmetric(self):
        p, q = softmax(logits(1)), softmax(logits(2))
        np.testing.assert_allclose(js_divergence(p, q), js_divergence(q, p))

    def test_bounded_by_ln2(self):
        # The bound is why this is readable next to a cross-entropy of 82.
        p = softmax(constant_logits(cls=0) * 50)
        q = softmax(constant_logits(cls=7) * 50)
        assert np.all(js_divergence(p, q) <= np.log(2) + 1e-9)

    def test_finite_for_disjoint_support(self):
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.0, 1.0]])
        assert np.isfinite(js_divergence(p, q)).all()


class TestBehavioral:
    def test_emits_only_the_reference_requiring_metrics(self):
        # Stage 5 already records the raw accuracies under audit="meta"; re-emitting them here
        # would double-count the behavioural family in the agreement analysis.
        out = get_audit("behavior").measure(ctx_for(logits()))
        assert {m for m, _ in out} == {
            "js_to_oracle",
            "js_to_original",
            "pred_agreement",
            "logit_l2",
        }
        for raw in ("retain_acc", "test_acc", "forget_acc", "ce_loss", "macro_f1"):
            assert all(m != raw for m, _ in out)

    def test_a_model_identical_to_an_oracle_scores_zero_divergence(self):
        # The gate: oracle-vs-oracle JS is ~0. Stage 7 gets this by passing held-out oracles
        # through the ordinary path, so it must hold with no special-casing.
        same = logits(55)
        ctx = ctx_for(same)
        ctx.oracle_logits = {"forget": np.stack([same, same])}
        ctx.config = {"prob_floor": 1e-12, "probe_sets": ["forget"]}
        out = get_audit("behavior").measure(ctx)
        assert out[("js_to_oracle", "forget")] == pytest.approx(0.0, abs=1e-12)
        assert out[("pred_agreement", "forget")] == pytest.approx(1.0)
        assert out[("logit_l2", "forget")] == pytest.approx(0.0, abs=1e-12)

    def test_disagreeing_model_scores_higher_than_an_agreeing_one(self):
        oracle = logits(55)
        near = oracle + np.random.default_rng(0).normal(scale=0.01, size=oracle.shape)
        far = logits(77)

        def js(target):
            ctx = ctx_for(target)
            ctx.oracle_logits = {"forget": np.stack([oracle])}
            ctx.config = {"prob_floor": 1e-12, "probe_sets": ["forget"]}
            return get_audit("behavior").measure(ctx)[("js_to_oracle", "forget")]

        assert js(near) < js(far)

    def test_does_not_crash_on_the_constant_predictor(self):
        # neggrad collapses to this. If the audit raised, the destructive control would drop out
        # of the comparison -- and it is the case that makes the audit-validity argument.
        out = get_audit("behavior").measure(ctx_for(constant_logits()))
        assert out, "produced no measurements at all"
        assert all(np.isfinite(v) for v in out.values())

    def test_records_undefined_rather_than_nan_poisoning_on_bad_logits(self):
        bad = logits(4)
        bad[0, 0] = np.inf
        out = get_audit("behavior").measure(ctx_for(bad))
        forget = {m: v for (m, p), v in out.items() if p == "forget"}
        assert forget and all(np.isnan(v) for v in forget.values())
        # Other probe sets are unaffected: one broken probe must not discard the whole model.
        assert np.isfinite(out[("js_to_oracle", "retain")])

    def test_skips_oracle_metrics_when_no_ensemble_is_present(self):
        out = get_audit("behavior").measure(ctx_for(logits(), oracles=0))
        assert {m for m, _ in out} == {"js_to_original"}

    def test_notes_carry_the_degeneracy_reason(self):
        assert "constant predictor" in get_audit("behavior").notes_for(ctx_for(constant_logits()))
        assert get_audit("behavior").notes_for(ctx_for(logits())) == ""


class TestRegistry:
    def test_behavior_is_registered(self):
        assert "behavior" in audit_names()

    def test_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="known:"):
            get_audit("no-such-audit")

    def test_duplicate_names_are_refused(self):
        # Two audits under one name would make the `audit` column ambiguous, and every
        # cross-family comparison groups on it.
        class A(Audit):
            name = "behavior"

            def measure(self, ctx):
                return {}

        with pytest.raises(ValueError, match="already registered"):
            register(A)

    def test_an_audit_must_name_itself(self):
        class Nameless(Audit):
            def measure(self, ctx):
                return {}

        with pytest.raises(ValueError, match="non-empty"):
            register(Nameless)
