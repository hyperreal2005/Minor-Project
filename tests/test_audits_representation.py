"""Layer 4 — representation similarity.

CKA has well-known mathematical properties (invariance to rotation and isotropic scaling, not to
arbitrary linear maps) and those are what is tested here, because they are what make it a
similarity measure at all. The degenerate cases get equal attention: a constant representation
must report *undefined*, never 0.0, since 0.0 reads as "maximally dissimilar" and would be
averaged into a family mean as if it were a measurement.
"""

import numpy as np
import pytest

from forgetcheck.audits import AuditContext, get_audit
from forgetcheck.audits.representation import (
    center_gram,
    gram_linear,
    linear_cka,
    rbf_cka,
)

N, D = 60, 16


def acts(seed=0, n=N, d=D):
    return np.random.default_rng(seed).normal(size=(n, d))


def ctx_for(target, oracles, layers=("layer1", "layer2"), degen_logits=None):
    return AuditContext(
        run_id="c10r18__unlearn__mem-high-3000__salun__train1",
        logits={"forget": degen_logits if degen_logits is not None else acts(99, n=N, d=10)},
        activations={ell: target for ell in layers},
        oracle_activations={ell: oracles for ell in layers},
        forget_kind="memstratum",
        forget_size=3000,
        config={"layers": list(layers), "rbf_sigma": "median"},
    )


class TestCKAProperties:
    def test_identical_representations_score_one(self):
        x = acts(0)
        assert linear_cka(x, x) == pytest.approx(1.0)
        assert rbf_cka(x, x) == pytest.approx(1.0)

    def test_invariant_to_isotropic_scaling(self):
        x = acts(1)
        assert linear_cka(x, 7.5 * x) == pytest.approx(1.0)

    def test_invariant_to_orthogonal_rotation(self):
        x = acts(2)
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(D, D)))
        assert linear_cka(x, x @ q) == pytest.approx(1.0)

    def test_invariant_to_translation(self):
        # Double-centring is what buys this; a missing centre would show up here.
        x = acts(4)
        assert linear_cka(x, x + 11.0) == pytest.approx(1.0)

    def test_symmetric(self):
        x, y = acts(5), acts(6)
        assert linear_cka(x, y) == pytest.approx(linear_cka(y, x))

    def test_bounded_in_unit_interval(self):
        for s in range(6):
            v = linear_cka(acts(s), acts(s + 100))
            assert -1e-9 <= v <= 1 + 1e-9

    def test_unrelated_representations_score_below_related_ones(self):
        x = acts(7)
        near = x + 0.05 * acts(8)
        far = acts(9)
        assert linear_cka(x, near) > linear_cka(x, far)

    def test_not_invariant_to_an_arbitrary_linear_map(self):
        # CKA is invariant to orthogonal transforms but NOT to general invertible ones. If this
        # ever passed, the implementation would be measuring something weaker than claimed.
        x = acts(10)
        a = np.random.default_rng(11).normal(size=(D, D)) * np.linspace(0.1, 10, D)
        assert linear_cka(x, x @ a) < 0.99

    def test_centering_a_gram_matrix_zeroes_its_row_means(self):
        g = center_gram(gram_linear(acts(12)))
        np.testing.assert_allclose(g.mean(axis=0), 0.0, atol=1e-8)


class TestDegenerateRepresentations:
    def test_a_constant_representation_is_undefined_not_zero(self):
        # A collapsed model. 0.0 would read as "maximally dissimilar" and be averaged in as a
        # measurement; NaN cannot be.
        const = np.ones((N, D))
        assert np.isnan(linear_cka(const, acts(0)))
        assert np.isnan(rbf_cka(const, acts(0)))

    def test_both_constant_is_also_undefined(self):
        assert np.isnan(linear_cka(np.ones((N, D)), np.full((N, D), 3.0)))

    def test_rbf_median_bandwidth_of_zero_does_not_divide_by_zero(self):
        # Identical rows give a median pairwise distance of zero.
        v = rbf_cka(np.ones((N, D)), acts(1))
        assert np.isnan(v) or np.isfinite(v)

    def test_one_undefined_oracle_does_not_poison_the_layer(self):
        target = acts(0)
        oracles = np.stack([np.ones((N, D)), acts(1), acts(2)])  # first is degenerate
        out = get_audit("representation").measure(ctx_for(target, oracles))
        assert np.isfinite(out[("cka_linear", "layer1")])

    def test_all_undefined_stays_undefined(self):
        out = get_audit("representation").measure(
            ctx_for(np.ones((N, D)), np.stack([np.ones((N, D))] * 2))
        )
        assert np.isnan(out[("cka_linear", "layer1")])

    def test_non_finite_activations_are_undefined(self):
        t = acts(0).copy()
        t[0, 0] = np.inf
        out = get_audit("representation").measure(ctx_for(t, np.stack([acts(1)])))
        assert all(np.isnan(out[(m, "layer1")]) for m in ("cka_linear", "cka_rbf", "activation_l2"))


class TestRepresentationAudit:
    def test_emits_every_metric_for_every_layer(self):
        out = get_audit("representation").measure(
            ctx_for(acts(0), np.stack([acts(1), acts(2)]), layers=("layer1", "layer2", "layer3"))
        )
        assert {p for _, p in out} == {"layer1", "layer2", "layer3"}
        assert {m for m, _ in out} == {"cka_linear", "cka_rbf", "activation_l2"}

    def test_a_model_matching_the_oracles_scores_near_one(self):
        # The oracle-vs-oracle baseline case: Stage 7 gets it by passing held-out oracles
        # through this same path, so it must hold with no special-casing.
        x = acts(0)
        out = get_audit("representation").measure(ctx_for(x, np.stack([x, x])))
        assert out[("cka_linear", "layer1")] == pytest.approx(1.0)
        assert out[("activation_l2", "layer1")] == pytest.approx(0.0, abs=1e-9)

    def test_a_divergent_model_scores_lower_than_a_close_one(self):
        oracles = np.stack([acts(1), acts(2)])
        close = 0.5 * (acts(1) + acts(2))
        far = acts(50) * 3.0
        a = get_audit("representation")
        assert (
            a.measure(ctx_for(close, oracles))[("cka_linear", "layer1")]
            > a.measure(ctx_for(far, oracles))[("cka_linear", "layer1")]
        )

    def test_layers_without_activations_are_skipped_not_faked(self):
        ctx = ctx_for(acts(0), np.stack([acts(1)]), layers=("layer1",))
        ctx.config = {"layers": ["layer1", "layer4"], "rbf_sigma": "median"}
        out = get_audit("representation").measure(ctx)
        assert {p for _, p in out} == {"layer1"}

    def test_notes_flag_a_collapsed_model(self):
        const_logits = np.full((N, 10), -20.0)
        const_logits[:, 3] = 20.0
        ctx = ctx_for(np.ones((N, D)), np.stack([acts(1)]), degen_logits=const_logits)
        assert "undefined" in get_audit("representation").notes_for(ctx)
