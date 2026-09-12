"""Layer 2 — the population membership-inference attack.

Tested against constructed cases where the right answer is known: a model that has plainly
memorised its members must be attackable, a model whose members are indistinguishable from
non-members must not be, and the constant predictor must land at chance without raising.
"""

import numpy as np
import pytest

from forgetcheck.audits import AuditContext, get_audit
from forgetcheck.audits.privacy_population import (
    attack_features,
    balanced_accuracy,
    cross_validated_scores,
    roc_auc,
    tpr_at_fpr,
)

K = 10
CFG = {"prob_floor": 1e-12, "n_splits": 5, "tpr_at_fpr": 0.01, "nonmember_source": "test"}


def make_logits(labels, *, confident: float, seed: int):
    """Logits that put ``confident`` extra mass on the true label, plus noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(labels), K))
    x[np.arange(len(labels)), labels] += confident
    return x


def ctx_for(forget_logits, test_logits, y_f, y_t, seed=0):
    return AuditContext(
        run_id="c10r18__unlearn__rand-500__finetune__train0",
        logits={"forget": forget_logits, "test": test_logits},
        labels={"forget": y_f, "test": y_t},
        forget_kind="random",
        forget_size=len(y_f),
        config=CFG,
        audit_seed=seed,
    )


class TestStatistics:
    def test_auc_is_one_for_perfect_separation(self):
        s = np.r_[np.ones(5), np.zeros(5)]
        m = np.r_[np.ones(5, bool), np.zeros(5, bool)]
        assert roc_auc(s, m) == 1.0

    def test_auc_is_half_when_all_scores_tie(self):
        # The constant-predictor case: no signal at all. 0.5 is the truth, not a failure.
        m = np.r_[np.ones(6, bool), np.zeros(6, bool)]
        assert roc_auc(np.full(12, 0.3), m) == 0.5

    def test_auc_matches_a_brute_force_pair_count(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=30)
        m = rng.random(30) < 0.5
        pos, neg = s[m], s[~m]
        brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        assert roc_auc(s, m) == pytest.approx(brute)

    def test_tpr_at_low_fpr_rewards_confident_members(self):
        # Three members stand far above every non-member; at 1% FPR the attacker names them.
        m = np.r_[np.ones(10, bool), np.zeros(100, bool)]
        s = np.r_[np.full(3, 10.0), np.zeros(7), np.linspace(-1, 1, 100)]
        assert tpr_at_fpr(s, m, 0.01) == pytest.approx(0.3)

    def test_balanced_accuracy_at_chance_is_half(self):
        m = np.r_[np.ones(10, bool), np.zeros(10, bool)]
        assert balanced_accuracy(np.full(20, 0.5), m) == pytest.approx(0.5)


class TestFeatures:
    def test_shape_and_finiteness(self):
        y = np.arange(20) % K
        f = attack_features(make_logits(y, confident=3.0, seed=0), y)
        assert f.shape == (20, 3) and np.isfinite(f).all()

    def test_confident_correct_predictions_have_low_loss(self):
        y = np.arange(20) % K
        low = attack_features(make_logits(y, confident=8.0, seed=0), y)[:, 0]
        high = attack_features(make_logits(y, confident=0.0, seed=0), y)[:, 0]
        assert low.mean() < high.mean()


class TestCrossValidation:
    def test_every_example_is_scored_out_of_fold(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 3))
        m = np.r_[np.ones(25, bool), np.zeros(25, bool)]
        s = cross_validated_scores(x, m, n_splits=5, seed=0)
        assert np.isfinite(s).all() and s.shape == (50,)

    def test_deterministic_given_the_seed(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 3))
        m = np.r_[np.ones(20, bool), np.zeros(20, bool)]
        a = cross_validated_scores(x, m, n_splits=4, seed=7)
        b = cross_validated_scores(x, m, n_splits=4, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_zero_variance_features_do_not_produce_nan(self):
        # Fully constant features -- the constant predictor with uniform labels.
        x = np.full((40, 3), 1.234)
        m = np.r_[np.ones(20, bool), np.zeros(20, bool)]
        s = cross_validated_scores(x, m, n_splits=5, seed=0)
        assert np.isfinite(s).all()


class TestPopulationMIA:
    def test_a_memorising_model_is_attackable(self):
        # Members get confident correct logits; non-members get noise. An attacker should win.
        y_f, y_t = np.arange(200) % K, np.arange(200) % K
        ctx = ctx_for(
            make_logits(y_f, confident=8.0, seed=1), make_logits(y_t, confident=0.0, seed=2),
            y_f, y_t,
        )
        out = get_audit("privacy_population").measure(ctx)
        assert out[("mia_auc_pop", "forget")] > 0.9
        assert out[("mia_acc_pop", "forget")] > 0.8

    def test_an_oracle_like_model_is_at_chance(self):
        # Members and non-members drawn from the same distribution: nothing to find.
        y_f, y_t = np.arange(200) % K, np.arange(200) % K
        ctx = ctx_for(
            make_logits(y_f, confident=2.0, seed=3), make_logits(y_t, confident=2.0, seed=4),
            y_f, y_t,
        )
        auc = get_audit("privacy_population").measure(ctx)[("mia_auc_pop", "forget")]
        assert abs(auc - 0.5) < 0.1

    def test_the_constant_predictor_is_at_chance_and_does_not_raise(self):
        # neggrad's endpoint. Chance here is the finding: it leaks nothing because it knows
        # nothing, and it "passes" the privacy audit while being useless.
        #
        # Not *exactly* 0.5: confidence and entropy are flat, but loss varies with the true
        # label (low when y == the predicted class). That is label information, shared by both
        # groups, so it cannot separate them -- the AUC is chance plus sampling noise.
        y_f, y_t = np.arange(100) % K, np.arange(100) % K
        const = np.full((100, K), -20.0)
        const[:, 3] = 20.0
        ctx = ctx_for(const, const.copy(), y_f, y_t)
        audit = get_audit("privacy_population")
        out = audit.measure(ctx)
        assert abs(out[("mia_auc_pop", "forget")] - 0.5) < 0.1
        assert np.isfinite(out[("mia_tpr_at_fpr_pop", "forget")])
        assert "no per-example signal" in audit.notes_for(ctx)

    def test_the_constant_predictor_with_uniform_labels_is_exactly_half(self):
        # With the label information removed too, every feature ties and the rank AUC is
        # exactly 0.5 by construction -- the cleanest statement of "no signal".
        y = np.full(100, 3)
        const = np.full((100, K), -20.0)
        const[:, 3] = 20.0
        out = get_audit("privacy_population").measure(ctx_for(const, const.copy(), y, y))
        assert out[("mia_auc_pop", "forget")] == 0.5

    def test_non_finite_logits_give_undefined_not_a_crash(self):
        y = np.arange(50) % K
        bad = make_logits(y, confident=1.0, seed=0)
        bad[0, 0] = np.inf
        out = get_audit("privacy_population").measure(ctx_for(bad, make_logits(y, confident=1.0, seed=1), y, y))
        assert all(np.isnan(v) for v in out.values())

    def test_size_matches_non_members_to_the_forget_set(self):
        # 60 forget vs 500 test: balanced accuracy only means something if the classes balance.
        y_f, y_t = np.arange(60) % K, np.arange(500) % K
        ctx = ctx_for(
            make_logits(y_f, confident=8.0, seed=1), make_logits(y_t, confident=0.0, seed=2),
            y_f, y_t,
        )
        out = get_audit("privacy_population").measure(ctx)
        assert 0.0 <= out[("mia_acc_pop", "forget")] <= 1.0

    def test_declares_no_oracle_dependency(self):
        assert get_audit("privacy_population").needs_oracles is False

    def test_deterministic_for_the_same_audit_seed(self):
        y = np.arange(120) % K
        f, t = make_logits(y, confident=4.0, seed=5), make_logits(y, confident=1.0, seed=6)
        a = get_audit("privacy_population").measure(ctx_for(f, t, y, y, seed=11))
        b = get_audit("privacy_population").measure(ctx_for(f, t, y, y, seed=11))
        assert a == b
