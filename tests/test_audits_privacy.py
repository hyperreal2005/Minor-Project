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


# --------------------------------------------------------------------------- Layer 3: RMIA


from forgetcheck.audits.privacy_rmia import offline_prior, rmia_scores, true_label_prob  # noqa: E402

RMIA_CFG = {"prob_floor": 1e-12, "gamma": 2.0, "tpr_at_fpr": 0.01, "offline_a": 0.3}


def rmia_ctx(f_log, t_log, y_f, y_t, ref_f, ref_t, in_f=None, seed=0, cfg=None):
    return AuditContext(
        run_id="c10r18__unlearn__rand-500__finetune__train0",
        logits={"forget": f_log, "test": t_log},
        labels={"forget": y_f, "test": y_t},
        reference_logits={"forget": ref_f, "test": ref_t},
        reference_in_mask={"forget": in_f} if in_f is not None else {},
        forget_kind="random",
        forget_size=len(y_f),
        config=cfg or RMIA_CFG,
        audit_seed=seed,
    )


def references(labels, n_refs, *, confident, seed, in_mask=None, in_confident=None):
    """Reference logits; where in_mask is True that model 'trained on' the example."""
    rng = np.random.default_rng(seed)
    out = np.stack([make_logits(labels, confident=confident, seed=seed + i) for i in range(n_refs)])
    if in_mask is not None and in_confident is not None:
        for j in range(n_refs):
            hit = np.flatnonzero(in_mask[j])
            out[j, hit, labels[hit]] += in_confident - confident
    return out


class TestRMIAStatistics:
    def test_true_label_prob_handles_both_shapes(self):
        y = np.arange(20) % K
        one = true_label_prob(make_logits(y, confident=3.0, seed=0), y)
        many = true_label_prob(np.stack([make_logits(y, confident=3.0, seed=0)] * 4), y)
        assert one.shape == (20,) and many.shape == (4, 20)
        np.testing.assert_allclose(many[0], one)

    def test_offline_prior_uses_only_out_models(self):
        # IN models are confident; OUT models are not. The prior must ignore the IN ones.
        refs = np.zeros((4, 3))
        refs[0] = 0.9; refs[1] = 0.9        # "IN"
        refs[2] = 0.1; refs[3] = 0.1        # "OUT"
        mask = np.array([[True] * 3, [True] * 3, [False] * 3, [False] * 3])
        got = offline_prior(refs, mask, a=1.0)          # a=1 -> exactly the OUT mean
        np.testing.assert_allclose(got, 0.1)

    def test_offline_prior_with_no_mask_averages_everything(self):
        refs = np.full((5, 3), 0.4)
        np.testing.assert_allclose(offline_prior(refs, None, a=1.0), 0.4)

    def test_an_example_every_reference_trained_on_falls_back_not_divides_by_zero(self):
        refs = np.full((3, 2), 0.8)
        mask = np.ones((3, 2), bool)
        got = offline_prior(refs, mask, a=0.3)
        assert np.isfinite(got).all()

    def test_a_zero_prior_does_not_produce_infinite_ratios(self):
        s = rmia_scores(np.array([0.5]), np.array([0.0]), np.array([0.5]), np.array([0.5]), gamma=2.0)
        assert np.isfinite(s).all()

    def test_scores_are_fractions(self):
        rng = np.random.default_rng(0)
        s = rmia_scores(rng.random(20), rng.random(20), rng.random(50), rng.random(50), gamma=2.0)
        assert s.shape == (20,) and ((s >= 0) & (s <= 1)).all()

    def test_an_empty_population_is_undefined_not_a_crash(self):
        s = rmia_scores(np.array([0.5]), np.array([0.5]), np.array([]), np.array([]), gamma=2.0)
        assert np.isnan(s).all()


class TestRMIA:
    @staticmethod
    def _case(n=200, n_refs=8, member_confidence=8.0, seed=0):
        """Target memorised its members; references saw a random half of them."""
        y_f, y_t = np.arange(n) % K, np.arange(2 * n) % K
        rng = np.random.default_rng(seed)
        in_f = rng.random((n_refs, n)) < 0.5
        return dict(
            f_log=make_logits(y_f, confident=member_confidence, seed=1),
            t_log=make_logits(y_t, confident=1.0, seed=2),
            y_f=y_f, y_t=y_t,
            ref_f=references(y_f, n_refs, confident=1.0, seed=10, in_mask=in_f, in_confident=8.0),
            ref_t=references(y_t, n_refs, confident=1.0, seed=40),
            in_f=in_f,
        )

    def test_a_memorising_model_is_attackable(self):
        out = get_audit("privacy_rmia").measure(rmia_ctx(**self._case()))
        assert out[("mia_auc_rmia", "forget")] > 0.75

    def test_a_model_that_treats_members_like_strangers_is_at_chance(self):
        # Target is equally (un)confident on members and non-members: nothing to detect.
        c = self._case(member_confidence=1.0)
        auc = get_audit("privacy_rmia").measure(rmia_ctx(**c))[("mia_auc_rmia", "forget")]
        assert abs(auc - 0.5) < 0.15

    def test_the_constant_predictor_is_at_chance_and_does_not_raise(self):
        c = self._case()
        const = np.full((len(c["y_f"]), K), -20.0); const[:, 3] = 20.0
        const_t = np.full((len(c["y_t"]), K), -20.0); const_t[:, 3] = 20.0
        c["f_log"], c["t_log"] = const, const_t
        ctx = rmia_ctx(**c)
        audit = get_audit("privacy_rmia")
        out = audit.measure(ctx)
        assert abs(out[("mia_auc_rmia", "forget")] - 0.5) < 0.15
        assert all(np.isfinite(v) for v in out.values())
        assert "no signal" in audit.notes_for(ctx)

    def test_missing_references_give_undefined_not_chance(self):
        # A machine without the shadows must not report a plausible-looking 0.5.
        c = self._case()
        ctx = rmia_ctx(**c)
        ctx.reference_logits = {}
        out = get_audit("privacy_rmia").measure(ctx)
        assert all(np.isnan(v) for v in out.values())

    def test_halving_the_references_is_the_stage_6_gate(self):
        # configs/audits.yaml: TPR at low FPR must be stable at 16 references as at 32.
        c = self._case(n_refs=16)
        full = get_audit("privacy_rmia").measure(rmia_ctx(**c))
        half = get_audit("privacy_rmia").measure(
            rmia_ctx(**c, cfg={**RMIA_CFG, "n_references": 8})
        )
        assert abs(full[("mia_auc_rmia", "forget")] - half[("mia_auc_rmia", "forget")]) < 0.15

    def test_non_finite_logits_give_undefined(self):
        c = self._case()
        c["f_log"] = c["f_log"].copy(); c["f_log"][0, 0] = np.inf
        out = get_audit("privacy_rmia").measure(rmia_ctx(**c))
        assert all(np.isnan(v) for v in out.values())

    def test_deterministic_for_the_same_audit_seed(self):
        c = self._case()
        a = get_audit("privacy_rmia").measure(rmia_ctx(**c, seed=5))
        b = get_audit("privacy_rmia").measure(rmia_ctx(**c, seed=5))
        assert a == b

    def test_declares_references_and_no_oracles(self):
        a = get_audit("privacy_rmia")
        assert a.needs_references is True and a.needs_oracles is False
