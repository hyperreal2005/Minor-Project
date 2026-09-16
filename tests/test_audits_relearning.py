"""Layer 5 — reversibility.

The anchor behaviour is this module's own self-test, and it is tested as such: the `original` arm
must normalise to ~1 and the `oracle` arm to ~0 *by construction*, so a protocol error shows up
here rather than as a surprising result in the paper.
"""

import numpy as np
import pytest

from forgetcheck.audits import AuditContext, get_audit
from forgetcheck.audits.relearning import (
    curve_auc,
    normalized_recovery,
    relearn_curve,
    steps_to_fraction,
)

STEPS = [0, 1, 2, 5, 10, 25, 50, 100]


def curve(vals, steps=STEPS, retain=None, test=None):
    out = {"steps": np.array(steps, float), "forget_acc": np.array(vals, float)}
    if retain is not None:
        out["retain_acc"] = np.array(retain, float)
    if test is not None:
        out["test_acc"] = np.array(test, float)
    return out


def ctx_for(arms, config=None):
    return AuditContext(
        run_id="c10r18__unlearn__mem-high-3000__salun__train1",
        logits={"forget": np.random.default_rng(0).normal(size=(20, 10))},
        relearn_curves=arms,
        forget_kind="memstratum",
        forget_size=3000,
        config=config or {"t80_threshold": 0.8, "max_utility_drop_pp": 5.0},
    )


class TestCurveAUC:
    def test_a_flat_curve_returns_its_level(self):
        assert curve_auc(STEPS, [0.7] * 8) == pytest.approx(0.7)

    def test_bounded_by_the_curve(self):
        v = [0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
        assert min(v) <= curve_auc(STEPS, v) <= max(v)

    def test_faster_recovery_scores_higher(self):
        fast = [0.1, 0.8, 0.9, 0.95, 0.96, 0.97, 0.97, 0.97]
        slow = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 0.9, 0.97]
        assert curve_auc(STEPS, fast) > curve_auc(STEPS, slow)

    def test_log_axis_does_not_let_the_tail_dominate(self):
        """The schedule is geometric; on a linear axis the 50->100 gap alone would outweigh
        everything before step 25, where the arms actually differ."""
        early = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        late = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]
        assert curve_auc(STEPS, early) > curve_auc(STEPS, late) + 0.3

    def test_too_few_points_is_undefined(self):
        assert np.isnan(curve_auc([0], [0.5]))

    def test_nans_are_dropped_not_propagated(self):
        v = [0.1, np.nan, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
        assert np.isfinite(curve_auc(STEPS, v))


class TestNormalizedRecovery:
    def test_matching_the_oracle_is_zero(self):
        assert normalized_recovery(0.3, 0.3, 0.9) == pytest.approx(0.0)

    def test_matching_the_original_is_one(self):
        assert normalized_recovery(0.9, 0.3, 0.9) == pytest.approx(1.0)

    def test_recovering_faster_than_the_original_exceeds_one(self):
        # A primed state: unlearning left the model somewhere the data is unusually easy again.
        assert normalized_recovery(1.05, 0.3, 0.9) > 1.0

    def test_coincident_anchors_are_undefined_not_enormous(self):
        # mem-low-3000: the oracle already scores 0.998 on its own forget set, so there is
        # almost no gap between the anchors and any ratio would be noise over noise.
        assert np.isnan(normalized_recovery(0.5, 0.9, 0.9))

    def test_a_missing_anchor_is_undefined(self):
        assert np.isnan(normalized_recovery(0.5, np.nan, 0.9))


class TestStepsToFraction:
    def test_interpolates_between_checkpoints(self):
        got = steps_to_fraction([0, 10], [0.0, 1.0], target=0.5)
        assert got == pytest.approx(5.0)

    def test_already_there_at_step_zero(self):
        assert steps_to_fraction([0, 10], [0.9, 1.0], target=0.5) == 0.0

    def test_never_reached_is_infinite_not_clipped(self):
        # "did not recover within the budget" is a real result; clipping to 100 would hide it.
        assert steps_to_fraction(STEPS, [0.1] * 8, target=0.8) == float("inf")


class TestRelearningAudit:
    @staticmethod
    def _arms(method_vals):
        return {
            "method": curve(method_vals, retain=[0.99] * 8, test=[0.93] * 8),
            "oracle": curve([0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8]),
            "original": curve([0.9, 0.95, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99]),
        }

    def test_the_original_arm_normalises_to_one(self):
        # The module's own gate. If this fails the protocol is wrong, not the finding.
        arms = self._arms([0.9, 0.95, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99])
        out = get_audit("relearning").measure(ctx_for(arms))
        assert out[("relearn_norm", "forget")] == pytest.approx(1.0, abs=1e-9)

    def test_the_oracle_arm_normalises_to_zero(self):
        arms = self._arms([0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8])
        out = get_audit("relearning").measure(ctx_for(arms))
        assert out[("relearn_norm", "forget")] == pytest.approx(0.0, abs=1e-9)

    def test_a_primed_model_exceeds_one(self):
        arms = self._arms([0.95, 0.99, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert get_audit("relearning").measure(ctx_for(arms))[("relearn_norm", "forget")] > 1.0

    def test_a_randinit_floor_lands_well_below_zero(self):
        # Proves the reintroduction data alone cannot manufacture recovery.
        arms = self._arms([0.0, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3])
        assert get_audit("relearning").measure(ctx_for(arms))[("relearn_norm", "forget")] < -0.1

    def test_utility_drop_is_reported_in_percentage_points(self):
        arms = self._arms([0.5] * 8)
        arms["method"]["test_acc"] = np.array([0.93, 0.90, 0.88, 0.88, 0.89, 0.90, 0.91, 0.92])
        out = get_audit("relearning").measure(ctx_for(arms))
        assert out[("relearn_utility_drop", "test")] == pytest.approx(5.0, abs=1e-9)

    def test_a_recovery_bought_by_wrecking_the_model_is_flagged(self):
        arms = self._arms([0.9] * 8)
        arms["method"]["test_acc"] = np.array([0.93, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        note = get_audit("relearning").notes_for(ctx_for(arms))
        assert "recovery void" in note

    def test_missing_anchors_leave_norm_undefined_and_say_so(self):
        arms = {"method": curve([0.5] * 8)}
        out = get_audit("relearning").measure(ctx_for(arms))
        assert np.isfinite(out[("relearn_auc", "forget")])
        assert np.isnan(out[("relearn_norm", "forget")])
        assert "anchors missing" in get_audit("relearning").notes_for(ctx_for(arms))

    def test_t80_uses_the_originals_step_zero_performance(self):
        arms = self._arms([0.0, 0.0, 0.0, 0.72, 0.9, 0.9, 0.9, 0.9])
        out = get_audit("relearning").measure(ctx_for(arms))
        # original step-0 is 0.9, so the target is 0.72, first reached exactly at step 5.
        assert out[("relearn_t80", "forget")] == pytest.approx(5.0)

    def test_declares_it_needs_weights(self):
        assert get_audit("relearning").needs_weights is True


class TestRelearnTrainer:
    """The trainer must give every arm an identical protocol -- that is its whole job."""

    @staticmethod
    def _setup():
        import torch

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 4))
        g = torch.Generator().manual_seed(1)
        batches = [(torch.randn(8, 12, generator=g), torch.randint(0, 4, (8,), generator=g))
                   for _ in range(3)]
        return model, batches

    def test_records_every_scheduled_step_including_zero(self):
        model, batches = self._setup()
        out = relearn_curve(model, batches, lambda m: {"forget_acc": 0.5},
                            eval_steps=[0, 1, 5], seed=0)
        np.testing.assert_array_equal(out["steps"], [0.0, 1.0, 5.0])
        assert out["forget_acc"].shape == (3,)

    def test_same_seed_and_batches_give_an_identical_curve(self):
        import torch

        seen = []

        def ev(m):
            seen.append(float(next(m.parameters()).detach().sum()))
            return {"forget_acc": seen[-1]}

        a_model, batches = self._setup()
        b_model, _ = self._setup()
        a = relearn_curve(a_model, batches, ev, eval_steps=[0, 2, 4], seed=7)
        b = relearn_curve(b_model, batches, ev, eval_steps=[0, 2, 4], seed=7)
        np.testing.assert_allclose(a["forget_acc"], b["forget_acc"])

    def test_the_model_is_left_in_eval_free_state_and_training_actually_moved_it(self):
        model, batches = self._setup()
        before = float(next(model.parameters()).detach().abs().sum())
        relearn_curve(model, batches, lambda m: {"forget_acc": 0.0}, eval_steps=[0, 10], seed=0)
        after = float(next(model.parameters()).detach().abs().sum())
        assert before != after, "relearning took no steps"

    def test_no_batches_still_produces_the_step_zero_point(self):
        model, _ = self._setup()
        out = relearn_curve(model, [], lambda m: {"forget_acc": 0.3}, eval_steps=[0, 5], seed=0)
        assert out["steps"].tolist() == [0.0]
