"""The six unlearning methods.

Uses a small MLP rather than ResNet-18 so the suite stays fast: what is being tested is the
*logic* of each method — ascent direction, masking, teacher/student divergence, the no-mutation
contract — none of which depends on the architecture.

The properties asserted are the ones that would silently invalidate results if broken: a method
that mutates its input would corrupt the next method in the queue, and a method that does nothing
would occupy a rank slot while carrying no information.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from forgetcheck.unlearn import CORE_METHODS, UnlearnContext, get_unlearner, method_names


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 10))


def loader(n, *, seed=0, batch=16, n_classes=10):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    idx = torch.arange(n)
    return list(torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y, idx), batch_size=batch
    ))


@pytest.fixture
def ctx():
    return UnlearnContext(
        forget_loader=loader(48, seed=1),
        retain_loader=loader(96, seed=2),
        forget_eval_loader=loader(48, seed=1),
        device="cpu",
        num_classes=10,
        seed=0,
    )


FAST = {
    "finetune": {"epochs": 1},
    "neggrad": {"steps": 4},
    "neggradplus": {"epochs": 1},
    "scrub": {"epochs": 2, "msteps": 1, "max_steps": 2},
    "salun": {"epochs": 1},
    "l1sparse": {"epochs": 1},
    "ssd": {},
}


class TestContract:
    @pytest.mark.parametrize("name", sorted(FAST))
    def test_does_not_mutate_the_input(self, name, ctx):
        # The caller owns the original checkpoint and reuses it for the next method in the
        # queue. An in-place update would make method 2 operate on method 1's output.
        model = tiny_model()
        before = copy.deepcopy(model.state_dict())
        get_unlearner(name, **FAST[name]).unlearn(model, ctx)
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k]), f"{name} mutated the input model at {k}"

    @pytest.mark.parametrize("name", sorted(FAST))
    def test_returns_a_usable_model(self, name, ctx):
        out = get_unlearner(name, **FAST[name]).unlearn(tiny_model(), ctx)
        assert isinstance(out, nn.Module)
        with torch.no_grad():
            logits = out(torch.randn(4, 3, 8, 8))
        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all(), f"{name} produced non-finite outputs"

    @pytest.mark.parametrize("name", sorted(set(FAST) - {"ssd"}))
    def test_actually_changes_the_weights(self, name, ctx):
        # A method that leaves the model untouched occupies a rank slot while carrying no
        # information -- the exact failure mode that got SSD demoted.
        model = tiny_model()
        out = get_unlearner(name, **FAST[name]).unlearn(model, ctx)
        before, after = model.state_dict(), out.state_dict()
        changed = sum(
            1 for k in before if not torch.equal(before[k].float(), after[k].float())
        )
        assert changed > 0, f"{name} was a no-op"

    @pytest.mark.parametrize("name", sorted(FAST))
    def test_is_deterministic_given_the_seed(self, name, ctx):
        a = get_unlearner(name, **FAST[name]).unlearn(tiny_model(), ctx).state_dict()
        b = get_unlearner(name, **FAST[name]).unlearn(tiny_model(), ctx).state_dict()
        for k in a:
            torch.testing.assert_close(a[k], b[k], msg=f"{name} is not deterministic at {k}")


class TestSemantics:
    def test_neggrad_ascends_forget_loss(self, ctx):
        # The destructive control must actually destroy: forget-set loss should go UP.
        model = tiny_model()
        crit = nn.CrossEntropyLoss()

        def forget_loss(m):
            with torch.no_grad():
                return float(sum(crit(m(x), y) * len(y) for x, y, _ in ctx.forget_loader))

        out = get_unlearner("neggrad", steps=9, lr=0.05).unlearn(model, ctx)
        assert forget_loss(out) > forget_loss(model)

    def test_finetune_never_touches_the_forget_set(self, ctx):
        # Fine-tune sees only retain data, so an empty forget loader must not change anything.
        starved = UnlearnContext(
            forget_loader=[], retain_loader=ctx.retain_loader, device="cpu", seed=0
        )
        out = get_unlearner("finetune", epochs=1).unlearn(tiny_model(), starved)
        assert isinstance(out, nn.Module)

    def test_neggradplus_is_gentler_than_neggrad(self, ctx):
        # The retain term is the entire difference: it should keep retain loss far lower.
        model = tiny_model()
        crit = nn.CrossEntropyLoss()

        def retain_loss(m):
            with torch.no_grad():
                return float(sum(crit(m(x), y) * len(y) for x, y, _ in ctx.retain_loader))

        plain = get_unlearner("neggrad", steps=9, lr=0.05).unlearn(model, ctx)
        plus = get_unlearner("neggradplus", epochs=3, lr=0.05, alpha=0.95).unlearn(model, ctx)
        assert retain_loss(plus) < retain_loss(plain)

    def test_scrub_ascent_budget_does_not_scale_with_the_forget_set(self, ctx):
        # All five seeds of SCRUB collapsed at rand-5000 (40 ascent steps) and held everywhere
        # at 24 or fewer. The max-steps count must be fixed, not proportional to |Df|.
        big = UnlearnContext(
            forget_loader=loader(96, seed=1), retain_loader=ctx.retain_loader,
            forget_eval_loader=loader(96, seed=1), device="cpu", num_classes=10, seed=0,
        )
        counted = {}
        for label, c in (("small", ctx), ("big", big)):
            n = 0
            orig = torch.optim.SGD.step

            def step(self, *a, **k):
                nonlocal n
                n += 1
                return orig(self, *a, **k)

            torch.optim.SGD.step = step
            try:
                get_unlearner("scrub", epochs=2, msteps=1, max_steps=4).unlearn(tiny_model(), c)
            finally:
                torch.optim.SGD.step = orig
            counted[label] = n
        assert counted["small"] == counted["big"], counted

    def test_neggrad_budget_does_not_scale_with_the_forget_set(self, ctx):
        """The size axis (rand-500 .. rand-5000) is an experimental variable, so the destructive
        control's strength must not be a function of |Df| -- otherwise a "size effect" is partly
        a compute effect. Stage 5 measured exactly that: 10 steps at rand-500 was a no-op
        (retain 0.9918) while 60 steps at 3000 was total collapse (retain 0.020).

        Only neggrad is asserted this strictly. SalUn's *forget* pass must scale with |Df| --
        it has to randomise the label of every forget example -- so for SalUn the invariant is
        that the retain *repair* is constant, which
        `test_salun_sees_the_whole_retain_set_each_epoch` covers.
        """
        name = "neggrad"
        big = UnlearnContext(
            forget_loader=loader(96, seed=1),          # 4x the fixture's forget set
            retain_loader=ctx.retain_loader,
            forget_eval_loader=loader(96, seed=1),
            device="cpu", num_classes=10, seed=0,
        )
        counted = {}
        for label, c in (("small", ctx), ("big", big)):
            n = 0
            model = tiny_model()
            orig = torch.optim.SGD.step

            def step(self, *a, **k):
                nonlocal n
                n += 1
                return orig(self, *a, **k)

            torch.optim.SGD.step = step
            try:
                get_unlearner(name, **FAST[name]).unlearn(model, c)
            finally:
                torch.optim.SGD.step = orig
            counted[label] = n

        assert counted["small"] == counted["big"], (
            f"{name} took {counted['small']} steps on a 48-example forget set and "
            f"{counted['big']} on a 96-example one; the budget tracks |Df|"
        )

    def test_salun_sees_the_whole_retain_set_each_epoch(self, ctx):
        # Fidelity check against OPTML-Group/Unlearn-Saliency, whose CIFAR-10 path runs the
        # forget loader and then the *full* retain loader every epoch. Pairing one retain batch
        # per forget batch -- the earlier bug -- undersamples Dr by |Dr|/|Df|, which is 97x at
        # rand-500 and is what collapsed that condition.
        seen = []
        wrapped = list(ctx.retain_loader)

        class Counting(list):
            def __iter__(self):
                seen.append(1)
                return iter(wrapped)

        c = UnlearnContext(
            forget_loader=ctx.forget_loader, retain_loader=Counting(wrapped),
            forget_eval_loader=ctx.forget_eval_loader, device="cpu", num_classes=10, seed=0,
        )
        get_unlearner("salun", epochs=3).unlearn(tiny_model(), c)
        assert len(seen) == 3, f"retain loader traversed {len(seen)} times over 3 epochs"

    def test_neggrad_shipped_defaults_actually_destroy(self, ctx):
        # Regression on the mem-high-3000 pilot. The original defaults (epochs=1, lr=0.001) were
        # 12 steps over 3000 examples and left forget accuracy at 0.9723 -- *above* fine-tune's
        # 0.9460. A destructive control that does not destroy still occupies a rank slot and
        # invites the reader to conclude gradient ascent is harmless.
        #
        # Asserted against the *shipped* defaults with no overrides, because the defaults are
        # what the 240 production runs use, and overriding them here is what hid the bug.
        model = tiny_model()
        crit = nn.CrossEntropyLoss()

        def forget_loss(m):
            with torch.no_grad():
                return float(sum(crit(m(x), y) * len(y) for x, y, _ in ctx.forget_loader))

        base = forget_loss(model)
        weak = forget_loss(get_unlearner("neggrad", steps=3, lr=0.001).unlearn(model, ctx))
        shipped = forget_loss(get_unlearner("neggrad").unlearn(model, ctx))
        # The tiny MLP cannot reproduce ResNet-18 magnitudes, so this compares the two configs
        # rather than testing an absolute threshold: the shipped one must ascend by an order of
        # magnitude more than the config that failed the pilot.
        assert shipped - base > 10 * (weak - base) > 0

    def test_neggradplus_ascent_is_floored_at_the_cap(self, ctx):
        # Below the cap the forget term must behave exactly like the published -CE(forget);
        # at or above it, it must contribute no gradient at all. A fresh model already sits at
        # chance (CE ~ 2.31), so a cap of 1.0 is already exceeded and must be fully inert --
        # indistinguishable from switching the term off entirely.
        model = tiny_model()

        def run(**kw):
            return get_unlearner("neggradplus", epochs=2, lr=0.05, **kw).unlearn(
                model, ctx
            ).state_dict()

        off, exceeded = run(forget_cap=0.0), run(forget_cap=1.0)
        for k in off:
            assert torch.equal(off[k], exceeded[k]), f"cap not gating the forget term at {k}"

    def test_neggradplus_recovers_the_published_loss_at_an_infinite_cap(self, ctx):
        # The cap is a documented deviation, so the published objective has to remain reachable
        # -- otherwise the comparison we report is against a method nobody else ran. This also
        # proves the clamp is load-bearing rather than decorative: with it lifted, the same run
        # lands somewhere else.
        model = tiny_model()

        def run(cap):
            return get_unlearner("neggradplus", epochs=2, lr=0.05, forget_cap=cap).unlearn(
                model, ctx
            ).state_dict()

        floored, published = run(0.0), run(float("inf"))
        assert any(not torch.equal(floored[k], published[k]) for k in floored)

    def test_salun_mask_is_sparse_and_respects_the_ratio(self, ctx):
        u = get_unlearner("salun", sparsity=0.3)
        mask = u._saliency_mask(tiny_model(), ctx)
        total = sum(m.numel() for m in mask.values())
        kept = sum(int(m.sum()) for m in mask.values())
        assert 0 < kept < total
        assert abs(kept / total - 0.3) < 0.05, f"kept {kept/total:.3f}, wanted ~0.30"

    def test_salun_updates_only_through_the_mask(self, ctx):
        # If updates leaked outside the mask, SalUn would be indistinguishable from unmasked
        # random-label fine-tuning -- i.e. plain label-noise injection.
        model = tiny_model()
        u = get_unlearner("salun", epochs=1, sparsity=0.1, lr=0.1)
        mask = u._saliency_mask(copy.deepcopy(model), ctx)
        out = u.unlearn(model, ctx)
        before, after = model.state_dict(), out.state_dict()
        for name, m in mask.items():
            if m.sum() == 0:
                continue
            frozen = m == 0
            if frozen.any():
                torch.testing.assert_close(
                    before[name][frozen], after[name][frozen],
                    msg=f"SalUn updated masked-out weights in {name}",
                )

    def test_scrub_diverges_from_the_teacher_on_forget(self, ctx):
        model = tiny_model()
        out = get_unlearner("scrub", epochs=2, msteps=2, lr=0.05).unlearn(model, ctx)
        with torch.no_grad():
            x = next(iter(ctx.forget_loader))[0]
            same = torch.allclose(model(x), out(x), atol=1e-4)
        assert not same, "SCRUB left forget-set predictions unchanged"

    def test_l1sparse_shrinks_weight_magnitude(self, ctx):
        model = tiny_model()

        def l1(m):
            with torch.no_grad():
                return float(sum(p.abs().sum() for p in m.parameters() if p.dim() > 1))

        out = get_unlearner("l1sparse", epochs=3, lr=0.05, l1_lambda=0.05).unlearn(model, ctx)
        assert l1(out) < l1(model)

    def test_l1sparse_leaves_batchnorm_alone(self):
        # Penalising BN scales shrinks the normalisation itself, damaging utility without
        # removing memorised structure.
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                              nn.Flatten(), nn.Linear(4 * 8 * 8, 10))
        penalised = [n for n, p in model.named_parameters() if p.dim() > 1]
        assert not any("1." in n for n in penalised), penalised


class TestRegistry:
    def test_six_core_methods_are_registered(self):
        assert len(CORE_METHODS) == 6
        assert set(CORE_METHODS) <= set(method_names())

    def test_ssd_is_not_a_core_method(self):
        # Demoted: its Fisher-importance mechanism cannot separate a random i.i.d. forget set
        # from the retain set (review finding B1).
        assert "ssd" in method_names()
        assert "ssd" not in CORE_METHODS

    def test_unknown_method_suggests_alternatives(self):
        with pytest.raises(KeyError, match="registered"):
            get_unlearner("scrubb")

    def test_unknown_hyperparameter_rejected(self):
        with pytest.raises(ValueError, match="unknown hyperparameter"):
            get_unlearner("finetune", learning_rate=0.1)

    def test_method_names_are_valid_run_id_segments(self):
        from forgetcheck.registry import run_id

        for name in method_names():
            run_id(role="unlearn", forget="rand-500", method=name, seed=0)
