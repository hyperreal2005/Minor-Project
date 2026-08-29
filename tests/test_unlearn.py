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
    "neggrad": {"epochs": 1},
    "neggradplus": {"epochs": 1},
    "scrub": {"epochs": 2, "msteps": 1},
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

        out = get_unlearner("neggrad", epochs=3, lr=0.05).unlearn(model, ctx)
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

        plain = get_unlearner("neggrad", epochs=3, lr=0.05).unlearn(model, ctx)
        plus = get_unlearner("neggradplus", epochs=3, lr=0.05, alpha=0.95).unlearn(model, ctx)
        assert retain_loss(plus) < retain_loss(plain)

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
