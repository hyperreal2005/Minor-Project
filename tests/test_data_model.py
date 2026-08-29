"""Data plumbing and model feature taps.

Uses a synthetic bundle rather than the real download so the suite stays fast and runs offline.
The properties tested are the ones that make results comparable across models: evaluation order
is deterministic, the retain set is always the exact complement of the forget set, and the canary
corruption is reproducible from the index array alone.
"""

import numpy as np
import pytest
import torch

from forgetcheck.data.cifar import (
    DataBundle,
    apply_canaries,
    canary_labels,
    make_loader,
)
from forgetcheck.models.resnet import (
    FEATURE_LAYERS,
    count_parameters,
    extract_features,
    feature_dims,
    make_resnet18,
)


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    n_tr, n_te = 400, 100
    return DataBundle(
        name="cifar10",
        train_x=rng.integers(0, 256, (n_tr, 32, 32, 3), dtype=np.uint8),
        train_y=rng.integers(0, 10, n_tr).astype(np.int64),
        test_x=rng.integers(0, 256, (n_te, 32, 32, 3), dtype=np.uint8),
        test_y=rng.integers(0, 10, n_te).astype(np.int64),
        sha="0" * 16,
        num_classes=10,
    )


class TestBundle:
    def test_retain_is_the_exact_complement(self, bundle):
        forget = np.array([0, 5, 399])
        retain = bundle.retain_indices(forget)
        assert len(retain) == bundle.n_train - len(forget)
        assert not set(retain) & set(forget)
        assert set(retain) | set(forget) == set(range(bundle.n_train))
        assert np.all(np.diff(retain) > 0)

    def test_float_images_rejected(self, bundle):
        with pytest.raises(ValueError, match="must be uint8"):
            DataBundle(
                name="cifar10",
                train_x=bundle.train_x.astype(np.float32),
                train_y=bundle.train_y,
                test_x=bundle.test_x,
                test_y=bundle.test_y,
                sha="x",
                num_classes=10,
            )

    def test_with_labels_shares_pixels(self, bundle):
        other = bundle.with_labels(np.zeros(bundle.n_train, dtype=np.int64))
        assert other.train_x is bundle.train_x  # no needless 180 MB copy
        assert other.train_y.sum() == 0

    def test_with_labels_length_checked(self, bundle):
        with pytest.raises(ValueError, match="wrong length"):
            bundle.with_labels(np.zeros(7, dtype=np.int64))


class TestCanaries:
    def test_label_never_equals_the_truth(self, bundle):
        idx = np.arange(bundle.n_train)
        wrong = canary_labels(bundle.train_y, idx)
        assert np.all(wrong != bundle.train_y)
        assert wrong.min() >= 0 and wrong.max() < 10

    def test_deterministic_from_indices_alone(self, bundle):
        idx = np.array([3, 17, 42])
        a = canary_labels(bundle.train_y, idx)
        b = canary_labels(bundle.train_y, idx)
        np.testing.assert_array_equal(a, b)

    def test_offset_range_is_collision_free_by_construction(self):
        # y_canary = (y + 1 + idx % 9) % 10. The offset lies in [1, 9] so it is never congruent
        # to 0 mod 10 -- the "re-draw on collision" the plan hedged with cannot be reached.
        y = np.tile(np.arange(10), 100)
        idx = np.arange(1000)
        wrong = canary_labels(y, idx)
        assert np.all(wrong != y)

    def test_apply_corrupts_only_the_canaries(self, bundle):
        idx = np.array([1, 2, 3])
        corrupted, wrong = apply_canaries(bundle, idx)
        changed = np.flatnonzero(corrupted.train_y != bundle.train_y)
        np.testing.assert_array_equal(changed, idx)
        np.testing.assert_array_equal(corrupted.train_y[idx], wrong)

    def test_original_bundle_is_untouched(self, bundle):
        before = bundle.train_y.copy()
        apply_canaries(bundle, np.array([0, 1]))
        np.testing.assert_array_equal(bundle.train_y, before)

    def test_cifar100_refused_rather_than_silently_wrong(self, bundle):
        # The formula assumes 10 classes; applying it to 100 would produce labels that collide.
        c100 = DataBundle(
            name="cifar100", train_x=bundle.train_x, train_y=bundle.train_y,
            test_x=bundle.test_x, test_y=bundle.test_y, sha="x", num_classes=100,
        )
        with pytest.raises(NotImplementedError, match="assumes 10 classes"):
            apply_canaries(c100, np.array([0]))


class TestLoaders:
    def test_eval_order_is_deterministic(self, bundle):
        # Activations from two models are only comparable if row i is the same example in both.
        def ids():
            loader = make_loader(bundle, train=False, batch_size=64)
            return torch.cat([b[2] for b in loader]).tolist()

        assert ids() == ids() == list(range(bundle.n_train))

    def test_eval_keeps_the_partial_last_batch(self, bundle):
        loader = make_loader(bundle, np.arange(70), train=False, batch_size=64)
        assert sum(len(b[1]) for b in loader) == 70

    def test_subset_indices_are_respected(self, bundle):
        want = np.array([5, 10, 250])
        loader = make_loader(bundle, want, train=False, batch_size=2)
        got = torch.cat([b[2] for b in loader]).numpy()
        np.testing.assert_array_equal(np.sort(got), want)

    def test_train_loader_shuffles(self, bundle):
        a = torch.cat([b[2] for b in make_loader(bundle, train=True, seed=0, batch_size=64)])
        assert a.tolist() != list(range(bundle.n_train))

    def test_train_order_reproducible_from_seed(self, bundle):
        def order(seed):
            torch.manual_seed(seed)
            return torch.cat(
                [b[2] for b in make_loader(bundle, train=True, seed=seed, batch_size=64)]
            ).tolist()

        assert order(3) == order(3)

    def test_normalisation_applied(self, bundle):
        x = next(iter(make_loader(bundle, train=False, batch_size=32)))[0]
        assert x.shape[1:] == (3, 32, 32)
        assert x.abs().max() < 4.0  # normalised, not raw 0-255

    def test_out_of_range_indices_rejected(self, bundle):
        with pytest.raises(IndexError, match="out of range"):
            make_loader(bundle, np.array([999999]), train=False)

    def test_bad_split_rejected(self, bundle):
        with pytest.raises(ValueError, match="split must be"):
            make_loader(bundle, train=False, split="validation")


class TestModel:
    def test_cifar_stem_is_patched(self):
        m = make_resnet18()
        assert m.conv1.kernel_size == (3, 3)
        assert m.conv1.stride == (1, 1)
        assert isinstance(m.maxpool, torch.nn.Identity)

    def test_parameter_count_matches_the_storage_budget(self):
        # ~11.17 M params is the figure the ~45 MB/checkpoint budget in plan §5 rests on.
        n = count_parameters(make_resnet18())
        assert 11_000_000 < n < 11_300_000, n

    def test_same_seed_gives_identical_weights(self):
        # Paired M0/oracle runs share train_seed; that must mean sharing an initialisation, or
        # the two models differ for a reason unrelated to the forget set.
        a = make_resnet18(seed=7).state_dict()
        b = make_resnet18(seed=7).state_dict()
        for k in a:
            assert torch.equal(a[k], b[k]), k

    def test_different_seeds_differ(self):
        a = make_resnet18(seed=1).state_dict()["conv1.weight"]
        b = make_resnet18(seed=2).state_dict()["conv1.weight"]
        assert not torch.equal(a, b)

    def test_spatial_resolution_survives_the_stem(self, bundle):
        # The ImageNet stem would leave 8x8 going into layer1, discarding most of what CIFAR has.
        m = make_resnet18()
        feats = {}
        m.layer1.register_forward_hook(lambda _m, _i, o: feats.__setitem__("l1", o.shape))
        m.eval()
        with torch.no_grad():
            m(torch.zeros(2, 3, 32, 32))
        assert feats["l1"][-2:] == (32, 32)


class TestFeatureExtraction:
    def test_all_taps_fire_with_expected_widths(self, bundle):
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(32), train=False, batch_size=16)
        feats, logits = extract_features(m, loader)
        assert set(feats) == set(FEATURE_LAYERS)
        dims = feature_dims()
        for name, tensor in feats.items():
            assert tensor.shape == (32, dims[name]), name
            assert tensor.dtype == torch.float32
        assert logits.shape == (32, 10)

    def test_activations_are_pooled_to_2d(self, bundle):
        # Raw layer1 output for 3000 probes is ~786 MB per model; pooling here is what keeps
        # the whole matrix under 2 GB (plan §5).
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(8), train=False, batch_size=8)
        feats, _ = extract_features(m, loader)
        assert all(t.ndim == 2 for t in feats.values())

    def test_hooks_are_removed_afterwards(self, bundle):
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(8), train=False, batch_size=8)
        extract_features(m, loader)
        assert len(m.layer1._forward_hooks) == 0
        assert len(m.avgpool._forward_hooks) == 0

    def test_hooks_removed_even_on_error(self, bundle):
        m = make_resnet18(seed=0)
        with pytest.raises(ValueError, match="no batches"):
            extract_features(m, [])
        assert len(m.layer1._forward_hooks) == 0

    def test_penultimate_duplicates_layer4_and_is_excluded_by_default(self, bundle):
        # In ResNet-18 avgpool is an AdaptiveAvgPool2d((1,1)) sitting directly after layer4 --
        # the same global average pool the taps already apply. Storing both would add 512 floats
        # per probe for zero information, ~35% of the project's activation storage.
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(16), train=False, batch_size=16)
        feats, _ = extract_features(m, loader, layers=("layer4", "penultimate"))
        torch.testing.assert_close(feats["penultimate"], feats["layer4"])
        assert "penultimate" not in FEATURE_LAYERS

    def test_default_taps_total_960_floats(self):
        assert sum(feature_dims().values()) == 960

    def test_subset_of_layers(self, bundle):
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(8), train=False, batch_size=8)
        feats, _ = extract_features(m, loader, layers=("layer4", "penultimate"))
        assert set(feats) == {"layer4", "penultimate"}

    def test_unknown_layer_rejected(self, bundle):
        m = make_resnet18(seed=0)
        with pytest.raises(ValueError, match="unknown feature layers"):
            extract_features(m, [], layers=("layer9",))

    def test_extraction_is_deterministic(self, bundle):
        m = make_resnet18(seed=0)
        loader = make_loader(bundle, np.arange(16), train=False, batch_size=8)
        a, _ = extract_features(m, loader)
        b, _ = extract_features(m, loader)
        for k in a:
            torch.testing.assert_close(a[k], b[k])
