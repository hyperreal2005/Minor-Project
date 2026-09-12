"""Stage 1 gate: the artifact store survives ephemeral sessions.

Kaggle sessions are time-limited and disposable. Two properties matter more than anything else
here: a killed session must leave either a complete artifact or none (never a truncated one that
reads as valid), and an audit must be able to prove it read the weights its record claims.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from forgetcheck.registry import ArtifactStore, StoreError, run_id


BASE = run_id(role="base", forget="full", seed=0)
ORACLE = run_id(role="oracle", forget="mem-high-3000", seed=205, seed_kind="oracle")
UNLEARN = run_id(role="unlearn", forget="mem-high-3000", method="scrub", seed=2)


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 10))


class TestCheckpoints:
    def test_round_trip_is_bit_exact_in_fp32(self, store, model):
        # fp32 is the default precisely so that relearning continues from the same weights it
        # would have had in memory. Rounding here would perturb the reversibility measurement.
        store.save_checkpoint(BASE, model.state_dict(), epochs=30, train_seed=0)
        loaded, meta = store.load_checkpoint(BASE)
        for k, v in model.state_dict().items():
            assert torch.equal(loaded[k], v), k
        assert meta.dtype == "float32"
        assert meta.epochs == 30 and meta.train_seed == 0

    def test_fp16_is_available_but_lossy(self, store, model):
        store.save_checkpoint(BASE, model.state_dict(), dtype="float16")
        loaded, meta = store.load_checkpoint(BASE)
        assert meta.dtype == "float16"
        ref = model.state_dict()["0.weight"]
        assert loaded["0.weight"].dtype == torch.float32  # widened on load
        assert not torch.equal(loaded["0.weight"], ref)
        assert torch.allclose(loaded["0.weight"], ref, atol=1e-3)

    def test_has_checkpoint_requires_metadata_too(self, store, model):
        # A checkpoint without metadata cannot be provenance-checked; treating it as complete
        # would let an unverifiable artifact into the results.
        assert not store.has_checkpoint(BASE)
        store.save_checkpoint(BASE, model.state_dict())
        assert store.has_checkpoint(BASE)
        store.meta_path(BASE).unlink()
        assert not store.has_checkpoint(BASE)

    def test_tampering_is_detected(self, store, model):
        store.save_checkpoint(BASE, model.state_dict())
        torch.save({"junk": torch.zeros(1)}, store.checkpoint_path(BASE))
        with pytest.raises(StoreError, match="sha mismatch"):
            store.load_checkpoint(BASE)
        store.load_checkpoint(BASE, verify=False)  # escape hatch, explicit

    def test_missing_checkpoint_raises(self, store):
        with pytest.raises(StoreError, match="no checkpoint"):
            store.load_checkpoint(BASE)

    def test_malformed_run_id_rejected_before_writing(self, store, model):
        with pytest.raises(ValueError):
            store.save_checkpoint("not-a-run-id", model.state_dict())
        assert not (store.root / "checkpoints").exists()

    def test_no_temp_files_left(self, store, model):
        store.save_checkpoint(BASE, model.state_dict())
        assert list((store.root / "checkpoints").glob("*.tmp")) == []

    def test_resumability_is_the_whole_point(self, store, model):
        # A run whose artifacts exist is skipped, so a session that dies mid-queue costs only
        # its in-flight run.
        assert not store.has_checkpoint(UNLEARN)
        store.save_checkpoint(UNLEARN, model.state_dict())
        assert store.has_checkpoint(UNLEARN)


class TestIteration:
    def test_filter_by_parsed_coordinates(self, store, model):
        for rid in (BASE, ORACLE, UNLEARN):
            store.save_checkpoint(rid, model.state_dict())
        assert list(store.iter_checkpoints(role="oracle")) == [ORACLE]
        assert set(store.iter_checkpoints(forget="mem-high-3000")) == {ORACLE, UNLEARN}
        assert len(list(store.iter_checkpoints())) == 3

    def test_foreign_files_are_ignored(self, store, model):
        store.save_checkpoint(BASE, model.state_dict())
        (store.root / "checkpoints" / "someone-elses-file.pt").write_bytes(b"x")
        assert list(store.iter_checkpoints()) == [BASE]


class TestActivations:
    def test_round_trip_with_probe_ids(self, store):
        acts = {"layer3": np.random.randn(64, 256), "layer4": np.random.randn(64, 512)}
        ids = np.arange(64)
        store.save_activations(BASE, acts, probe_ids=ids)
        loaded, probe_ids = store.load_activations(BASE)
        assert set(loaded) == {"layer3", "layer4"}
        assert loaded["layer4"].shape == (64, 512)
        assert loaded["layer3"].dtype == np.float32
        np.testing.assert_array_equal(probe_ids, ids)
        np.testing.assert_allclose(loaded["layer3"], acts["layer3"], atol=1e-2)

    def test_unpooled_activations_rejected(self, store):
        # Raw conv output for 3000 probes is ~786 MB per model; pooling is mandatory, not an
        # optimisation (plan §5).
        with pytest.raises(StoreError, match="Pool over spatial dims"):
            store.save_activations(BASE, {"layer1": np.zeros((8, 64, 32, 32))})

    def test_layers_must_share_a_probe_count(self, store):
        # Otherwise a later CKA silently compares different example sets.
        with pytest.raises(StoreError, match="disagree on probe count"):
            store.save_activations(
                BASE, {"layer3": np.zeros((64, 256)), "layer4": np.zeros((32, 512))}
            )

    def test_probe_ids_length_checked(self, store):
        with pytest.raises(StoreError, match="probe_ids length"):
            store.save_activations(
                BASE, {"layer3": np.zeros((64, 256))}, probe_ids=np.arange(10)
            )

    def test_empty_set_rejected(self, store):
        with pytest.raises(StoreError, match="empty activation set"):
            store.save_activations(BASE, {})

    def test_fp16_default_halves_storage(self, store):
        acts = {"layer4": np.random.randn(512, 512)}
        store.save_activations(BASE, acts)
        small = store.activations_path(BASE).stat().st_size
        store.save_activations(ORACLE, acts, dtype="float32")
        assert small < store.activations_path(ORACLE).stat().st_size


class TestForgetSets:
    def test_round_trip(self, store):
        idx = np.array([3, 17, 42, 99])
        sha = store.save_forget_set("mem-high-3000", idx, meta={"stratum": "high"})
        loaded, meta = store.load_forget_set("mem-high-3000")
        np.testing.assert_array_equal(loaded, idx)
        assert meta["stratum"] == "high"
        assert len(sha) == 16

    def test_unsorted_rejected(self, store):
        # The array must be canonical, or its hash is not a stable identity for the forget set.
        with pytest.raises(StoreError, match="sorted ascending"):
            store.save_forget_set("f", np.array([5, 1, 9]))

    def test_duplicates_rejected(self, store):
        with pytest.raises(StoreError, match="duplicates"):
            store.save_forget_set("f", np.array([1, 1, 2]))

    def test_hash_is_content_stable(self, store):
        idx = np.array([1, 2, 3])
        a = store.save_forget_set("f1", idx)
        b = store.save_forget_set("f2", idx)
        assert a == b, "identical index arrays must hash identically"


def test_usage_reporting(store, model):
    store.save_checkpoint(BASE, model.state_dict())
    store.save_activations(BASE, {"layer4": np.zeros((16, 32))})
    usage = store.usage()
    assert usage["checkpoints"][0] == 2  # .pt and .json
    assert usage["activations"][0] == 1
    assert usage["forget_sets"] == (0, 0.0)

class TestStaleHyperparameters:
    """A run_id names *what* was unlearned, not how -- so a method fix reuses the same id.

    Without a hyperparameter check, re-running after a fix hits `has_checkpoint`, skips, and
    leaves the broken weights in place while reporting success. Stage 5 turned up two method
    defects whose fixes would have been swallowed exactly this way.
    """

    def test_metadata_round_trips_the_hyperparameter_sha(self, tmp_path):
        import torch

        from forgetcheck.registry import ArtifactStore

        store = ArtifactStore(tmp_path)
        rid = "c10r18__unlearn__rand-500__neggrad__train0"
        store.save_checkpoint(rid, {"w": torch.zeros(2)}, hparams_sha="aaaa1111")
        assert store.load_meta(rid).hparams_sha == "aaaa1111"

    def test_a_config_change_changes_the_sha(self):
        from forgetcheck.registry import config_sha
        from forgetcheck.unlearn import get_unlearner

        def sha(**kw):
            return config_sha(get_unlearner("neggrad", **kw).signature())

        assert sha() == sha(), "sha must be stable for an unchanged config"
        assert sha() != sha(steps=7), "a changed step budget must change the sha"
        assert sha() != sha(lr=0.5), "a changed lr must change the sha"

    def test_an_implementation_version_bump_changes_the_sha_without_touching_hparams(self):
        """SalUn's retain-exposure fix changed the loop, not the defaults, and its 16 pre-fix
        checkpoints on account 1 read as current. `version` is how a behaviour change with the
        same hyperparameters becomes visible to the guard."""
        from forgetcheck.registry import config_sha
        from forgetcheck.unlearn.methods import FineTune

        class Revised(FineTune):
            version = 2

        v1, v2 = FineTune().signature(), Revised().signature()
        assert "impl_version" not in v1, "version 1 is omitted so existing shas stay valid"
        assert v2["impl_version"] == 2
        assert {k: v for k, v in v2.items() if k != "impl_version"} == v1
        assert config_sha(v1) != config_sha(v2)
