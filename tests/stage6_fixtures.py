"""Shared fixtures for the Stage 6 runner tests: a synthetic bundle, a minimal context, and a
store seeded with real ResNet-18 checkpoints.

A plain module, not a package import. `tests/` has no `__init__.py`, so pytest puts the
directory itself on `sys.path` and `import stage6_fixtures` works however pytest is launched.
The previous `from tests.test_audits_e2e import ...` worked only under `python -m pytest`,
which adds the working directory to the path; bare `pytest` -- what CI runs -- does not, and
two commits went red on `ModuleNotFoundError: No module named 'tests'`.
"""

import numpy as np
import yaml

from forgetcheck.config import find_configs


N_TRAIN, N_TEST, K = 240, 120, 10


def _bundle():
    from forgetcheck.data.cifar import DataBundle

    rng = np.random.default_rng(0)
    return DataBundle(
        name="cifar10",
        train_x=rng.integers(0, 256, size=(N_TRAIN, 32, 32, 3), dtype=np.uint8),
        train_y=(np.arange(N_TRAIN) % K).astype(np.int64),
        test_x=rng.integers(0, 256, size=(N_TEST, 32, 32, 3), dtype=np.uint8),
        test_y=(np.arange(N_TEST) % K).astype(np.int64),
        sha="synthetic",
        num_classes=K,
    )


class _Ctx:
    """The slice of `forgetcheck.config.Context` the runner touches, over synthetic data."""

    def __init__(self, tmp_path, forget_id="rand-500", n_forget=40):
        from forgetcheck.data.forget_sets import spec_by_id
        from forgetcheck.registry import ArtifactStore

        self.root = tmp_path
        self.store = ArtifactStore(tmp_path / "artifacts")
        self.records_dir = tmp_path / "results" / "records"
        self.records_dir.mkdir(parents=True)
        self.bundle = _bundle()
        self._spec = spec_by_id(forget_id)
        self._fidx = np.arange(n_forget, dtype=np.int64)
        self.base = {
            "oracles": {"paired_seeds": [0]},
            "shadows": {"count": 2, "subset_fraction": 0.5},
        }
        self.seeds = {"audit": 0}
        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        cfg["relearning"]["eval_steps"] = [0, 1, 2]
        cfg["relearning"]["reintroduction_size"] = 16
        cfg["relearning"]["batch_size"] = 8
        cfg["representation"]["probe_size"] = 60
        cfg["representation"]["retain_probe_size"] = 60
        cfg["sde"]["n_draws"] = 10
        cfg["privacy"]["population"]["n_splits"] = 2
        self.audits = cfg

    def spec(self, forget_id):
        return self._spec

    def forget_indices(self, forget_id):
        return self._fidx


def _seed_store(ctx):
    """Write every checkpoint the runner will look for, from one random ResNet-18."""
    import torch

    from forgetcheck.models.resnet import make_resnet18
    from forgetcheck.registry import run_id
    from forgetcheck.unlearn import base_run_id_for

    torch.manual_seed(0)
    state = make_resnet18(num_classes=K).state_dict()
    ids = [
        "c10r18__unlearn__rand-500__finetune__train0",
        run_id(role="oracle", forget="rand-500", seed=0, seed_kind="train"),
        base_run_id_for(ctx.spec("rand-500"), 0),
        run_id(role="shadow", forget="full", seed=0, seed_kind="shadow"),
        run_id(role="shadow", forget="full", seed=1, seed_kind="shadow"),
    ]
    for rid in ids:
        ctx.store.save_checkpoint(rid, state, train_seed=0, hparams_sha="e2e")
    return ids[0]
