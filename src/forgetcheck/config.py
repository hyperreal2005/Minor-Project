"""Configuration loading and the run context.

One place that resolves ``configs/*.yaml`` into live objects — the data bundle, the memorization
scores, the artifact store — so that every CLI command starts from an identical, verified state.
Both hashes are checked on load: a machine whose CIFAR download or memorization file differs from
everyone else's fails immediately rather than producing results nobody can compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data.cifar import DataBundle, load_cifar
from .data.forget_sets import ForgetSpec, all_specs, materialise, spec_by_id
from .data.memorization import load_scores
from .registry import ArtifactStore
from .train.loop import TrainConfig

__all__ = ["Context", "find_configs", "load_yaml"]


def find_configs(start: Path | None = None) -> Path:
    """Locate the ``configs/`` directory by walking up from here, then from the cwd."""
    seeds = [Path(__file__).resolve()]
    if start is not None:
        seeds.insert(0, Path(start).resolve() / "_")
    seeds.append(Path.cwd().resolve() / "_")
    for seed in seeds:
        for parent in seed.parents:
            if (parent / "configs" / "base.yaml").is_file():
                return parent / "configs"
    raise FileNotFoundError(
        "could not locate configs/base.yaml. Run from the project root, or pass --configs."
    )


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@dataclass  # not slots=True: cached_property needs __dict__, and one long-lived Context
            # per process makes the memory saving meaningless anyway
class Context:
    """Everything a command needs, resolved once.

    Heavy members are lazy: a command that only lists work should not pay for a 180 MB dataset
    load, and on Kaggle that difference is the whole start-up cost of a session.
    """

    configs: Path
    device: str = "cpu"
    root: Path = Path(".")
    _base: dict = None
    _audits: dict = None

    def __post_init__(self) -> None:
        self._base = load_yaml(self.configs / "base.yaml")
        self._audits = load_yaml(self.configs / "audits.yaml")

    # -- raw config ----------------------------------------------------------

    @property
    def base(self) -> dict:
        return self._base

    @property
    def audits(self) -> dict:
        return self._audits

    @property
    def seeds(self) -> dict:
        return self._base["seeds"]

    @property
    def train_config(self) -> TrainConfig:
        t = dict(self._base["training"])
        allowed = set(TrainConfig.__slots__)
        return TrainConfig(**{k: v for k, v in t.items() if k in allowed})

    # -- resolved artefacts --------------------------------------------------

    @cached_property
    def bundle(self) -> DataBundle:
        d = self._base["dataset"]
        return load_cifar(
            self.root / d["root"],
            name=d["name"],
            download=True,
            expect_sha=d.get("expect_sha"),
        )

    @cached_property
    def memorization(self) -> np.ndarray:
        m = self._base["memorization"]
        path = self.root / m["path"]
        scores = load_scores(path, n_train=self._base["dataset"]["n_train"])
        expect = m.get("expect_sha")
        if expect:
            from .registry import file_sha

            actual = file_sha(path)
            if actual != expect:
                raise RuntimeError(
                    f"memorization scores hash to {actual}, expected {expect}. The file differs "
                    "from the one the rest of the project used; the difficulty axis would not "
                    "be comparable."
                )
        return scores

    @cached_property
    def store(self) -> ArtifactStore:
        return ArtifactStore(self.root / self._base["paths"]["artifacts"])

    @property
    def records_dir(self) -> Path:
        p = self.root / self._base["paths"]["records"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- forget sets ---------------------------------------------------------

    def spec(self, forget_id: str) -> ForgetSpec:
        return spec_by_id(forget_id)

    def forget_indices(self, forget_id: str) -> np.ndarray:
        """Resolve (and cache) a condition's index array, verified against the store."""
        spec = self.spec(forget_id)
        kw = {"memorization": self.memorization} if spec.kind == "memstratum" else {}
        return materialise(
            spec, self.bundle.n_train, store=self.store, **kw
        )

    def all_forget_ids(self) -> tuple[str, ...]:
        return tuple(all_specs())

    @property
    def primary_condition(self) -> str:
        return self._base["oracles"]["primary_condition"]

    def describe(self) -> dict[str, Any]:
        d = self._base["dataset"]
        return {
            "dataset": d["name"],
            "arch": self._base["model"]["arch"],
            # Report what is *available*, not just what this invocation asked for. `status` takes
            # no --device, so echoing the CLI default printed "device: cpu" straight after a run
            # that had plainly used a GPU -- true of the status call, useless as information, and
            # alarming to read.
            "accelerator": _accelerator(),
            "requested_device": self.device,
            "artifacts": str(self.store.root),
            "records": str(self.records_dir),
            "epochs": self.train_config.epochs,
            "train_seeds": self.seeds["train"],
            "primary_condition": self.primary_condition,
        }


def _accelerator() -> str:
    """What torch can actually see. Cheap: torch is already imported by anything that trains."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu only"
    except Exception:
        return "unknown"
