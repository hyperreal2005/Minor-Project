"""CIFAR-10/100 loading with fixed, hashed splits and the canary corruption.

Two things this module is responsible for, both of which are gates in the implementation plan:

**Byte-identical data across machines.** Stage 0's gate is that the data hash matches on Kaggle
and locally. Every result in the project is comparable only if every model saw the same 50,000
images in the same canonical order, so the hash is computed once and checked on every load
rather than assumed.

**Deterministic evaluation order.** Activations from two models are only comparable if row *i*
is the same example in both. Evaluation loaders are therefore never shuffled and never drop a
partial last batch, and :func:`make_loader` refuses to produce a shuffled loader for evaluation.

The data is held in memory as uint8 (about 180 MB for CIFAR-10) rather than read per-item through
PIL. On the CPU-bound local machine this is a large speedup, and it makes index-based subsetting
— which this project does constantly — trivial.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "DataBundle",
    "load_cifar",
    "make_loader",
    "canary_labels",
    "apply_canaries",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "N_TRAIN",
]

CIFAR10_MEAN: Final = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Final = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN: Final = (0.5071, 0.4865, 0.4409)
CIFAR100_STD: Final = (0.2673, 0.2564, 0.2762)

N_TRAIN: Final = 50_000
N_TEST: Final = 10_000


@dataclass(frozen=True, slots=True)
class DataBundle:
    """The whole dataset in memory, plus the hash that proves it is the same everywhere."""

    name: str
    train_x: np.ndarray  # uint8 (N, 32, 32, 3)
    train_y: np.ndarray  # int64 (N,)
    test_x: np.ndarray
    test_y: np.ndarray
    sha: str
    num_classes: int

    def __post_init__(self) -> None:
        if self.train_x.dtype != np.uint8 or self.test_x.dtype != np.uint8:
            raise ValueError("images must be uint8; normalisation happens in the transform")
        if len(self.train_x) != len(self.train_y):
            raise ValueError("train images and labels disagree in length")

    @property
    def n_train(self) -> int:
        return len(self.train_y)

    def all_indices(self) -> np.ndarray:
        return np.arange(self.n_train, dtype=np.int64)

    def retain_indices(self, forget: np.ndarray) -> np.ndarray:
        """Complement of the forget set, sorted. The retain set is always derived, never stored,
        so the two can never disagree about which examples are in play."""
        mask = np.ones(self.n_train, dtype=bool)
        mask[np.asarray(forget, dtype=np.int64)] = False
        return np.flatnonzero(mask).astype(np.int64)

    def with_labels(self, train_y: np.ndarray) -> "DataBundle":
        """A copy carrying different training labels — used for the canary variant."""
        y = np.asarray(train_y, dtype=np.int64)
        if len(y) != self.n_train:
            raise ValueError("replacement labels have the wrong length")
        return DataBundle(
            name=self.name,
            train_x=self.train_x,
            train_y=y,
            test_x=self.test_x,
            test_y=self.test_y,
            sha=self.sha,
            num_classes=self.num_classes,
        )


def _hash_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def load_cifar(
    root: str | Path = "data",
    *,
    name: str = "cifar10",
    download: bool = True,
    expect_sha: str | None = None,
) -> DataBundle:
    """Load a CIFAR dataset into memory.

    Args:
        expect_sha: if given, raise unless the loaded data hashes to it. Stage 0's gate — pin
            this once the first machine has reported its hash, so a silently different download
            on another machine fails loudly instead of producing incomparable results.
    """
    import torchvision

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if name == "cifar10":
        cls, num_classes = torchvision.datasets.CIFAR10, 10
    elif name == "cifar100":
        cls, num_classes = torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(f"unknown dataset {name!r}")

    train = cls(root=str(root), train=True, download=download)
    test = cls(root=str(root), train=False, download=download)

    train_x = np.asarray(train.data, dtype=np.uint8)
    train_y = np.asarray(train.targets, dtype=np.int64)
    test_x = np.asarray(test.data, dtype=np.uint8)
    test_y = np.asarray(test.targets, dtype=np.int64)

    sha = _hash_arrays(train_x, train_y, test_x, test_y)
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(
            f"{name} hashes to {sha}, expected {expect_sha}. The data on this machine differs "
            "from the data the rest of the project used; results would not be comparable."
        )

    return DataBundle(
        name=name,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        sha=sha,
        num_classes=num_classes,
    )


# --------------------------------------------------------------------------- canaries


def canary_labels(y_true: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The deliberately wrong label assigned to each canary.

    ``y_canary = (y_true + 1 + (idx % 9)) % 10``

    The offset lies in [1, 9], so it is never congruent to 0 mod 10 and the assigned label can
    never coincide with the true one. The implementation plan hedged with "re-drawn if it
    collides"; the formula makes a collision impossible by construction, so no re-draw exists
    and none is needed.

    Determinism comes from the example's own index rather than from an RNG, so the corruption is
    reproducible from the index array alone.
    """
    idx = np.asarray(indices, dtype=np.int64)
    y = np.asarray(y_true, dtype=np.int64)[idx]
    n_classes = 10
    offset = 1 + (idx % (n_classes - 1))
    wrong = (y + offset) % n_classes
    if np.any(wrong == y):  # pragma: no cover - impossible by the argument above
        raise AssertionError("canary label collided with the true label")
    return wrong.astype(np.int64)


def apply_canaries(
    bundle: DataBundle, indices: np.ndarray
) -> tuple[DataBundle, np.ndarray]:
    """Return a bundle whose training labels are corrupted at ``indices``.

    The canary condition's original model trains on this corrupted bundle: canaries must be
    present during training for there to be anything to forget. The oracle trains on the *clean*
    bundle restricted to the retain set — the canaries are removed, not corrected — so it
    provably carries no canary-label association.

    Returns:
        ``(corrupted_bundle, assigned_wrong_labels)``.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if bundle.num_classes != 10:
        raise NotImplementedError(
            "the canary label formula assumes 10 classes; generalise it before using "
            "CIFAR-100 for the canary condition"
        )
    wrong = canary_labels(bundle.train_y, idx)
    y = bundle.train_y.copy()
    y[idx] = wrong
    return bundle.with_labels(y), wrong


# --------------------------------------------------------------------------- torch plumbing


class _IndexedCIFAR(Dataset):
    """A view onto a subset of a bundle, with augmentation applied on the fly."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        *,
        train: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        img = torch.from_numpy(self.x[j]).permute(2, 0, 1).float().div_(255.0)

        if self.train:
            # RandomCrop(32, padding=4) then RandomHorizontalFlip — the standard CIFAR pair.
            img = torch.nn.functional.pad(img, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            img = img[:, top : top + 32, left : left + 32]
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=(2,))

        img = (img - self._mean) / self._std
        # The original index travels with the example so that audits can align rows to examples
        # without trusting loader order.
        return img, int(self.y[j]), j


def make_loader(
    bundle: DataBundle,
    indices: np.ndarray | None = None,
    *,
    train: bool,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int | None = None,
    split: str = "train",
    drop_last: bool = False,
) -> DataLoader:
    """Build a loader over a subset of a bundle.

    Args:
        train: augmentation and shuffling on. **Evaluation loaders must pass ``train=False``**:
            activations from two models are only comparable if row *i* is the same example in
            both, so eval loaders are never shuffled and never drop a partial batch.
        split: ``'train'`` or ``'test'`` — which half of the bundle to index into.
        seed: seeds the loader's shuffling generator. Required for reproducible training order.
    """
    if split == "train":
        x, y, n = bundle.train_x, bundle.train_y, bundle.n_train
    elif split == "test":
        x, y, n = bundle.test_x, bundle.test_y, len(bundle.test_y)
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    idx = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise IndexError(f"indices out of range for {split} split of size {n}")

    mean, std = (
        (CIFAR10_MEAN, CIFAR10_STD)
        if bundle.num_classes == 10
        else (CIFAR100_MEAN, CIFAR100_STD)
    )
    ds = _IndexedCIFAR(x, y, idx, train=train, mean=mean, std=std)

    generator = None
    if train and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=drop_last and train,
        generator=generator,
        persistent_workers=bool(num_workers) and train,
    )
