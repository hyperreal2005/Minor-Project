"""Probe sets: the examples every audit looks at.

Separated from the runner because this is where comparability is won or lost. Two models'
outputs are only comparable row-for-row if they were evaluated on the same examples in the same
order, and two *conditions* are only comparable if their probe sets were drawn the same way. A
probe set that varied per model would turn every cross-model number into a comparison of probe
sets — the same class of error as a protocol that drifts between arms.

So every probe set here is a **pure function of (condition, config)**. No model, no account, no
wall-clock, nothing that could differ between the machine that audits account 1's shard and the
machine that audits account 3's.

Three probe sets, and the sizes are deliberate:

``forget``
    The whole forget set, 500–5000 examples. Never subsampled: it is the object of study, and at
    rand-500 a sample would leave too few for the split-half statistic SDE needs.

``retain``
    A fixed sample, because the real retain set is 45–49.5k and every audit would pay for it on
    every model. The sample is drawn from a generator seeded by the condition, so it is identical
    across models, accounts and reruns — but *different* across conditions, which is correct:
    each condition has its own retain set.

``test``
    The whole 10k test split. Not subsampled, because two audits need it at full size: the
    population attack matches non-members to the forget set (up to 5000), and RMIA needs three
    disjoint roles — members, non-members and the population z — out of test alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ProbeSpec", "build_probes", "probe_seed"]


def probe_seed(forget_id: str, base: int = 0) -> int:
    """A stable seed for one condition's probe draw.

    Hashed from the condition name rather than taken from an enumeration, so adding a condition
    later cannot renumber the existing ones and silently redraw probe sets that results already
    depend on. ``hash()`` is not used: it is salted per process.
    """
    import hashlib

    digest = hashlib.sha256(f"probe::{forget_id}::{base}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """Which example indices each audit probe covers, for one condition."""

    forget: np.ndarray
    retain: np.ndarray
    #: Indices into the *test* split, not the train split. Kept separate for that reason.
    test: np.ndarray
    #: The mixed probe the representation audit uses: equal parts forget, retain and test.
    #: Stored as (split, index) pairs because it spans two splits.
    mixed_train: np.ndarray
    mixed_test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"forget": self.forget, "retain": self.retain, "test": self.test}


def build_probes(
    *,
    forget_indices: np.ndarray,
    n_train: int,
    n_test: int,
    forget_id: str,
    config: Any = None,
) -> ProbeSpec:
    """Construct one condition's probe sets. Deterministic given its arguments."""
    cfg = dict(config or {})
    retain_size = int(cfg.get("retain_probe_size", 3000))
    mixed_size = int(cfg.get("probe_size", 3000))

    forget_indices = np.asarray(forget_indices, dtype=np.int64)
    rng = np.random.default_rng(probe_seed(forget_id))

    mask = np.ones(n_train, dtype=bool)
    mask[forget_indices] = False
    retain_pool = np.flatnonzero(mask)
    retain = np.sort(rng.choice(retain_pool, min(retain_size, retain_pool.size), replace=False))

    test = np.arange(n_test, dtype=np.int64)

    # The mixed probe is equal parts forget/retain/test, so CKA is not dominated by whichever
    # probe happens to be largest, and is drawn from the sets above so a model is only ever run
    # over one union of examples.
    #
    # The total is held at exactly `mixed_size` across every condition, which matters more than
    # the balance: CKA's value depends on the number of probes, so a 2500-probe condition and a
    # 3000-probe one would not be comparable. rand-500 has only 500 forget examples, well short
    # of a third, so the shortfall is backfilled from retain and test rather than accepted as a
    # smaller probe. The balance is then reported in the record's `n_probe`, and the imbalance is
    # a property of the condition, not of the model.
    take = lambda pool, k: rng.choice(pool, min(k, pool.size), replace=False)  # noqa: E731
    per = max(1, mixed_size // 3)

    m_forget = take(forget_indices, per)
    shortfall = per - m_forget.size
    # Split any shortfall evenly between the two pools that can absorb it.
    m_retain = take(retain, per + shortfall - shortfall // 2)
    m_test = take(test, mixed_size - m_forget.size - m_retain.size)

    mixed_train = np.sort(np.concatenate([m_forget, m_retain]))
    mixed_test = np.sort(m_test)

    return ProbeSpec(
        forget=np.sort(forget_indices),
        retain=retain,
        test=test,
        mixed_train=mixed_train,
        mixed_test=mixed_test,
    )
