"""Diagnostic: does sharing an initialisation make a model look more retrain-like?

Paired oracles are trained with the same `train_seed` as the M0 they pair with, so oracle *s* and
M0 *s* start from **identical weights** (`make_resnet18(seed=...)`, by design — it is what makes
the paired relearning anchors a controlled comparison). Every unlearned model starts from its M0.

The oracle-referenced audits then compare each unlearned model against all five paired oracles —
one of which is its initialisation twin — while the calibration band is built from oracle-vs-
oracle comparisons, which never contain a twin. If a shared initialisation keeps two networks
measurably closer, the unlearned models and M0 enjoy an advantage the band does not see, and
the audits read them as more retrain-like than they are: false negatives, in the direction that
flatters every method.

This measures it from the cached outputs, no GPU: for each unlearned model, its distance to its
twin oracle against its distance to the four others, and both against oracle-vs-oracle distance.
The decision-relevant number is the **shift of the ensemble mean** the twin causes, relative to
the band's half-width — if it is a sizeable fraction, verdicts can flip and the oracle-referenced
audits should exclude the twin, exactly as leave-one-out excludes an oracle from its own band.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

__all__ = ["twin_effect", "feature_cka"]


def feature_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA in feature space: ``||YᵀX||² / (||XᵀX|| ||YᵀY||)`` on centred columns.

    Identical to the Gram-matrix form for the linear kernel, and ``d×d`` rather than ``n×n`` —
    512 × 512 instead of 3000 × 3000 at layer4, which is what makes a few hundred pairs cheap.
    """
    x = np.asarray(x, dtype=np.float64); x = x - x.mean(axis=0)
    y = np.asarray(y, dtype=np.float64); y = y - y.mean(axis=0)
    num = float(np.linalg.norm(y.T @ x) ** 2)
    den = float(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))
    return num / den if den > 0 else float("nan")


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-example L2 between two activation matrices -- the audits' `activation_l2`."""
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=1).mean())


def _js(a: np.ndarray, b: np.ndarray) -> float:
    from ..audits.base import softmax
    from ..audits.behavioral import js_divergence

    return float(js_divergence(softmax(a, floor=1e-12), softmax(b, floor=1e-12)).mean())


def twin_effect(
    store,
    *,
    forget_id: str,
    train_seeds: Iterable[int],
    methods: Iterable[str],
    layer: str = "layer4",
    problems: list | None = None,
) -> list[dict]:
    """One row per metric (forget-probe JS, `layer` CKA) for one condition.

    Positive `twin_advantage_sd` means the twin oracle is *closer* than the other oracles, in
    units of the oracle-vs-oracle spread. `ensemble_shift` is how much the twin moves the
    five-oracle mean the audits actually use (one fifth of the twin/non-twin gap).

    Every cache that cannot be used is appended to ``problems`` as ``(run_id, reason)``, so a
    skipped condition says why rather than leaving it to be guessed.
    """
    from ..registry import run_id

    seeds = list(train_seeds)
    oracle_ids = {s: run_id(role="oracle", forget=forget_id, seed=s, seed_kind="train")
                  for s in seeds}

    def skip(rid, why):
        if problems is not None:
            problems.append((rid, why))

    def load(rid):
        if not (store.has_outputs(rid) and store.has_activations(rid)):
            return skip(rid, "not cached in this session")
        try:
            logits, _ = store.load_outputs(rid, forget_id=forget_id)
            acts, _ = store.load_activations(rid)
        except Exception as exc:  # unreadable cache: skip it, never fail the diagnostic
            return skip(rid, f"unreadable ({type(exc).__name__})")
        if "forget" not in logits or layer not in acts:
            return skip(rid, f"no forget logits or no {layer}")
        if not (np.isfinite(logits["forget"]).all() and np.isfinite(acts[layer]).all()):
            # an fp16-overflowed cache (the destroyed control's): skip, never NaN
            return skip(rid, "non-finite (fp16 overflow)")
        return logits["forget"], acts[layer]

    oracles = {s: load(r) for s, r in oracle_ids.items()}
    oracles = {s: v for s, v in oracles.items() if v is not None}
    if len(oracles) < 3:
        return []

    # Raw activation L2 as well as CKA. CKA is blind to how neurons are ordered; raw L2 compares
    # neuron i with neuron i, which only means something when the two networks' neurons
    # correspond -- as a shared initialisation can make them. So the twin can matter far more
    # for L2 than for CKA, and the audits' activation_l2 is exactly this distance.
    pairs = {k: {"twin": [], "other": [], "oo": []} for k in ("js", "cka", "l2")}
    for s, u in combinations(sorted(oracles), 2):
        pairs["js"]["oo"].append(_js(oracles[s][0], oracles[u][0]))
        pairs["cka"]["oo"].append(feature_cka(oracles[s][1], oracles[u][1]))
        pairs["l2"]["oo"].append(_l2(oracles[s][1], oracles[u][1]))

    for m in methods:
        for s in seeds:
            got = load(f"c10r18__unlearn__{forget_id}__{m}__train{s}")
            if got is None or s not in oracles:
                continue
            for t, (o_logits, o_acts) in oracles.items():
                kind = "twin" if t == s else "other"
                pairs["js"][kind].append(_js(got[0], o_logits))
                pairs["cka"][kind].append(feature_cka(got[1], o_acts))
                pairs["l2"][kind].append(_l2(got[1], o_acts))

    rows = []
    for metric, closer_is in (("js", -1.0), ("cka", 1.0), ("l2", -1.0)):
        p = pairs[metric]
        if not p["twin"] or not p["other"] or len(p["oo"]) < 2:
            continue
        twin, other = float(np.mean(p["twin"])), float(np.mean(p["other"]))
        oo_sd = float(np.std(p["oo"], ddof=1))
        advantage = closer_is * (twin - other)  # > 0: the twin is closer
        rows.append({
            "forget_id": forget_id,
            "metric": {"js": "js_forget", "cka": f"cka_{layer}", "l2": f"l2_{layer}"}[metric],
            "n_twin_pairs": len(p["twin"]),
            "twin_mean": twin,
            "other_mean": other,
            "oracle_oracle_mean": float(np.mean(p["oo"])),
            "oracle_oracle_sd": oo_sd,
            "twin_advantage_sd": advantage / oo_sd if oo_sd > 0 else float("nan"),
            "ensemble_shift": advantage / len(oracles),
        })
    return rows
