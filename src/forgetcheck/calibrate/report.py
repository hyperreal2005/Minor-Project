"""Stage 7: from audit records to calibrated verdicts and audit-validity rates.

Three kinds of model pass through the same six audits, and each answers a different question:

* **Oracles** -- retrains that provably never saw the forget set. Known negatives. They form the
  null band per metric and measure each audit's false-positive rate.
* **M0** -- the original model, per condition. Known positives: it trained on the forget set.
  They measure each audit's power, and whether a condition can discriminate at all.
* **The canary condition's models** -- scored against ground truth measured directly
  (`canary.py`), so an audit's verdicts can be checked model by model.

Outputs, all Parquet under ``results/calibration/``:

``bands``           per (condition, audit, metric, probe): the retrain prediction interval.
``validity``        per (condition, audit, metric, probe): calibrated FPR (leave-one-out, should
                    be ≈ nominal), native FPR (the audit's own rule on retrains), power on M0,
                    low-discriminability, each with its n and an exact 95% interval.
``flags``           per unlearned model and metric: the calibrated and native verdicts, the
                    normalized oracle gap (the registered effect size) and the utility guard.
                    What Stage 8's agreement analysis consumes.
``canary``          per canary-condition model: the ground-truth statistics and label.
``canary_validity`` per audit metric: agreement with the ground truth, with and without the
                    utility guard.
``utility``         per model and condition: test accuracy against the retrain mean.
``relearning_vs_accuracy``  per condition: does relearning flag anything the starting
                    accuracy does not?

Per condition throughout, with the *n* stated -- seven conditions rest on five retrains, and
suppressing them would erase the difficulty axis the project is partly about.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bands import (MIN_BAND_N, Band, clopper_pearson, coverage_for, flag, loo_flags, nominal_fpr,
                    prediction_band)
from .native import native_flag, native_rule_for

__all__ = ["CalibrationConfig", "calibrate", "load_audit_records", "recover_training_metrics",
           "write_tables"]

#: Metrics that are verdicts, not measurements. A normal band over {0, 1} is meaningless; these
#: are scored by their native rule only.
VERDICT_METRICS = frozenset({"sde_verdict"})

#: Rows that are bookkeeping, not measurements of the model.
NOT_MEASUREMENTS = frozenset({"audit_undefined"})

#: Metrics that are tautological for a *paired* oracle audited as a candidate: it is its own
#: oracle anchor, so its value is fixed by construction. Only the ensemble oracles -- independent
#: retrains with borrowed anchors -- form the band for these.
SELF_ANCHORED = frozenset({"relearn_norm"})

GROUND_TRUTH = ("canary_acc", "canary_prob", "canary_top_wrong")

#: One value per condition describing the *protocol*, not any candidate: no band applies. The
#: random-init relearning floor feeds the relearning protocol check instead.
PROTOCOL_CONSTANTS = frozenset({"relearn_randinit_auc"})

#: Metrics defined relative to M0 itself. M0's value is fixed by construction (its divergence
#: from itself is 0), so "power on M0" would be a tautology and is not reported. The first real
#: calibration showed js_to_original detecting M0 at 100% everywhere, for exactly that reason.
M0_SELF_REFERENCED = frozenset({"js_to_original", "cka_to_original"})

#: Guard metrics say whether *another* measurement can be trusted -- did relearning wreck the
#: model? -- not whether anything was forgotten. They get a band (the guard needs a retrain
#: reference) but are never scored as audits: no power on M0, no canary verdicts. Scored, the
#: first real run put relearn_utility_drop at balanced accuracy 0.50, reading as an audit at
#: chance when it is not an audit.
GUARDS = frozenset({"relearn_utility_drop"})

#: Metrics whose values are proportions -- of examples, pairs or operating points -- so a raw
#: difference reads directly as points and `min_effect` applies. The rest (divergences,
#: distances, CKA, HSIC) have no natural unit and keep the registered sd rule alone.
PROPORTION_METRICS = frozenset({
    "relearn_auc", "pred_agreement",
    "mia_acc_pop", "mia_auc_pop", "mia_tpr_at_fpr_pop",
    "mia_acc_rmia", "mia_auc_rmia", "mia_tpr_at_fpr_rmia",
})


@dataclass(frozen=True)
class CalibrationConfig:
    width_sd: float = 2.0
    low_discriminability_sd: float = 1.0
    alpha: float = 0.05
    tpr_fpr: float = 0.01
    primary_condition: str = "mem-high-3000"
    band_oracle_seeds: tuple[int, ...] = ()
    holdout_oracle_seeds: tuple[int, ...] = ()
    min_effect: float = 0.02
    utility_guard_pp: float = 5.0

    @property
    def coverage(self) -> float:
        return coverage_for(self.width_sd)

    @classmethod
    def from_context(cls, ctx) -> "CalibrationConfig":
        cal = ctx.audits.get("calibration", {})
        rmia = ctx.audits.get("privacy", {}).get("rmia", {})
        return cls(
            width_sd=float(cal.get("band_width_sd", 2.0)),
            low_discriminability_sd=float(cal.get("low_discriminability_sd", 1.0)),
            tpr_fpr=float(rmia.get("tpr_at_fpr", 0.01)),
            primary_condition=ctx.primary_condition,
            band_oracle_seeds=tuple(cal.get("band_oracle_seeds", ())),
            holdout_oracle_seeds=tuple(cal.get("probe_oracle_seeds", ())),
            min_effect=float(cal.get("min_effect", 0.02)),
            utility_guard_pp=float(cal.get("utility_guard_pp", 5.0)),
        )


# --------------------------------------------------------------------------- loading


def load_audit_records(records_dir):
    """All records, deduplicated to the latest write per measurement.

    Two legitimate sources of duplicates: relearning-anchor rows (an oracle's or M0's
    `relearn_auc` recorded while auditing *other* models) and the same model's row when it is
    later audited as a candidate; and first-pass anchor shards superseded by per-condition ones.
    The latest write wins, which is the candidate row in the first case and the per-condition
    shard in the second.
    """
    from ..registry import read_records

    df = read_records(records_dir)
    key = ["run_id", "forget_id", "audit", "metric", "probe_set"]
    df = df.sort_values("timestamp").drop_duplicates(key, keep="last")
    return df.reset_index(drop=True)


#: The training-time measurements the utility guard and the relearning check read.
TRAINING_METRICS = (("test_acc", "test"), ("forget_acc", "forget"))


def recover_training_metrics(df, store):
    """Rows for training-time metrics that have no record, recovered from checkpoint metadata.

    Stages 3 and 5 wrote each model's final test and forget accuracy twice, from one
    evaluation: as `meta` records, and into the checkpoint's metadata file. With every dataset
    attached, the first calibration to need them found 0 of the 94 training shards and 16 of
    the 240 unlearning shards in the records directory -- while every metadata file was there,
    because no checkpoint loads without one, and Stage 6 loaded them all. Only (model, metric)
    pairs with no record are recovered, so a real record always wins. Base models need only
    their test accuracy: a clean M0 has no forget set of its own. Returns ``(frame, n_models)``.
    """
    from ..data.forget_sets import spec_by_id
    from ..registry import StoreError, make_record, parse_run_id, records_frame

    meta_rows = df[df["audit"] == "meta"]
    have = set(zip(meta_rows["run_id"], meta_rows["metric"]))
    rows, models = [], set()
    for rid in sorted(df.loc[df["audit"] != "meta", "run_id"].unique()):
        key = parse_run_id(rid)
        wanted = [(m, p) for m, p in TRAINING_METRICS if (rid, m) not in have
                  and not (key.role == "base" and m == "forget_acc")]
        if not wanted:
            continue
        try:
            meta = store.load_meta(rid)
        except StoreError:
            continue
        final = meta.final_metrics or {}
        if key.role == "base":
            fields = {"forget_id": key.forget, "forget_kind": "none", "forget_size": 0}
        else:
            fields = spec_by_id(key.forget).as_record_fields()
        provenance = {"hparams": meta.hparams_sha, "checkpoint_sha": meta.sha,
                      "notes": "recovered from checkpoint metadata"}
        if meta.git_commit:
            provenance["git_commit"] = meta.git_commit
        if meta.saved_at:
            provenance["timestamp"] = meta.saved_at
        for metric, probe in wanted:
            v = final.get(metric)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
                continue
            n = int(fields["forget_size"]) if probe == "forget" else 10_000  # CIFAR-10 test set
            rows.append(make_record(run_id=rid, audit="meta", metric=metric, probe_set=probe,
                                    value=float(v), n_probe=n, **provenance, **fields))
            models.add(rid)
    return (records_frame(rows) if rows else df.iloc[0:0]), len(models)


# --------------------------------------------------------------------------- helpers


def _rate(flags) -> tuple[int, int]:
    known = [f for f in flags if f is not None]
    return int(sum(known)), len(known)


def _with_ci(prefix: str, k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {
        f"{prefix}": (k / n) if n else float("nan"),
        f"{prefix}_k": k,
        f"{prefix}_n": n,
        f"{prefix}_lo": lo,
        f"{prefix}_hi": hi,
    }


def _band_members(oracles, metric: str):
    """The oracle rows that may form the band for ``metric``.

    For self-anchored metrics only the ensemble oracles -- independent retrains, identified by a
    non-null ``oracle_seed`` -- count; a paired oracle is its own anchor there.

    A vectorised column test, deliberately not ``DataFrame.apply(..., axis=1)``. On an EMPTY
    frame ``apply`` returns an empty *float* Series rather than a boolean one, and pandas treats a
    non-boolean key as a list of column names -- so ``oracles[mask]`` came back with no columns
    at all, and the first Kaggle calibration died on ``oracles["value"]``. The frame was empty
    for `relearn_norm` at conditions where every oracle's value was undefined (anchor gap under
    `min_anchor_gap`) while some unlearned models' were not. ``notna()`` is also the null test
    that works for NaN and pd.NA alike.
    """
    if metric in SELF_ANCHORED:
        return oracles[oracles["oracle_seed"].notna()]
    return oracles


# --------------------------------------------------------------------------- the calibration


def calibrate(df, *, registry, config: CalibrationConfig) -> dict:
    """Build every Stage 7 table from a records DataFrame. Pure: no IO, no GPU."""
    import pandas as pd

    meas = df[~df["metric"].isin(NOT_MEASUREMENTS | PROTOCOL_CONSTANTS)
              & ~df["metric"].isin(GROUND_TRUTH)]
    meas = meas[meas["audit"] != "meta"]

    utility = _utility(df, config)
    damaged = {(u.forget_id, u.run_id): bool(u.damaged) for u in utility.itertuples()}
    drop_pp = {(u.forget_id, u.run_id): float(u.drop_pp) for u in utility.itertuples()}

    bands, validity, flags = [], [], []
    group_cols = ["forget_id", "audit", "metric", "probe_set"]
    for (cond, audit, metric, probe), g in meas.groupby(group_cols, sort=True):
        if metric not in registry:
            continue
        direction = registry[metric].direction
        rule = native_rule_for(audit, metric)

        oracles = _band_members(g[g["role"] == "oracle"], metric)
        m0 = g[g["role"] == "base"]
        unlearned = g[g["role"] == "unlearn"]

        retired = bool(registry[metric].retired)
        row = {"forget_id": cond, "audit": audit, "metric": metric, "probe_set": probe,
               "direction": direction, "native_rule": rule, "retired": retired}
        n_probe = int(g["n_probe"].iloc[0]) if len(g) else 0

        # ---- native rule: the audit as practitioners use it -----------------------------
        def nat(values):
            return [native_flag(rule, float(v), n_probe=n_probe, alpha=config.alpha,
                                fpr=config.tpr_fpr) for v in values] if rule else []

        if rule:
            row.update(_with_ci("native_fpr", *_rate(nat(oracles["value"]))))
            row.update(_with_ci("native_tpr", *_rate(nat(m0["value"]))))

        # ---- calibrated band -------------------------------------------------------------
        if metric in VERDICT_METRICS:
            band = Band(len(oracles), float("nan"), float("nan"), float("nan"), float("nan"))
        else:
            band = prediction_band(oracles["value"], coverage=config.coverage)
        bands.append({**{k: row[k] for k in ("forget_id", "audit", "metric", "probe_set",
                                             "direction")},
                      "n": band.n, "mean": band.mean, "sd": band.sd, "lo": band.lo,
                      "hi": band.hi, "coverage": config.coverage})

        if band.defined:
            row["nominal_fpr"] = nominal_fpr(direction, config.coverage)
            row.update(_with_ci("calibrated_fpr", *_rate(
                loo_flags(oracles["value"].to_numpy(), direction=direction,
                          coverage=config.coverage))))
            if metric not in M0_SELF_REFERENCED | GUARDS:
                row.update(_with_ci("m0_tpr", *_rate(
                    [flag(float(v), band, direction) for v in m0["value"]])))
            elif metric in M0_SELF_REFERENCED and len(m0):
                # No power and no discriminability test -- M0's value is fixed by construction --
                # but its gap is still the anchor of the signed scale: 1 = M0, 0 = a retrain,
                # negative = farther from M0 than a retrain is. H3 needs exactly that sign.
                row["m0_gap_raw"] = float(m0["value"].mean()) - band.mean
            if len(m0) and band.sd == band.sd and metric not in M0_SELF_REFERENCED | GUARDS:
                gap = abs(float(m0["value"].mean()) - band.mean)
                # A zero-width band is a constant metric: every retrain gave the same value.
                # If M0 gives it too, that is 0/0 -- nothing to discriminate -- not infinity.
                # relearn_t80 at six conditions: every model reaches 80% at step 0.
                if band.sd > 0:
                    row["m0_gap_sd"] = gap / band.sd
                else:
                    row["m0_gap_sd"] = float("inf") if gap > 0 else float("nan")
                row["low_discriminability"] = bool(
                    gap < config.low_discriminability_sd * band.sd or gap == 0)
                row["m0_gap_raw"] = float(m0["value"].mean()) - band.mean
                if metric in PROPORTION_METRICS:
                    row["below_min_effect"] = bool(gap < config.min_effect)
                    row["min_effect"] = config.min_effect

            # Pre-registered split, primary condition only: band from the band seeds, tested
            # on the three held-out seeds. Coarse (n = 3) and reported as a check on the LOO rate.
            if cond == config.primary_condition and config.holdout_oracle_seeds:
                seeds = oracles["oracle_seed"]
                train = oracles[seeds.isin(config.band_oracle_seeds)]
                held = oracles[seeds.isin(config.holdout_oracle_seeds)]
                b9 = prediction_band(train["value"], coverage=config.coverage)
                row.update(_with_ci("holdout_fpr", *_rate(
                    [flag(float(v), b9, direction) for v in held["value"]])))

        validity.append(row)

        # ---- the registered effect size (implementation plan §4.4) --------------------------
        # The normalized oracle gap G = |m(Mu) - mean m(Mr)| / |m(M0) - mean m(Mr)|: 0 retrain-
        # like, 1 M0-like. Undefined in a marked cell -- M0 inside the retrain noise, or, for a
        # proportion, less than `min_effect` from the retrains -- where the plan's "+ epsilon"
        # would otherwise turn a vanishing denominator into a huge, meaningless score.
        denom = float("nan")
        if band.defined and len(m0) and metric not in GUARDS:
            d = abs(float(m0["value"].mean()) - band.mean)
            if d > 0 and not (row.get("low_discriminability") or row.get("below_min_effect")):
                denom = d

        # ---- per-model verdicts, for Stage 8 -----------------------------------------------
        for _, r in unlearned.iterrows():
            v = float(r["value"])
            key = (cond, r["run_id"])
            flags.append({
                "run_id": r["run_id"], "forget_id": cond, "method": r["method"],
                "train_seed": r.get("train_seed"), "audit": audit, "metric": metric,
                "probe_set": probe, "value": v,
                "calibrated_flag": flag(v, band, direction) if band.defined else None,
                "native_flag": nat([v])[0] if rule else None,
                "oracle_gap": abs(v - band.mean) / denom,
                "utility_drop_pp": drop_pp.get(key, float("nan")),
                "damaged": damaged.get(key),
                "retired": retired,
            })

    canary, canary_validity = _canary(df, meas, registry, config, damaged)
    return {
        "bands": pd.DataFrame(bands),
        "validity": pd.DataFrame(validity),
        "flags": pd.DataFrame(flags),
        "canary": canary,
        "canary_validity": canary_validity,
        "relearning_protocol": _relearning_protocol(df),
        "utility": utility,
        "relearning_vs_accuracy": _relearning_vs_accuracy(df, damaged, config),
    }


def _utility(df, config):
    """Test accuracy against the retrain mean, per model and condition: the utility guard.

    A retrain-referenced audit flags whatever is unlike a retrain, and a broken model is as
    unlike one as a model that remembers: at the canary condition every calibrated audit
    flagged the destroyed control. A flag on a model more than `utility_guard_pp` points below
    the retrain mean is read as damage, not retained influence. Stages 3 and 5 recorded
    `test_acc` for every model as a meta row; the run_id identifies the model whatever forget_id
    that row carries (M0's is "full", or its canary variant).
    """
    import pandas as pd

    cols = ["forget_id", "run_id", "role", "method", "test_acc", "retrain_test_acc", "drop_pp",
            "damaged", "guard_pp"]
    acc = df[(df["metric"] == "test_acc") & (df["probe_set"] == "test")]
    acc = acc.drop_duplicates("run_id", keep="last").set_index("run_id")["value"].astype(float)
    audited = (df[(df["audit"] != "meta") & df["role"].isin(["oracle", "base", "unlearn"])]
               [["forget_id", "run_id", "role", "method"]]
               .drop_duplicates(["forget_id", "run_id"]))
    rows = []
    for cond, m in audited.groupby("forget_id", sort=True):
        ref = acc.reindex(m.loc[m["role"] == "oracle", "run_id"]).dropna()
        if ref.empty:
            continue
        ref_mean = float(ref.mean())
        for x in m.itertuples(index=False):
            a = acc.get(x.run_id)
            if a is None or a != a:
                continue
            drop = 100.0 * (ref_mean - float(a))
            rows.append({"forget_id": cond, "run_id": x.run_id, "role": x.role,
                         "method": x.method, "test_acc": float(a), "retrain_test_acc": ref_mean,
                         "drop_pp": drop, "damaged": bool(drop > config.utility_guard_pp),
                         "guard_pp": config.utility_guard_pp})
    return pd.DataFrame(rows, columns=cols)


def _relearning_vs_accuracy(df, damaged, config):
    """Does relearning flag anything the starting accuracy does not?

    `relearn_auc` averages forget accuracy over the relearning curve *including step 0*, so it
    contains the model's starting accuracy. Where retrains already get the forget set right
    (rand-*, mem-low: 0.93-0.998) the curve starts near the top with nowhere to climb, and the
    score is the starting accuracy by another name. The audit's distinctive claim is hidden
    knowledge: a model whose accuracy looks retrain-like but which relearns faster. So, per
    condition, against the same retrains: how many unlearned models does relearning flag, how
    many does starting accuracy flag, and how many does *only* relearning flag.

    Starting accuracy is the curve's step 0 -- accuracy on the whole forget set under the
    condition's labels. Stages 3 and 5 recorded exactly that as meta `forget_acc`, except at the
    canary condition, where Stage 3 scored the retrains on the clean labels; there the
    ground-truth pass's `canary_acc` is the same quantity for every role. Damaged models are
    left out: a collapsed network relearning is just training.
    """
    import pandas as pd

    rel = df[(df["audit"] == "relearning") & (df["metric"] == "relearn_auc")
             & (df["probe_set"] == "forget") & df["role"].isin(["oracle", "unlearn"])]
    rows = []
    for cond, r in rel.groupby("forget_id", sort=True):
        canary = "forget_kind" in r and str(r["forget_kind"].iloc[0]) == "canary"
        metric, probe = ("canary_acc", "canary") if canary else ("forget_acc", "forget")
        s = df[(df["metric"] == metric) & (df["probe_set"] == probe)]
        start = s.drop_duplicates("run_id", keep="last").set_index("run_id")["value"].astype(float)
        r = r.drop_duplicates("run_id", keep="last")
        keep = r["run_id"].isin(start.index).to_numpy() & np.array(
            [not damaged.get((cond, rid), False) for rid in r["run_id"]], dtype=bool)
        r = r[keep]
        o, u = r[r["role"] == "oracle"], r[r["role"] == "unlearn"]
        if len(o) < MIN_BAND_N:
            continue
        o_start, u_start = start[o["run_id"]].to_numpy(), start[u["run_id"]].to_numpy()
        o_auc, u_auc = o["value"].to_numpy(float), u["value"].to_numpy(float)
        b_start = prediction_band(o_start, coverage=config.coverage)
        b_auc = prediction_band(o_auc, coverage=config.coverage)
        by_start = [bool(flag(float(x), b_start, "closer_to_oracle")) for x in u_start]
        by_auc = [bool(flag(float(x), b_auc, "closer_to_oracle")) for x in u_auc]
        x, y = np.concatenate([o_start, u_start]), np.concatenate([o_auc, u_auc])
        corr = (float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0
                else float("nan"))
        # Which side of the retrains a relearning-only flag falls on is the whole question:
        # faster than a retrain is hidden knowledge; slower is damage or over-forgetting, which
        # relearning sees and accuracy does not -- but it is not what the audit claims to find.
        only = [a and not b for a, b in zip(by_auc, by_start)]
        fast = [o and float(v) > b_auc.mean for o, v in zip(only, u_auc)]
        faster = sum(fast)

        def tally(sel):
            from collections import Counter

            c = Counter(m for s, m in zip(sel, u["method"]) if s)
            return ", ".join(f"{m} {n}" for m, n in sorted(c.items()))
        rows.append({
            "forget_id": cond, "start_metric": metric,
            "n_retrain": len(o), "n_unlearned": len(u),
            "retrain_start": float(o_start.mean()), "retrain_relearn": float(o_auc.mean()),
            "unlearned_start": float(u_start.mean()) if len(u) else float("nan"),
            "unlearned_relearn": float(u_auc.mean()) if len(u) else float("nan"),
            "r_relearn_vs_start": corr,
            "flagged_by_start": sum(by_start),
            "flagged_by_relearn": sum(by_auc),
            "flagged_by_relearn_only": sum(only),
            "relearn_only_faster": faster,
            "relearn_only_slower": sum(only) - faster,
            "faster_methods": tally(fast),
            "slower_methods": tally([o and not f for o, f in zip(only, fast)]),
        })
    return pd.DataFrame(rows)


def _relearning_protocol(df):
    """The relearning layer's own self-test, per condition: random-init < retrain < M0.

    The gate the plan wrote for Stage 6 ("original ~1, oracle ~0") holds by definition for the
    normalised metric, so it checks nothing. The version with content compares raw curve AUCs:
    a freshly initialised network relearning the same 500 examples must recover *less* than a
    retrain (else the reintroduction data alone manufactures recovery), and M0 at least as much
    as a retrain. `m0_minus_oracle_sd` says whether M0 and the retrains are separable at all
    here -- where they are not, reversibility is not measurable at this condition.
    """
    import pandas as pd

    g = df[df["audit"] == "relearning"]
    rows = []
    for cond, c in g.groupby("forget_id", sort=True):
        floor = c.loc[c["metric"] == "relearn_randinit_auc", "value"]
        oracle = c.loc[(c["metric"] == "relearn_auc") & (c["role"] == "oracle"), "value"]
        m0 = c.loc[(c["metric"] == "relearn_auc") & (c["role"] == "base"), "value"]
        if oracle.empty:
            continue
        o_mean = float(oracle.mean())
        o_sd = float(oracle.std(ddof=1)) if len(oracle) > 1 else float("nan")
        f = float(floor.mean()) if len(floor) else float("nan")
        m = float(m0.mean()) if len(m0) else float("nan")
        kind = c["forget_kind"].iloc[0] if "forget_kind" in c and len(c) else ""
        rows.append({
            "forget_id": cond,
            # At the canary condition the reintroduced labels are *wrong*, and they contradict
            # the retrain's knowledge of the true label, which a random-init network does not
            # have. So there the floor is not a floor: "retrain no faster than random init" is
            # the expected outcome -- the retrain carries no residual association -- not a
            # protocol failure.
            "note": ("canary: floor ~ retrain expected (wrong labels vs the retrain's true-label "
                     "prior)" if kind == "canary" else ""),
            "randinit_auc": f,
            "oracle_auc_mean": o_mean, "oracle_auc_sd": o_sd, "oracle_n": int(len(oracle)),
            "m0_auc_mean": m, "m0_n": int(len(m0)),
            "floor_below_oracle": bool(f < o_mean) if f == f else None,
            "m0_minus_oracle_sd": (m - o_mean) / o_sd if (m == m and o_sd and o_sd > 0) else float("nan"),
        })
    return pd.DataFrame(rows)


def _confusion(items) -> dict:
    """Counts and rates for ``(flagged, truly_retains, role)`` triples.

    False positives are split by who they are. A retrain flagged is an invalid audit; an
    unlearned model with no residual association flagged -- in practice the destroyed control --
    is an audit that cannot tell *damage* from *retained influence*.
    """
    tp = sum(1 for f, t, _ in items if f and t)
    fp = sum(1 for f, t, _ in items if f and not t)
    tn = sum(1 for f, t, _ in items if not f and not t)
    fn = sum(1 for f, t, _ in items if not f and t)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "tnr": tnr,
            "fp_retrain": sum(1 for f, t, r in items if f and not t and r == "oracle"),
            "fp_other": sum(1 for f, t, r in items if f and not t and r != "oracle"),
            "balanced_accuracy": np.nanmean([tpr, tnr]), "n": len(items)}


def _canary(df, meas, registry, config, damaged=None):
    """Ground truth at the canary condition, and every audit's agreement with it."""
    import pandas as pd

    damaged = damaged or {}
    gt = df[df["metric"].isin(GROUND_TRUTH) & (df["probe_set"] == "canary")]
    if gt.empty:
        return pd.DataFrame(), pd.DataFrame()
    cond = gt["forget_id"].iloc[0]
    wide = gt.pivot_table(index=["run_id", "role", "method"], columns="metric",
                          values="value").reset_index()

    # Ground truth: oracles never saw the canaries (negative); M0 trained on them (positive);
    # an unlearned model is positive if it still prefers the assigned wrong label, among wrong
    # labels only, more than the retrains do.
    top = wide.set_index("run_id")["canary_top_wrong"] if "canary_top_wrong" in wide else None
    oracle_top = wide.loc[wide["role"] == "oracle", "canary_top_wrong"] if top is not None else []
    top_band = prediction_band(oracle_top, coverage=config.coverage)

    def truth(r):
        if r["role"] == "oracle":
            return False
        if r["role"] == "base":
            return True
        if not top_band.defined:
            return None
        return bool(r["canary_top_wrong"] > top_band.hi)

    wide["ground_truth"] = wide.apply(truth, axis=1)
    wide["top_wrong_band_hi"] = top_band.hi
    labelled = wide.set_index("run_id")["ground_truth"]
    role_of = wide.set_index("run_id")["role"]

    rows = []
    at = meas[meas["forget_id"] == cond]
    for (audit, metric, probe), g in at.groupby(["audit", "metric", "probe_set"]):
        if metric not in registry or metric in GUARDS:
            continue
        direction = registry[metric].direction
        rule = native_rule_for(audit, metric)
        oracles = _band_members(g[g["role"] == "oracle"], metric)
        n_probe = int(g["n_probe"].iloc[0])

        verdicts: dict[str, dict[str, bool | None]] = {"calibrated": {}, "native": {}}
        if metric not in VERDICT_METRICS:
            band = prediction_band(oracles["value"], coverage=config.coverage)
            loo = loo_flags(oracles["value"].to_numpy(), direction=direction,
                            coverage=config.coverage)
            verdicts["calibrated"].update(dict(zip(oracles["run_id"], loo)))
            for _, r in g[g["role"] != "oracle"].iterrows():
                verdicts["calibrated"][r["run_id"]] = flag(float(r["value"]), band, direction)
        if rule:
            for _, r in g.iterrows():
                verdicts["native"][r["run_id"]] = native_flag(
                    rule, float(r["value"]), n_probe=n_probe, alpha=config.alpha,
                    fpr=config.tpr_fpr)

        for kind, v in verdicts.items():
            # M0 is excluded for metrics defined against M0 itself: its value is fixed by
            # construction, so counting it would score the audit on a tautology.
            rids = [rid for rid in v if not (metric in M0_SELF_REFERENCED
                                             and role_of.get(rid) == "base")]
            items = [(rid, v[rid], labelled.get(rid), role_of.get(rid)) for rid in rids
                     if v[rid] is not None and labelled.get(rid) is not None]
            if not items:
                continue
            # The utility guard: a flag on a damaged model is read as damage, not retention.
            hurt = {rid for rid, *_ in items if damaged.get((cond, rid), False)}
            guarded = _confusion([(f and rid not in hurt, t, r) for rid, f, t, r in items])
            rows.append({"audit": audit, "metric": metric, "probe_set": probe, "kind": kind,
                         "retired": bool(registry[metric].retired),
                         **_confusion([(f, t, r) for _, f, t, r in items]),
                         "n_damaged": len(hurt),
                         "fp_other_guarded": guarded["fp_other"],
                         "ba_guarded": guarded["balanced_accuracy"]})
    return wide, pd.DataFrame(rows)


def write_tables(tables: dict, out_dir) -> list[Path]:
    """Write each table as Parquet, atomically. Returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, t in tables.items():
        path = out / f"{name}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        t.to_parquet(tmp, index=False)
        tmp.replace(path)
        paths.append(path)
    return paths
