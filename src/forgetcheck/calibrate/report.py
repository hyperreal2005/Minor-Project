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
``flags``           per unlearned model and metric: the calibrated and native verdicts. What
                    Stage 8's agreement analysis consumes.
``canary``          per canary-condition model: the ground-truth statistics and label.
``canary_validity`` per audit metric: agreement with the ground truth.

Per condition throughout, with the *n* stated -- seven conditions rest on five retrains, and
suppressing them would erase the difficulty axis the project is partly about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bands import Band, clopper_pearson, coverage_for, flag, loo_flags, nominal_fpr, prediction_band
from .native import native_flag, native_rule_for

__all__ = ["CalibrationConfig", "calibrate", "load_audit_records", "write_tables"]

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


@dataclass(frozen=True)
class CalibrationConfig:
    width_sd: float = 2.0
    low_discriminability_sd: float = 1.0
    alpha: float = 0.05
    tpr_fpr: float = 0.01
    primary_condition: str = "mem-high-3000"
    band_oracle_seeds: tuple[int, ...] = ()
    holdout_oracle_seeds: tuple[int, ...] = ()

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


def _is_ensemble_oracle(row) -> bool:
    return row["role"] == "oracle" and not _isnan(row.get("oracle_seed"))


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or (x != x)
    except Exception:
        return False


# --------------------------------------------------------------------------- the calibration


def calibrate(df, *, registry, config: CalibrationConfig) -> dict:
    """Build every Stage 7 table from a records DataFrame. Pure: no IO, no GPU."""
    import pandas as pd

    meas = df[~df["metric"].isin(NOT_MEASUREMENTS) & ~df["metric"].isin(GROUND_TRUTH)]
    meas = meas[meas["audit"] != "meta"]

    bands, validity, flags = [], [], []
    group_cols = ["forget_id", "audit", "metric", "probe_set"]
    for (cond, audit, metric, probe), g in meas.groupby(group_cols, sort=True):
        if metric not in registry:
            continue
        direction = registry[metric].direction
        rule = native_rule_for(audit, metric)

        oracles = g[g["role"] == "oracle"]
        if metric in SELF_ANCHORED:
            oracles = oracles[oracles.apply(_is_ensemble_oracle, axis=1)]
        m0 = g[g["role"] == "base"]
        unlearned = g[g["role"] == "unlearn"]

        row = {"forget_id": cond, "audit": audit, "metric": metric, "probe_set": probe,
               "direction": direction, "native_rule": rule}
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
            row.update(_with_ci("m0_tpr", *_rate(
                [flag(float(v), band, direction) for v in m0["value"]])))
            if len(m0) and band.sd == band.sd:
                gap = abs(float(m0["value"].mean()) - band.mean)
                row["m0_gap_sd"] = gap / band.sd if band.sd > 0 else float("inf")
                row["low_discriminability"] = bool(gap < config.low_discriminability_sd * band.sd)

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

        # ---- per-model verdicts, for Stage 8 -----------------------------------------------
        for _, r in unlearned.iterrows():
            v = float(r["value"])
            flags.append({
                "run_id": r["run_id"], "forget_id": cond, "method": r["method"],
                "train_seed": r.get("train_seed"), "audit": audit, "metric": metric,
                "probe_set": probe, "value": v,
                "calibrated_flag": flag(v, band, direction) if band.defined else None,
                "native_flag": nat([v])[0] if rule else None,
            })

    canary, canary_validity = _canary(df, meas, registry, config)
    return {
        "bands": pd.DataFrame(bands),
        "validity": pd.DataFrame(validity),
        "flags": pd.DataFrame(flags),
        "canary": canary,
        "canary_validity": canary_validity,
    }


def _canary(df, meas, registry, config):
    """Ground truth at the canary condition, and every audit's agreement with it."""
    import pandas as pd

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

    rows = []
    at = meas[meas["forget_id"] == cond]
    for (audit, metric, probe), g in at.groupby(["audit", "metric", "probe_set"]):
        if metric not in registry:
            continue
        direction = registry[metric].direction
        rule = native_rule_for(audit, metric)
        oracles = g[g["role"] == "oracle"]
        if metric in SELF_ANCHORED:
            oracles = oracles[oracles.apply(_is_ensemble_oracle, axis=1)]
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
            pairs = [(v[rid], labelled.get(rid)) for rid in v
                     if v[rid] is not None and labelled.get(rid) is not None]
            if not pairs:
                continue
            tp = sum(1 for f, t in pairs if f and t)
            fp = sum(1 for f, t in pairs if f and not t)
            tn = sum(1 for f, t in pairs if not f and not t)
            fn = sum(1 for f, t in pairs if not f and t)
            tpr = tp / (tp + fn) if tp + fn else float("nan")
            tnr = tn / (tn + fp) if tn + fp else float("nan")
            rows.append({"audit": audit, "metric": metric, "probe_set": probe, "kind": kind,
                         "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "tnr": tnr,
                         "balanced_accuracy": np.nanmean([tpr, tnr]), "n": len(pairs)})
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
