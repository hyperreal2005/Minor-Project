"""The Stage 7 report, printed from ``results/calibration/*.parquet``.

`forgetcheck calibrate` prints it after writing the tables; the notebook's results cell calls it
too. It lives in the package, not in a notebook cell, because a Kaggle notebook's cells are
frozen when it is imported and ``git pull`` does not touch them: the first re-run after the
canary table gained its counts printed the *old* table from the new data. The install cell
always pulls the current code, so the report is always the current one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["print_report"]


def _read(out: Path, name: str):
    import pandas as pd

    path = out / f"{name}.parquet"
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _section(title: str, *notes: str) -> None:
    print(title)
    for n in notes:
        print(n)
    print()


def print_report(out_dir="results/calibration") -> None:
    import pandas as pd

    out = Path(out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 40)

    v = _read(out, "validity")
    if v.empty:
        print(f"no validity table under {out}; run `forgetcheck calibrate` first")
        return
    fwd = v[v["probe_set"].isin(["forget", "layer4"])]
    nat = fwd[fwd["native_rule"].notna()] if "native_rule" in fwd else fwd.iloc[0:0]

    def pivot(frame, col, scale=1.0, digits=2):
        if col not in frame or frame[col].notna().sum() == 0:
            return "(none)"
        return (frame.pivot_table(index="metric", columns="forget_id", values=col) * scale).round(digits)

    _section("NATIVE false-positive rate -- each retrain-free audit's own rule, applied to genuine "
             "retrains",
             "(a valid rule flags ~5%; far above that means the rule, as commonly used, is broken)")
    print(pivot(nat, "native_fpr"), "\n")

    _section("NATIVE POWER -- fraction of M0 each retrain-free audit's own rule detects")
    print(pivot(nat, "native_tpr"), "\n")

    _section("POWER -- fraction of M0 (which trained on the forget set) each calibrated audit "
             "detects")
    print(pivot(fwd, "m0_tpr"), "\n")

    _section("CALIBRATED false-positive rate on held-out retrains -- should sit near nominal "
             "(0.02-0.05)", "(this checks the calibration itself, not the audits)")
    print(pivot(fwd, "calibrated_fpr"), "\n")

    if "below_min_effect" in fwd:
        prop = fwd[fwd["below_min_effect"].notna()]
        marked = prop[prop["below_min_effect"].astype(bool)]
        floor = 100.0 * float(prop["min_effect"].dropna().iloc[0]) if "min_effect" in prop else 2.0
        _section("EFFECT SIZE -- M0's raw gap from the retrain mean, proportion metrics, in points",
                 f"(below {floor:g} points the cell is marked, not scored -- the retrains sit at "
                 "a ceiling or M0 barely differs)")
        print(pivot(prop, "m0_gap_raw", scale=100.0, digits=1), "\n")
        if len(marked):
            print("marked:", ", ".join(f"{r.metric} @ {r.forget_id}"
                                       for r in marked.itertuples()), "\n")

    if "m0_gap_sd" in fwd:
        # Signed, because the direction is evidence: activation_l2 separates M0 from the
        # retrains by 16-42 sd at every condition, mem-low included, where nothing else does.
        # Negative there means M0 is *closer* to the oracle ensemble than a retrain is --
        # the signature of M0's initialisation twin in the ensemble, not of the forget set.
        sign = np.sign(fwd["m0_gap_raw"]) if "m0_gap_raw" in fwd else 1.0
        _section("EFFECT SIZE, every metric -- M0 minus the retrain mean, in retrain standard "
                 "deviations",
                 "(|value| under 1: marked by the registered safeguard; negative: M0 below the "
                 "retrains -- for a distance such as JS or L2, closer to the ensemble than a "
                 "retrain is)")
        print(pivot(fwd.assign(m0_gap_sd=sign * fwd["m0_gap_sd"]), "m0_gap_sd", digits=1), "\n")

    p = _read(out, "relearning_protocol")
    if len(p):
        _section("RELEARNING PROTOCOL -- random-init must relearn less than a retrain; M0 at least "
                 "as much")
        print(p.round(3).to_string(index=False), "\n")

    rv = _read(out, "relearning_vs_accuracy")
    _section("RELEARNING vs STARTING ACCURACY -- does relearning flag any unlearned model that "
             "its forget accuracy alone does not?",
             "(flagged_by_relearn_only = 0: relearning adds nothing to plain accuracy at that "
             "condition; damaged models excluded)")
    if len(rv):
        print(rv.round(3).to_string(index=False), "\n")
    missing = sorted(set(v["forget_id"]) - set(rv["forget_id"] if len(rv) else ()))
    if missing:
        print(f"not computed for {', '.join(missing)}: their starting accuracies are Stage 3/5 "
              f"records (the forgetcheck-artifacts dataset)\n")

    u = _read(out, "utility")
    if len(u):
        label = u["method"].where(u["role"] == "unlearn",
                                  u["role"].map({"oracle": "retrain", "base": "M0"}))
        guard = float(u["guard_pp"].iloc[0]) if "guard_pp" in u else 5.0
        _section("UTILITY -- worst test-accuracy drop below the retrain mean, in points, over "
                 "seeds",
                 f"(the guard marks a model 'damaged' above {guard:g} points; a flag on it is "
                 "read as damage, not retention)")
        print(u.assign(label=label).pivot_table(index="label", columns="forget_id",
                                                values="drop_pp", aggfunc="max").round(1), "\n")
    else:
        print("UTILITY -- no test_acc records found: attach the forgetcheck-artifacts dataset "
              "(stages 3-5)\n")

    c = _read(out, "canary")
    if len(c):
        _section("CANARY GROUND TRUTH -- oracles should sit near 1/9 on canary_top_wrong, M0 "
                 "near 1")
        c["ground_truth"] = c["ground_truth"].map({True: 1.0, False: 0.0})  # None -> NaN
        cols = [k for k in ("canary_top_wrong", "canary_acc", "canary_prob", "ground_truth")
                if k in c]
        print(c.groupby(["role", "method"])[cols].mean(numeric_only=False).round(3), "\n")

    cv = _read(out, "canary_validity")
    if len(cv):
        _section("CANARY VALIDITY -- each audit's verdicts against the ground truth",
                 "(fp_retrain: retrains flagged -- the audit is invalid; fp_other: models with no "
                 "residual association flagged,",
                 " in practice the destroyed control; ba_guarded: balanced accuracy once flags on "
                 "damaged models are read as damage;",
                 " n below 40 (35 for js_to_original, which excludes M0): some verdicts undefined)")
        cols = ["audit", "metric", "probe_set", "kind", "n", "tp", "fn", "tn", "fp_retrain",
                "fp_other", "n_damaged", "balanced_accuracy", "ba_guarded"]
        print(cv[[k for k in cols if k in cv]].round(2).to_string(index=False))
