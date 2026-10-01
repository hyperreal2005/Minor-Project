"""Stage 8: every analysis table from Stage 7's calibration tables, and the printed report.

Input is ``results/calibration/`` only -- `flags` (per unlearned model and metric: value, G,
verdicts, guard) and `validity` (which cells are marked). Nothing is read from checkpoints,
caches or by hand, so every number traces to a record shard (implementation plan §8, gate), and
the analysis runs anywhere the ~2 MB of calibration tables are: Kaggle, or a laptop.
"""

from __future__ import annotations

from pathlib import Path

from .agreement import (PRIMARY, REGISTERED_PAIRS, AnalysisConfig, correlation_matrix,
                        correlations, difficulty_spread, disagreement, instance_tables,
                        method_summary, patterns, privacy_contrast, rank_tables)
from .mixed_effects import fit_main, interaction_test

__all__ = ["analyse", "write_tables", "print_report"]


def analyse(cal_dir, config: AnalysisConfig | None = None) -> dict:
    import pandas as pd

    config = config or AnalysisConfig()
    cal = Path(cal_dir)
    flags = pd.read_parquet(cal / "flags.parquet")
    validity = pd.read_parquet(cal / "validity.parquet")
    G, V, excluded = instance_tables(flags, validity)

    summary = method_summary(flags, G, V)
    mixed = [fit_main(G, PRIMARY[f]) for f in PRIMARY if PRIMARY[f] in G]
    inter = [interaction_test(G, PRIMARY[f]) for f in PRIMARY if PRIMARY[f] in G]
    retired = sorted(set(flags.loc[flags["retired"].astype(bool), "metric"])) \
        if "retired" in flags else []
    return {
        "inventory": pd.DataFrame([{
            "instances": int(len(G)),
            "methods": ", ".join(sorted(G.index.get_level_values("method").unique())),
            "conditions": int(G.index.get_level_values("forget_id").nunique()),
            "metrics": int(G.shape[1]),
            **{f"excluded: {k}": v for k, v in excluded.items()},
            "retired": ", ".join(retired),
        }]),
        "method_summary": summary,
        "correlations": correlations(G, config),
        "correlation_matrix": correlation_matrix(G),
        "disagreement": disagreement(V),
        "patterns": patterns(G, V),
        "privacy_contrast": privacy_contrast(G, V, config),
        "difficulty": difficulty_spread(G),
        "interaction": pd.DataFrame(inter),
        "mixed_effects": pd.concat([m for m in mixed if len(m)], ignore_index=True)
        if any(len(m) for m in mixed) else pd.DataFrame(),
        "rank_tables": rank_tables(summary),
    }


def write_tables(tables: dict, out_dir) -> list[Path]:
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


# --------------------------------------------------------------------------- the report


def _h(title: str, *notes: str) -> None:
    print(title)
    for n in notes:
        print(n)
    print()


def _ci(v, lo, hi) -> str:
    return "  --" if v != v else f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def print_report(t: dict) -> None:
    import pandas as pd

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 40)

    inv = t["inventory"].iloc[0]
    print("STAGE 8 -- AGREEMENT ANALYSIS, on the normalized oracle gap G (0 = like a retrain, "
          "1 = like M0)\n")
    for k, v in inv.items():
        print(f"  {k}: {v}")
    print()

    s = t["method_summary"]
    _h("RQ1 -- how close to a retrain? Mean G over seeds, per method and condition, on each "
       "family's primary metric", "(blank: the cell is marked -- M0 indistinguishable or at a "
       "ceiling -- so G is undefined there)")
    for fam, metric in PRIMARY.items():
        part = s[s["metric"] == metric]
        if len(part):
            print(f"{fam} ({metric})")
            print(part.pivot_table(index="method", columns="forget_id", values="G_mean")
                  .round(2).to_string(), "\n")

    c = t["correlations"]
    _h("RQ2-RQ5 -- the registered comparisons (master reference §15.2): instance-level rank "
       "correlation of G",
       "(pooled: every instance, as registered. within: n-weighted mean of the per-condition "
       "values -- where pooled",
       " exceeds within, the excess comes from differences between conditions, not agreement "
       "between audits on the same models)")
    rows = []
    for pair in dict.fromkeys(c["pair"]):
        p = c[(c["pair"] == pair) & (c["scope"] == "pooled")]
        w = c[(c["pair"] == pair) & (c["scope"] == "within")]
        if not len(p):
            continue
        p, w = p.iloc[0], w.iloc[0] if len(w) else None
        rows.append({"pair": pair, "n": p["n"],
                     "tau pooled [95% CI]": _ci(p.get("tau", float("nan")), p.get("tau_lo"),
                                                p.get("tau_hi")),
                     "rho pooled [95% CI]": _ci(p.get("rho", float("nan")), p.get("rho_lo"),
                                                p.get("rho_hi")),
                     "tau within": round(float(w["tau"]), 2) if w is not None and "tau" in w
                     and w["tau"] == w["tau"] else float("nan")})
    print(pd.DataFrame(rows).to_string(index=False), "\n")
    per = c[~c["scope"].isin(["pooled", "within"])]
    if len(per):
        print("tau per condition (n ~25 each; descriptive):")
        print(per.pivot_table(index="pair", columns="scope", values="tau").round(2).to_string(),
              "\n")

    pc = t["privacy_contrast"]
    if len(pc):
        _h("H2b -- does the population attack call models more private than RMIA?",
           "(mean of G_pop - G_rmia per instance: negative = population attack more lenient; "
           "flag rates on the same instances)")
        cols = [k for k in ("scope", "n", "mean_diff", "lo", "hi", "share_pop_more_lenient",
                            "wilcoxon_p", "flag_rate_pop", "flag_rate_rmia") if k in pc]
        print(pc[cols].round(3).to_string(index=False), "\n")

    d = t["disagreement"]
    if len(d):
        _h("Pass/fail disagreement (§15.3) -- share of instances where two audits' calibrated "
           "verdicts contradict",
           "(registered pairs; marked cells not scored. Read with the flag rates: two audits "
           "that flag everything agree by saturation)")
        reg = d[d["registered"]]
        tab = reg.pivot_table(index="pair", columns="scope", values="disagree").round(2)
        tab = tab[["pooled"] + [c for c in tab.columns if c != "pooled"]]
        print(tab.to_string(), "\n")
        print("flag rate of each primary audit, pooled:")
        fr = {}
        for r in reg[reg["scope"] == "pooled"].itertuples():
            a, b = r.pair.split(" vs ")
            fr[a], fr[b] = r.flag_rate_a, r.flag_rate_b
        print("  " + ", ".join(f"{k} {v:.2f}" for k, v in fr.items()), "\n")

    pt = t["patterns"]
    if len(pt):
        _h("RQ3, H1, RQ4 -- the patterns the hypotheses name, counted over instances",
           "(RQ3: behaviourally retrain-like (G < 0.5) yet representationally M0-like (G > 0.5). "
           "H1: passes behaviour, flagged by",
           " representation. RQ4: passes behaviour and both privacy audits, flagged by "
           "relearning)")
        print(pt.to_string(index=False), "\n")

    dif = t["difficulty"]
    if len(dif):
        _h("H4 -- do methods differ more where memorization is high? Share of G variance "
           "explained by method (eta^2), per condition",
           "(H4 predicts largest at mem-high, smallest at mem-low, the designed negative control)")
        print(dif.pivot_table(index="family", columns="forget_id", values="eta_sq").round(2)
              .to_string(), "\n")
        inter = t["interaction"]
        if len(inter) and "lr" in inter:
            print("method x condition interaction on the difficulty axis (likelihood-ratio test):")
            print(inter.round(4).to_string(index=False), "\n")

    me = t["mixed_effects"]
    if len(me):
        _h("CONFIRMATORY -- G ~ method + condition + (1 | seed), per primary metric",
           f"(reference: method {me['method_ref'].iloc[0]}, condition "
           f"{me['condition_ref'].iloc[0]}; coefficients in units of G)")
        for metric, part in me.groupby("metric", sort=False):
            r0 = part.iloc[0]
            if r0["term"] == "no finite maximum":
                print(f"{metric}  n={r0['n']}  NO FIT: no optimizer found a finite maximum\n")
                continue
            print(f"{metric}  n={r0['n']}  seed variance={r0['seed_var']:.4f}  "
                  f"optimizer={r0['optimizer']}"
                  + ("" if r0["converged"] else "  NOT CONVERGED")
                  + (f"  [{r0['warnings']}]" if r0["warnings"] else ""))
            for r in part.itertuples():
                print(f"    {r.term:28s} {r.coef:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]  p={r.p:.3g}")
            print()

    rk = t["rank_tables"]
    if len(rk):
        _h("RANK TABLES -- DESCRIPTIVE ONLY (§15.1): methods ranked by mean G per condition, "
           "1 = most retrain-like", "(never inferential: six ranked units cannot carry a test)")
        for fam, part in rk.groupby("family", sort=False):
            print(f"{fam} ({part['metric'].iloc[0]})")
            print(part.pivot_table(index="method", columns="forget_id", values="rank")
                  .astype("Int64").to_string(), "\n")

    m = t["correlation_matrix"]
    if len(m):
        _h("FULL MATRIX -- pooled tau between every pair of analysed metrics (descriptive)")
        full = pd.concat([m, m.rename(columns={"metric_a": "metric_b", "metric_b": "metric_a"})])
        print(full.pivot_table(index="metric_a", columns="metric_b", values="tau").round(2)
              .to_string())
