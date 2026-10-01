"""Stage 8: every analysis table from Stage 7's calibration tables, and the printed report.

Input is ``results/calibration/`` only -- `flags` (per unlearned model and metric: value, G,
verdicts, guard), `validity` (which cells are marked, M0's gap), `bands` (the retrain means) and
`canary` (the ground truth). Nothing is read from checkpoints, caches or by hand, so every number
traces to a record shard (implementation plan §8, gate), and the analysis runs anywhere the few
MB of calibration tables are: Kaggle, or a laptop.

The report is ordered by the research questions and hypotheses of the master reference §7-8.
"""

from __future__ import annotations

from pathlib import Path

from .agreement import (H3_METRICS, H3_PAIRS, PRIMARY, AnalysisConfig,
                        agreement_by_condition, canary_continuous, h3_controlled,
                        correlation_matrix, correlations, difficulty_spread, disagreement,
                        instance_tables, method_summary, patterns, privacy_contrast, rank_tables,
                        rq4_by_method, signed_gaps)
from .mixed_effects import fit_main, interaction_test

__all__ = ["analyse", "write_tables", "print_report"]


def analyse(cal_dir, config: AnalysisConfig | None = None) -> dict:
    import pandas as pd

    config = config or AnalysisConfig()
    cal = Path(cal_dir)

    def read(name):
        p = cal / f"{name}.parquet"
        return pd.read_parquet(p) if p.is_file() else None

    flags, validity = read("flags"), read("validity")
    bands, canary = read("bands"), read("canary")
    G, V, excluded = instance_tables(flags, validity)
    S = signed_gaps(flags, validity, bands, G.index) if bands is not None else None

    summary = method_summary(flags, G, V, S)
    corr = correlations(G, config)
    have_h3 = S is not None and all(m in S for m in H3_METRICS.values())
    h3_metrics = {**PRIMARY, **H3_METRICS}
    h3 = (correlations(S, config, pairs=H3_PAIRS, metrics=h3_metrics, salt=100)
          if have_h3 else pd.DataFrame())
    h3_abs = (correlations(G, config, pairs=H3_PAIRS, metrics=h3_metrics, salt=100)
              if have_h3 else pd.DataFrame())
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
        "correlations": corr,
        "h3": h3,
        "h3_absolute": h3_abs,
        "h3_controlled": h3_controlled(S) if have_h3 else pd.DataFrame(),
        "correlation_matrix": correlation_matrix(G),
        "agreement_by_condition": agreement_by_condition(corr),
        "disagreement": disagreement(V),
        "patterns": patterns(G, V),
        "rq4_by_method": rq4_by_method(G),
        "privacy_contrast": privacy_contrast(G, V, config),
        "canary_continuous": canary_continuous(G, canary, config),
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


def _per_family(s, value: str, digits: int = 2):
    for fam, metric in PRIMARY.items():
        part = s[s["metric"] == metric]
        if len(part):
            print(f"{fam} ({metric})")
            print(part.pivot_table(index="method", columns="forget_id", values=value)
                  .round(digits).to_string(), "\n")


def print_report(t: dict) -> None:
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 40)

    inv = t["inventory"].iloc[0]
    print("STAGE 8 -- AGREEMENT ANALYSIS, on the normalized oracle gap G (0 = like a retrain, "
          "1 = like M0, above 1 = farther from a retrain than M0 is)\n")
    for k, v in inv.items():
        print(f"  {k}: {v}")
    print()

    s = t["method_summary"]
    # ------------------------------------------------------------------ RQ1
    _h("RQ1 -- how close to a retrain? (1) calibrated verdicts: share of healthy unlearned "
       "models each primary audit flags as unlike every retrain")
    fr = s[s["metric"].isin(PRIMARY.values())].groupby(["metric", "forget_id"]).apply(
        lambda d: np.average(d["flag_rate"], weights=d["n"]) if d["flag_rate"].notna().any()
        else np.nan, include_groups=False).unstack()
    print(fr.round(2).to_string(), "\n")
    _h("(2) share of models farther from a retrain than the original model is (G > 1)",
       "(above 0.5: unlearning moved most models AWAY from retraining, past where M0 already "
       "was)")
    beyond = s[s["metric"].isin(PRIMARY.values())].groupby(["metric", "forget_id"]).apply(
        lambda d: np.average(d["share_beyond_m0"], weights=d["n"])
        if d["share_beyond_m0"].notna().any() else np.nan, include_groups=False).unstack()
    print(beyond.round(2).to_string(), "\n")
    _h("(3) median G per method and condition (medians: G's tail is the denominator)",
       "(blank: the cell is marked, so G is undefined there)")
    _per_family(s, "G_median")
    if "signed_median" in s and s["signed_median"].notna().any():
        _h("(4) median SIGNED G: negative = pushed to the side opposite M0, somewhere neither "
           "the retrains nor M0 are")
        _per_family(s, "signed_median")

    # ------------------------------------------------------------------ RQ2-RQ5 correlations
    c = t["correlations"]
    _h("RQ2-RQ5 -- the registered comparisons (master reference §15.2): instance-level rank "
       "correlation of G",
       "(within = n-weighted mean of the per-condition values. Pooling mixes conditions whose "
       "G levels differ -- the denominator -- so",
       " within is the cleaner read of whether audits agree on the same models; pooled is "
       "reported as registered)")
    rows = []
    for pair in dict.fromkeys(c["pair"]):
        p = c[(c["pair"] == pair) & (c["scope"] == "pooled")]
        w = c[(c["pair"] == pair) & (c["scope"] == "within")]
        if not len(p):
            continue
        p, w = p.iloc[0], (w.iloc[0] if len(w) else None)
        rows.append({"pair": pair, "n": p["n"],
                     "tau pooled [95% CI]": _ci(p.get("tau", np.nan), p.get("tau_lo"),
                                                p.get("tau_hi")),
                     "rho pooled [95% CI]": _ci(p.get("rho", np.nan), p.get("rho_lo"),
                                                p.get("rho_hi")),
                     "tau within": round(float(w["tau"]), 2) if w is not None and "tau" in w
                     and w["tau"] == w["tau"] else np.nan})
    print(pd.DataFrame(rows).to_string(index=False), "\n")
    per = c[~c["scope"].isin(["pooled", "within"])].copy()
    if len(per):
        per["tau [95% CI]"] = [_ci(a, lo, hi) for a, lo, hi in
                               zip(per.get("tau"), per.get("tau_lo"), per.get("tau_hi"))]
        print("tau per condition, with its interval (n ~25 each):")
        print(per.pivot_table(index="pair", columns="scope", values="tau [95% CI]",
                              aggfunc="first").to_string(), "\n")

    # ------------------------------------------------------------------ H3
    h3 = t.get("h3")
    if h3 is not None and len(h3):
        _h("H3 -- do models whose representations stay closer to M0 relearn more like M0?",
           "(SIGNED scale, the primary test: cka_to_original from a retrain's similarity to M0 (0) "
           "up to M0 (1), negative = farther from M0",
           " than a retrain; against signed relearning G. H3 predicts tau > 0. The absolute-G "
           "version, a sensitivity check, is beside it)")
        h = h3.copy()
        h["tau [95% CI]"] = [_ci(a, lo, hi) for a, lo, hi in
                             zip(h.get("tau"), h.get("tau_lo"), h.get("tau_hi"))]
        ha = t.get("h3_absolute")
        if ha is not None and len(ha):
            h = h.merge(ha[["scope", "tau"]].rename(columns={"tau": "tau |G|"}), on="scope",
                        how="left")
        cols = [k for k in ("scope", "n", "tau [95% CI]", "tau |G|") if k in h]
        print(h[cols].round(2).to_string(index=False), "\n")
        hc = t.get("h3_controlled")
        if hc is not None and len(hc):
            r = hc.iloc[0]
            print(f"beyond method identity: across the 5 seeds of each method x condition, mean "
                  f"tau {r['cell_mean_tau']:+.2f} ({int(r['cells_positive'])}/{int(r['cells'])} "
                  f"cells positive);")
            print(f"relearning ~ representation-to-M0 + method + condition + (1 | seed): slope "
                  f"{r['slope']:+.3f} [{r['slope_lo']:+.3f}, {r['slope_hi']:+.3f}], "
                  f"p={r['slope_p']:.2g}\n")

    # ------------------------------------------------------------------ H2b
    pc = t["privacy_contrast"]
    if len(pc):
        _h("H2b -- is the population attack more lenient than RMIA?",
           "(verdicts: of the models the two attacks disagree on, how many does the population "
           "attack pass -- exact McNemar.",
           " G: mean of G_pop - G_rmia, negative = population attack sees the model nearer a "
           "retrain)")
        cols = [k for k in ("scope", "n", "pop_passes_rmia_flags", "pop_flags_rmia_passes",
                            "mcnemar_p", "flag_rate_pop", "flag_rate_rmia", "mean_diff", "lo",
                            "hi") if k in pc]
        print(pc[cols].round(3).to_string(index=False), "\n")

    # ------------------------------------------------------------------ RQ3 / H1 / RQ4
    pt = t["patterns"]
    if len(pt):
        _h("RQ3, H1, RQ4 -- the patterns the hypotheses name, counted over instances",
           "(RQ3: behaviour G < 0.5 yet representation G > 0.5. H1: passes behaviour, flagged by "
           "representation.",
           " RQ4: passes behaviour and both privacy audits, flagged by relearning; rq4g: RMIA "
           "G < 0.5 yet relearning G > 0.5)")
        print(pt.to_string(index=False), "\n")
    rq = t.get("rq4_by_method")
    if rq is not None and len(rq):
        print("rq4g by method (RMIA calls it retrain-like, relearning calls it M0-like), all "
              "conditions:")
        print(rq.groupby("method")[["rmia_retrain_like", "relearning_M0_like"]].sum()
              .to_string(), "\n")

    # ------------------------------------------------------------------ RQ6, by degree
    cc = t.get("canary_continuous")
    if cc is not None and len(cc):
        _h("RQ6 by degree -- does each audit's G rise with how much canary association a model "
           "kept (canary_top_wrong)?",
           "(Kendall tau over the healthy canary models: 5 methods x 5 seeds, so mostly an "
           "ordering of methods. Negative = the audit ranks models backwards)")
        print(cc.round(3).to_string(index=False), "\n")

    # ------------------------------------------------------------------ H4
    dif = t["difficulty"]
    if len(dif):
        _h("H4 -- do methods differ more where memorization is high? SD of the method means of "
           "log G per condition",
           "(denominator-free; H4 predicts largest at mem-high, smallest at mem-low, the "
           "designed negative control)")
        print(dif.pivot_table(index="family", columns="forget_id", values="method_sd_log")
              .round(2).to_string(), "\n")
        inter = t["interaction"]
        if len(inter) and "lr" in inter:
            print("method x condition on the difficulty axis (mem-low/med/high-3000 vs rand-3000), "
                  "likelihood-ratio test on log G:")
            cols = [k for k in ("metric", "n", "conditions", "lr", "df", "p") if k in inter]
            print(inter[cols].round(4).to_string(index=False), "\n")
    ag = t.get("agreement_by_condition")
    if ag is not None and len(ag):
        _h("H4, second half -- how much do the audit families agree at each condition? Mean "
           "per-condition tau over the registered pairs")
        print(ag.round(2).to_string(index=False), "\n")

    # ------------------------------------------------------------------ confirmatory
    me = t["mixed_effects"]
    if len(me):
        _h("CONFIRMATORY -- log G ~ method + condition + (1 | seed), per primary metric",
           f"(reference: method {me['method_ref'].dropna().iloc[0]}, condition "
           f"{me['condition_ref'].dropna().iloc[0]}. Method ratio = exp(coef): how many times "
           "farther from the retrains",
           " than fine-tune, at every condition alike. Condition terms carry M0's gap and are "
           "not difficulty effects)")
        for metric, part in me.groupby("metric", sort=False):
            r0 = part.iloc[0]
            if r0["term"] == "no finite maximum":
                print(f"{metric}  n={r0['n']}  NO FIT: no optimizer found a finite maximum\n")
                continue
            print(f"{metric}  n={r0['n']}  seed variance={r0['seed_var']:.4f}  "
                  f"optimizer={r0['optimizer']}" + ("" if r0["converged"] else "  NOT CONVERGED"))
            for r in part.itertuples():
                if r.term.startswith("method"):
                    print(f"    {r.term:24s} x{np.exp(r.coef):.2f}  "
                          f"[x{np.exp(r.lo):.2f}, x{np.exp(r.hi):.2f}]  p={r.p:.2g}")
            print()

    # ------------------------------------------------------------------ verdict disagreement
    d = t["disagreement"]
    if len(d):
        _h("Pass/fail disagreement (§15.3) -- share of instances where two audits' calibrated "
           "verdicts contradict",
           "(marked cells not scored. Read with RQ1's flag rates: audits that flag everything "
           "agree by saturation)")
        reg = d[d["registered"]]
        tab = reg.pivot_table(index="pair", columns="scope", values="disagree").round(2)
        tab = tab[["pooled"] + [x for x in tab.columns if x != "pooled"]]
        print(tab.to_string(), "\n")

    # ------------------------------------------------------------------ descriptive
    rk = t["rank_tables"]
    if len(rk):
        _h("RANK TABLES -- DESCRIPTIVE ONLY (§15.1): methods ranked by median G per condition, "
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
