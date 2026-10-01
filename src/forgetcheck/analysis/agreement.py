"""Stage 8: do the audits agree, model by model? (master reference §15, implementation plan §8)

The unit of analysis is the model instance -- one unlearned model at one condition -- never the
method: Kendall's tau over six method ranks cannot reach p < 0.05 for most outcomes, while ~200
instances can (configs/audits.yaml, `agreement`). Every comparison is on the registered common
scale, the normalized oracle gap ``G = |m(Mu) - mean m(Mr)| / |m(M0) - mean m(Mr)|`` (plan §4.4):
0 is retrain-like, 1 is M0-like, and the calibration already left it undefined in every cell the
safeguards mark, so a ceiling or an indistinguishable M0 never enters an agreement statistic.

**Who is in.** Unlearned models only, minus the damaged ones (the utility guard: all 40 `neggrad`
models and three at mem-low) -- a destroyed network is unlike a retrain on every audit, and would
manufacture agreement out of damage. Minus retired metrics (`activation_l2`).

**Pooled, per condition, and within.** The registered statistic pools every instance. Pooling
across conditions can manufacture agreement nobody measured -- if every metric happens to sit
higher at one condition than another, the pooled correlation picks up the difference between
conditions, not agreement between audits on the same models. So each correlation is also
computed per condition and as the n-weighted mean of those (`within`); where pooled exceeds
within, the excess is the conditions talking, not the audits.

**G above 1, and G's tail.** G > 1 is a result, by the plan's own definition: the method moved
the model *farther* from a retrain than the original model was. Its magnitude, though, is set by
the denominator. Where M0 is statistically separable from the retrains but only just -- JS at
mem-low (0.0002), CKA at the random sets (0.002-0.007) -- G reaches 30 to 150. So magnitudes are
summarised by medians and by the share beyond M0 (G > 1), never by means; rank statistics are
untouched by it; and the confirmatory models are fitted on log G, where the denominator is a
per-condition constant that the condition term absorbs exactly (`mixed_effects`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "AnalysisConfig", "PRIMARY", "REGISTERED_PAIRS", "instance_tables", "kendall_tau_b",
    "correlations", "correlation_matrix", "disagreement", "patterns", "privacy_contrast",
    "method_summary", "rank_tables", "difficulty_spread", "signed_gaps", "canary_continuous",
    "rq4_by_method", "agreement_by_condition", "H3_METRICS", "H3_PAIRS",
]

#: One metric per audit family for the registered comparisons, fixed before any Stage 8 result.
#: Privacy is compared AUC against AUC so the two attacks are measured like for like; the TPR
#: variants (RMIA's is the registry's "headline") are in the full matrix. cka_linear is the
#: representation audit's declared primary; relearn_auc is the only reversibility metric defined
#: at every condition. SDE's family is registered as unranked, so it is secondary throughout.
PRIMARY = {
    "behavior": "js_to_oracle",
    "privacy_weak": "mia_auc_pop",
    "privacy_strong": "mia_auc_rmia",
    "representation": "cka_linear",
    "reversibility": "relearn_auc",
}

#: Master reference §15.2: behaviour vs privacy, behaviour vs representation, privacy vs
#: relearning, representation vs relearning, and within privacy population vs per-example.
REGISTERED_PAIRS = (
    ("behavior", "privacy_weak"), ("behavior", "privacy_strong"),
    ("behavior", "representation"),
    ("privacy_weak", "reversibility"), ("privacy_strong", "reversibility"),
    ("representation", "reversibility"),
    ("privacy_weak", "privacy_strong"),
)

#: H3, added with its metric on 1 Oct 2026: "models retaining representations closer to the
#: original may exhibit faster relearning". G cannot test it -- it measures distance from the
#: retrains, not proximity to M0 -- so it gets the representation-to-M0 measure, against the
#: reversibility primary. H3 predicts a positive tau.
H3_METRICS = {"representation_to_original": "cka_to_original"}
H3_PAIRS = (("representation_to_original", "reversibility"),)

#: Each metric enters on one probe: the forget set, or layer4 for representation.
PROBES = ("forget", "layer4")
#: A verdict, not a measurement: no band, no G. Scored in Stage 7's native validity instead.
NOT_ANALYSED = frozenset({"sde_verdict"})
KEYS = ["run_id", "forget_id", "method", "train_seed"]


@dataclass(frozen=True)
class AnalysisConfig:
    resamples: int = 10_000
    ci: float = 0.95
    seed: int = 0

    @classmethod
    def from_context(cls, ctx) -> "AnalysisConfig":
        a = ctx.audits.get("agreement", {})
        seeds = getattr(ctx, "seeds", {}) or {}
        return cls(resamples=int(a.get("bootstrap_resamples", 10_000)),
                   ci=float(a.get("bootstrap_ci", 0.95)), seed=int(seeds.get("audit", 0)))


# --------------------------------------------------------------------------- instances


def instance_tables(flags, validity):
    """Wide tables, one row per analysed instance: ``G`` (oracle gap) and ``V`` (calibrated
    verdict, None where the cell is marked or the verdict undefined), plus who was left out."""
    f = flags[flags["probe_set"].isin(PROBES) & ~flags["metric"].isin(NOT_ANALYSED)].copy()
    if "retired" in f:
        f = f[~f["retired"].astype(bool)]
    excluded = {"damaged models": int(f.loc[f["damaged"].fillna(False).astype(bool),
                                             "run_id"].nunique())}
    f = f[~f["damaged"].fillna(False).astype(bool)]

    marks = validity[["forget_id", "audit", "metric", "probe_set"]].copy()
    marked = np.zeros(len(validity), dtype=bool)
    for col in ("low_discriminability", "below_min_effect"):
        if col in validity:
            marked |= validity[col].fillna(False).astype(bool).to_numpy()
    marks["marked"] = marked
    f = f.merge(marks, on=["forget_id", "audit", "metric", "probe_set"], how="left")
    f["marked"] = f["marked"].fillna(False).astype(bool)
    f["verdict"] = [None if (m or v is None or v != v) else bool(v)
                    for m, v in zip(f["marked"], f["calibrated_flag"])]

    # An exact reshape, not pivot_table: that aggregates, and silently drops a metric whose
    # verdicts are all undefined. Each (instance, metric) is one row once the probe is fixed.
    idx = f.set_index(KEYS + ["metric"])
    G = idx["oracle_gap"].astype(float).unstack("metric")
    V = idx["verdict"].astype(object).unstack("metric").reindex(index=G.index, columns=G.columns)
    G.columns.name = V.columns.name = None
    excluded["marked cells (metric x condition)"] = int(
        f.loc[f["marked"], ["forget_id", "metric"]].drop_duplicates().shape[0])
    return G, V, excluded


# --------------------------------------------------------------------------- rank statistics


def kendall_tau_b(x, y) -> float:
    """Kendall's tau-b (ties corrected), exact, O(n^2). Equals scipy's on any input; written out
    so the bootstrap can run it on stacked resamples without 10^4 Python-level calls."""
    return float(_tau_b(np.asarray(x, float)[None], np.asarray(y, float)[None])[0])


def _tau_b(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise tau-b for stacked samples ``X, Y`` of shape ``(b, n)``."""
    n = X.shape[1]
    iu = np.triu_indices(n, 1)
    sx = np.sign(X[:, :, None] - X[:, None, :])[:, iu[0], iu[1]]
    sy = np.sign(Y[:, :, None] - Y[:, None, :])[:, iu[0], iu[1]]
    num = (sx * sy).sum(axis=1)
    den = np.sqrt((sx != 0).sum(axis=1) * (sy != 0).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    rx, ry = rankdata(X, axis=1), rankdata(Y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(axis=1) / den, np.nan)


def _bootstrap(x, y, config: AnalysisConfig, *, salt: int) -> dict:
    """Percentile intervals for tau-b and Spearman, resampling instances. Seeded per pair so
    adding a pair never changes another's interval."""
    n = len(x)
    rng = np.random.default_rng([config.seed, salt])
    lo_q, hi_q = (1 - config.ci) / 2, 1 - (1 - config.ci) / 2
    taus, rhos = [], []
    chunk = max(1, min(config.resamples, 4_000_000 // max(1, n * n)))
    for start in range(0, config.resamples, chunk):
        b = min(chunk, config.resamples - start)
        idx = rng.integers(0, n, size=(b, n))
        X, Y = x[idx], y[idx]
        taus.append(_tau_b(X, Y))
        rhos.append(_spearman(X, Y))
    t, r = np.concatenate(taus), np.concatenate(rhos)
    return {"tau_lo": float(np.nanquantile(t, lo_q)), "tau_hi": float(np.nanquantile(t, hi_q)),
            "rho_lo": float(np.nanquantile(r, lo_q)), "rho_hi": float(np.nanquantile(r, hi_q))}


def _pair(G, a: str, b: str):
    if a not in G or b not in G:
        return np.array([]), np.array([]), G.iloc[0:0]
    m = G[a].notna() & G[b].notna()
    return G.loc[m, a].to_numpy(), G.loc[m, b].to_numpy(), G[m]


def correlations(G, config: AnalysisConfig, pairs=REGISTERED_PAIRS, metrics=None,
                 salt: int = 0) -> "pd.DataFrame":
    """The registered comparisons (§15.2): tau-b and Spearman with bootstrap intervals, pooled
    and per condition, plus the n-weighted within-condition mean. ``metrics`` maps the labels
    in ``pairs`` to metric names (default: the family primaries)."""
    import pandas as pd

    metrics = metrics or PRIMARY
    rows = []
    for k, (fa, fb) in enumerate(pairs, start=salt):
        a, b = metrics[fa], metrics[fb]
        x, y, sub = _pair(G, a, b)
        per = []
        conds = sorted(sub.index.get_level_values("forget_id").unique())
        for j, cond in enumerate(conds):
            sel = sub.index.get_level_values("forget_id") == cond
            xc, yc = x[sel], y[sel]
            row = {"pair": f"{fa} vs {fb}", "metric_a": a, "metric_b": b, "scope": cond,
                   "n": int(len(xc))}
            if len(xc) >= 5:
                row.update(tau=kendall_tau_b(xc, yc), rho=float(_spearman(xc[None], yc[None])[0]),
                           **_bootstrap(xc, yc, config, salt=1000 * (k + 1) + j))
            per.append(row)
        pooled = {"pair": f"{fa} vs {fb}", "metric_a": a, "metric_b": b, "scope": "pooled",
                  "n": int(len(x))}
        if len(x) >= 5:
            pooled.update(tau=kendall_tau_b(x, y), rho=float(_spearman(x[None], y[None])[0]),
                          **_bootstrap(x, y, config, salt=k + 1))
        scored = [r for r in per if "tau" in r and r["tau"] == r["tau"]]
        within = {"pair": f"{fa} vs {fb}", "metric_a": a, "metric_b": b, "scope": "within",
                  "n": int(sum(r["n"] for r in scored))}
        if scored:
            w = np.array([r["n"] for r in scored], float)
            within["tau"] = float(np.average([r["tau"] for r in scored], weights=w))
            within["rho"] = float(np.average([r["rho"] for r in scored], weights=w))
        rows += [pooled, within] + per
    return pd.DataFrame(rows)


def correlation_matrix(G) -> "pd.DataFrame":
    """Every analysed metric against every other: pooled tau-b, and the n-weighted mean of the
    per-condition values (``tau_within``), with n. Descriptive: the registered comparisons are
    the primary pairs above; this is the full picture around them."""
    import pandas as pd

    rows = []
    for a, b in combinations(sorted(G.columns), 2):
        x, y, sub = _pair(G, a, b)
        conds = sub.index.get_level_values("forget_id")
        per = [(int((conds == c).sum()), kendall_tau_b(x[conds == c], y[conds == c]))
               for c in sorted(set(conds)) if (conds == c).sum() >= 5]
        per = [(n, t) for n, t in per if t == t]
        rows.append({"metric_a": a, "metric_b": b, "n": int(len(x)),
                     "tau": kendall_tau_b(x, y) if len(x) >= 5 else float("nan"),
                     "tau_within": (float(np.average([t for _, t in per],
                                                     weights=[n for n, _ in per]))
                                    if per else float("nan"))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- verdicts


def disagreement(V) -> "pd.DataFrame":
    """§15.3: the share of instances on which two audits' calibrated verdicts contradict --
    one flags the model as unlike a retrain, the other passes it. Marked cells and undefined
    verdicts are not scored. Each audit's own flag rate is reported beside the rate: at a
    condition where both flag everything, zero disagreement is agreement by saturation."""
    import pandas as pd

    rows = []
    for fa, fb in combinations(PRIMARY, 2):
        a, b = PRIMARY[fa], PRIMARY[fb]
        if a not in V or b not in V:
            continue
        both = V[[a, b]].dropna()
        for scope, part in [("pooled", both)] + [
                (c, both[both.index.get_level_values("forget_id") == c])
                for c in sorted(both.index.get_level_values("forget_id").unique())]:
            va, vb = part[a].astype(bool), part[b].astype(bool)
            rows.append({"pair": f"{fa} vs {fb}", "registered": (fa, fb) in REGISTERED_PAIRS,
                         "scope": scope, "n": int(len(part)),
                         "disagree": float((va != vb).mean()) if len(part) else float("nan"),
                         "flag_rate_a": float(va.mean()) if len(part) else float("nan"),
                         "flag_rate_b": float(vb.mean()) if len(part) else float("nan")})
    return pd.DataFrame(rows)


def patterns(G, V) -> "pd.DataFrame":
    """RQ3, H1 and RQ4 as counts over instances, per condition and pooled.

    * RQ3 -- behaviourally closer to a retrain than to M0 (G < 0.5) yet representationally
      closer to M0 (G > 0.5).
    * H1 -- passes the behavioural audit, flagged by the representation audit.
    * RQ4 -- passes behaviour and both privacy audits, flagged by relearning.
    """
    import pandas as pd

    beh, rep, rel = PRIMARY["behavior"], PRIMARY["representation"], PRIMARY["reversibility"]
    pw, ps = PRIMARY["privacy_weak"], PRIMARY["privacy_strong"]
    rows = []
    conds = sorted(G.index.get_level_values("forget_id").unique())
    for scope in ["pooled"] + conds:
        g = G if scope == "pooled" else G[G.index.get_level_values("forget_id") == scope]
        v = V.reindex(g.index)
        row = {"scope": scope, "n_instances": int(len(g))}
        if beh in g and rep in g:
            base = g[beh].notna() & g[rep].notna()
            closer = base & (g[beh] < 0.5)
            row["rq3_behaviour_retrain_like"] = int(closer.sum())
            row["rq3_of_which_representation_M0_like"] = int((closer & (g[rep] > 0.5)).sum())
        if beh in v and rep in v:
            ok = v[beh].notna() & v[rep].notna()
            passed = ok & (v[beh] == False)  # noqa: E712 -- None/True/False column
            row["h1_pass_behaviour"] = int(passed.sum())
            row["h1_of_which_flagged_by_representation"] = int((passed & (v[rep] == True)).sum())
        if all(c in v for c in (beh, pw, ps, rel)):
            ok = v[[beh, pw, ps, rel]].notna().all(axis=1)
            clean = ok & (v[beh] == False) & (v[pw] == False) & (v[ps] == False)  # noqa: E712
            row["rq4_pass_behaviour_and_privacy"] = int(clean.sum())
            row["rq4_of_which_flagged_by_relearning"] = int((clean & (v[rel] == True)).sum())
        if ps in g and rel in g:
            # The verdict form above is empty whenever behaviour flags everything. On G: the
            # strong privacy attack calls the model retrain-like, relearning calls it M0-like.
            both = g[ps].notna() & g[rel].notna()
            private = both & (g[ps] < 0.5)
            row["rq4g_rmia_retrain_like"] = int(private.sum())
            row["rq4g_of_which_relearning_M0_like"] = int((private & (g[rel] > 0.5)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def rq4_by_method(G) -> "pd.DataFrame":
    """RQ4 on G, by method and condition: models the strong privacy attack calls retrain-like
    (G < 0.5) that relearning calls M0-like (G > 0.5)."""
    import pandas as pd

    ps, rel = PRIMARY["privacy_strong"], PRIMARY["reversibility"]
    if ps not in G or rel not in G:
        return pd.DataFrame()
    g = G[[ps, rel]].dropna()
    private = g[ps] < 0.5
    hit = private & (g[rel] > 0.5)
    out = pd.DataFrame({"rmia_retrain_like": private, "relearning_M0_like": hit})
    return (out.groupby(level=["forget_id", "method"]).sum().astype(int).reset_index())


def privacy_contrast(G, V, config: AnalysisConfig) -> "pd.DataFrame":
    """H2b: does the population attack call models more private than the per-example one?

    Paired over instances: ``d = G_pop - G_rmia``. Negative means the population attack sees
    the model as closer to a retrain. Mean with a bootstrap interval, the share of instances
    where the population attack is the more lenient, a Wilcoxon signed-rank p, and each attack's
    calibrated flag rate on the same instances.
    """
    import pandas as pd
    from scipy.stats import binomtest, wilcoxon

    pw, ps = PRIMARY["privacy_weak"], PRIMARY["privacy_strong"]
    rows = []
    if pw not in G or ps not in G:
        return pd.DataFrame(rows)
    both = G[[pw, ps]].dropna()
    conds = sorted(both.index.get_level_values("forget_id").unique())
    for j, scope in enumerate(["pooled"] + conds):
        part = both if scope == "pooled" else both[both.index.get_level_values("forget_id") == scope]
        d = (part[pw] - part[ps]).to_numpy()
        row = {"scope": scope, "n": int(len(d))}
        if len(d) >= 5:
            rng = np.random.default_rng([config.seed, 7, j])
            boots = d[rng.integers(0, len(d), size=(config.resamples, len(d)))].mean(axis=1)
            q = (1 - config.ci) / 2
            row.update(mean_diff=float(d.mean()), lo=float(np.quantile(boots, q)),
                       hi=float(np.quantile(boots, 1 - q)),
                       share_pop_more_lenient=float((d < 0).mean()),
                       wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        vv = V.reindex(part.index)[[pw, ps]].dropna()
        row["flag_rate_pop"] = float(vv[pw].astype(bool).mean()) if len(vv) else float("nan")
        row["flag_rate_rmia"] = float(vv[ps].astype(bool).mean()) if len(vv) else float("nan")
        # The verdict form of H2b, which is what "ranks methods as more private" amounts to
        # once a band decides pass/fail: of the models the attacks disagree on, does the
        # population attack pass more of them? Exact McNemar on the discordant pairs.
        lenient = int(((vv[pw] == False) & (vv[ps] == True)).sum())  # noqa: E712
        strict = int(((vv[pw] == True) & (vv[ps] == False)).sum())  # noqa: E712
        row["pop_passes_rmia_flags"], row["pop_flags_rmia_passes"] = lenient, strict
        row["mcnemar_p"] = (float(binomtest(lenient, lenient + strict, 0.5).pvalue)
                            if lenient + strict else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- per method


def signed_gaps(flags, validity, bands, index):
    """G with its sign: ``(m(Mu) - mean m(Mr)) / (m(M0) - mean m(Mr))``. Positive is on M0's
    side of the retrains (1 = at M0, above 1 = beyond it); negative is the opposite side --
    an unlearned model pushed somewhere neither the retrains nor M0 are. The registered G is
    its absolute value; the sign says which kind of "far" a large G is."""
    import pandas as pd

    key = ["forget_id", "audit", "metric", "probe_set"]
    f = flags[flags["probe_set"].isin(PROBES) & flags["oracle_gap"].notna()]
    f = f.merge(bands[key + ["mean"]].rename(columns={"mean": "band_mean"}), on=key, how="left")
    f = f.merge(validity[key + ["m0_gap_raw"]], on=key, how="left")
    f["signed"] = (f["value"] - f["band_mean"]) / f["m0_gap_raw"]
    S = f.set_index(KEYS + ["metric"])["signed"].astype(float).unstack("metric")
    S.columns.name = None
    return S.reindex(index=index)


def method_summary(flags, G, V, S=None) -> "pd.DataFrame":
    """RQ1 and §16.2's "mean and standard deviation for every main metric across seeds":
    per condition, method and metric -- the raw value, G (median, and the share beyond M0),
    the signed G's median, and the share flagged. Medians, because G's tail is the denominator."""
    import pandas as pd

    raw = flags[flags["probe_set"].isin(PROBES)].set_index(KEYS + ["metric"])["value"]
    rows = []
    for metric in G.columns:
        g, v = G[metric], V[metric]
        r = raw.xs(metric, level="metric").reindex(G.index)
        s = S[metric] if S is not None and metric in S else pd.Series(float("nan"), index=G.index)
        df = pd.DataFrame({"G": g, "S": s, "flag": v.map({True: 1.0, False: 0.0}), "value": r})
        for (cond, method), part in df.groupby(level=["forget_id", "method"]):
            gg = part["G"].dropna()
            rows.append({"forget_id": cond, "method": method, "metric": metric,
                         "n": int(part["value"].notna().sum()),
                         "value_mean": part["value"].mean(), "value_sd": part["value"].std(),
                         "G_median": gg.median() if len(gg) else float("nan"),
                         "G_mean": part["G"].mean(), "G_sd": part["G"].std(),
                         "share_beyond_m0": float((gg > 1).mean()) if len(gg) else float("nan"),
                         "signed_median": part["S"].median(),
                         "flag_rate": part["flag"].mean()})
    return pd.DataFrame(rows)


def rank_tables(summary) -> "pd.DataFrame":
    """§15.1, DESCRIPTIVE ONLY: methods ranked per condition by median G on each family's
    primary metric, 1 = most retrain-like. Never an inferential statistic (master reference
    §16.2)."""
    import pandas as pd

    rows = []
    for fam, metric in PRIMARY.items():
        part = summary[summary["metric"] == metric]
        for cond, c in part.groupby("forget_id"):
            c = c.dropna(subset=["G_median"])
            ranks = c["G_median"].rank(method="min")
            for (_, r), k in zip(c.iterrows(), ranks):
                rows.append({"family": fam, "metric": metric, "forget_id": cond,
                             "method": r["method"], "G_median": r["G_median"], "rank": int(k)})
    return pd.DataFrame(rows)


def difficulty_spread(G) -> "pd.DataFrame":
    """H4, descriptively: how far apart the methods are at each condition, per primary metric.

    On log G, where it is free of the denominator: within one condition M0's gap is a constant
    shift of every model's log G, so the SD of the method means compares like with like across
    conditions. (eta^2, the share of variance the method explains, is kept but saturates at
    0.75-0.99 everywhere -- seeds agree so closely that it cannot separate conditions.)
    H4 predicts the spread largest at mem-high and smallest at mem-low."""
    import pandas as pd

    rows = []
    for fam, metric in PRIMARY.items():
        if metric not in G:
            continue
        logG = np.log(G[metric].where(G[metric] > 0))
        for cond, part in logG.dropna().groupby(level="forget_id"):
            by = part.groupby(level="method")
            means = by.mean()
            ss_tot = float(((part - part.mean()) ** 2).sum())
            ss_between = float((by.size() * (means - part.mean()) ** 2).sum())
            rows.append({"family": fam, "metric": metric, "forget_id": cond, "n": int(len(part)),
                         "methods": int(len(means)), "method_sd_log": float(means.std()),
                         "eta_sq": ss_between / ss_tot if ss_tot > 0 else float("nan")})
    return pd.DataFrame(rows)


def agreement_by_condition(corr) -> "pd.DataFrame":
    """H4's second half: how much do the audit families agree at each condition? The mean
    per-condition tau over the registered pairs, and how many of them are negative."""
    import pandas as pd

    per = corr[~corr["scope"].isin(["pooled", "within"])].dropna(subset=["tau"])
    rows = []
    for cond, c in per.groupby("scope"):
        rows.append({"forget_id": cond, "pairs": int(len(c)), "mean_tau": float(c["tau"].mean()),
                     "negative_pairs": int((c["tau"] < 0).sum()),
                     "ci_excludes_zero": int(((c["tau_lo"] > 0) | (c["tau_hi"] < 0)).sum())})
    return pd.DataFrame(rows)


def canary_continuous(G, canary, config: AnalysisConfig | None = None) -> "pd.DataFrame":
    """RQ6 by degree, not verdict: does each audit's G order the canary models by how much of
    the canary association they actually kept?

    Stage 7 scored verdicts against the ground truth, and every healthy canary model retains
    (all positives) -- a verdict cannot rank them. ``canary_top_wrong`` measures the amount, so
    an audit that measures retention should rise with it: Kendall's tau over the healthy canary
    models. The 25 are five methods by five seeds, so the ordering is mostly between methods.
    """
    import pandas as pd
    from scipy.stats import kendalltau

    if canary is None or not len(canary) or "canary_top_wrong" not in canary:
        return pd.DataFrame()
    truth = canary.set_index("run_id")["canary_top_wrong"]
    conds = set(G.index.get_level_values("forget_id"))
    cond = next((c for c in conds if c.startswith("canary")), None)
    if cond is None:
        return pd.DataFrame()
    gc = G.xs(cond, level="forget_id")
    rows = []
    for metric in gc.columns:
        s = gc[metric].dropna()
        t = np.asarray(s.index.get_level_values("run_id").map(truth), float)
        ok = np.isfinite(t)
        if ok.sum() >= 10 and np.std(s.to_numpy()[ok]) > 0:
            x, y = s.to_numpy()[ok], t[ok]
            r = kendalltau(x, y)
            row = {"metric": metric, "n": int(ok.sum()), "tau": float(r.statistic),
                   "p": float(r.pvalue)}
            if config is not None:
                row.update({k: v for k, v in _bootstrap(
                    x, y, config, salt=500 + sorted(gc.columns).index(metric)).items()
                    if k.startswith("tau")})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("tau", ascending=False, ignore_index=True)
