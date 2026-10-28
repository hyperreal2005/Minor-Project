"""What-ifs: change a rule of the analysis and watch the conclusions move."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as C
import data as D
import stats

PROPORTION = {m for m, spec in D.METRICS.items() if spec[3]}


# ----------------------------------------------------------------------------- A: band size
def band_size():
    st.subheader("A · How wide should a band built from a few retrains be?")
    st.markdown("A band is built from *n* retrains and used to judge a *new* model. The obvious "
                "rule, mean ± 2 standard deviations, ignores that the mean and the spread are "
                "themselves estimated from only *n* models, so it is too narrow and flags genuine "
                "retrains too often. A **prediction interval** widens with small *n* to hold its "
                "promise. These rates are exact, not simulated.")
    a, b = st.columns([1, 1])
    n = a.slider("Retrains in the band (n)", 3, 20, 4, key="lab_n",
                 help="Each leave-one-out band at a five-retrain forget set is built from 4.")
    k = b.slider("Fixed rule: mean ± k · sd, with k =", 1.5, 3.0, 2.0, 0.05, key="lab_k")
    ns = np.arange(3, 21)
    naive = [stats.naive_fpr(int(x), k) for x in ns]
    nominal = 2 * (1 - __import__("scipy").stats.norm.cdf(k))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=naive, mode="lines+markers", name=f"mean ± {k:.2f} sd",
                             line=dict(color=D.RED, width=2.5),
                             hovertemplate="n = %{x}: %{y:.1%} of genuine retrains flagged"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=[nominal] * len(ns), mode="lines",
                             name="prediction interval (same nominal rate)",
                             line=dict(color=D.BLUE, width=2.5),
                             hovertemplate="%{y:.1%} at every n<extra></extra>"))
    fig.add_vline(x=n, line=dict(color=D.INK, dash="dot"))
    fig.update_xaxes(title="retrains in the band", dtick=1)
    fig.update_yaxes(title="genuine retrains flagged", tickformat=".0%", rangemode="tozero")
    C.style(fig, height=360)
    st.plotly_chart(fig, width="stretch")
    m = stats.pi_multiplier(n, 1 - nominal)
    st.markdown(f"With **n = {n}**, mean ± {k:.2f} sd flags **{stats.naive_fpr(n, k):.1%}** of "
                f"genuine retrains against a promised **{nominal:.1%}** — "
                f"**{stats.naive_fpr(n, k) / nominal:.1f}×** too many, and every excess flag would "
                f"be read as the audit being invalid. The prediction interval keeps "
                f"{nominal:.1%} by using k = {m:.2f} instead.")


# ----------------------------------------------------------------------------- B: band rule on data
def _edges(row, rule, coverage, kfix):
    n, mean, sd = int(row["n"]), row["mean"], row["sd"]
    if n < 3 or not (sd == sd):
        return np.nan, np.nan
    k = stats.pi_multiplier(n, coverage) if rule == "pi" else kfix
    return mean - k * sd, mean + k * sd


def band_rule():
    st.subheader("B · Change the band rule and see who gets flagged")
    a, b, c = st.columns([1.3, 1, 1])
    rule = a.radio("Band rule", ["prediction interval (as used)", "mean ± k · sd"],
                   key="lab_rule", horizontal=True)
    rule = "pi" if rule.startswith("prediction") else "fixed"
    coverage = b.select_slider("Coverage", options=[0.80, 0.90, 0.95, 0.9545, 0.99, 0.999],
                               value=0.9545, key="lab_cov", disabled=rule != "pi")
    kfix = c.slider("k", 1.0, 3.0, 2.0, 0.1, key="lab_kfix", disabled=rule != "fixed")
    bands = D.table("bands").set_index(["forget_id", "metric", "probe_set"])
    flags = D.healthy_flags()
    has_c = D.has("candidates")
    cand = D.table("candidates") if has_c else pd.DataFrame()

    rows_u, rows_r, rows_m = [], [], []
    for fam, (metric, probe) in D.PRIMARY.items():
        for cnd in D.CONDITIONS:
            key = (cnd, metric, probe)
            if key not in bands.index:
                continue
            b_ = bands.loc[key]
            cl = D.cell(cnd, metric, probe)
            if cl.get("marked"):
                rows_u.append((fam, cnd, np.nan))
                continue
            lo, hi = _edges(b_, rule, coverage, kfix)
            g = flags[(flags["forget_id"] == cnd) & (flags["metric"] == metric)
                      & (flags["probe_set"] == probe)]
            fl = [D.flag_value(v, lo, hi, b_["direction"]) for v in g["value"]]
            rows_u.append((fam, cnd, np.mean(fl) if fl else np.nan))
            if has_c:
                cc = cand[(cand["forget_id"] == cnd) & (cand["metric"] == metric)
                          & (cand["probe_set"] == probe)]
                mem = cc[cc["band_member"]]["value"].to_numpy()
                for i in range(len(mem)):  # leave-one-out, as the calibration does it
                    rest = np.delete(mem, i)
                    if len(rest) < 3:
                        continue
                    lo_i, hi_i = _edges({"n": len(rest), "mean": rest.mean(),
                                         "sd": rest.std(ddof=1)}, rule, coverage, kfix)
                    rows_r.append((fam, cnd, D.flag_value(mem[i], lo_i, hi_i, b_["direction"])))
                for v in cc[cc["role"] == "base"]["value"]:
                    rows_m.append((fam, cnd, D.flag_value(v, lo, hi, b_["direction"])))

    u = pd.DataFrame(rows_u, columns=["family", "cond", "share"])
    z = [[u[(u["family"] == f) & (u["cond"] == cnd)]["share"].mean() for cnd in D.CONDITIONS]
         for f in D.PRIMARY]
    text = [["marked" if v != v else f"{v:.0%}" for v in r] for r in z]
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[x] for x in D.CONDITIONS],
                              [D.FAMILY_LABEL[f] for f in D.PRIMARY], text=text, height=300,
                              colorbar="unlearned models flagged"), width="stretch")
    if has_c and rows_r:
        r = pd.DataFrame(rows_r, columns=["family", "cond", "flag"])
        m = pd.DataFrame(rows_m, columns=["family", "cond", "flag"])
        x1, x2, x3 = st.columns(3)
        x1.metric("Genuine retrains flagged (leave-one-out)", f"{r['flag'].mean():.1%}",
                  help=f"{int(r['flag'].sum())} of {len(r)} retrain checks on the five primary audits")
        x2.metric("Original models flagged (power)", f"{m['flag'].mean():.0%}" if len(m) else "—")
        x3.metric("Unlearned models flagged", f"{np.nanmean(u['share']):.0%}")
        st.caption("Widen the band and fewer genuine retrains are flagged, but unlearned models and "
                   "originals slip through too; narrow it and the audit starts accusing retrains. "
                   "The prediction interval at 95.45% is the rule registered for the study.")
    else:
        st.caption("Once the per-retrain values are in (notebook 06 re-run), this panel also "
                   "recomputes the false-positive rate on the genuine retrains and the power on the "
                   "originals under the rule you pick.")


# ----------------------------------------------------------------------------- C: damage guard
def damage_guard():
    st.subheader("C · Where should \"damaged\" start?")
    st.markdown("A model whose test accuracy falls far below the retrains' is treated as "
                "**damaged**, and a flag on it is read as damage, not memory. Move the threshold.")
    u = D.table("utility")
    u = u[u["role"] == "unlearn"].copy()
    t = st.slider("Damaged = test accuracy more than this many points below the retrains", 0.5,
                  20.0, float(D.guard_points()), 0.1, key="lab_guard")
    u["damaged_now"] = u["drop_pp"] > t
    healthy_methods = u[u["method"] != "neggrad"]
    stable_lo = healthy_methods.loc[healthy_methods["drop_pp"] <= 5.0, "drop_pp"].max()
    stable_hi = u.loc[u["drop_pp"] > 5.0, "drop_pp"].min()
    fig = go.Figure()
    for meth in D.METHODS:
        dm = u[u["method"] == meth]
        fig.add_trace(go.Scatter(
            x=dm["drop_pp"], y=[D.COND_LABEL[c] for c in dm["forget_id"]], mode="markers",
            name=D.METHOD_LABEL[meth],
            marker=dict(size=10, color=D.METHOD_COLOR[meth], symbol=D.METHOD_SYMBOL[meth],
                        line=dict(color=D.INK, width=[2 if d else 0 for d in dm["damaged_now"]])),
            hovertext=[f"{D.METHOD_LABEL[meth]} seed {int(s)} at {D.COND_LABEL[c]}: "
                       f"{d:.2f} points below the retrains" for s, c, d in
                       zip(dm["run_id"].str[-1].astype(int), dm["forget_id"], dm["drop_pp"])],
            hoverinfo="text"))
    fig.add_vrect(x0=stable_lo, x1=stable_hi, fillcolor="rgba(12,163,12,0.15)", line_width=0,
                  annotation_text="every model classified the same way", annotation_position="top")
    fig.add_vline(x=t, line=dict(color=D.CRITICAL, width=2))
    fig.update_xaxes(type="log", title="test accuracy below the retrains (points, log scale)",
                     range=[np.log10(0.05), np.log10(100)], minor=dict(showgrid=False),
                     tickvals=[0.1, 0.3, 1, 3, 5, 10, 30, 100],
                     ticktext=["0.1", "0.3", "1", "3", "5", "10", "30", "100"])
    fig.update_yaxes(categoryorder="array", categoryarray=[D.COND_LABEL[c] for c in D.CONDITIONS][::-1])
    C.style(fig, height=430)
    st.plotly_chart(fig, width="stretch")
    n_dmg = int(u["damaged_now"].sum())
    by_m = u[u["damaged_now"]].groupby("method").size()
    st.markdown(f"At **{t:.1f} points**: **{n_dmg}** of 240 unlearned models are damaged ("
                + ", ".join(f"{D.METHOD_LABEL[m]} {int(v)}" for m, v in by_m.items()) + "). "
                f"Only thresholds between **{stable_lo:.2f}** and **{stable_hi:.2f}** classify every "
                f"model as the registered 5-point guard does: the healthy model nearest the line "
                f"loses {stable_lo:.2f} points, the nearest damaged one {stable_hi:.2f}.")

    # Effect on the canary scoring.
    st.markdown("**What it does to the canary scoring**")
    grid = np.unique(np.concatenate([np.geomspace(0.5, 100, 90), [t]]))
    counts = [canary_perfect(float(x)) for x in grid]
    n_read = canary_perfect(t, total=True)
    fig = go.Figure(go.Scatter(x=grid, y=counts, mode="lines", line=dict(color=D.BLUE, width=2.5),
                               line_shape="hv",
                               hovertemplate="threshold %{x:.2f} pts: %{y} perfect<extra></extra>"))
    fig.add_vline(x=t, line=dict(color=D.CRITICAL, width=2))
    fig.update_xaxes(type="log", title="damage threshold (points)", minor=dict(showgrid=False),
                     tickvals=[0.5, 1, 3, 5, 10, 30, 100], ticktext=["0.5", "1", "3", "5", "10", "30",
                                                                     "100"])
    fig.update_yaxes(title="perfect canary readings", range=[0, n_read + 2])
    C.style(fig, height=280, legend=False)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"At {t:.1f} points, **{canary_perfect(t)} of {n_read}** calibrated canary readings "
                "score a perfect balanced accuracy. Too low a threshold calls healthy models damaged "
                "and throws their correct flags away; above the destroyed control's 83–86-point loss "
                "its flags count as memory again, and most readings fall back to 0.75.")


@st.cache_data(show_spinner=False)
def canary_perfect(t: float, total: bool = False) -> int:
    """Calibrated canary readings at a balanced accuracy of 1.00 when "damaged" starts at t."""
    u = D.table("utility")
    drop = u[u["role"] == "unlearn"].set_index("run_id")["drop_pp"]
    f = D.table("flags")
    f = f[f["forget_id"] == "canary-500"]
    cn = D.table("canary").set_index("run_id")
    cv = D.table("canary_validity")
    cv = cv[(~cv["retired"].astype(bool)) & (cv["kind"] == "calibrated")]
    n, perfect = 0, 0
    for _, r in cv.iterrows():
        g = f[(f["metric"] == r["metric"]) & (f["probe_set"] == r["probe_set"])]
        if not len(g):
            continue
        n += 1
        truth = g["run_id"].map(cn["ground_truth"]).astype(bool)
        flagged = g["calibrated_flag"].astype(bool)
        verdict = flagged & ~(g["run_id"].map(drop) > t)
        # M0 and the retrains are never damaged; their counts come from the calibration.
        p_u, n_u = int(truth.sum()), int((~truth).sum())
        m0_tp = int(r["tp"]) - int((flagged & truth).sum())
        p_m0 = int(r["tp"] + r["fn"]) - p_u
        n_ret = int(r["n"]) - int(r["tp"] + r["fn"]) - n_u
        tpr = (int((verdict & truth).sum()) + m0_tp) / (p_u + p_m0)
        tnr = ((n_u - int((verdict & ~truth).sum())) + (n_ret - int(r["fp_retrain"]))) / (n_u + n_ret)
        perfect += 0.5 * (tpr + tnr) >= 0.999
    return n if total else int(perfect)


# ----------------------------------------------------------------------------- D: minimum effect
def min_effect():
    st.subheader("D · When is a difference too small to score?")
    st.markdown("A cell is **marked** (left out) when the original model can't be told from a "
                "retrain: within one retrain standard deviation, or — for proportions, where a "
                "ceiling can make 0.002 look like many standard deviations — within a minimum "
                "number of points.")
    floor = st.slider("Minimum effect for proportion metrics (points)", 0.0, 6.0, 2.0, 0.1,
                      key="lab_floor")
    v = D.table("validity")
    v = v[v["metric"].isin(PROPORTION) & (v["probe_set"] == "forget") & v["m0_gap_raw"].notna()]
    metrics = [m for m in D.ANALYSED if m in set(v["metric"])]
    z, text, hover = [], [], []
    for m in metrics:
        zr, tr, hr = [], [], []
        for cnd in D.CONDITIONS:
            r = v[(v["metric"] == m) & (v["forget_id"] == cnd)]
            if not len(r):
                zr.append(np.nan), tr.append(""), hr.append("")
                continue
            r = r.iloc[0]
            gap = abs(r["m0_gap_raw"]) * 100
            sd_rule = D._true(r.get("low_discriminability"))
            marked = sd_rule or gap < floor
            zr.append(0.0 if marked else 1.0)
            tr.append(f"{gap:.1f}")
            hr.append(f"{D.mlabel(m)}, {D.COND_LABEL[cnd]}: original {gap:.2f} points from the "
                      f"retrains — " + ("marked (within 1 sd)" if sd_rule else
                                         "marked (below the floor)" if marked else "scored"))
        z.append(zr), text.append(tr), hover.append(hr)
    scale = [[0, D.NEUTRAL], [1, "#9ec5f4"]]
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS],
                              [D.mlabel(m) for m in metrics], text=text, hover=hover, scale=scale,
                              zmin=0, zmax=1, colorbar="scored", height=120 + 36 * len(metrics)),
                    width="stretch")
    gaps = np.sort((v["m0_gap_raw"].abs() * 100).to_numpy())
    lo = gaps[gaps < 2.0].max() if (gaps < 2.0).any() else np.nan
    hi = gaps[gaps >= 2.0].min() if (gaps >= 2.0).any() else np.nan
    n_marked = int(sum(1 for r in z for x in r if x == 0.0))
    st.caption(f"Cell text: how far the original sits from the retrains, in points. At a {floor:.1f}-"
               f"point floor, {n_marked} of these cells are marked. No cell lies between "
               f"{lo:.1f} and {hi:.1f} points, so any floor in that range gives the same marks as "
               "the registered 2 points.")


def render():
    st.title("Calibration lab")
    st.markdown("The study's verdicts rest on four rules fixed before the results were read. Here "
                "you can change each one and watch what moves — and why the registered value is "
                "where it is.")
    tabs = st.tabs(["A · band size", "B · band rule", "C · damage guard", "D · minimum effect"])
    with tabs[0]:
        band_size()
    with tabs[1]:
        band_rule()
    with tabs[2]:
        damage_guard()
    with tabs[3]:
        min_effect()
