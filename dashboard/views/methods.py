"""Methods and forget-set difficulty: no single audit ranks the methods."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as C
import data as D
import stats


def ratio_heatmap():
    me = D.table("mixed_effects")
    m = me[me["term"].astype(str).str.startswith("method")]
    methods = [k for k in D.HEALTHY_METHODS if k != "finetune"]
    z, text, hover, bold = [], [], [], []
    for meth in methods:
        zr, tr, hr, br = [], [], [], []
        for fam, (metric, _) in D.PRIMARY.items():
            r = m[(m["metric"] == metric) & (m["term"] == f"method: {meth}")]
            if not len(r):
                zr.append(np.nan), tr.append(""), hr.append(""), br.append(False)
                continue
            r = r.iloc[0]
            zr.append(r["coef"] / np.log(2))
            tr.append(f"×{np.exp(r['coef']):.2f}")
            hr.append(f"{D.METHOD_LABEL[meth]}, {D.FAMILY_LABEL[fam]}: ×{np.exp(r['coef']):.2f} "
                      f"[{np.exp(r['lo']):.2f}, {np.exp(r['hi']):.2f}] of fine-tuning's distance")
            br.append(bool(r["lo"] > 0 or r["hi"] < 0))
        z.append(zr), text.append(tr), hover.append(hr), bold.append(br)
    fig = C.heatmap(z, [D.FAMILY_LABEL[f] for f in D.PRIMARY], [D.METHOD_LABEL[k] for k in methods],
                    text=text, hover=hover, bold=bold, scale=C.div_scale(neg=D.BLUE, pos=D.RED),
                    zmin=-1.6, zmax=1.6, colorbar="log₂ ratio", height=300)
    return fig


def render():
    st.title("Methods & difficulty")
    st.subheader("Which method looks most like a retrain depends on the audit")
    st.markdown("From the confirmatory mixed models, `log G ~ method + condition + (1 | seed)`: "
                "each method's distance from the retrains relative to fine-tuning, averaged over "
                "forget sets. **Blue** = nearer a retrain than fine-tuning, **red** = farther; "
                "bold = the 95% interval excludes ×1.")
    st.plotly_chart(ratio_heatmap(), width="stretch")
    st.caption("SalUn and NegGrad+ are the most retrain-like to RMIA and relearning and the least "
               "to behavior and representation; SCRUB is best on representation; the population "
               "attack separates no method from fine-tuning. Seeds barely matter (seed variance "
               "below 10⁻⁸).")

    st.subheader("How far, and in which direction, per forget set")
    a, b = st.columns([1, 1])
    fam = a.selectbox("Audit", list(D.PRIMARY), format_func=D.FAMILY_LABEL.get, key="meth_fam")
    stat = b.radio("Show", ["median G", "share beyond the original (G > 1)", "share flagged"],
                   horizontal=True, key="meth_stat")
    metric = D.PRIMARY[fam][0]
    ms = D.table("method_summary")
    ms = ms[ms["metric"] == metric]
    col = {"median G": "G_median", "share beyond the original (G > 1)": "share_beyond_m0",
           "share flagged": "flag_rate"}[stat]
    z, text, hover = [], [], []
    for meth in D.HEALTHY_METHODS:
        zr, tr, hr = [], [], []
        for cnd in D.CONDITIONS:
            r = ms[(ms["method"] == meth) & (ms["forget_id"] == cnd)]
            v = float(r[col].iloc[0]) if len(r) and r[col].iloc[0] == r[col].iloc[0] else np.nan
            zr.append(v)
            tr.append("" if v != v else (f"{v:.2f}" if col == "G_median" else f"{v:.0%}"))
            hr.append(f"{D.METHOD_LABEL[meth]}, {D.COND_LABEL[cnd]}: "
                      + ("marked" if v != v else tr[-1]))
        z.append(zr), text.append(tr), hover.append(hr)
    if col == "G_median":
        zz = np.log10(np.clip(np.asarray(z, float), 1e-3, None))
        fig = C.heatmap(zz, [D.COND_LABEL[c] for c in D.CONDITIONS],
                        [D.METHOD_LABEL[m] for m in D.HEALTHY_METHODS], text=text, hover=hover,
                        scale=C.div_scale(neg=D.BLUE, pos=D.RED), zmin=-1.2, zmax=1.2,
                        colorbar="log₁₀ G", height=330)
    else:
        fig = C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS],
                        [D.METHOD_LABEL[m] for m in D.HEALTHY_METHODS], text=text, hover=hover,
                        height=330, colorbar="share")
    st.plotly_chart(fig, width="stretch")
    if col == "G_median":
        st.caption("G = 0 is a retrain, 1 the original. Blue: partway to a retrain; red: past the "
                   "original — on representation most models end up farther from every retrain "
                   "than the original was.")

    st.subheader("Forgetting and utility, method by method")
    t = D.table("training")
    cnd = stats.condition_select("meth_cond")
    tt = t[t["forget_id"] == cnd]
    measure = st.radio("Measure", ["forget-set accuracy", "test accuracy"], horizontal=True,
                       key="meth_measure")
    col = "forget_acc" if measure.startswith("forget") else "test_acc"
    fig = go.Figure()
    rt = tt[tt["role"] == "oracle"][col]
    if len(rt):
        fig.add_vrect(x0=rt.min(), x1=rt.max(), fillcolor=D.BAND_FILL, line_width=0)
        fig.add_vline(x=rt.mean(), line=dict(color=D.RETRAIN, dash="dash"),
                      annotation_text=f"retrains {rt.mean():.3f}", annotation_position="top")
    for meth in D.METHODS:
        dm = tt[(tt["role"] == "unlearn") & (tt["method"] == meth)]
        fig.add_trace(go.Scatter(
            x=dm[col], y=[D.METHOD_LABEL[meth]] * len(dm), mode="markers",
            marker=dict(size=13, color=D.METHOD_COLOR[meth], symbol=D.METHOD_SYMBOL[meth],
                        line=dict(color="white", width=1)),
            hovertext=[f"{D.METHOD_LABEL[meth]} seed {int(s)}: {v:.3f}"
                       for s, v in zip(dm["train_seed"], dm[col])], hoverinfo="text",
            showlegend=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=measure)
    C.style(fig, height=360, legend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey band: the range of genuine retrains. The destroyed control sits near 10% "
               "on everything: it 'forgets' by forgetting everything.")

    st.subheader("Does difficulty change how far apart the methods are?")
    dfc = D.table("difficulty")
    z, text = [], []
    fams = [f for f in D.PRIMARY]
    for f in fams:
        zr, tr = [], []
        for cnd in D.CONDITIONS:
            r = dfc[(dfc["family"] == f) & (dfc["forget_id"] == cnd)]
            v = float(r["method_sd_log"].iloc[0]) if len(r) else np.nan
            zr.append(v), tr.append("" if v != v else f"{v:.2f}")
        z.append(zr), text.append(tr)
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS],
                              [D.FAMILY_LABEL[f] for f in fams], text=text, zmin=0, zmax=2.2,
                              colorbar="SD of method means (log G)", height=300),
                    width="stretch")
    st.caption("The spread between methods is largest at mem-high only for the population attack. "
               "Behavior spreads most at mem-low, through the methods' side effects on easy "
               "examples rather than through forgetting.")
