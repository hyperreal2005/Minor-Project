"""Are the audits right? Native rules vs calibrated bands, and the canary ground truth."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as C
import data as D
import stats
from views.tour import NATIVE, native_vs_calibrated


def _rate_matrix(col: str, metrics):
    v = D.table("validity")
    z, text, hover = [], [], []
    for metric, probe, lab in metrics:
        zr, tr, hr = [], [], []
        for c in D.CONDITIONS:
            r = v[(v["metric"] == metric) & (v["probe_set"] == probe) & (v["forget_id"] == c)]
            if not len(r) or not (r.iloc[0].get(f"{col}_n") == r.iloc[0].get(f"{col}_n")) \
                    or not r.iloc[0].get(f"{col}_n"):
                zr.append(np.nan), tr.append("—"), hr.append(f"{lab}, {D.COND_LABEL[c]}: not scored")
                continue
            r = r.iloc[0]
            k, n = int(r[f"{col}_k"]), int(r[f"{col}_n"])
            lo, hi = r.get(f"{col}_lo"), r.get(f"{col}_hi")
            zr.append(k / n), tr.append(f"{k}/{n}")
            hr.append(f"{lab}, {D.COND_LABEL[c]}: {k} of {n}"
                      + (f" (95% CI {lo:.0%}–{hi:.0%})" if lo == lo else ""))
        z.append(zr), text.append(tr), hover.append(hr)
    return z, text, hover


def render():
    st.title("Audit validity")
    st.markdown("A **retrain** never saw the forget set, so an audit that flags one is wrong. The "
                "**original model** did see it, so an audit that misses it cannot see anything. "
                "These two known answers score every audit.")

    st.subheader("1 · The retrain-free audits' own rules")
    kind = st.radio("Show", ["genuine retrains flagged (false positives)",
                             "original models flagged (power)"], horizontal=True, key="val_kind")
    col = "native_fpr" if kind.startswith("genuine") else "native_tpr"
    z, text, hover = _rate_matrix(col, [(m, "forget", lab) for m, lab in NATIVE])
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS], [l for _, l in NATIVE],
                              text=text, hover=hover, height=320,
                              colorbar="share flagged"), width="stretch")
    st.caption("A rule is *valid* at a condition when it flags at most 1 of 5 retrains and at least "
               "4 of 5 originals. The population attack's AUC rule is valid only on the random sets; "
               "its 1%-FPR rule nowhere. At mem-low missing the original is the right answer: there "
               "is nothing to see.")

    st.subheader("2 · The same audits, calibrated against retrains")
    families = st.multiselect(
        "Audit families", [D.FAMILY_LABEL[f] for f in D.FAMILY_ORDER],
        default=[D.FAMILY_LABEL[f] for f in D.FAMILY_ORDER], key="val_fams")
    fam_keys = [f for f in D.FAMILY_ORDER if D.FAMILY_LABEL[f] in families]
    v = D.table("validity")
    metrics = []
    for f in fam_keys:
        for m in [m for m, spec in D.METRICS.items() if spec[1] == f]:
            probe = D.default_probe(m)
            vm = v[(v["metric"] == m) & (v["probe_set"] == probe)]
            if len(vm) and not vm["retired"].astype(bool).any()                     and (vm["calibrated_fpr_n"].fillna(0) > 0).any():
                metrics.append((m, probe, D.mlabel(m)))
    show = st.radio("Show", ["retrains flagged (false-positive rate)", "originals flagged (power)"],
                    horizontal=True, key="val_cal")
    z, text, hover = _rate_matrix("calibrated_fpr" if show.startswith("retrains") else "m0_tpr",
                                  metrics)
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS], [m[2] for m in metrics],
                              text=text, hover=hover, height=110 + 34 * len(metrics),
                              colorbar="share flagged"), width="stretch")
    fpr = stats.pooled_calibrated_fpr()
    (k1, n1), (k2, n2) = fpr["five"], fpr["high"]
    st.caption(f"Leave-one-out: each retrain against a band built from the others. Pooled: "
               f"{k1}/{n1} ({k1 / n1:.1%}) at the five-retrain conditions, {k2}/{n2} "
               f"({k2 / n2:.1%}) at mem-high; nominal 2.3% (one-sided) to 4.6% (two-sided). "
               "Raw activation distance is retired and not shown.")

    st.subheader("3 · Scored against the canary ground truth")
    st.markdown("At the canary set every healthy unlearned model **still remembers** and the "
                "retrains and the destroyed control **do not**, so each audit's verdicts on those "
                "40 models can be scored as balanced accuracy.")
    guard = st.toggle("Apply the utility guard (a flag on a damaged model reads as damage)",
                      value=True, key="val_guard")
    cv = D.table("canary_validity")
    cv = cv[~cv["retired"].astype(bool)].copy()
    cv["label"] = [f"{D.mlabel(m)} · {p}" + (" (native rule)" if k == "native" else "")
                   for m, p, k in zip(cv["metric"], cv["probe_set"], cv["kind"])]
    cv["shown"] = cv["ba_guarded"] if guard else cv["balanced_accuracy"]
    cv = cv.sort_values("shown")
    fig = go.Figure(go.Bar(
        x=cv["shown"], y=cv["label"], orientation="h",
        marker=dict(color=[D.GOOD if s >= 0.999 else D.BLUE if s >= 0.7 else D.CRITICAL
                           for s in cv["shown"]]),
        text=[f"{s:.2f}" for s in cv["shown"]], textposition="outside",
        hovertext=[f"{lab}: {s:.2f}<br>remembering models caught {int(tp)}/{int(tp + fn)}; "
                   f"retrains flagged {int(fr)}; destroyed flagged "
                   f"{int(fg if guard else fo)}" for lab, s, tp, fn, fr, fo, fg in
                   zip(cv["label"], cv["shown"], cv["tp"], cv["fn"], cv["fp_retrain"],
                       cv["fp_other"], cv["fp_other_guarded"])], hoverinfo="text"))
    fig.add_vline(x=0.5, line=dict(color=D.MUTED, dash="dot"), annotation_text="chance")
    fig.update_xaxes(range=[0, 1.12], title="balanced accuracy")
    C.style(fig, height=26 * len(cv) + 90, legend=False)
    st.plotly_chart(fig, width="stretch")
    n_perfect = int((cv["shown"] >= 0.999).sum())
    st.caption(f"{n_perfect} of {len(cv)} readings perfect {'with' if guard else 'without'} the "
               "guard. Early-layer CKA fails either way: only late layers carry forget-set-specific "
               "structure.")

    st.subheader("4 · Does the audit track *how much* a model remembers?")
    cc = D.table("canary_continuous").sort_values("tau", ascending=False)
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        fig = go.Figure()
        for r in cc.itertuples():
            clear = r.tau_lo > 0 or r.tau_hi < 0
            fig.add_trace(go.Scatter(
                x=[r.tau], y=[D.mlabel(r.metric)], mode="markers",
                error_x=dict(type="data", symmetric=False, array=[r.tau_hi - r.tau],
                             arrayminus=[r.tau - r.tau_lo], color=D.BLUE, thickness=1.5),
                marker=dict(size=10, color=D.BLUE if clear else "white",
                            line=dict(color=D.BLUE, width=2)),
                hovertext=f"{D.mlabel(r.metric)}: τ {r.tau:+.2f} [{r.tau_lo:+.2f}, {r.tau_hi:+.2f}]",
                hoverinfo="text", showlegend=False))
        fig.add_vline(x=0, line=dict(color=D.AXIS))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Kendall τ with measured canary memory", range=[-0.6, 1])
        C.style(fig, height=520, legend=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        metric = st.selectbox("Look at one metric", list(cc["metric"]), format_func=D.mlabel,
                              key="val_deg_metric")
        probe = D.default_probe(metric)
        f = D.healthy_flags()
        f = f[(f["forget_id"] == "canary-500") & (f["metric"] == metric) & (f["probe_set"] == probe)]
        cn = D.table("canary")[["run_id", "canary_top_wrong"]]
        d = f.merge(cn, on="run_id")
        fig = go.Figure()
        for meth in D.HEALTHY_METHODS:
            dm = d[d["method"] == meth]
            fig.add_trace(C.method_trace(
                dm, "canary_top_wrong", "oracle_gap", meth, size=12,
                hover=[f"{D.METHOD_LABEL[meth]} seed {int(s)}: memory {x:.1%}, G {y:.2f}"
                       for s, x, y in zip(dm["train_seed"], dm["canary_top_wrong"], dm["oracle_gap"])]))
        fig.update_xaxes(title="canary memory: share preferring the planted label", tickformat=".0%")
        fig.update_yaxes(title=f"G on {D.mlabel(metric)}")
        C.style(fig, height=470)
        st.plotly_chart(fig, width="stretch")
        r = cc[cc["metric"] == metric].iloc[0]
        st.caption(f"τ {r['tau']:+.2f} [{r['tau_lo']:+.2f}, {r['tau_hi']:+.2f}] over the 25 healthy "
                   "canary models (5 methods × 5 seeds, so largely an ordering of methods).")
