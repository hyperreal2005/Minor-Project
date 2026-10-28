"""Relearning: does forgotten behaviour come back, and what predicts it?"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts as C
import data as D
import stats

H3_CONDS = ["canary-500", "mem-high-3000", "mem-med-3000", "rand-2500", "rand-3000", "rand-5000"]


def h3_figure(methods=None, height=560):
    s = D.table("signed_gaps")
    h3 = D.table("h3").set_index("scope")
    methods = methods or D.HEALTHY_METHODS
    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.04, vertical_spacing=0.12,
                        subplot_titles=[f"{D.COND_LABEL[c]} · τ {h3.loc[c, 'tau']:+.2f}"
                                        if c in h3.index else D.COND_LABEL[c] for c in H3_CONDS])
    for k, cnd in enumerate(H3_CONDS):
        r, c = divmod(k, 3)
        part = s[(s["forget_id"] == cnd)].dropna(subset=["cka_to_original", "relearn_auc"])
        for meth in methods:
            dm = part[part["method"] == meth]
            fig.add_trace(C.method_trace(
                dm, "cka_to_original", "relearn_auc", meth, size=9, showlegend=k == 0,
                hover=[f"{D.METHOD_LABEL[meth]} seed {int(sd)}<br>representation {x:+.2f}, "
                       f"relearning {y:+.2f}" for sd, x, y in
                       zip(dm["train_seed"], dm["cka_to_original"], dm["relearn_auc"])]),
                row=r + 1, col=c + 1)
        for v in (0, 1):
            fig.add_vline(x=v, line=dict(color=D.AXIS, width=1), row=r + 1, col=c + 1)
            fig.add_hline(y=v, line=dict(color=D.AXIS, width=1, dash="dot"), row=r + 1, col=c + 1)
    fig.update_xaxes(title_text="", row=2)
    C.style(fig, height=height)
    fig.update_annotations(font=dict(size=13, color=D.INK))
    fig.add_annotation(text="representation: similarity to its own original (signed G: 0 = a "
                            "retrain's level, 1 = the original's)", x=0.5, y=-0.1,
                       xref="paper", yref="paper", showarrow=False, font=dict(color=D.INK_2))
    fig.add_annotation(text="relearning (signed G)", x=-0.06, y=0.5, xref="paper", yref="paper",
                       textangle=-90, showarrow=False, font=dict(color=D.INK_2))
    fig.update_layout(margin=dict(l=60, r=10, t=95, b=70),
                      legend=dict(y=1.13, yanchor="bottom"))
    return fig


def curves_view():
    cv = D.table("curves")
    st.subheader("Watch forgotten behaviour come back")
    st.markdown("Each unlearned model is fine-tuned on 500 of its forget-set images for 100 steps, "
                "under the same protocol as its anchors: the **paired retrain** (grey), the "
                "**original** (black) and a **randomly initialised** network (dotted).")
    if st.button("Show the hidden-knowledge case: SalUn at rand-500", icon=":material/visibility:"):
        st.session_state["rl_cond"] = "rand-500"
        st.session_state["rl_methods"] = ["salun"]
    a, b, c = st.columns([2, 2, 1.2])
    with a:
        cnd = stats.condition_select("rl_cond", default="mem-high-3000",
                                     options=[x for x in D.CONDITIONS if x in set(cv["forget_id"])])
    kw = {} if "rl_methods" in st.session_state else {"default": ["salun", "finetune"]}
    methods = b.multiselect("Methods", D.METHODS, format_func=D.METHOD_LABEL.get, key="rl_methods",
                            **kw)
    norm = c.toggle("Scale 0 = retrain, 1 = original", value=False, key="rl_norm",
                    help="Each step's accuracy placed between the mean retrain and the mean "
                         "original curve at that step.")
    part = cv[cv["forget_id"] == cnd]
    anchors = part[part["arm"].isin(["oracle", "original"])]
    mean_rt = part[part["arm"] == "oracle"].groupby("step")["forget_acc"].mean()
    mean_m0 = part[part["arm"] == "original"].groupby("step")["forget_acc"].mean()

    def scale(s, acc):
        if not norm:
            return acc
        lo, hi = mean_rt.reindex(s).to_numpy(), mean_m0.reindex(s).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return (acc - lo) / (hi - lo)

    fig = go.Figure()
    x_of = lambda s: np.asarray(s) + 1  # noqa: E731 -- log axis needs step 0 at 1
    for arm, colour, dash, name in (("oracle", D.RETRAIN, "solid", "paired retrains"),
                                    ("original", D.ORIGINAL, "solid", "originals"),
                                    ("randinit", D.MUTED, "dot", "random init")):
        for k, (rid, g) in enumerate(part[part["arm"] == arm].groupby("run_id")):
            g = g.sort_values("step")
            fig.add_trace(go.Scatter(x=x_of(g["step"]), y=scale(g["step"], g["forget_acc"]),
                                     mode="lines", line=dict(color=colour, dash=dash, width=1.5),
                                     opacity=0.7, name=name, legendgroup=arm, showlegend=k == 0,
                                     hovertext=f"{name}: {rid}", hoverinfo="text+y"))
    for meth in methods:
        for k, (rid, g) in enumerate(part[(part["arm"] == "method") & (part["method"] == meth)]
                                     .groupby("run_id")):
            g = g.sort_values("step")
            fig.add_trace(go.Scatter(
                x=x_of(g["step"]), y=scale(g["step"], g["forget_acc"]), mode="lines+markers",
                line=dict(color=D.METHOD_COLOR[meth], width=2.5),
                marker=dict(symbol=D.METHOD_SYMBOL[meth], size=7), name=D.METHOD_LABEL[meth],
                legendgroup=meth, showlegend=k == 0,
                hovertext=[f"{D.METHOD_LABEL[meth]} seed {int(sd)}, step {int(st_)}: "
                           f"{a:.3f}" for sd, st_, a in zip(g["train_seed"], g["step"], g["forget_acc"])],
                hoverinfo="text"))
    fig.update_xaxes(type="log", title="relearning step (+1, log scale)",
                     tickvals=[1, 2, 3, 6, 11, 26, 51, 101],
                     ticktext=["0", "1", "2", "5", "10", "25", "50", "100"])
    fig.update_yaxes(title="forget-set accuracy" if not norm else "0 = retrain, 1 = original")
    C.style(fig, height=520)
    st.plotly_chart(fig, width="stretch")
    m = part[(part["arm"] == "method")].drop_duplicates("run_id")
    if m["recorded_auc"].notna().any():
        d = (m["auc"] - m["recorded_auc"]).abs()
        st.caption(f"These curves are a re-run of the identical protocol for display. GPU training "
                   f"is not bit-reproducible: their AUCs differ from the recorded ones by a median "
                   f"of {d.median():.4f} (max {d.max():.4f}).")


def auc_vs_start():
    st.subheader("Relearning mostly repeats where the model starts")
    st.markdown("The relearning curve begins at the model's current forget accuracy, so its area "
                "largely restates that starting point. Where models *look* forgotten, though, it "
                "adds information: at rand-500, four models start inside the retrain band yet "
                "relearn faster than every retrain — all four are SalUn.")
    cnd = stats.condition_select("rl_start_cond", default="rand-500")
    f = D.healthy_flags()
    f = f[(f["forget_id"] == cnd) & (f["metric"] == "relearn_auc") & (f["probe_set"] == "forget")]
    t = D.table("training")[["run_id", "forget_acc"]]
    d = f.merge(t, on="run_id")
    rt = D.table("training")
    rt = rt[(rt["role"] == "oracle") & (rt["forget_id"] == cnd)]["forget_acc"]
    band = D.cell(cnd, "relearn_auc", "forget")
    fig = go.Figure()
    if len(rt):
        fig.add_vrect(x0=rt.min(), x1=rt.max(), fillcolor=D.BAND_FILL, line_width=0,
                      annotation_text="retrains' start", annotation_position="top left")
    if band:
        fig.add_hrect(y0=band["lo"], y1=band["hi"], fillcolor=D.BAND_FILL, line_width=0,
                      annotation_text="retrains' relearning", annotation_position="bottom right")
    for meth in D.HEALTHY_METHODS:
        dm = d[d["method"] == meth]
        fig.add_trace(C.method_trace(dm, "forget_acc", "value", meth, size=11, hover=[
            f"{D.METHOD_LABEL[meth]} seed {int(s)}: start {a:.3f}, relearning AUC {v:.3f}"
            for s, a, v in zip(dm["train_seed"], dm["forget_acc"], dm["value"])]))
    fig.update_xaxes(title="starting forget-set accuracy")
    fig.update_yaxes(title="relearning AUC")
    C.style(fig, height=460)
    st.plotly_chart(fig, width="stretch")
    rv = D.table("relearning_vs_accuracy")
    r = rv[rv["forget_id"] == cnd]
    if len(r):
        r = r.iloc[0]
        st.caption(f"Correlation between relearning AUC and starting accuracy here: "
                   f"{r['r_relearn_vs_start']:.3f}. Flagged by starting accuracy "
                   f"{int(r['flagged_by_start'])}, by relearning {int(r['flagged_by_relearn'])}, by "
                   f"relearning only {int(r['flagged_by_relearn_only'])}.")


def render():
    st.title("Relearning")
    if D.has("curves"):
        curves_view()
    else:
        st.info("Step-by-step relearning curves appear here once notebook 07's output is in "
                "(`forgetcheck_curves.zip` → `results/curves/`, then `prepare_data.py`).",
                icon=":material/hourglass_top:")
    auc_vs_start()
    st.subheader("What predicts relearning? Closeness to the original (H3)")
    st.markdown("Measured as distance *from the retrains*, representation does not predict "
                "relearning — most models move away from both the retrains and the original. "
                "Measured as similarity to each model's **own original**, it does, at every forget "
                "set where relearning can be scored.")
    sel = st.multiselect("Methods", D.HEALTHY_METHODS, default=D.HEALTHY_METHODS,
                         format_func=D.METHOD_LABEL.get, key="h3_methods")
    st.plotly_chart(h3_figure(sel or D.HEALTHY_METHODS), width="stretch")
    hc = D.table("h3_controlled").iloc[0]
    h3 = D.table("h3").set_index("scope")
    st.caption(f"Within-condition τ {h3.loc['within', 'tau']:+.2f}; controlling for method and "
               f"condition, slope {hc['slope']:+.2f} [{hc['slope_lo']:+.2f}, {hc['slope_hi']:+.2f}]. "
               "Filter to one method: the relation holds within it. The signed scale was chosen "
               "after the first run (exploratory).")
