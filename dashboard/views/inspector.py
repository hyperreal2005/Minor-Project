"""One model's report card: every audit as a strip against the retrains' band."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts as C
import data as D
import stats

DEFAULT_METRICS = [("js_to_oracle", "forget"), ("pred_agreement", "forget"),
                   ("mia_auc_pop", "forget"), ("mia_tpr_at_fpr_pop", "forget"),
                   ("mia_auc_rmia", "forget"), ("mia_tpr_at_fpr_rmia", "forget"),
                   ("cka_linear", "layer4"), ("cka_to_original", "layer4"),
                   ("relearn_auc", "forget"), ("relearn_t80", "forget"),
                   ("sde_hsic", "forget"), ("sde_margin", "forget")]


def _verdict(flag, native, damaged):
    if flag is None or flag != flag:
        return "not scored"
    if damaged and flag:
        return "flagged — read as damage"
    return "unlike every retrain" if flag else "within the retrain band"


def render():
    st.title("Model inspector")
    st.markdown("Pick any of the 240 unlearned models. Each audit is drawn as a strip: the grey "
                "band is where genuine retrains fall, the black diamond is the original model, "
                "faint marks are the other unlearned models at the same forget set, and the large "
                "mark is the model you picked.")
    a, b, c, d = st.columns([2.2, 1.4, 0.8, 1.1])
    with a:
        cond = stats.condition_select("insp_cond")
    method = b.selectbox("Method", D.METHODS, format_func=D.METHOD_LABEL.get, key="insp_method")
    seed = c.selectbox("Seed", [0, 1, 2, 3, 4], key="insp_seed")
    show_all = d.toggle("Every metric and probe", value=False, key="insp_all")

    f = D.table("flags")
    here = f[f["forget_id"] == cond]
    mine = here[(here["method"] == method) & (here["train_seed"] == seed)]
    if not len(mine):
        st.warning("No records for that model.")
        return
    rid = mine["run_id"].iloc[0]
    u = D.table("utility")
    ur = u[(u["run_id"] == rid) & (u["forget_id"] == cond)]
    st.caption(f"`{rid}` — {D.METHOD_BLURB[method]}.")

    m1, m2, m3 = st.columns(3)
    if len(ur):
        ur = ur.iloc[0]
        m1.metric("Test accuracy", f"{ur['test_acc']:.1%}",
                  f"{-ur['drop_pp']:+.1f} pts vs retrains", delta_color="normal")
        m2.metric("Damaged?", "yes" if ur["damaged"] else "no",
                  help=f"Damaged = more than {ur['guard_pp']:.0f} points below the retrains' test "
                       "accuracy. A flag on a damaged model is read as damage, not memory.")
    if cond == "canary-500":
        cn = D.table("canary")
        cr = cn[cn["run_id"] == rid]
        if len(cr):
            cr = cr.iloc[0]
            m3.metric("Still remembers the canaries?", "yes" if cr["ground_truth"] else "no",
                      help=f"Prefers the planted label for {cr['canary_top_wrong']:.1%} of canaries "
                           f"(retrain band ends at {cr['top_wrong_band_hi']:.1%}; chance 11.1%).")

    pairs = DEFAULT_METRICS if not show_all else sorted(
        {(m, p) for m, p in zip(here["metric"], here["probe_set"]) if m in D.METRICS},
        key=lambda mp: (D.FAMILY_ORDER.index(D.METRICS[mp[0]][1]), mp[0], mp[1]))
    pairs = [mp for mp in pairs if len(here[(here["metric"] == mp[0]) & (here["probe_set"] == mp[1])])]

    cand = D.table("candidates") if D.has("candidates") else pd.DataFrame()
    rows = len(pairs)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.55 / max(rows, 1),
                        subplot_titles=[f"{D.mlabel(m)}" + (f" · {p}" if p != "forget" else "")
                                        for m, p in pairs])
    table_rows = []
    shown = set()
    for i, (metric, probe) in enumerate(pairs, start=1):
        cl = D.cell(cond, metric, probe)
        g = here[(here["metric"] == metric) & (here["probe_set"] == probe)]
        if cl:
            C.band_shape(fig, cl.get("lo"), cl.get("hi"), row=i, col=1)
            fig.add_trace(go.Scatter(x=[cl["mean"]], y=[0], mode="markers", name="retrain mean",
                                     marker=dict(symbol="line-ns-open", size=18, color=D.RETRAIN,
                                                 line=dict(width=3)),
                                     hovertext=f"retrains: mean {cl['mean']:.4g}, band "
                                               f"{cl['lo']:.4g}–{cl['hi']:.4g} (n = {int(cl['n'])})",
                                     hoverinfo="text", showlegend="rm" not in shown,
                                     legendgroup="rm"), row=i, col=1)
            shown.add("rm")
            if cl.get("m0") == cl.get("m0"):
                fig.add_trace(go.Scatter(x=[cl["m0"]], y=[0], mode="markers", name="original (M0)",
                                         marker=dict(symbol="diamond", size=12, color=D.ORIGINAL),
                                         hovertext=f"original: {cl['m0']:.4g}", hoverinfo="text",
                                         showlegend="m0" not in shown, legendgroup="m0"),
                              row=i, col=1)
                shown.add("m0")
        if len(cand):
            rr = cand[(cand["forget_id"] == cond) & (cand["metric"] == metric)
                      & (cand["probe_set"] == probe) & (cand["role"] == "oracle")]
            if len(rr):
                fig.add_trace(go.Scatter(x=rr["value"], y=np.full(len(rr), -0.35), mode="markers",
                                         name="each retrain", marker=dict(size=6, color=D.RETRAIN),
                                         hovertext=[f"retrain {r}: {v:.4g}" for r, v in
                                                    zip(rr["run_id"], rr["value"])],
                                         hoverinfo="text", showlegend="rt" not in shown,
                                         legendgroup="rt"), row=i, col=1)
                shown.add("rt")
        others = g[g["run_id"] != rid]
        jitter = np.random.default_rng(i).uniform(0.18, 0.42, len(others))
        for meth in D.METHODS:
            o = others[others["method"] == meth]
            if not len(o):
                continue
            fig.add_trace(go.Scatter(
                x=o["value"], y=jitter[others["method"].to_numpy() == meth], mode="markers",
                name=D.METHOD_LABEL[meth], opacity=0.35,
                marker=dict(size=7, color=D.METHOD_COLOR[meth], symbol=D.METHOD_SYMBOL[meth]),
                hovertext=[f"{D.METHOD_LABEL[meth]} seed {int(s)}: {v:.4g}"
                           for s, v in zip(o["train_seed"], o["value"])], hoverinfo="text",
                showlegend=meth not in shown, legendgroup=meth), row=i, col=1)
            shown.add(meth)
        me_ = g[g["run_id"] == rid]
        if len(me_):
            r = me_.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r["value"]], y=[0], mode="markers", name="this model",
                marker=dict(size=17, color=D.METHOD_COLOR[method], symbol=D.METHOD_SYMBOL[method],
                            line=dict(color=D.INK, width=2)),
                hovertext=f"this model: {r['value']:.4g}", hoverinfo="text",
                showlegend="me" not in shown, legendgroup="me"), row=i, col=1)
            shown.add("me")
            table_rows.append({
                "audit": D.FAMILY_LABEL[D.METRICS[metric][1]], "metric": D.mlabel(metric),
                "probe": probe, "value": r["value"],
                "retrain band": (f"{cl['lo']:.4g} – {cl['hi']:.4g}" if cl and cl["lo"] == cl["lo"]
                                 else "—"),
                "original": (cl.get("m0") if cl and cl.get("m0") is not None else np.nan),
                "calibrated verdict": "retired" if r["retired"] else _verdict(
                    r["calibrated_flag"], r["native_flag"], bool(r["damaged"])),
                "native rule": ("—" if r["native_flag"] is None or r["native_flag"] != r["native_flag"]
                                else ("says: leaks / in training" if r["native_flag"] else "says: fine")),
                "G": (r["oracle_gap"] if r["oracle_gap"] is not None else np.nan),
                "what it measures": D.METRICS[metric][2],
            })
        fig.update_yaxes(visible=False, range=[-0.6, 0.6], row=i, col=1)
    C.style(fig, height=max(320, 92 * rows + 60))
    fig.update_annotations(font=dict(size=12, color=D.INK_2), xanchor="left", x=0)
    st.plotly_chart(fig, width="stretch")
    if not len(cand):
        st.caption("Individual retrains appear here once the per-retrain values are in "
                   "(notebook 06 re-run); the band is drawn from their summary meanwhile.")

    st.subheader("Verdicts")
    tdf = pd.DataFrame(table_rows)
    st.dataframe(tdf, hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn(format="%.4g"),
                                "original": st.column_config.NumberColumn(format="%.4g"),
                                "G": st.column_config.NumberColumn(
                                    format="%.2f", help="0 = like a retrain, 1 = like the "
                                    "original, >1 = farther than the original")})
