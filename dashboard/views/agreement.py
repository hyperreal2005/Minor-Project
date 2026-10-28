"""Do two audits put the same models in the same order?"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as C
import data as D
import stats

PAIR_LABEL = {"behavior": "Behavior", "privacy_weak": "Population", "privacy_strong": "RMIA",
              "representation": "Representation", "reversibility": "Relearning"}


@st.cache_data(show_spinner=False)
def gaps() -> pd.DataFrame:
    """G per healthy model, one column per analysed metric (at its headline probe)."""
    f = D.healthy_flags()
    keep = [m in D.ANALYSED and p == D.default_probe(m) for m, p in zip(f["metric"], f["probe_set"])]
    f = f[keep]
    return f.pivot_table(index=["forget_id", "run_id", "method", "train_seed"], columns="metric",
                         values="oracle_gap").reset_index()


def render():
    st.title("Agreement explorer")
    st.markdown("Each audit turns a model into a distance from the retrains, **G** (0 = like a "
                "retrain, 1 = like the original). Two audits *agree* when they put the same models "
                "in the same order. The unit is the model, not the method: there are 197 healthy "
                "unlearned models.")
    g = gaps()
    a, b, c, d = st.columns([1.3, 1.3, 1.6, 1])
    x = a.selectbox("Audit metric on x", D.ANALYSED, index=D.ANALYSED.index("mia_auc_rmia"),
                    format_func=D.mlabel, key="agr_x")
    y = b.selectbox("Audit metric on y", D.ANALYSED, index=D.ANALYSED.index("relearn_auc"),
                    format_func=D.mlabel, key="agr_y")
    with c:
        scope = st.selectbox("Models", ["all forget sets (pooled)"] + D.CONDITIONS,
                             format_func=lambda s: s if s.startswith("all") else D.COND_LABEL[s],
                             key="agr_scope")
    log = d.toggle("Log axes", value=True, key="agr_log",
                   help="G has a long right tail wherever the original is barely different from a "
                        "retrain.")
    data = g if scope.startswith("all") else g[g["forget_id"] == scope]
    data = data.dropna(subset=[x, y])
    if not len(data):
        st.warning("These two metrics are never scored on the same models here (marked cells).")
        return

    pooled = stats.kendall(data[x], data[y])
    per = {cnd: stats.kendall(gg[x], gg[y]) for cnd, gg in data.groupby("forget_id") if len(gg) >= 5}
    ns = {cnd: len(gg) for cnd, gg in data.groupby("forget_id")}
    within = (np.nansum([per[k] * ns[k] for k in per]) / np.sum([ns[k] for k in per
                                                                    if per[k] == per[k]])
              if per else np.nan)
    m1, m2, m3 = st.columns(3)
    m1.metric("Kendall τ, " + ("pooled" if scope.startswith("all") else D.COND_LABEL[scope]),
              f"{pooled:+.2f}")
    if scope.startswith("all"):
        m2.metric("Within-condition τ (n-weighted)", f"{within:+.2f}",
                  delta=f"{within - pooled:+.2f} vs pooled", delta_color="off",
                  help="Pooling lets every audit shift together with the forget set, which can "
                       "manufacture agreement that no single forget set shows.")
    m3.metric("Models", f"{len(data)}")

    fig = go.Figure()
    colour_by = st.radio("Colour by", ["method", "forget set"], horizontal=True, key="agr_colour")
    if colour_by == "method":
        for meth in D.HEALTHY_METHODS:
            dm = data[data["method"] == meth]
            fig.add_trace(C.method_trace(dm, x, y, meth, size=10, hover=[
                f"{D.METHOD_LABEL[meth]} · {D.COND_LABEL[f]} · seed {int(s)}<br>"
                f"{D.mlabel(x)} G {gx:.2f}<br>{D.mlabel(y)} G {gy:.2f}"
                for f, s, gx, gy in zip(dm["forget_id"], dm["train_seed"], dm[x], dm[y])]))
    else:
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
                   "#e34948"]
        for k, cnd in enumerate(D.CONDITIONS):
            dm = data[data["forget_id"] == cnd]
            if not len(dm):
                continue
            fig.add_trace(go.Scatter(x=dm[x], y=dm[y], mode="markers", name=D.COND_LABEL[cnd],
                                     marker=dict(size=10, color=palette[k], line=dict(color="white",
                                                                                     width=1)),
                                     hovertext=[f"{D.COND_LABEL[cnd]} · {D.METHOD_LABEL[m]}"
                                                for m in dm["method"]], hoverinfo="text"))
    for v in (1,):
        fig.add_vline(x=v, line=dict(color=D.ORIGINAL, dash="dot", width=1))
        fig.add_hline(y=v, line=dict(color=D.ORIGINAL, dash="dot", width=1))
    fig.update_xaxes(title=f"G on {D.mlabel(x)}", type="log" if log else "linear")
    fig.update_yaxes(title=f"G on {D.mlabel(y)}", type="log" if log else "linear")
    C.style(fig, height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dotted lines: G = 1, the original model's distance. Models beyond them sit farther "
               "from every retrain than the original did.")

    if scope.startswith("all") and per:
        bar = go.Figure(go.Bar(
            x=[D.COND_LABEL[k] for k in per], y=list(per.values()),
            marker=dict(color=[D.BLUE if v >= 0 else D.RED for v in per.values()]),
            text=[f"{v:+.2f}" for v in per.values()], textposition="outside",
            hovertext=[f"{D.COND_LABEL[k]}: τ {v:+.2f} over {ns[k]} models" for k, v in per.items()],
            hoverinfo="text"))
        bar.add_hline(y=pooled, line=dict(color=D.INK, dash="dash"),
                      annotation_text=f"pooled {pooled:+.2f}", annotation_position="bottom left")
        bar.update_yaxes(range=[-1, 1], title="τ within the forget set")
        C.style(bar, height=300, legend=False)
        st.plotly_chart(bar, width="stretch")

    st.subheader("The registered pairs, forget set by forget set")
    corr = D.table("correlations")
    perc = corr[~corr["scope"].isin(["pooled", "within"])]
    pairs = list(dict.fromkeys(perc["pair"]))
    conds = [cnd for cnd in D.CONDITIONS if cnd in set(perc["scope"])]
    z, text, hover, bold = [], [], [], []
    for p in pairs:
        zr, tr, hr, br = [], [], [], []
        for cnd in conds:
            r = perc[(perc["pair"] == p) & (perc["scope"] == cnd)]
            t = float(r["tau"].iloc[0]) if len(r) else np.nan
            lo = float(r["tau_lo"].iloc[0]) if len(r) else np.nan
            hi = float(r["tau_hi"].iloc[0]) if len(r) else np.nan
            zr.append(t), tr.append(f"{t:+.2f}" if t == t else "marked")
            hr.append(f"{p}, {D.COND_LABEL[cnd]}: τ {t:+.2f} [{lo:+.2f}, {hi:+.2f}]" if t == t
                      else "marked")
            br.append(bool(lo == lo and (lo > 0 or hi < 0)))
        z.append(zr), text.append(tr), hover.append(hr), bold.append(br)
    names = [" vs ".join(PAIR_LABEL.get(s, s) for s in p.split(" vs ")) for p in pairs]
    mean = np.nanmean(np.asarray(z, float), axis=0)
    z.append(list(mean)), text.append([f"{m:+.2f}" for m in mean])
    hover.append([f"mean over pairs at {D.COND_LABEL[c]}: {m:+.2f}" for c, m in zip(conds, mean)])
    bold.append([False] * len(conds))
    st.plotly_chart(C.heatmap(z, [D.COND_LABEL[c] for c in conds], names + ["Mean over pairs"],
                              text=text, hover=hover, bold=bold, scale="div", zmin=-1, zmax=1,
                              colorbar="Kendall τ", height=400), width="stretch")
    st.caption("Bold: the 95% bootstrap interval excludes 0. Audits agree most where memorization "
               "is strongest (mem-high, canary) and least at mem-med: the reverse of what was "
               "predicted (H4).")
