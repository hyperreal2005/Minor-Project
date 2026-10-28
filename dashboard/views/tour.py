"""Six steps, one claim each, each with its interactive element preset. For the presentation."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as C
import data as D
import stats

STEPS = [
    ("The right answer is a retrain, not \"low forget accuracy\"",
     "A model retrained without a dog image still recognises it: it learned dogs from thousands "
     "of others. So forgetting means *behaving like a model that never saw the data*, not being "
     "wrong on it. Rank the methods both ways and watch the destroyed model move."),
    ("The audits' own rules flag genuine retrains",
     "A membership attack's usual rule says: AUC significantly above 0.5 means the data leaks. "
     "But a retrain never saw the forget set, and on atypical forget sets its AUC is far above "
     "0.5. Switch the reference point and watch the verdicts flip."),
    ("Calibrating against retrains fixes it",
     "Each audit's own rule, applied to models that provably never saw the forget set, against "
     "the same audit read as \"outside the band genuine retrains occupy\". A valid rule flags "
     "about 2–5%."),
    ("\"Unlike a retrain\" is not \"remembers\"",
     "On the canary set the answer is known: every healthy model still remembers, the retrains "
     "and the destroyed control do not. A distance from the retrains also flags the destroyed "
     "model. Read verdicts with a utility guard and see what changes."),
    ("Which method looks best depends on which audit you ask",
     "Each method's distance from the retrains relative to fine-tuning, from the confirmatory "
     "mixed models. Pick an audit; the order changes."),
    ("Representations near the original relearn faster",
     "Each point is a healthy unlearned model: how close its internal representation stayed to "
     "its own original, against how fast it relearns the forgotten data. The relation holds "
     "within every method."),
]


def render():
    i = st.session_state.setdefault("tour_step", 0)
    st.progress((i + 1) / len(STEPS), text=f"Step {i + 1} of {len(STEPS)}")
    title, blurb = STEPS[i]
    st.markdown(f"<div class='claim'>{title}</div>", unsafe_allow_html=True)
    st.markdown(blurb)
    [step1, step2, step3, step4, step5, step6][i]()
    st.divider()
    a, _, b = st.columns([1, 4, 1])
    if a.button("← Previous", disabled=i == 0, width="stretch"):
        st.session_state["tour_step"] = i - 1
        st.rerun()
    if b.button("Next →" if i < len(STEPS) - 1 else "Finish", type="primary",
                width="stretch"):
        if i < len(STEPS) - 1:
            st.session_state["tour_step"] = i + 1
            st.rerun()
        else:
            import nav
            st.switch_page(nav.HOME)


# ----------------------------------------------------------------------------- step 1
def step1():
    cond = "mem-high-3000"
    t = D.table("training")
    t = t[t["forget_id"] == cond]
    retrain = t[t["role"] == "oracle"]["forget_acc"]
    base = D.table("training")
    m0 = base[(base["role"] == "base") & (base["forget_id"] == cond)]["forget_acc"]
    m0v = float(m0.mean()) if len(m0) else 1.0
    rmean = float(retrain.mean())
    un = t[t["role"] == "unlearn"].groupby("method")["forget_acc"].mean()
    G = (un - rmean).abs() / abs(m0v - rmean)
    mode = st.radio("Rank methods by", ["lower forget accuracy (naive)",
                                         "distance from a retrain (G)"], horizontal=True)
    naive = mode.startswith("lower")
    order = un.sort_values().index if naive else G.sort_values().index
    vals = [un[m] for m in order] if naive else [G[m] for m in order]
    fig = go.Figure(go.Bar(
        y=[D.METHOD_LABEL[m] for m in order], x=vals, orientation="h",
        marker=dict(color=[D.METHOD_COLOR[m] for m in order]),
        text=[f"{v:.3f}" if naive else f"G {v:.2f}" for v in vals], textposition="outside",
        hovertext=[f"{D.METHOD_LABEL[m]}: forget accuracy {un[m]:.3f}, G {G[m]:.2f}" for m in order],
        hoverinfo="text"))
    if naive:
        fig.add_vline(x=rmean, line=dict(color=D.RETRAIN, dash="dash"),
                      annotation_text=f"retrain {rmean:.3f}", annotation_position="top")
        fig.add_vline(x=m0v, line=dict(color=D.ORIGINAL, dash="dot"),
                      annotation_text=f"original {m0v:.2f}", annotation_position="top")
    else:
        fig.add_vline(x=0, line=dict(color=D.RETRAIN, dash="dash"), annotation_text="retrain",
                      annotation_position="top")
        fig.add_vline(x=1, line=dict(color=D.ORIGINAL, dash="dot"), annotation_text="original",
                      annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="forget-set accuracy" if naive else "G (0 = retrain, 1 = original)")
    C.style(fig, height=380, legend=False)
    st.plotly_chart(fig, width="stretch")
    best = order[0]
    st.caption(f"Ranked first: **{D.METHOD_LABEL[best]}**. mem-high (the 3,000 most memorized "
               f"images): a retrain gets {rmean:.1%} of them right, the original "
               f"{m0v:.0%}.")


# ----------------------------------------------------------------------------- step 2
def step2():
    ref = st.radio("Read the population attack's AUC against", ["0.5 (the usual rule)",
                                                                  "the genuine retrains"],
                   horizontal=True)
    usual = ref.startswith("0.5")
    rows = []
    for cond in D.CONDITIONS:
        c = D.cell(cond, "mia_auc_pop", "forget")
        if c:
            rows.append((cond, c))
    fig = go.Figure()
    y = [D.COND_LABEL[cond] for cond, _ in rows]
    if usual:
        fig.add_vline(x=0.5, line=dict(color=D.CRITICAL, width=2),
                      annotation_text="0.5 = no leakage", annotation_position="bottom right")
    else:
        for cond, c in rows:
            fig.add_shape(type="rect", x0=c["lo"], x1=c["hi"], y0=D.COND_LABEL[cond],
                          y1=D.COND_LABEL[cond], yref="y", line=dict(color=D.RETRAIN, width=14),
                          opacity=0.45)
    fig.add_trace(go.Scatter(
        x=[c["mean"] for _, c in rows], y=y, mode="markers", name="genuine retrains (mean)",
        marker=dict(color=D.RETRAIN, size=13, symbol="line-ns-open", line=dict(width=3)),
        hovertext=[f"{D.COND_LABEL[k]}: retrains' AUC {c['mean']:.3f} "
                   f"(band {c['lo']:.3f}–{c['hi']:.3f})" for k, c in rows], hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[c.get("m0") for _, c in rows], y=y, mode="markers", name="original model (mean)",
        marker=dict(color=D.ORIGINAL, size=11, symbol="diamond"),
        hovertext=[f"{D.COND_LABEL[k]}: original's AUC {c.get('m0'):.3f}" for k, c in rows],
        hoverinfo="text"))
    fig.update_xaxes(title="population membership-inference AUC", range=[0.4, 1.02])
    fig.update_yaxes(autorange="reversed")
    C.style(fig, height=400)
    st.plotly_chart(fig, width="stretch")
    if usual:
        flagged = [f"{D.COND_LABEL[k]} {C.pct(c['native_fpr_k'], c['native_fpr_n'])}"
                   for k, c in rows if c.get("native_fpr_n")]
        st.error("Genuine retrains called **leaking** by the 0.5 rule: " + ", ".join(flagged) +
                 ". The attack separates atypical or mislabelled images from test images whether "
                 "or not the model saw them.", icon=":material/error:")
    else:
        st.success("Against the retrains' own band, a retrain is flagged only at the nominal rate "
                   "— and note that at mem-high and the canary the original sits *below* the "
                   "retrains: the model that trained on the data looks less member-like.",
                   icon=":material/check_circle:")


# ----------------------------------------------------------------------------- step 3
NATIVE = [("mia_auc_pop", "Population: AUC > 0.5"), ("mia_tpr_at_fpr_pop", "Population: TPR@1%"),
          ("mia_auc_rmia", "RMIA: AUC > 0.5"), ("mia_tpr_at_fpr_rmia", "RMIA: TPR@1%"),
          ("sde_verdict", "SDE: own verdict")]


def native_vs_calibrated(kind: str):
    v = D.table("validity")
    v = v[v["probe_set"] == "forget"].set_index(["metric", "forget_id"])
    z, text, hover = [], [], []
    rows = NATIVE if kind == "native" else [n for n in NATIVE if n[0] != "sde_verdict"] + [
        ("sde_hsic", "SDE: HSIC")]
    for metric, lab in rows:
        zr, tr, hr = [], [], []
        for c in D.CONDITIONS:
            if (metric, c) not in v.index:
                zr.append(np.nan), tr.append(""), hr.append("")
                continue
            r = v.loc[(metric, c)]
            col = "native_fpr" if kind == "native" else "calibrated_fpr"
            k, n = r.get(f"{col}_k"), r.get(f"{col}_n")
            if not (n == n and n):
                zr.append(np.nan), tr.append("—"), hr.append(f"{lab}, {D.COND_LABEL[c]}: not scored")
                continue
            zr.append(k / n)
            tr.append(f"{int(k)}/{int(n)}")
            hr.append(f"{lab}, {D.COND_LABEL[c]}: {int(k)} of {int(n)} genuine retrains flagged")
        z.append(zr), text.append(tr), hover.append(hr)
    return C.heatmap(z, [D.COND_LABEL[c] for c in D.CONDITIONS], [lab for _, lab in rows],
                     text=text, hover=hover, colorbar="share of retrains flagged", height=330)


def step3():
    kind = st.radio("Verdict", ["native rule", "calibrated against retrains"], horizontal=True)
    st.plotly_chart(native_vs_calibrated("native" if kind == "native rule" else "calibrated"),
                    width="stretch")
    fpr = stats.pooled_calibrated_fpr()
    (k1, n1), (k2, n2) = fpr["five"], fpr["high"]
    st.caption(f"Over every calibrated audit: {k1}/{n1} retrain checks flagged at the "
               f"five-retrain conditions ({k1 / n1:.1%}) and {k2}/{n2} at mem-high "
               f"({k2 / n2:.1%}), against a nominal 2.3–4.6%. Each retrain is compared with a "
               "band built from the *other* retrains (leave-one-out).")


# ----------------------------------------------------------------------------- step 4
def step4():
    guard = st.toggle("Read a flag on a damaged model as damage (the utility guard)", value=False)
    cv = D.table("canary_validity")
    cv = cv[~cv["retired"].astype(bool)].copy()
    pick = [("js_to_oracle", "forget", "calibrated", "JS divergence to retrains"),
            ("mia_auc_pop", "forget", "calibrated", "Population AUC, calibrated"),
            ("mia_auc_pop", "forget", "native", "Population AUC, native rule"),
            ("mia_auc_rmia", "forget", "calibrated", "RMIA AUC, calibrated"),
            ("mia_auc_rmia", "forget", "native", "RMIA AUC, native rule"),
            ("cka_linear", "layer1", "calibrated", "Linear CKA, layer 1"),
            ("cka_linear", "layer4", "calibrated", "Linear CKA, layer 4"),
            ("cka_to_original", "layer4", "calibrated", "CKA to original, layer 4"),
            ("relearn_auc", "forget", "calibrated", "Relearning AUC"),
            ("sde_verdict", "forget", "native", "SDE verdict, native")]
    labels, vals, hover = [], [], []
    for metric, probe, kind, lab in pick:
        r = cv[(cv["metric"] == metric) & (cv["probe_set"] == probe) & (cv["kind"] == kind)]
        if not len(r):
            continue
        r = r.iloc[0]
        ba = r["ba_guarded"] if guard else r["balanced_accuracy"]
        labels.append(lab)
        vals.append(ba)
        hover.append(f"{lab}: balanced accuracy {ba:.2f}<br>models that remember caught "
                     f"{int(r['tp'])}/{int(r['tp'] + r['fn'])}; retrains flagged "
                     f"{int(r['fp_retrain'])}; destroyed flagged "
                     f"{int(r['fp_other_guarded'] if guard else r['fp_other'])}")
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", hovertext=hover, hoverinfo="text",
                           marker=dict(color=[D.GOOD if v >= 0.999 else D.BLUE if v >= 0.7
                                              else D.CRITICAL for v in vals]),
                           text=[f"{v:.2f}" for v in vals], textposition="outside"))
    fig.add_vline(x=0.5, line=dict(color=D.MUTED, dash="dot"), annotation_text="chance")
    fig.update_xaxes(range=[0, 1.12], title="balanced accuracy against the canary ground truth")
    fig.update_yaxes(autorange="reversed")
    C.style(fig, height=430, legend=False)
    st.plotly_chart(fig, width="stretch")
    k = stats.kpis()
    st.caption(f"Across all {k['canary_n']} audit readings: {k['canary_perfect_unguarded']} "
               f"perfect without the guard, {k['canary_perfect']} with it. Without it, most audits "
               "score exactly 0.75: they catch every model that remembers and pass every retrain, "
               "but also flag the 5 destroyed models.")


# ----------------------------------------------------------------------------- step 5
def step5():
    me = D.table("mixed_effects")
    fam = st.segmented_control("Audit", [D.FAMILY_LABEL[f] for f in D.PRIMARY],
                               default="RMIA", key="tour_ratio_family")
    fam = next((f for f in D.PRIMARY if D.FAMILY_LABEL[f] == fam), "privacy_strong")
    metric = D.PRIMARY[fam][0]
    m = me[(me["metric"] == metric) & me["term"].astype(str).str.startswith("method")].copy()
    m["method"] = m["term"].str.split(": ", n=1).str[1]
    m = pd.concat([m, pd.DataFrame([{"method": "finetune", "coef": 0.0, "lo": 0.0, "hi": 0.0}])])
    m = m.sort_values("coef")
    fig = go.Figure()
    for r in m.itertuples():
        ratio, lo, hi = np.exp(r.coef), np.exp(r.lo), np.exp(r.hi)
        fig.add_trace(go.Scatter(
            x=[ratio], y=[D.METHOD_LABEL[r.method]], mode="markers",
            error_x=dict(type="data", symmetric=False, array=[hi - ratio], arrayminus=[ratio - lo],
                         color=D.METHOD_COLOR[r.method], thickness=2),
            marker=dict(size=14, color=D.METHOD_COLOR[r.method], symbol=D.METHOD_SYMBOL[r.method]),
            hovertext=f"{D.METHOD_LABEL[r.method]}: ×{ratio:.2f} [{lo:.2f}, {hi:.2f}] of "
                      "fine-tuning's distance", hoverinfo="text", showlegend=False))
    fig.add_vline(x=1, line=dict(color=D.MUTED, dash="dash"), annotation_text="fine-tuning")
    fig.update_xaxes(type="log", title="distance from the retrains relative to fine-tuning "
                     "(left = more retrain-like)", tickvals=[0.25, 0.5, 1, 2, 4],
                     ticktext=["×¼", "×½", "×1", "×2", "×4"], range=[np.log10(0.25), np.log10(4)])
    fig.update_yaxes(autorange="reversed")
    C.style(fig, height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Most retrain-like under {D.FAMILY_LABEL[fam]}: "
               f"**{D.METHOD_LABEL[m.iloc[0]['method']]}**. Try RMIA and then Representation: "
               "SalUn and NegGrad+ swap from best to worst.")


# ----------------------------------------------------------------------------- step 6
def step6():
    from views.relearning import h3_figure
    st.plotly_chart(h3_figure(), width="stretch")
    h3 = D.table("h3").set_index("scope")
    hc = D.table("h3_controlled").iloc[0]
    st.caption(f"Within-condition Kendall τ {h3.loc['within', 'tau']:+.2f}; in a mixed model "
               f"controlling for method and condition the slope is {hc['slope']:+.2f} "
               f"[{hc['slope_lo']:+.2f}, {hc['slope_hi']:+.2f}]. 0 = a retrain's level, "
               "1 = the original's. (The signed scale was chosen after the first run: exploratory.)")
