import streamlit as st

import data as D
import stats


def render():
    k = stats.kpis()
    st.title("T.R.A.C.E.")
    st.markdown("<div class='claim'>When can machine-unlearning audits be trusted?</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Machine unlearning asks a trained model to *forget* part of its training data. Whether it "
        "worked is judged by **audits**, and audits disagree. This study scores each audit against "
        "cases where the right answer is known: **genuine retrains** (models trained from scratch "
        "without the forget set, which must not be flagged), the **original models** (which did "
        "see it), and **canaries** (deliberately mislabelled images whose residual memory can be "
        "measured directly). CIFAR-10, ResNet-18, five unlearning methods plus a destroyed "
        "control, eight forget sets, five seeds: 240 unlearned models and 52 retrains.")

    five, high = k["fpr"]["five"], k["fpr"]["high"]
    cards = [
        (f"{k['native_k']} / {k['native_n']}",
         "genuine retrains called <b>leaking</b> by the population attack's own rule (mem-high)",
         "A retrain never saw the forget set, so every one of these flags is wrong."),
        (f"{five[0] / five[1]:.1%} · {high[0] / high[1]:.1%}",
         "false-positive rate once the audits are <b>calibrated against retrains</b>",
         f"{five[0]}/{five[1]} checks at the five-retrain forget sets, {high[0]}/{high[1]} at "
         "mem-high; nominal 2.3–4.6%."),
        (f"{k['canary_perfect']} / {k['canary_n']}",
         "canary readings <b>perfect</b> once read with the damage guard",
         f"Only {k['canary_perfect_unguarded']} without it: a destroyed model is also unlike a "
         "retrain."),
        (f"{k['retrain_like']} / {k['retrain_like_n']}",
         "healthy unlearned models that pass <b>every</b> calibrated audit",
         "Outside mem-low, the negative control, where there is nothing to forget."),
        (f"τ {k['h3_tau']:+.2f}",
         "representations near the original <b>predict faster relearning</b> (H3)",
         "Within-condition Kendall τ; it holds within every method."),
    ]
    cols = st.columns(5)
    for col, (value, label, note) in zip(cols, cards):
        col.markdown(
            f"<div style='border:1px solid #e1e0d9;border-radius:10px;padding:14px 16px;"
            f"background:#ffffff;height:100%'>"
            f"<div style='font-size:2rem;font-weight:650;color:#0b0b0b'>{value}</div>"
            f"<div style='font-size:0.95rem;color:#0b0b0b;margin-top:4px'>{label}</div>"
            f"<div style='font-size:0.8rem;color:#52514e;margin-top:6px'>{note}</div></div>",
            unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(str(D.ASSETS / "design.png"), caption="Study design")
    with right:
        st.subheader("Four things to take away")
        st.markdown(
            "1. **An audit's own decision rule cannot be trusted on realistic forget sets.** "
            "Applied to genuine retrains, the population attack's rule calls all of them "
            "\"leaking\" whenever the forget set was chosen by memorization or mislabelling: it "
            "detects *distribution shift*, not membership.\n"
            "2. **Calibrated against retrains, the same audits are valid**, but must be read "
            "with utility: a broken model is also unlike a retrain.\n"
            "3. **No unlearning method reproduces a retrain, and which one looks best depends on "
            "the audit you ask.** No single audit can rank them.\n"
            "4. **Relearning mostly repeats starting accuracy**, yet it exposes what a strong "
            "attack misses, and representations that stay near the original predict it.")
        st.write("")
        b1, b2 = st.columns(2)
        import nav
        if b1.button("Start the guided tour", type="primary", icon=":material/play_arrow:",
                     width="stretch"):
            st.session_state["tour_step"] = 0
            st.switch_page(nav.TOUR)
        if b2.button("Inspect a model", icon=":material/search:", width="stretch"):
            st.switch_page(nav.INSPECTOR)

    m = D.meta()
    extras = []
    if not m.get("has_candidates"):
        extras.append("per-retrain values (notebook 06 re-run) — the retrain views use band "
                      "summaries until then")
    if not m.get("curves_rows"):
        extras.append("step-by-step relearning curves (notebook 07)")
    if extras:
        st.info("Waiting for: " + "; ".join(extras) + ". Everything else is live.",
                icon=":material/hourglass_top:")
