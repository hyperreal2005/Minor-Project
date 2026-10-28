"""What the forget sets look like: memorization, and real images."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import charts as C
import data as D
import stats


def render():
    st.title("Forget sets")
    st.markdown("**Memorization** of a training image is how much worse a model does on it when "
                "it is left out of training (Feldman & Zhang's definition; the CIFAR-10 scores are "
                "those released by Zhao et al.). Typical images score 0 — the model gets them right "
                "from the rest of its class. Atypical ones score near 1.")
    mem = D.memorization()
    cnd = stats.condition_select("fs_cond", default="mem-high-3000")
    imgs, meta = D.gallery()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mem, xbins=dict(start=0, end=1.0001, size=0.025),
                               name="all 50,000 training images", marker_color=D.AXIS,
                               hovertemplate="score %{x}: %{y} images<extra></extra>"))
    fi = D.table("forget_indices")
    if len(fi):
        idx = fi.loc[fi["forget_id"] == cnd, "index"].to_numpy()
        fig.add_trace(go.Histogram(x=mem[idx], xbins=dict(start=0, end=1.0001, size=0.025),
                                   name=f"{D.COND_LABEL[cnd]} ({len(idx):,} images)",
                                   marker_color=D.BLUE, opacity=0.85,
                                   hovertemplate="score %{x}: %{y} images<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="log", title="images (log scale)")
    fig.update_xaxes(title="memorization score")
    C.style(fig, height=330)
    st.plotly_chart(fig, width="stretch")
    share = (mem > 0.5).mean()
    st.caption(f"{share:.1%} of CIFAR-10's training images score above 0.5, and the median is "
               f"{np.median(mem):.3f}: a random forget set is a mixture, not an easy case.")

    if imgs is None:
        st.info("Image gallery unavailable: CIFAR-10 was not on the machine that built the snapshot.")
        return
    st.subheader(f"{D.COND_LABEL[cnd]}: a random sample of 24 of its images")
    st.caption(D.COND_BLURB[cnd])
    m = meta[meta["forget_id"] == cnd]
    cols = st.columns(8)
    for k, (i, r) in enumerate(m.iterrows()):
        img = Image.fromarray(imgs[i]).resize((96, 96), Image.NEAREST)
        cap = D.CIFAR_CLASSES[r["label"]]
        if r["planted"] >= 0:
            cap = f"is {cap}, labelled {D.CIFAR_CLASSES[r['planted']]}"
        cap += f" · mem {r['memorization']:.2f}"
        cols[k % 8].image(img, caption=cap, width="stretch")
    if cnd.startswith("canary"):
        st.caption("Canaries keep their image but get a deliberately wrong label before the original "
                   "model is trained. Only memorization can fit a wrong label, so a model that still "
                   "prefers it remembers — by construction.")
