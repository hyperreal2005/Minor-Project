"""T.R.A.C.E. dashboard: a visualization layer over the completed study.

    venv\\Scripts\\streamlit.exe run dashboard\\app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="T.R.A.C.E. dashboard", page_icon=":material/fact_check:",
                   layout="wide", initial_sidebar_state="expanded")

import data as D  # noqa: E402
import nav  # noqa: E402

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1400px;}
    div[data-testid="stMetricValue"] {font-size: 2.1rem;}
    .claim {font-size: 1.35rem; font-weight: 600; margin: 0.2rem 0 0.6rem 0;}
    .muted {color: #52514e;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    m = D.meta()
    st.caption(f"Data snapshot {m.get('built', '?')} · commit {m.get('commit', '?')}")
    with st.expander("Glossary", expanded=False):
        st.markdown(
            "- **Original (M0)** — the model trained on everything, forget set included.\n"
            "- **Retrain** — trained from scratch *without* the forget set: the right answer.\n"
            "- **Band** — the range genuine retrains occupy on a metric (95.45% prediction "
            "interval).\n"
            "- **Flagged** — outside the band: *unlike every retrain*.\n"
            "- **Native rule** — an audit's own pass/fail rule, with no retrains.\n"
            "- **G** — distance from the retrains as a share of the original's: 0 = retrain, "
            "1 = original, > 1 = beyond it.\n"
            "- **Damaged** — test accuracy more than 5 points below the retrains.\n"
            "- **Marked** — the original can't be told from a retrain on that metric, so it "
            "isn't scored.\n"
            "- **Canary** — a training image given a deliberately wrong label.")

nav.navigation().run()
