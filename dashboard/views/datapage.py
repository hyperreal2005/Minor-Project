"""The tables behind every view, downloadable, and where each record came from."""
import streamlit as st

import data as D

DESCRIBE = {
    "flags": "Every unlearned model's value, verdicts and G on every audit metric",
    "bands": "The retrains' band for every (forget set, metric, probe)",
    "validity": "False-positive rate, power and discriminability for every audit cell",
    "candidates": "Every retrain's and original's own audit values",
    "canary": "Canary ground truth for the 40 canary models",
    "canary_validity": "Each audit scored against the canary ground truth",
    "utility": "Test accuracy against the retrains, and the damage guard",
    "training": "Training-time test, retain and forget accuracy of every model",
    "correlations": "Agreement between audit families: pooled, per condition, within",
    "mixed_effects": "The confirmatory mixed models",
    "method_summary": "Per method and forget set: medians of G, share beyond the original",
    "h3": "H3: representation-to-original vs relearning",
    "signed_gaps": "Signed G per healthy model and metric",
    "curves": "Relearning curves, step by step",
    "provenance": "When every audit's records were written, and by which commit",
}


def render():
    st.title("Data & provenance")
    m = D.meta()
    st.markdown(f"Snapshot built {m.get('built', '?')} from commit `{m.get('commit', '?')}`. Every "
                "chart in this dashboard reads these tables; nothing is typed in by hand. The full "
                "write-up is `docs/RESULTS.md`.")
    names = [n for n in DESCRIBE if D.has(n)]
    name = st.selectbox("Table", names, format_func=lambda n: f"{n} — {DESCRIBE[n]}")
    df = D.table(name)
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df, width="stretch", height=480)
    st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name}.csv", mime="text/csv", icon=":material/download:")
    if name != "provenance" and D.has("provenance"):
        with st.expander("Provenance: when each audit's records were written"):
            st.dataframe(D.table("provenance"), width="stretch")
