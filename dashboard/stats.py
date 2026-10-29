"""Computations shared by several views. Every headline is computed from the tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sps

import data as D


@st.cache_data(show_spinner=False)
def pooled_calibrated_fpr() -> dict:
    """The calibrated audits' pooled false-positive rate, split as the paper reports it."""
    v = D.table("validity")
    s = v[(v["calibrated_fpr_n"].fillna(0) > 0) & v["probe_set"].isin(["forget", "layer4"])]
    s = s[~s["retired"].astype(bool)]
    hi = s["forget_id"] == "mem-high-3000"
    k1, n1 = s.loc[~hi, "calibrated_fpr_k"].sum(), s.loc[~hi, "calibrated_fpr_n"].sum()
    k2, n2 = s.loc[hi, "calibrated_fpr_k"].sum(), s.loc[hi, "calibrated_fpr_n"].sum()
    return {"five": (int(k1), int(n1)), "high": (int(k2), int(n2))}


@st.cache_data(show_spinner=False)
def kpis() -> dict:
    v = D.table("validity")
    r = v[(v["forget_id"] == "mem-high-3000") & (v["metric"] == "mia_auc_pop")
          & (v["probe_set"] == "forget")].iloc[0]
    cv = D.table("canary_validity")
    cv = cv[~cv["retired"].astype(bool)]
    h3 = D.table("h3").set_index("scope")
    # Models outside the negative control that pass every scored primary audit.
    f = D.healthy_flags()
    keep = [D.PRIMARY.get(D.METRICS[m][1]) == (m, p) if m in D.METRICS else False
            for m, p in zip(f["metric"], f["probe_set"])]
    sel = f[keep]
    sel = sel[sel["oracle_gap"].notna() & (sel["forget_id"] != "mem-low-3000")]
    passed = sel.groupby("run_id")["calibrated_flag"].agg(lambda s: not s.astype(bool).any())
    return {
        "native_k": int(r["native_fpr_k"]), "native_n": int(r["native_fpr_n"]),
        "fpr": pooled_calibrated_fpr(),
        "canary_perfect": int((cv["ba_guarded"] >= 0.9999).sum()), "canary_n": len(cv),
        "canary_perfect_unguarded": int((cv["balanced_accuracy"] >= 0.9999).sum()),
        "retrain_like": int(passed.sum()), "retrain_like_n": int(passed.size),
        "h3_tau": float(h3.loc["within", "tau"]),
    }


def kendall(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(sps.kendalltau(x[ok], y[ok]).statistic)


def auc_above_chance(auc: float, n: int, alpha: float = 0.05) -> bool:
    """The population attack's native rule: AUC significantly above 0.5, one-sided, with the
    null standard error for n members against n non-members (as calibrate/native.py)."""
    se = np.sqrt((2 * n + 1) / (12 * n * n))
    return bool((auc - 0.5) / se > sps.norm.ppf(1 - alpha))


def loo_flag(values, i: int, direction: str, coverage: float = 0.9545) -> bool:
    """Retrain i against a prediction band built from the other retrains (leave-one-out)."""
    rest = np.delete(np.asarray(values, float), i)
    if len(rest) < 3:
        return False
    k = pi_multiplier(len(rest), coverage)
    lo, hi = rest.mean() - k * rest.std(ddof=1), rest.mean() + k * rest.std(ddof=1)
    return D.flag_value(float(values[i]), lo, hi, direction)


def pi_multiplier(n: int, coverage: float) -> float:
    """k in mean +- k*s for a prediction interval for one new retrain."""
    if n < 2:
        return float("nan")
    return float(sps.t.ppf((1 + coverage) / 2, n - 1) * np.sqrt(1 + 1 / n))


def naive_fpr(n: int, k: float = 2.0, two_sided: bool = True) -> float:
    """Exact false-positive rate of a mean +- k*sd band built from n retrains, for a new retrain:
    (X - mean) / (s * sqrt(1 + 1/n)) is t with n - 1 degrees of freedom."""
    if n < 2:
        return float("nan")
    q = k / np.sqrt(1 + 1 / n)
    tail = 1 - sps.t.cdf(q, n - 1)
    return float(2 * tail if two_sided else tail)


@st.cache_data(show_spinner=False)
def wide_values(cond: str) -> pd.DataFrame:
    """Every unlearned model at a condition: one row, one column per (metric, probe)."""
    f = D.table("flags")
    f = f[f["forget_id"] == cond]
    w = f.pivot_table(index=["run_id", "method", "train_seed"], columns=["metric", "probe_set"],
                      values="value")
    return w


def condition_select(key: str, *, default: str = "mem-high-3000", options=None,
                     label: str = "Forget set") -> str:
    opts = options or D.CONDITIONS
    # A preset written to session state (e.g. by a "show me" button) wins; passing an index as
    # well would make Streamlit warn about two sources for one widget's value.
    if key in st.session_state and st.session_state[key] not in opts:
        del st.session_state[key]
    kw = {} if key in st.session_state else {"index": opts.index(default) if default in opts else 0}
    return st.selectbox(label, opts, format_func=lambda c: f"{D.COND_LABEL[c]} — {D.COND_BLURB[c]}",
                        key=key, **kw)
