"""Shared constants, labels and cached loaders for the dashboard.

Every number the dashboard shows is read or computed from the snapshot in dashboard/data/
(built by prepare_data.py from the result tables). Nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
ASSETS = Path(__file__).resolve().parent / "assets"

# --------------------------------------------------------------------------- conditions
CONDITIONS = ["canary-500", "mem-high-3000", "mem-med-3000", "mem-low-3000",
              "rand-500", "rand-2500", "rand-3000", "rand-5000"]
COND_LABEL = {"canary-500": "canary", "mem-high-3000": "mem-high", "mem-med-3000": "mem-med",
              "mem-low-3000": "mem-low", "rand-500": "rand-500", "rand-2500": "rand-2500",
              "rand-3000": "rand-3000", "rand-5000": "rand-5000"}
COND_BLURB = {
    "canary-500": "500 images given a deliberately wrong label before training: residual memory "
                  "is measurable directly (the ground truth)",
    "mem-high-3000": "the 3,000 most memorized images: where an original and a retrain differ most "
                     "(the primary condition)",
    "mem-med-3000": "the 3,000 images with memorization nearest 0.5",
    "mem-low-3000": "the 3,000 least memorized images: nothing to forget (the negative control)",
    "rand-500": "500 random images (1% of the training set)",
    "rand-2500": "2,500 random images (5%)",
    "rand-3000": "3,000 random images: bridges the size and difficulty axes",
    "rand-5000": "5,000 random images (10%)",
}

# --------------------------------------------------------------------------- methods
METHODS = ["finetune", "l1sparse", "neggradplus", "salun", "scrub", "neggrad"]
HEALTHY_METHODS = METHODS[:5]
METHOD_LABEL = {"finetune": "Fine-tuning", "l1sparse": "L1-sparse", "neggradplus": "NegGrad+",
                "salun": "SalUn", "scrub": "SCRUB", "neggrad": "NegGrad (destroyed)"}
# Fixed categorical order from the validated palette (dataviz reference; passes the CVD and
# normal-vision checks on adjacent pairs). Three slots are below 3:1 on white, so every method
# also carries a marker symbol and a legend: colour never carries identity alone.
METHOD_COLOR = {"finetune": "#2a78d6", "l1sparse": "#eb6834", "neggradplus": "#1baf7a",
                "salun": "#eda100", "scrub": "#e87ba4", "neggrad": "#6250d6"}
METHOD_SYMBOL = {"finetune": "circle", "l1sparse": "square", "neggradplus": "diamond",
                 "salun": "triangle-up", "scrub": "cross", "neggrad": "x"}
METHOD_BLURB = {
    "finetune": "keep training the original on the retain set only",
    "l1sparse": "fine-tune on the retain set with an L1 penalty that sparsifies the weights",
    "neggradplus": "gradient ascent on the forget set while descending on the retain set "
                   "(forget loss capped at chance level)",
    "salun": "update only the 50% most forget-salient weights, with random labels on the forget set",
    "scrub": "teacher-student: stay close to the original on retain data, move away on forget data",
    "neggrad": "unbounded gradient ascent on the forget set: wrecks the model on purpose (a control)",
}

# --------------------------------------------------------------------------- colours
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RETRAIN = "#898781"
BAND_FILL = "rgba(137,135,129,0.22)"
ORIGINAL = "#0b0b0b"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

# --------------------------------------------------------------------------- audits and metrics
FAMILY_LABEL = {"behavior": "Behavior", "privacy_weak": "Population MIA",
                "privacy_strong": "RMIA", "representation": "Representation",
                "reversibility": "Relearning", "independence": "SDE"}
FAMILY_ORDER = list(FAMILY_LABEL)
PRIMARY = {"behavior": ("js_to_oracle", "forget"), "privacy_weak": ("mia_auc_pop", "forget"),
           "privacy_strong": ("mia_auc_rmia", "forget"), "representation": ("cka_linear", "layer4"),
           "reversibility": ("relearn_auc", "forget")}

# metric -> (label, family, plain-language description, proportion?)
METRICS = {
    "js_to_oracle": ("JS divergence to retrains", "behavior",
                     "Jensen-Shannon divergence between the model's predicted class probabilities and "
                     "the retrains' (0 = identical).", False),
    "pred_agreement": ("Prediction agreement", "behavior",
                       "Share of images on which the model predicts the same class as the retrains.", True),
    "logit_l2": ("Logit L2 distance", "behavior",
                 "L2 distance between the model's logits and the retrains'.", False),
    "js_to_original": ("JS divergence to original", "behavior",
                       "JS divergence to the model's own original: how far it moved from where it "
                       "started.", False),
    "mia_auc_pop": ("Population MIA AUC", "privacy_weak",
                    "One logistic attacker separating forget images from test images by loss, "
                    "confidence and entropy: AUC.", True),
    "mia_acc_pop": ("Population MIA accuracy", "privacy_weak",
                    "The same attacker's balanced accuracy.", True),
    "mia_tpr_at_fpr_pop": ("Population MIA TPR@1%FPR", "privacy_weak",
                           "Share of forget images caught when only 1% of test images are wrongly "
                           "accused.", True),
    "mia_auc_rmia": ("RMIA AUC", "privacy_strong",
                     "Per-example attack comparing the model with 32 reference models: AUC.", True),
    "mia_acc_rmia": ("RMIA accuracy", "privacy_strong", "RMIA's accuracy.", True),
    "mia_tpr_at_fpr_rmia": ("RMIA TPR@1%FPR", "privacy_strong",
                            "RMIA's true-positive rate at a 1% false-positive rate.", True),
    "cka_linear": ("Linear CKA", "representation",
                   "Similarity of internal activations to the retrains' (1 = identical geometry).", False),
    "cka_rbf": ("RBF-CKA", "representation",
                "CKA with an RBF kernel: the second representation measure.", False),
    "cka_to_original": ("CKA to original", "representation",
                        "Linear CKA to the model's own original: closeness to where it started "
                        "(the H3 measure).", False),
    "activation_l2": ("Activation L2 (retired)", "representation",
                      "Raw activation distance. Retired: it measures a shared initialisation, not "
                      "forgetting.", False),
    "relearn_auc": ("Relearning AUC", "reversibility",
                    "Average forget-set accuracy while relearning 500 reintroduced forget images "
                    "for 100 steps.", True),
    "relearn_t80": ("Relearning T80", "reversibility",
                    "Steps to recover 80% of the original's starting forget accuracy.", False),
    "relearn_norm": ("Normalized recovery", "reversibility",
                     "Relearning on a scale from the paired retrain (0) to the original (1).", False),
    "relearn_utility_drop": ("Relearning utility drop", "reversibility",
                             "Test-accuracy drop during relearning (the relearning guard).", False),
    "sde_hsic": ("SDE HSIC", "independence",
                 "Statistical dependence (HSIC) between the outputs on two halves of the forget "
                 "set.", False),
    "sde_margin": ("SDE margin", "independence",
                   "How much closer the forget set looks to in-training than to out-of-training "
                   "data.", False),
    "sde_verdict": ("SDE verdict", "independence",
                    "SDE's own verdict: does the forget set look in-training?", False),
}
ANALYSED = ["js_to_oracle", "pred_agreement", "logit_l2", "js_to_original", "mia_auc_pop",
            "mia_acc_pop", "mia_tpr_at_fpr_pop", "mia_auc_rmia", "mia_acc_rmia",
            "mia_tpr_at_fpr_rmia", "cka_linear", "cka_rbf", "cka_to_original", "relearn_auc",
            "relearn_norm", "relearn_t80", "sde_hsic", "sde_margin"]
DIRECTION = {}  # filled from the bands table: lower_better / higher_better / closer_to_oracle

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
                 "ship", "truck"]


def mlabel(metric: str) -> str:
    return METRICS.get(metric, (metric,))[0]


def default_probe(metric: str) -> str:
    return "layer4" if metric.startswith("cka") or metric == "activation_l2" else "forget"


# --------------------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    p = DATA / f"{name}.parquet"
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def has(name: str) -> bool:
    return (DATA / f"{name}.parquet").is_file()


@st.cache_data(show_spinner=False)
def meta() -> dict:
    p = DATA / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


@st.cache_data(show_spinner=False)
def gallery():
    p = DATA / "gallery_images.npy"
    if not p.is_file():
        return None, pd.DataFrame()
    return np.load(p), table("gallery")


@st.cache_data(show_spinner=False)
def memorization() -> np.ndarray:
    return np.load(DATA / "memorization.npy")


def healthy_flags() -> pd.DataFrame:
    f = table("flags")
    return f[~f["damaged"].fillna(False).astype(bool)]


@st.cache_data(show_spinner=False)
def cell(cond: str, metric: str, probe: str) -> dict:
    """The band and M0 for one (condition, metric, probe) cell; empty dict if absent."""
    b = table("bands")
    v = table("validity")
    key = (b["forget_id"] == cond) & (b["metric"] == metric) & (b["probe_set"] == probe)
    if not key.any():
        return {}
    row = b[key].iloc[0].to_dict()
    vr = v[(v["forget_id"] == cond) & (v["metric"] == metric) & (v["probe_set"] == probe)]
    if len(vr):
        vr = vr.iloc[0]
        gap = vr.get("m0_gap_raw", np.nan)
        row["m0"] = row["mean"] + gap if gap == gap else np.nan
        row["marked"] = _true(vr.get("low_discriminability")) or _true(vr.get("below_min_effect"))
        for k in ("native_rule", "native_fpr_k", "native_fpr_n", "native_tpr_k", "native_tpr_n",
                  "calibrated_fpr_k", "calibrated_fpr_n", "m0_tpr_k", "m0_tpr_n", "nominal_fpr",
                  "m0_gap_sd", "m0_gap_raw", "retired"):
            row[k] = vr.get(k)
    return row


def _true(x) -> bool:
    """A table cell as a boolean: None and NaN (no test made) are False."""
    try:
        return bool(x) and not (isinstance(x, float) and x != x)
    except (TypeError, ValueError):
        return False


def flag_value(value: float, lo: float, hi: float, direction: str) -> bool:
    """The calibration's flag rule: one-sided for a direction, two-sided for closer-to-retrains."""
    if not (lo == lo and hi == hi):
        return False
    if direction == "lower_better":
        return value > hi
    if direction == "higher_better":
        return value < lo
    return value < lo or value > hi


def guard_points() -> float:
    u = table("utility")
    return float(u["guard_pp"].iloc[0]) if len(u) else 5.0
