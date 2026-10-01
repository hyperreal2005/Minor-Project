"""Stage 8's figures, drawn from the calibration and analysis tables alone.

One finding per figure, each traceable to a table (implementation plan §8: no hand-edited
values). Static, for the paper: PDF for print, PNG for review.

Visual rules, fixed here so every figure reads as one system: one-hue sequential scale for
magnitude, blue <-> gray <-> red for polarity (red negative, blue positive -- the pair passes the
CVD and contrast checks), hairline recessive grids, a surface gap between heatmap cells rather
than borders, cell text in ink chosen by the cell's luminance, never in a data colour. Wherever
an interval decides the reading, a hollow marker says "the interval crosses the null", so
significance is never carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["make_figures"]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, GAP = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

COND_ORDER = ["canary-500", "mem-high-3000", "mem-med-3000", "mem-low-3000",
              "rand-500", "rand-2500", "rand-3000", "rand-5000"]
COND_LABEL = {c: c.replace("-3000", "") if c.startswith("mem") else c.replace("-500", "")
              if c.startswith("canary") else c for c in COND_ORDER}
FAMILY_ORDER = ["behavior", "privacy_weak", "privacy_strong", "representation",
                "reversibility", "independence"]
FAMILY_LABEL = {"behavior": "behaviour", "privacy_weak": "population MIA",
                "privacy_strong": "RMIA", "representation": "representation",
                "reversibility": "relearning", "independence": "SDE"}


def _style():
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _cmap_sequential():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("fc_seq", SEQUENTIAL).with_extremes(bad=GAP)


def _cmap_diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("fc_div", [RED, NEUTRAL, BLUE]).with_extremes(
        bad=GAP)


def _ink_for(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.5 else INK


def _heatmap(ax, values, *, cmap, vmin, vmax, labels=None, bold=None, missing="--"):
    """Cells with a surface gap between them; text in ink chosen by the cell's luminance."""
    from matplotlib.colors import Normalize

    v = np.asarray(values, dtype=float)
    norm = Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(np.ma.masked_invalid(v), cmap=cmap, norm=norm,
                         edgecolors=GAP, linewidth=1.5)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            x, y = j + 0.5, i + 0.5
            if not np.isfinite(v[i, j]):
                ax.text(x, y, missing, ha="center", va="center", color=MUTED, fontsize=6.5)
                continue
            text = labels[i][j] if labels is not None else f"{v[i, j]:.2f}"
            weight = "bold" if bold is not None and bold[i][j] else "normal"
            ax.text(x, y, text, ha="center", va="center", fontsize=6.5, fontweight=weight,
                    color=_ink_for(cmap(norm(v[i, j]))))
    ax.set_xlim(0, v.shape[1])
    ax.set_ylim(v.shape[0], 0)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return mesh


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _title(fig, title: str, subtitle: str = "") -> None:
    """Title and subtitle above everything already drawn -- measured, so rotated or top-side
    tick labels can never collide with them -- left-aligned to the content, wrapped to it."""
    import textwrap

    # Measured in inches, placed in figure fractions: a tight-cropped save shifts artists that
    # live in figure coordinates but not ones placed in inches (dpi_scale_trans) -- the first
    # version put every title exactly one crop-offset away from where it was measured.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = fig.get_tightbbox(r)
    w_in, h_in = fig.get_size_inches()
    x = box.x0 / w_in
    y = (box.y1 + 0.05) / h_in
    if subtitle:
        width = max(40, int(box.width * 15.5))
        t = fig.text(x, y, textwrap.fill(subtitle, width), transform=fig.transFigure,
                     ha="left", va="bottom", fontsize=7, color=INK_2, linespacing=1.3)
        y += (t.get_window_extent(r).height / fig.dpi + 0.03) / h_in
    fig.text(x, y, title, transform=fig.transFigure, ha="left", va="bottom",
             fontsize=8.5, fontweight="bold", color=INK)


# --------------------------------------------------------------------------- the figures


NATIVE = [("mia_auc_pop", "population MIA, AUC > 0.5"),
          ("mia_tpr_at_fpr_pop", "population MIA, TPR at 1% FPR"),
          ("mia_auc_rmia", "RMIA, AUC > 0.5"),
          ("mia_tpr_at_fpr_rmia", "RMIA, TPR at 1% FPR"),
          ("sde_verdict", "SDE verdict")]


def fig_native_validity(validity, out: Path) -> list[Path]:
    """The audits' own decision rules, applied to retrains that never saw the forget set."""
    import matplotlib.pyplot as plt

    v = validity[validity["probe_set"] == "forget"].set_index(["metric", "forget_id"])
    conds = [c for c in COND_ORDER if c in set(validity["forget_id"])]
    vals, labs = [], []
    for metric, _ in NATIVE:
        row, lab = [], []
        for c in conds:
            if (metric, c) in v.index and v.loc[(metric, c), "native_fpr_n"] > 0:
                r = v.loc[(metric, c)]
                row.append(r["native_fpr"])
                lab.append(f"{int(r['native_fpr_k'])}/{int(r['native_fpr_n'])}")
            else:
                row.append(np.nan)
                lab.append("")
        vals.append(row)
        labs.append(lab)

    scored = validity[(validity["calibrated_fpr_n"].fillna(0) > 0)
                      & validity["probe_set"].isin(["forget", "layer4"])]
    if "retired" in scored:
        scored = scored[~scored["retired"].astype(bool)]
    hi = scored["forget_id"] == "mem-high-3000"
    k1, n1 = scored.loc[~hi, "calibrated_fpr_k"].sum(), scored.loc[~hi, "calibrated_fpr_n"].sum()
    k2, n2 = scored.loc[hi, "calibrated_fpr_k"].sum(), scored.loc[hi, "calibrated_fpr_n"].sum()

    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    mesh = _heatmap(ax, vals, cmap=_cmap_sequential(), vmin=0, vmax=1, labels=labs)
    ax.set_xticks(np.arange(len(conds)) + 0.5, [COND_LABEL[c] for c in conds])
    ax.set_yticks(np.arange(len(NATIVE)) + 0.5, [lab for _, lab in NATIVE])
    ax.xaxis.tick_top()
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("share of genuine retrains flagged", fontsize=6.5, color=INK_2)
    _title(fig, "The audits' own decision rules flag genuine retrains",
           f"Retrains never saw the forget set, so a valid rule flags about 5%. Calibrated "
           f"against the retrains instead, every audit together flags {k1 / n1:.1%} of them at "
           f"the five-retrain conditions ({int(k1)}/{int(n1)}) and {k2 / n2:.1%} at mem-high "
           f"({int(k2)}/{int(n2)}).")
    return _save(fig, out, "fig1_native_validity")


def fig_canary_degree(cc, registry, out: Path) -> list[Path]:
    """Does each audit's G rise with how much canary association a model actually kept?"""
    import matplotlib.pyplot as plt

    if cc is None or not len(cc):
        return []
    d = cc.dropna(subset=["tau"]).sort_values("tau").reset_index(drop=True)
    fam = [FAMILY_LABEL.get(registry[m].family, registry[m].family) for m in d["metric"]]
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(d) + 0.8))
    y = np.arange(len(d))
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.grid(axis="x", zorder=0)
    for i, r in d.iterrows():
        colour = BLUE if r["tau"] > 0 else RED
        lo, hi = r.get("tau_lo", np.nan), r.get("tau_hi", np.nan)
        clear = np.isfinite(lo) and (lo > 0 or hi < 0)
        if np.isfinite(lo):
            ax.plot([lo, hi], [i, i], color=colour, linewidth=1.2, solid_capstyle="round",
                    zorder=2)
        ax.plot(r["tau"], i, "o", markersize=4.5, markeredgewidth=1.2, color=colour,
                markerfacecolor=colour if clear else "white", zorder=3)
    ax.set_yticks(y, [f"{m}  ({f})" for m, f in zip(d["metric"], fam)])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Kendall tau between the audit's G and measured canary memory")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _title(fig, "Behaviour, representation and relearning track memory; membership "
                "inference does not",
           f"Healthy canary models (n = {int(d['n'].max())}: 5 methods x 5 seeds, so mostly an "
           "ordering of methods). Bars: 95% bootstrap interval; hollow: it crosses 0.")
    return _save(fig, out, "fig2_canary_degree")


def fig_method_ratios(me, out: Path) -> list[Path]:
    """The confirmatory models: each method's distance from the retrains, relative to
    fine-tune's, per audit."""
    import matplotlib.pyplot as plt

    from .agreement import PRIMARY

    if me is None or not len(me):
        return []
    m = me[me["term"].astype(str).str.startswith("method")]
    fams = [f for f in PRIMARY if PRIMARY[f] in set(m["metric"])]
    methods = sorted({t.split(": ", 1)[1] for t in m["term"]})
    fig, axes = plt.subplots(1, len(fams), figsize=(7.0, 1.7), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        part = m[m["metric"] == PRIMARY[fam]].set_index("term")
        ax.axvline(1, color=AXIS, linewidth=0.8, zorder=1)
        ax.grid(axis="x", zorder=0)
        for i, meth in enumerate(methods):
            key = f"method: {meth}"
            if key not in part.index:
                continue
            r = part.loc[key]
            lo, mid, hi = np.exp(r["lo"]), np.exp(r["coef"]), np.exp(r["hi"])
            clear = lo > 1 or hi < 1
            ax.plot([lo, hi], [i, i], color=BLUE, linewidth=1.2, solid_capstyle="round",
                    zorder=2)
            ax.plot(mid, i, "o", markersize=4.5, markeredgewidth=1.2, color=BLUE,
                    markerfacecolor=BLUE if clear else "white", zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4)
        ax.set_xticks([0.5, 1, 2], ["x1/2", "x1", "x2"])
        ax.minorticks_off()
        ax.set_title(FAMILY_LABEL[fam], color=INK, pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(methods)), methods)
    axes[0].set_ylim(len(methods) - 0.5, -0.5)
    fig.supxlabel("distance from the retrains, relative to fine-tune (log G ratio)",
                  fontsize=7, color=INK_2, y=-0.06)
    _title(fig, "Which method looks most like a retrain depends on the audit",
           "log G ~ method + condition + (1 | seed); left of x1 = nearer a retrain than "
           "fine-tune. Bars: 95% CI; hollow: it crosses x1.")
    return _save(fig, out, "fig3_method_ratios")


def fig_agreement_by_condition(corr, out: Path) -> list[Path]:
    """Per-condition tau for the registered pairs: where do the audit families agree?"""
    import matplotlib.pyplot as plt

    per = corr[~corr["scope"].isin(["pooled", "within"])]
    if not len(per):
        return []
    pairs = list(dict.fromkeys(per["pair"]))
    conds = [c for c in COND_ORDER if c in set(per["scope"])]
    lab = {f: FAMILY_LABEL[f] for f in FAMILY_LABEL}
    vals, labels, bold = [], [], []
    for p in pairs:
        row, lr, br = [], [], []
        for c in conds:
            r = per[(per["pair"] == p) & (per["scope"] == c)]
            t = float(r["tau"].iloc[0]) if len(r) and "tau" in r else np.nan
            row.append(t)
            lr.append(f"{t:+.2f}" if t == t else "")
            lo = float(r["tau_lo"].iloc[0]) if len(r) and "tau_lo" in r else np.nan
            hi = float(r["tau_hi"].iloc[0]) if len(r) and "tau_hi" in r else np.nan
            br.append(bool(np.isfinite(lo) and (lo > 0 or hi < 0)))
        vals.append(row)
        labels.append(lr)
        bold.append(br)
    names = [" vs ".join(lab.get(x, x) for x in p.split(" vs ")) for p in pairs]
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    mesh = _heatmap(ax, vals, cmap=_cmap_diverging(), vmin=-1, vmax=1, labels=labels,
                    bold=bold, missing="marked")
    ax.set_xticks(np.arange(len(conds)) + 0.5, [COND_LABEL[c] for c in conds])
    ax.set_yticks(np.arange(len(pairs)) + 0.5, names)
    ax.xaxis.tick_top()
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("Kendall tau of G", fontsize=6.5, color=INK_2)
    _title(fig, "Audits agree where memorization is high, and invert at the canary",
           "Instance-level Kendall tau per condition (n ~25 each); bold: the 95% bootstrap "
           "interval excludes 0.")
    return _save(fig, out, "fig4_agreement_by_condition")


def fig_beyond_m0(summary, out: Path) -> list[Path]:
    """Where unlearning leaves models farther from a retrain than the original model was."""
    import matplotlib.pyplot as plt

    from .agreement import PRIMARY

    s = summary[summary["metric"].isin(PRIMARY.values())]
    if not len(s):
        return []
    conds = [c for c in COND_ORDER if c in set(s["forget_id"])]
    vals, labels = [], []
    for fam, metric in PRIMARY.items():
        part = s[s["metric"] == metric]
        row, lr = [], []
        for c in conds:
            pc = part[(part["forget_id"] == c) & part["share_beyond_m0"].notna()]
            share = (float(np.average(pc["share_beyond_m0"], weights=pc["n"]))
                     if len(pc) else np.nan)
            row.append(share)
            lr.append(f"{share:.0%}" if share == share else "")
        vals.append(row)
        labels.append(lr)
    fig, ax = plt.subplots(figsize=(6.0, 1.75))
    mesh = _heatmap(ax, vals, cmap=_cmap_sequential(), vmin=0, vmax=1, labels=labels,
                    missing="marked")
    ax.set_xticks(np.arange(len(conds)) + 0.5, [COND_LABEL[c] for c in conds])
    ax.set_yticks(np.arange(len(PRIMARY)) + 0.5,
                  [f"{FAMILY_LABEL[f]} ({m})" for f, m in PRIMARY.items()])
    ax.xaxis.tick_top()
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("share with G > 1", fontsize=6.5, color=INK_2)
    _title(fig, "Where unlearning moves models farther from a retrain than the original was",
           "Share of healthy unlearned models with G > 1 per condition. marked: M0 barely "
           "differs from a retrain there (inside the retrains' noise, or under 2 points), so G "
           "is not scored.")
    return _save(fig, out, "fig5_beyond_m0")


def fig_matrix(matrix, registry, out: Path) -> list[Path]:
    """Within-condition tau between every pair of analysed metrics, grouped by family."""
    import matplotlib.pyplot as plt

    if matrix is None or not len(matrix) or "tau_within" not in matrix:
        return []
    metrics = sorted(set(matrix["metric_a"]) | set(matrix["metric_b"]),
                     key=lambda m: (FAMILY_ORDER.index(registry[m].family)
                                    if registry[m].family in FAMILY_ORDER else 99, m))
    k = len(metrics)
    pos = {m: i for i, m in enumerate(metrics)}
    M = np.full((k, k), np.nan)
    for r in matrix.itertuples():
        i, j = sorted((pos[r.metric_a], pos[r.metric_b]))
        M[j, i] = r.tau_within  # lower triangle only
    M = M[1:, :-1]  # the strict lower triangle: drop the empty first row and last column
    rows, cols = metrics[1:], metrics[:-1]

    def short(v):
        return "" if not np.isfinite(v) else f"{v:.1f}".replace("0.", ".").replace("1.0", "1")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    labels = [[short(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    mesh = _heatmap(ax, M, cmap=_cmap_diverging(), vmin=-1, vmax=1, labels=labels, missing="")
    ax.set_xticks(np.arange(len(cols)) + 0.5, cols, rotation=55, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)) + 0.5, rows)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("within-condition Kendall tau", fontsize=6.5, color=INK_2)
    _title(fig, "Agreement between every pair of audit metrics",
           "n-weighted mean of per-condition tau of G, metrics grouped by audit family.")
    return _save(fig, out, "fig6_matrix")


def fig_h3(flags, h3, out: Path) -> list[Path]:
    """H3: per condition, each model's representation-to-M0 G against its relearning G."""
    import matplotlib.pyplot as plt

    if flags is None or h3 is None or not len(h3):
        return []
    f = flags[~flags["damaged"].fillna(False).astype(bool)]
    rep = f[(f["metric"] == "cka_to_original") & (f["probe_set"] == "layer4")]
    rel = f[(f["metric"] == "relearn_auc") & (f["probe_set"] == "forget")]
    key = ["run_id", "forget_id"]
    d = rep[key + ["oracle_gap"]].merge(rel[key + ["oracle_gap"]], on=key,
                                        suffixes=("_rep", "_rel")).dropna()
    conds = [c for c in COND_ORDER if c in set(d["forget_id"])]
    if not conds:
        return []
    per = h3.set_index("scope")
    fig, axes = plt.subplots(1, len(conds), figsize=(1.25 * len(conds) + 0.4, 1.8),
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        part = d[d["forget_id"] == c]
        ax.grid(zorder=0)
        ax.plot(part["oracle_gap_rep"], part["oracle_gap_rel"], "o", markersize=3.6,
                color=BLUE, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        stat = ""
        if c in per.index and "tau" in per and per.loc[c, "tau"] == per.loc[c, "tau"]:
            r = per.loc[c]
            stat = f"\n\u03c4 {r['tau']:+.2f} [{r['tau_lo']:+.2f}, {r['tau_hi']:+.2f}]"
        # The statistic rides in the panel title, never on top of the points.
        ax.set_title(COND_LABEL[c] + stat, color=INK, pad=3, fontsize=7, linespacing=1.4)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("relearning G")
    fig.supxlabel("representation-to-M0 G (0 = a retrain's similarity to M0, 1 = M0)",
                  fontsize=7, color=INK_2, y=-0.08)
    w = per.loc["within"] if "within" in per.index else None
    within = (f"within-condition tau {w['tau']:+.2f}" if w is not None and "tau" in w
              and w["tau"] == w["tau"] else "")
    _title(fig, "H3: do models whose representations stay closer to M0 relearn more like M0?",
           f"Each point is a healthy unlearned model; H3 predicts a rising cloud. {within}.")
    return _save(fig, out, "fig7_h3")


def make_figures(analysis_dir, calibration_dir, out_dir) -> list[Path]:
    import pandas as pd

    from ..registry.metrics import default_registry

    _style()
    an, cal, out = Path(analysis_dir), Path(calibration_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def read(d, name):
        p = d / f"{name}.parquet"
        return pd.read_parquet(p) if p.is_file() else None

    reg = default_registry()
    paths: list[Path] = []
    validity = read(cal, "validity")
    if validity is not None:
        paths += fig_native_validity(validity, out)
    paths += fig_canary_degree(read(an, "canary_continuous"), reg, out)
    paths += fig_method_ratios(read(an, "mixed_effects"), out)
    corr = read(an, "correlations")
    if corr is not None:
        paths += fig_agreement_by_condition(corr, out)
    summary = read(an, "method_summary")
    if summary is not None:
        paths += fig_beyond_m0(summary, out)
    paths += fig_matrix(read(an, "correlation_matrix"), reg, out)
    paths += fig_h3(read(cal, "flags"), read(an, "h3"), out)
    return paths
