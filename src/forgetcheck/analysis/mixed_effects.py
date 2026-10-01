"""Stage 8 confirmatory tests (master reference §16.2).

``G ~ method + forget_condition + (1 | train_seed)``, one model per primary metric, on the
normalized oracle gap. The random intercept is the seed: the five unlearned models of a method at
a condition share their M0, so they are repeated measures, not independent draws.

The reference levels are chosen so the coefficients answer the design's two axes directly.
Method: fine-tune, the simplest baseline. Condition: ``rand-3000``, so the three ``mem-*-3000``
coefficients are the difficulty axis at fixed size (H4) and the other random sizes are the size
axis.

H4 also gets a test of its own: on the difficulty axis alone (the three strata and rand-3000),
does adding the method x condition interaction improve the fit -- do methods differ by different
amounts at different memorization levels? A likelihood-ratio test between the two models,
both fitted by maximum likelihood (REML likelihoods are not comparable across fixed effects).
"""

from __future__ import annotations

import warnings

__all__ = ["fit_main", "interaction_test", "DIFFICULTY_AXIS"]

DIFFICULTY_AXIS = ("mem-low-3000", "mem-med-3000", "mem-high-3000", "rand-3000")
METHOD_REF = "finetune"
CONDITION_REF = "rand-3000"


def _frame(G, metric, conditions=None):
    d = G[metric].dropna().rename("G").reset_index()
    if conditions is not None:
        d = d[d["forget_id"].isin(conditions)]
    d["train_seed"] = d["train_seed"].astype(int).astype(str)
    return d


def _ref(levels, preferred):
    levels = sorted(set(levels))
    return preferred if preferred in levels else levels[0]


#: Tried in turn; the best finite maximum wins. statsmodels' default, L-BFGS, can walk into a
#: degenerate point when the seed variance is at zero: on the realistic fixture it reported a
#: log-likelihood of +inf, an intercept of exactly 0 and a standard error of 4e5 -- printed as
#: a result. Powell and Nelder-Mead found the true maximum there (the OLS intercept, as they
#: must when the random effect vanishes).
OPTIMIZERS = ("lbfgs", "powell", "nm")


def _fit(formula, data, *, reml):
    """The converged fit with the highest finite likelihood and finite standard errors."""
    import numpy as np
    import statsmodels.formula.api as smf

    best, best_notes, best_method = None, [], ""
    for method in OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = smf.mixedlm(formula, data, groups=data["train_seed"]).fit(
                    reml=reml, method=method)
            except Exception:  # noqa: BLE001 -- a failed optimizer is just not a candidate
                continue
        sane = (res.converged and np.isfinite(res.llf)
                and np.isfinite(np.asarray(res.bse_fe, float)).all())
        if sane and (best is None or res.llf > best.llf):
            best, best_method = res, method
            best_notes = sorted({type(w.message).__name__ for w in caught})
    if best is None:
        raise RuntimeError(f"no optimizer found a finite maximum for {formula!r}")
    return best, best_notes, best_method


def _term(name: str) -> str:
    """``C(method, Treatment('finetune'))[T.scrub]`` -> ``method: scrub``."""
    if name.startswith("C(") and "[T." in name:
        factor = name[2:name.index(",")] if "," in name else name[2:name.index(")")]
        level = name[name.index("[T.") + 3:-1]
        return f"{'condition' if factor == 'forget_id' else factor}: {level}"
    return name


def fit_main(G, metric) -> "pd.DataFrame":
    """Fixed effects with 95% intervals for one metric; one row per term."""
    import pandas as pd

    d = _frame(G, metric)
    if d["method"].nunique() < 2 or d["forget_id"].nunique() < 2 or d["train_seed"].nunique() < 2:
        return pd.DataFrame()
    m_ref, c_ref = _ref(d["method"], METHOD_REF), _ref(d["forget_id"], CONDITION_REF)
    formula = (f"G ~ C(method, Treatment('{m_ref}')) + C(forget_id, Treatment('{c_ref}'))")
    try:
        res, notes, method = _fit(formula, d, reml=True)
    except RuntimeError:
        return pd.DataFrame([{"metric": metric, "term": "no finite maximum", "n": int(len(d))}])
    ci = res.conf_int()
    rows = []
    for name in res.fe_params.index:
        rows.append({"metric": metric, "term": _term(name), "coef": float(res.fe_params[name]),
                     "lo": float(ci.loc[name, 0]), "hi": float(ci.loc[name, 1]),
                     "p": float(res.pvalues[name]), "n": int(len(d)),
                     "seed_var": float(res.cov_re.iloc[0, 0]),
                     "method_ref": m_ref, "condition_ref": c_ref,
                     "converged": bool(res.converged), "optimizer": method,
                     "warnings": ", ".join(notes)})
    return pd.DataFrame(rows)


def interaction_test(G, metric) -> dict:
    """H4: likelihood-ratio test of method x condition on the difficulty axis."""
    from scipy.stats import chi2

    d = _frame(G, metric, DIFFICULTY_AXIS)
    out = {"metric": metric, "n": int(len(d)), "conditions": int(d["forget_id"].nunique())}
    if d["method"].nunique() < 2 or d["forget_id"].nunique() < 2 or d["train_seed"].nunique() < 2:
        return out
    try:
        add, n1, _ = _fit("G ~ C(method) + C(forget_id)", d, reml=False)
        inter, n2, _ = _fit("G ~ C(method) * C(forget_id)", d, reml=False)
    except RuntimeError:
        out["note"] = "no finite maximum"
        return out
    df = len(inter.fe_params) - len(add.fe_params)
    lr = max(0.0, 2.0 * (inter.llf - add.llf))
    out.update(lr=float(lr), df=int(df), p=float(chi2.sf(lr, df)) if df > 0 else float("nan"),
               converged=bool(add.converged and inter.converged),
               warnings=", ".join(sorted(set(n1) | set(n2))))
    return out
