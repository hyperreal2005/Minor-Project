"""Every dashboard page renders from the committed snapshot without an exception."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("streamlit")
pytest.importorskip("plotly")

DASH = Path(__file__).resolve().parents[1] / "dashboard"
VIEWS = ["home", "tour", "inspector", "validity", "agreement", "methods", "relearning",
         "forgetsets", "lab", "datapage"]


@pytest.mark.parametrize("view", VIEWS)
def test_page_renders(view):
    from streamlit.testing.v1 import AppTest

    if not (DASH / "data" / "flags.parquet").is_file():
        pytest.skip("no dashboard snapshot; run dashboard/prepare_data.py")
    code = (f"import sys\nsys.path.insert(0, r'{DASH}')\nfrom views import {view}\n"
            f"{view}.render()\n")
    at = AppTest.from_string(code, default_timeout=180)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_headline_numbers_reproduce_the_results():
    """The landing page's numbers are computed, and must match RESULTS.md."""
    import sys

    sys.path.insert(0, str(DASH))
    import stats

    k = stats.kpis()
    assert (k["native_k"], k["native_n"]) == (17, 17)
    assert k["fpr"] == {"five": (14, 595), "high": (11, 260)}
    assert (k["canary_perfect"], k["canary_n"], k["canary_perfect_unguarded"]) == (23, 39, 4)
    assert (k["retrain_like"], k["retrain_like_n"]) == (0, 175)
    assert round(k["h3_tau"], 2) == 0.64


def test_the_band_rule_maths():
    import sys

    sys.path.insert(0, str(DASH))
    import stats

    # A prediction interval keeps its nominal rate at any n; mean +- 2 sd does not.
    assert abs(stats.naive_fpr(1000, 2.0) - 0.0455) < 0.002
    assert stats.naive_fpr(4, 2.0) > 3 * 0.0455
    assert stats.pi_multiplier(4, 0.9545) > 3.5


def test_per_retrain_verdicts_reproduce_the_published_rates():
    """Step 2 of the tour judges every retrain itself; it must give the calibration's counts."""
    import sys

    sys.path.insert(0, str(DASH))
    import data as D
    import stats

    if not (DASH / "data" / "candidates.parquet").is_file():
        pytest.skip("no per-retrain values in the snapshot")
    c, v = D.table("candidates"), D.table("validity")
    for cond, n in D.COND_SIZE.items():
        rr = c[(c["forget_id"] == cond) & (c["metric"] == "mia_auc_pop")
               & (c["probe_set"] == "forget") & c["band_member"]]
        vr = v[(v["forget_id"] == cond) & (v["metric"] == "mia_auc_pop")
               & (v["probe_set"] == "forget")].iloc[0]
        native = sum(stats.auc_above_chance(x, n) for x in rr["value"])
        loo = sum(stats.loo_flag(rr["value"].to_numpy(), i, "closer_to_oracle")
                  for i in range(len(rr)))
        assert native == int(vr["native_fpr_k"]), cond
        assert loo == int(vr["calibrated_fpr_k"]), cond
