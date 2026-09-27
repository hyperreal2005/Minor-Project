"""The shared-initialisation diagnostic, on constructed caches where the answer is known."""

import numpy as np
import pytest

from forgetcheck.audits.representation import linear_cka
from forgetcheck.calibrate.twins import feature_cka, twin_effect
from forgetcheck.registry import ArtifactStore, run_id

COND, N, K, D = "rand-500", 80, 10, 16


def test_feature_space_cka_equals_the_gram_form():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    y = 0.6 * x + 0.4 * y
    assert feature_cka(x, y) == pytest.approx(linear_cka(x, y), rel=1e-9)


def _store(tmp_path, *, twin: bool, n_oracles: int = 5):
    """Five oracles; one unlearned model per seed, built either as a near-copy of its same-seed
    oracle (a strong twin effect) or independently (none)."""
    rng = np.random.default_rng(1)
    st = ArtifactStore(tmp_path)
    oracles = {}
    for s in range(n_oracles):
        logits, acts = rng.normal(size=(N, K)) * 2, rng.normal(size=(N, D))
        oracles[s] = (logits, acts)
        rid = run_id(role="oracle", forget=COND, seed=s, seed_kind="train")
        st.save_outputs(rid, {"forget": logits}, {"forget": np.arange(N) % K}, forget_id=COND)
        st.save_activations(rid, {"layer4": acts}, probe_ids=np.arange(N))
    for s in range(5):
        if twin and s in oracles:
            logits = oracles[s][0] + rng.normal(scale=0.05, size=(N, K))
            acts = oracles[s][1] + rng.normal(scale=0.05, size=(N, D))
        else:
            logits, acts = rng.normal(size=(N, K)) * 2, rng.normal(size=(N, D))
        rid = f"c10r18__unlearn__{COND}__finetune__train{s}"
        st.save_outputs(rid, {"forget": logits}, {"forget": np.arange(N) % K}, forget_id=COND)
        st.save_activations(rid, {"layer4": acts}, probe_ids=np.arange(N))
    return st


def _by_metric(rows):
    return {r["metric"]: r for r in rows}


def test_a_near_copy_of_the_twin_is_detected_on_both_metrics(tmp_path):
    rows = _by_metric(twin_effect(_store(tmp_path, twin=True), forget_id=COND,
                                  train_seeds=range(5), methods=["finetune"]))
    assert rows["js_forget"]["twin_advantage_sd"] > 5
    assert rows["cka_layer4"]["twin_advantage_sd"] > 5
    assert rows["js_forget"]["twin_mean"] < rows["js_forget"]["other_mean"]
    assert rows["js_forget"]["ensemble_shift"] > 0


def test_independent_models_show_no_twin_advantage(tmp_path):
    rows = _by_metric(twin_effect(_store(tmp_path, twin=False), forget_id=COND,
                                  train_seeds=range(5), methods=["finetune"]))
    assert abs(rows["js_forget"]["twin_advantage_sd"]) < 2
    assert abs(rows["cka_layer4"]["twin_advantage_sd"]) < 2


def test_too_few_cached_oracles_is_skipped_not_an_error(tmp_path):
    assert twin_effect(_store(tmp_path, twin=True, n_oracles=2), forget_id=COND,
                       train_seeds=range(5), methods=["finetune"]) == []


def test_an_unreadable_cache_file_is_skipped(tmp_path):
    st = _store(tmp_path, twin=True)
    st.outputs_path(f"c10r18__unlearn__{COND}__finetune__train0").write_bytes(b"")
    rows = _by_metric(twin_effect(st, forget_id=COND, train_seeds=range(5), methods=["finetune"]))
    assert rows["js_forget"]["n_twin_pairs"] == 4
