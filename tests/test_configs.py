"""Configs stay consistent with the code they configure.

Config drift is quiet: a layer renamed in one place and not the other produces an audit that
runs, writes plausible numbers, and measures the wrong thing. These checks make the coupling
explicit so it fails at test time instead.
"""

from pathlib import Path

import pytest
import yaml

from forgetcheck.data.forget_sets import PRIMARY_CONDITION, all_specs
from forgetcheck.models.resnet import FEATURE_LAYERS
from forgetcheck.registry import default_registry

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def base():
    return yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audits():
    return yaml.safe_load((CONFIGS / "audits.yaml").read_text(encoding="utf-8"))


def test_all_configs_parse():
    for name in ("base", "audits", "metrics"):
        assert yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


class TestRepresentation:
    def test_layers_match_the_model_taps(self, audits):
        assert set(audits["representation"]["layers"]) == set(FEATURE_LAYERS)

    def test_registry_declares_every_layer_as_a_probe_set(self, audits):
        reg = default_registry()
        declared = set(reg["cka_linear"].probe_sets)
        assert set(audits["representation"]["layers"]) <= declared

    def test_configured_measures_are_registered_metrics(self, audits):
        reg = default_registry()
        for key in ("primary_measure", "second_measure"):
            name = audits["representation"][key]
            assert name in reg, f"{key}={name!r} is not in the metric registry"
            assert reg[name].family == "representation"

    def test_two_distinct_measures(self, audits):
        # CKA alone is not a safe sole instrument: its values move without corresponding change
        # in functional behaviour [26]. A second, mechanistically different measure is required.
        assert audits["representation"]["primary_measure"] != audits["representation"]["second_measure"]


class TestSeeds:
    def test_streams_do_not_overlap(self, base):
        # Mixing streams silently couples effects the analysis assumes independent.
        train = set(base["seeds"]["train"])
        oracle = set(base["seeds"]["oracle"])
        assert not train & oracle
        assert base["seeds"]["selection"] not in train | oracle
        assert base["seeds"]["audit"] not in train | oracle

    def test_holdout_oracles_are_ensemble_members(self, base):
        assert set(base["seeds"]["oracle_holdout"]) <= set(base["seeds"]["oracle"])

    def test_holdout_excluded_from_the_band(self, base, audits):
        # A probe that helped define the band cannot test the band.
        band = set(audits["calibration"]["band_oracle_seeds"])
        probe = set(audits["calibration"]["probe_oracle_seeds"])
        assert not band & probe
        assert probe == set(base["seeds"]["oracle_holdout"])
        assert band | probe == set(base["seeds"]["oracle"])

    def test_paired_oracle_seeds_match_train_seeds(self, base):
        # A paired oracle shares its M0's initialisation; that is what makes the pair a
        # controlled comparison rather than two unrelated models.
        assert base["oracles"]["paired_seeds"] == base["seeds"]["train"]


class TestConditions:
    def test_primary_condition_exists(self, base):
        assert base["oracles"]["primary_condition"] in all_specs()
        assert base["oracles"]["primary_condition"] == PRIMARY_CONDITION.forget_id

    def test_selection_seed_matches_the_spec_default(self, base):
        for spec in all_specs().values():
            assert spec.selection_seed == base["seeds"]["selection"]


class TestRelearning:
    def test_arms_cover_both_anchors(self, audits):
        # Without M0 and the oracle there is no scale, and a raw recovery curve conflates
        # "retained structure" with "started closer".
        arms = set(audits["relearning"]["arms"])
        assert {"method", "oracle", "original"} <= arms

    def test_eval_steps_start_at_zero_and_increase(self, audits):
        steps = audits["relearning"]["eval_steps"]
        assert steps[0] == 0, "step 0 is the pre-relearning baseline"
        assert steps == sorted(steps) and len(set(steps)) == len(steps)


class TestAgreement:
    def test_unit_of_analysis_is_the_model_instance(self, audits):
        # Ranking six methods cannot reach significance (min two-sided p = 0.0028 at n=6;
        # 0.0833 at the four of v1.0). The correlation must run over model instances.
        assert audits["agreement"]["unit"] == "model_instance"

    def test_rank_tables_are_descriptive_only(self, audits):
        assert audits["agreement"]["rank_tables"] == "descriptive_only"


def test_shadow_count_matches_the_rmia_reference_count(base, audits):
    assert base["shadows"]["count"] == audits["privacy"]["rmia"]["n_references"]
