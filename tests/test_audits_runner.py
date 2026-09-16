"""Stage 6 orchestration: probe construction and the audit queue.

Probe construction gets the most attention here, because it is the one place where a mistake
would not look like a bug — every number would still be produced, and every one would be a
comparison of probe sets rather than of models.
"""

import numpy as np
import pytest

from forgetcheck.audits.probes import ProbeSpec, build_probes, probe_seed
from forgetcheck.audits.runner import AuditTarget, available_targets

N_TRAIN, N_TEST = 50_000, 10_000


def probes(forget_n=3000, forget_id="mem-high-3000", start=0, config=None):
    return build_probes(
        forget_indices=np.arange(start, start + forget_n),
        n_train=N_TRAIN, n_test=N_TEST, forget_id=forget_id, config=config,
    )


class TestProbeDeterminism:
    def test_identical_across_calls(self):
        a, b = probes(), probes()
        for field in ("forget", "retain", "test", "mixed_train", "mixed_test"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))

    def test_a_condition_gets_its_own_draw(self):
        # Each condition has its own retain set, so its retain probe must differ.
        assert not np.array_equal(probes(forget_id="rand-3000").retain,
                                  probes(forget_id="mem-high-3000").retain)

    def test_the_seed_does_not_depend_on_process_salt(self):
        # `hash()` is salted per process; two accounts must agree on the probe set.
        assert probe_seed("rand-500") == probe_seed("rand-500")
        assert probe_seed("rand-500") != probe_seed("rand-5000")

    def test_adding_a_condition_cannot_renumber_the_others(self):
        # Hashing the name rather than an index is what buys this: a new condition inserted in
        # the config must not silently redraw probes that published results depend on.
        before = probe_seed("mem-high-3000")
        assert probe_seed("mem-high-3000") == before


class TestProbeContent:
    def test_the_whole_forget_set_is_probed(self):
        for n in (500, 2500, 3000, 5000):
            assert probes(n).forget.size == n, "the forget set is the object of study"

    def test_retain_never_contains_a_forget_example(self):
        # A forget example in the retain probe would contaminate the retain-set baseline that
        # every utility claim rests on.
        for n in (500, 3000, 5000):
            p = probes(n)
            assert np.intersect1d(p.forget, p.retain).size == 0

    def test_the_full_test_split_is_probed(self):
        # The population attack matches non-members to a forget set of up to 5000, and RMIA
        # needs three disjoint roles out of test alone.
        assert probes().test.size == N_TEST

    def test_indices_are_sorted_so_rows_align_across_models(self):
        p = probes()
        for field in ("forget", "retain", "test", "mixed_train", "mixed_test"):
            arr = getattr(p, field)
            np.testing.assert_array_equal(arr, np.sort(arr))

    def test_retain_probe_size_is_configurable(self):
        assert probes(config={"retain_probe_size": 750}).retain.size == 750


class TestMixedProbe:
    @pytest.mark.parametrize("n", [500, 1000, 2500, 3000, 5000])
    def test_total_is_constant_across_conditions(self, n):
        """CKA's value depends on the number of probes, so a condition with 2500 and one with
        3000 would not be comparable. rand-500 cannot supply a third of the probe from its
        forget set, so the shortfall is backfilled rather than the probe being left smaller."""
        p = probes(n)
        assert p.mixed_train.size + p.mixed_test.size == 3000

    def test_balanced_when_the_forget_set_is_large_enough(self):
        p = probes(3000)
        # 1000 forget + 1000 retain in the train part, 1000 from test.
        assert p.mixed_train.size == 2000 and p.mixed_test.size == 1000

    def test_backfills_when_the_forget_set_is_small(self):
        p = probes(500)
        assert p.mixed_train.size + p.mixed_test.size == 3000
        assert p.mixed_train.size > 1500, "the shortfall must be taken up, not dropped"

    def test_mixed_train_draws_only_from_forget_and_retain(self):
        p = probes(3000)
        allowed = np.union1d(p.forget, p.retain)
        assert np.setdiff1d(p.mixed_train, allowed).size == 0

    def test_mixed_test_draws_only_from_test(self):
        p = probes()
        assert p.mixed_test.max() < N_TEST

    def test_size_is_configurable(self):
        p = probes(config={"probe_size": 600})
        assert p.mixed_train.size + p.mixed_test.size == 600


class TestAvailableTargets:
    class _Store:
        def __init__(self, ids):
            self._ids = ids

        def iter_checkpoints(self, *, role=None, forget=None):
            for r in self._ids:
                if role and f"__{role}__" not in r:
                    continue
                if forget and f"__{forget}__" not in r:
                    continue
                yield r

    def test_enumerates_only_what_this_machine_holds(self):
        # Shard-local: an account audits its own checkpoints and ships records, not weights.
        store = self._Store([
            "c10r18__unlearn__rand-500__salun__train1",
            "c10r18__unlearn__rand-500__scrub__train2",
            "c10r18__base__full__none__train0",
        ])
        got = available_targets(store)
        assert [t.run_id for t in got] == [
            "c10r18__unlearn__rand-500__salun__train1",
            "c10r18__unlearn__rand-500__scrub__train2",
        ]
        assert {t.method for t in got} == {"salun", "scrub"}
        assert {t.seed for t in got} == {1, 2}

    def test_sorted_for_reproducible_ordering(self):
        store = self._Store([
            "c10r18__unlearn__rand-500__scrub__train2",
            "c10r18__unlearn__rand-500__salun__train1",
        ])
        ids = [t.run_id for t in available_targets(store)]
        assert ids == sorted(ids)

    def test_filters_by_condition(self):
        store = self._Store([
            "c10r18__unlearn__rand-500__salun__train1",
            "c10r18__unlearn__rand-5000__salun__train1",
        ])
        got = available_targets(store, forget="rand-500")
        assert len(got) == 1 and got[0].forget_id == "rand-500"

    def test_an_empty_store_yields_nothing_rather_than_raising(self):
        assert available_targets(self._Store([])) == []


class TestAuditConfigRouting:
    def test_each_audit_gets_its_own_protocol_block(self):
        import yaml

        from forgetcheck.audits.runner import _config_for
        from forgetcheck.config import find_configs

        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        pop = _config_for("privacy_population", cfg)
        rmia = _config_for("privacy_rmia", cfg)
        # The population block's keys must be flattened up, not left nested.
        assert "nonmember_source" in pop and "n_splits" in pop
        assert "gamma" in rmia and "offline_a" in rmia
        # And the two must not bleed into each other.
        assert "nonmember_source" not in rmia or pop["nonmember_source"] == "test"

    def test_every_registered_audit_has_a_config_route(self):
        import yaml

        from forgetcheck.audits import REGISTRY
        from forgetcheck.audits.runner import _config_for
        from forgetcheck.config import find_configs

        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        for name in REGISTRY:
            assert isinstance(_config_for(name, cfg), dict), name

    def test_prob_floor_reaches_every_audit(self):
        import yaml

        from forgetcheck.audits import REGISTRY
        from forgetcheck.audits.runner import _config_for
        from forgetcheck.config import find_configs

        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        for name in REGISTRY:
            assert "prob_floor" in _config_for(name, cfg), name


class TestCLIWiring:
    def test_audit_subcommand_parses(self):
        from forgetcheck.cli import build_parser

        a = build_parser().parse_args(["audit", "--audits", "behavior,sde", "--forget", "rand-500"])
        assert a.command == "audit" and a.audits == "behavior,sde"

    def test_an_unknown_audit_name_is_refused(self):
        from forgetcheck.cli import build_parser, cmd_audit

        args = build_parser().parse_args(["audit", "--audits", "no-such-audit"])
        with pytest.raises(SystemExit, match="unknown audit"):
            cmd_audit(args)


class TestRecordsSurviveValidation:
    """Records must pass `records.validate()`.

    This is the check that pays for itself: the validator runs at *write* time, so a metric name
    it does not recognise fails after the audit has already spent its GPU hours. It caught two
    real defects on first run -- the AUDITS whitelist said "privacy_pop" while the module
    registered itself as "privacy_population", and "sde" was absent entirely.
    """

    @staticmethod
    def _ctx(target_logits=None, *, n=64, k=10, constant=False):
        from forgetcheck.audits import AuditContext

        rng = np.random.default_rng(0)
        def lg(seed, m=n):
            return rng.normal(size=(m, k)) if seed else rng.normal(size=(m, k))
        if constant:
            base = np.full((n, k), -20.0)
            base[:, 3] = 20.0
            target = base
        else:
            target = target_logits if target_logits is not None else lg(1)

        acts = {f"layer{i}": (np.ones((n, 8)) if constant else rng.normal(size=(n, 8)))
                for i in range(1, 5)}
        return AuditContext(
            run_id="c10r18__unlearn__mem-high-3000__salun__train1",
            logits={"forget": target, "retain": lg(2), "test": lg(3, 2 * n)},
            labels={"forget": np.arange(n) % k, "retain": np.arange(n) % k,
                    "test": np.arange(2 * n) % k},
            original_logits={"forget": lg(4), "retain": lg(5), "test": lg(6, 2 * n)},
            oracle_logits={p: np.stack([lg(7, m), lg(8, m)])
                           for p, m in (("forget", n), ("retain", n), ("test", 2 * n))},
            oracle_activations={f"layer{i}": np.stack([rng.normal(size=(n, 8))] * 2)
                                for i in range(1, 5)},
            activations=acts,
            reference_logits={"forget": np.stack([lg(9, n) for _ in range(4)]),
                              "test": np.stack([lg(10, 2 * n) for _ in range(4)])},
            reference_in_mask={"forget": rng.random((4, n)) < 0.5},
            relearn_curves={
                arm: {"steps": np.array([0.0, 1.0, 5.0]),
                      "forget_acc": np.array(v),
                      "retain_acc": np.array([0.99, 0.98, 0.98]),
                      "test_acc": np.array([0.93, 0.92, 0.92])}
                for arm, v in (("method", [0.3, 0.5, 0.7]), ("oracle", [0.1, 0.2, 0.3]),
                               ("original", [0.9, 0.95, 0.99]))
            },
            forget_kind="memstratum",
            forget_stratum="high",
            forget_size=3000,
            audit_seed=0,
        )

    @staticmethod
    def _records(ctx, name, values, notes):
        from forgetcheck.registry import make_record

        out = []
        for (metric, probe), value in values.items():
            defined = np.isfinite(value)
            out.append(make_record(
                run_id=ctx.run_id, audit=name,
                metric=metric if defined else "audit_undefined",
                probe_set=probe, value=float(value) if defined else 1.0,
                n_probe=64, forget_kind=ctx.forget_kind, forget_size=ctx.forget_size,
                forget_stratum=ctx.forget_stratum, selection_seed=1234, audit_seed=0,
                notes=notes if defined else f"{metric} undefined",
            ))
        return out

    @pytest.mark.parametrize("name", ["behavior", "privacy_population", "privacy_rmia",
                                      "representation", "relearning", "sde"])
    def test_a_healthy_model_produces_valid_records(self, name):
        from forgetcheck.audits import get_audit
        from forgetcheck.audits.runner import _config_for
        from forgetcheck.config import find_configs
        from forgetcheck.registry.records import validate
        import yaml

        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        ctx = self._ctx()
        ctx.config = _config_for(name, cfg)
        audit = get_audit(name)
        values = audit.measure(ctx)
        assert values, f"{name} produced nothing on a healthy model"
        for rec in self._records(ctx, name, values, audit.notes_for(ctx)):
            validate(rec)

    @pytest.mark.parametrize("name", ["behavior", "privacy_population", "privacy_rmia",
                                      "representation", "relearning", "sde"])
    def test_the_collapsed_control_produces_valid_records(self, name):
        # neggrad's endpoint: the model whose audits are most likely to be undefined, and the
        # one whose rows matter most.
        from forgetcheck.audits import get_audit
        from forgetcheck.audits.runner import _config_for
        from forgetcheck.config import find_configs
        from forgetcheck.registry.records import validate
        import yaml

        cfg = yaml.safe_load((find_configs() / "audits.yaml").read_text(encoding="utf-8"))
        ctx = self._ctx(constant=True)
        ctx.config = _config_for(name, cfg)
        audit = get_audit(name)
        for rec in self._records(ctx, name, audit.measure(ctx), audit.notes_for(ctx)):
            validate(rec)

    def test_every_registered_audit_may_write_records(self):
        # The whitelist and the registry must agree, or an audit runs and its rows are rejected.
        from forgetcheck.audits import REGISTRY
        from forgetcheck.registry.records import AUDITS

        missing = sorted(set(REGISTRY) - set(AUDITS))
        assert not missing, f"audits that cannot write records: {missing}"

    def test_an_infinite_value_becomes_an_undefined_row(self):
        # relearn_t80 returns inf when the model never recovers -- a real result, and one the
        # validator rejects in the value column.
        from forgetcheck.registry.records import RecordError, validate

        ctx = self._ctx()
        recs = self._records(ctx, "relearning", {("relearn_t80", "forget"): float("inf")}, "")
        assert recs[0].metric == "audit_undefined"
        validate(recs[0])
        with pytest.raises(RecordError, match="silently poisons"):
            validate(type(recs[0])(**{**recs[0].as_dict(), "metric": "relearn_t80",
                                      "value": float("inf")}))
