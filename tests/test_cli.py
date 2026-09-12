"""The CLI and work sharding.

The property that matters most here is the one that makes a dozen Kaggle accounts safe: every
account derives the identical work list from config alone, and takes a disjoint, complete stripe
of it with no coordination. If that broke, accounts would silently duplicate or drop work and
nobody would notice until the matrix came up short.

Training is exercised in ``test_end_to_end`` behind a ``slow`` marker, because a real run costs
minutes on CPU.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from forgetcheck.cli import STAGES, WorkItem, build_parser, plan_stage, shard
from forgetcheck.registry import parse_run_id
from forgetcheck.config import Context, find_configs

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ctx():
    return Context(configs=find_configs(REPO), root=REPO)


@pytest.fixture
def tiny_ctx(tmp_path):
    """A context with a one-seed, one-epoch config, for exercising the plumbing cheaply."""
    cfgs = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfgs)

    base = yaml.safe_load((cfgs / "base.yaml").read_text(encoding="utf-8"))
    base["training"]["epochs"] = 1
    base["training"]["amp"] = False
    base["training"]["channels_last"] = False
    base["seeds"]["train"] = [0]
    base["oracles"]["paired_seeds"] = [0]
    base["seeds"]["oracle"] = [200, 201]
    base["shadows"]["count"] = 2
    base["dataset"]["root"] = str(REPO / "data")
    base["memorization"]["path"] = str(REPO / "data/memorization/cifar10_memorization.npy")
    base["paths"]["artifacts"] = str(tmp_path / "artifacts")
    base["paths"]["records"] = str(tmp_path / "records")
    (cfgs / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")

    return Context(configs=cfgs, root=tmp_path)


class TestParser:
    def test_every_stage_is_reachable(self):
        p = build_parser()
        for stage in STAGES:
            args = p.parse_args(["queue", "--stage", str(stage)])
            assert args.stage == stage

    def test_unknown_method_rejected_at_parse_time(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["unlearn", "--forget", "rand-500", "--method", "nope", "--seed", "0"]
            )

    def test_account_out_of_range_rejected(self):
        from forgetcheck.cli import main

        with pytest.raises(SystemExit, match=r"account must be in"):
            main(["queue", "--stage", "3", "--account", "5", "--of", "3"])


@pytest.mark.requires_data
class TestPlanning:
    def test_stage_sizes_match_the_design(self, ctx):
        # 10 base (5 seeds x clean/canary) + 40 paired oracles (8 conditions x 5) + 12 ensemble.
        assert len(plan_stage(ctx, 3)) == 62
        assert len(plan_stage(ctx, 4)) == 32          # shadows
        assert len(plan_stage(ctx, 5)) == 6 * 8 * 5   # methods x conditions x seeds

    def test_run_ids_are_unique(self, ctx):
        for stage in STAGES:
            items = plan_stage(ctx, stage)
            assert len({i.run_id for i in items}) == len(items), f"stage {stage}"

    def test_canary_gets_its_own_base_model(self, ctx):
        # Canaries must be present during training for there to be anything to forget.
        base = [i.run_id for i in plan_stage(ctx, 3) if i.kind == "base"]
        assert any("__base__full__" in r for r in base)
        assert any("__base__canary-500__" in r for r in base)

    def test_ensemble_oracles_only_at_the_primary_condition(self, ctx):
        ens = [i for i in plan_stage(ctx, 3) if i.kind == "oracle-ensemble"]
        assert len(ens) == 12
        assert all(ctx.primary_condition in i.run_id for i in ens)

    def test_unknown_stage_explains_itself(self, ctx):
        with pytest.raises(SystemExit, match="queueable stages"):
            plan_stage(ctx, 99)


@pytest.mark.requires_data
class TestSharding:
    @pytest.mark.parametrize("of", [1, 2, 3, 5, 12])
    def test_disjoint_and_complete(self, ctx, of):
        items = plan_stage(ctx, 5)
        shards = [shard(items, account=a, of=of) for a in range(1, of + 1)]
        ids = [{i.run_id for i in s} for s in shards]
        assert sum(len(s) for s in ids) == len(set().union(*ids)), "accounts overlap"
        assert set().union(*ids) == {i.run_id for i in items}, "work was dropped"

    @pytest.mark.parametrize("of", [3, 12])
    def test_balanced(self, ctx, of):
        # Hashing was tried first and left one account of twelve with nothing to do on stage 4.
        for stage in STAGES:
            items = plan_stage(ctx, stage)
            sizes = [len(shard(items, account=a, of=of)) for a in range(1, of + 1)]
            assert max(sizes) - min(sizes) <= 1, f"stage {stage} unbalanced: {sizes}"

    def test_deterministic(self, ctx):
        items = plan_stage(ctx, 5)
        a = [i.run_id for i in shard(items, account=2, of=3)]
        b = [i.run_id for i in shard(items, account=2, of=3)]
        assert a == b

    def test_independent_of_plan_order(self, ctx):
        # Accounts must agree even if their plan lists were built in a different order.
        items = plan_stage(ctx, 5)
        forward = {i.run_id for i in shard(items, account=1, of=4)}
        reversed_ = {i.run_id for i in shard(list(reversed(items)), account=1, of=4)}
        assert forward == reversed_

    def test_single_account_gets_everything(self, ctx):
        items = plan_stage(ctx, 4)
        assert len(shard(items, account=1, of=1)) == len(items)

    def test_bad_account_rejected(self, ctx):
        with pytest.raises(ValueError, match=r"account must be in"):
            shard(plan_stage(ctx, 4), account=0, of=3)

    def test_a_filter_narrows_the_share_and_never_redraws_it(self, ctx):
        """Reproduces the account-1 re-run: `--methods neggrad,salun,scrub --account 1 --of 3`
        listed neggrad seeds 0 and 3 instead of the account's seed 2, because filtering ran
        before sharding and striping a 3-method list lands on different seeds than a 6-method
        one. Whatever the filter, the result must be a subset of the account's unfiltered share.
        """
        from forgetcheck.cli import filter_items

        items = plan_stage(ctx, 5)
        for account in (1, 2, 3):
            share = {i.run_id for i in shard(items, account=account, of=3)}
            narrowed = {
                i.run_id
                for i in filter_items(shard(items, account=account, of=3),
                                      methods=["neggrad", "salun", "scrub"])
            }
            assert narrowed <= share, f"account {account}: filter produced items outside its share"
            assert narrowed == {r for r in share if parse_run_id(r).method in ("neggrad", "salun", "scrub")}

            # The wrong order -- what cmd_queue used to do -- provably redraws the share.
            redrawn = {
                i.run_id
                for i in shard(filter_items(items, methods=["neggrad", "salun", "scrub"]),
                               account=account, of=3)
            }
            assert redrawn != narrowed, "the bug this test guards against would not reproduce"


@pytest.mark.requires_data
class TestContext:
    def test_hashes_are_verified(self, ctx):
        assert ctx.bundle.sha == ctx.base["dataset"]["expect_sha"]
        assert len(ctx.memorization) == 50_000

    def test_wrong_data_hash_is_fatal(self, tiny_ctx):
        tiny_ctx.base["dataset"]["expect_sha"] = "0" * 16
        tiny_ctx.__dict__.pop("bundle", None)  # drop any cached_property value
        with pytest.raises(RuntimeError, match="differs from the data"):
            _ = tiny_ctx.bundle

    def test_wrong_memorization_hash_is_fatal(self, tiny_ctx):
        tiny_ctx.base["memorization"]["expect_sha"] = "0" * 16
        with pytest.raises(RuntimeError, match="hash to"):
            _ = tiny_ctx.memorization

    def test_accepts_string_paths(self, tiny_ctx):
        # A dataclass field typed `Path` accepts a str happily, and the failure then surfaces
        # much later as "unsupported operand type(s) for /: 'str' and 'str'" from a property
        # that looks unrelated. Callers legitimately pass strings -- notebooks especially.
        from forgetcheck.config import Context

        ctx = Context(configs=str(tiny_ctx.configs), root=str(tiny_ctx.root))
        assert isinstance(ctx.configs, Path) and isinstance(ctx.root, Path)
        assert ctx.store.root  # the property that used to blow up
        assert ctx.records_dir

    def test_forget_indices_are_cached_and_verified(self, tiny_ctx):
        a = tiny_ctx.forget_indices("mem-high-3000")
        b = tiny_ctx.forget_indices("mem-high-3000")   # second call hits the cache check
        np.testing.assert_array_equal(a, b)
        assert tiny_ctx.store.has_forget_set("mem-high-3000")


@pytest.mark.slow
@pytest.mark.requires_data
def test_end_to_end(tiny_ctx):
    """Train one M0, unlearn from it, and confirm the records join.

    The whole chain in miniature: identity, checkpointing, resumption, and the record contract.
    Slow because it trains a real ResNet-18 on CPU.
    """
    from forgetcheck.registry import read_records
    from forgetcheck.train import base_task, run_task
    from forgetcheck.unlearn import run_unlearn

    ctx = tiny_ctx
    spec = ctx.spec("rand-500")
    fidx = ctx.forget_indices("rand-500")

    task = base_task(ctx.bundle, seed=0)
    object.__setattr__(task, "indices", np.arange(1500))  # keep it to ~30s on CPU
    rid = run_task(task, ctx.train_config, store=ctx.store, records_dir=ctx.records_dir,
                   forget_indices=fidx)
    assert rid and ctx.store.has_checkpoint(rid)
    assert run_task(task, ctx.train_config, store=ctx.store,
                    records_dir=ctx.records_dir, forget_indices=fidx) is None, "should skip"

    urid = run_unlearn(
        method="finetune", spec=spec, forget_indices=fidx[:100], bundle=ctx.bundle,
        seed=0, store=ctx.store, records_dir=ctx.records_dir,
        hparams={"epochs": 1}, batch_size=128,
    )
    assert ctx.store.has_checkpoint(urid)

    df = read_records(ctx.records_dir)
    assert set(df["run_id"]) == {rid, urid}
    assert {"test_acc", "retain_acc", "forget_acc", "forget_loss"} <= set(df["metric"])
    assert df["value"].notna().all()
    # The unlearned model must record which original it came from.
    assert rid in ctx.store.load_meta(urid).notes


@pytest.mark.requires_data
def test_unlearning_without_its_base_model_fails_loudly(tiny_ctx):
    from forgetcheck.unlearn import run_unlearn

    with pytest.raises(FileNotFoundError, match="Train stage 3 before"):
        run_unlearn(
            method="finetune", spec=tiny_ctx.spec("rand-500"),
            forget_indices=tiny_ctx.forget_indices("rand-500"),
            bundle=tiny_ctx.bundle, seed=0, store=tiny_ctx.store,
            records_dir=tiny_ctx.records_dir,
        )


class TestForceGuard:
    def test_force_without_a_filter_is_refused(self):
        # One mistyped flag must not be able to queue 240 runs of recomputation.
        from forgetcheck.cli import build_parser, cmd_queue

        args = build_parser().parse_args(["queue", "--stage", "5", "--force"])
        with pytest.raises(SystemExit, match="--force must be combined"):
            cmd_queue(args)


class TestFilterMatching:
    """filter_items must match conditions exactly, never as substrings."""

    @staticmethod
    def _items(*forgets):
        return [
            WorkItem(f"c10r18__unlearn__{f}__scrub__train0", "unlearn", lambda: None)
            for f in forgets
        ]

    def test_a_list_of_conditions_selects_exactly_those(self):
        from forgetcheck.cli import filter_items

        got = filter_items(self._items("rand-500", "rand-5000", "rand-2500"),
                           forget=["rand-500", "rand-5000"])
        assert sorted(parse_run_id(i.run_id).forget for i in got) == ["rand-500", "rand-5000"]

    def test_a_bare_string_does_not_substring_match(self):
        # "rand-500" is a substring of "rand-5000"; passing a string must not admit both.
        from forgetcheck.cli import filter_items

        got = filter_items(self._items("rand-500", "rand-5000"), forget="rand-5000")
        assert [parse_run_id(i.run_id).forget for i in got] == ["rand-5000"]


class TestExecuteReporting:
    """The summary line must match what was actually listed.

    `--dry-run` previously printed a list of [todo] items and then reported "would run 0",
    because the dry-run branch never counted them. Purely a reporting bug -- nothing was
    mis-planned -- but misleading enough that someone could conclude no work was scheduled.
    """

    @staticmethod
    def _store(present=(), shas=None):
        shas = shas or {}

        class Meta:
            def __init__(self, sha):
                self.hparams_sha = sha

        class S:
            def has_checkpoint(self, rid):
                return rid in present

            def load_meta(self, rid):
                return Meta(shas.get(rid, ""))
        return S()

    @staticmethod
    def _items(n, fail_at=()):
        def make(i):
            def run():
                if i in fail_at:
                    raise RuntimeError(f"boom {i}")
            return WorkItem(f"c10r18__base__full__none__train{i}", "base", run)
        return [make(i) for i in range(n)]

    def test_dry_run_counts_todo_items(self, capsys):
        from forgetcheck.cli import _execute

        _execute(self._items(5), dry_run=True, store=self._store())
        out = capsys.readouterr().out
        assert out.count("[todo]") == 5
        assert "would run 5, 0 already present (5 total)" in out

    def test_dry_run_separates_have_from_todo(self, capsys):
        from forgetcheck.cli import _execute

        present = {"c10r18__base__full__none__train0", "c10r18__base__full__none__train1"}
        _execute(self._items(5), dry_run=True, store=self._store(present))
        out = capsys.readouterr().out
        assert out.count("[have]") == 2 and out.count("[todo]") == 3
        assert "would run 3, 2 already present (5 total)" in out

    def test_a_checkpoint_from_an_older_configuration_is_stale_not_have(self, capsys):
        """Reproduces the Stage 5 re-run that reported "would run 0, 80 already present".

        The stale-hyperparameter guard lived inside run_unlearn, but _execute skipped on
        has_checkpoint before ever calling it -- so the guard was unreachable from the CLI, and
        a real run would have skipped all 80 exactly as the dry run did. The check has to live
        at the layer that decides what runs.
        """
        from forgetcheck.cli import _execute

        rid = "c10r18__unlearn__rand-500__neggrad__train2"
        item = WorkItem(rid, "unlearn", lambda: None, hparams_sha="new-config")
        store = self._store(present={rid}, shas={rid: "old-config"})

        _execute([item], dry_run=True, store=store)
        out = capsys.readouterr().out
        assert "[stale]" in out and "[have]" not in out
        assert "would run 1 (1 stale, from an older configuration), 0 already present" in out

    def test_a_stale_item_actually_runs(self, capsys):
        from forgetcheck.cli import _execute

        rid = "c10r18__unlearn__rand-500__salun__train1"
        ran = []
        item = WorkItem(rid, "unlearn", lambda: ran.append(rid), hparams_sha="new")
        _execute([item], dry_run=False, store=self._store(present={rid}, shas={rid: "old"}))
        assert ran == [rid], "a stale checkpoint must be recomputed, not kept"
        assert "ran 1 (1 stale, from an older configuration)" in capsys.readouterr().out

    def test_a_matching_sha_is_still_skipped(self):
        # The other 56 of account 1's runs: same configuration, must cost only a metadata read.
        from forgetcheck.cli import _execute

        rid = "c10r18__unlearn__rand-500__scrub__train2"
        ran = []
        item = WorkItem(rid, "unlearn", lambda: ran.append(rid), hparams_sha="same")
        _execute([item], dry_run=False, store=self._store(present={rid}, shas={rid: "same"}))
        assert ran == []

    def test_items_without_a_sha_are_judged_on_presence_alone(self):
        # Base/oracle/shadow items do not carry one; their behaviour is unchanged.
        from forgetcheck.cli import _execute

        rid = "c10r18__base__full__none__train0"
        ran = []
        item = WorkItem(rid, "base", lambda: ran.append(rid))
        _execute([item], dry_run=False, store=self._store(present={rid}))
        assert ran == []

    def test_force_recomputes_a_current_checkpoint(self, capsys):
        # The SalUn case: implementation changed, hyperparameters did not, sha matches, and the
        # only honest way to say "redo it anyway" is to say so explicitly.
        from forgetcheck.cli import _execute

        rid = "c10r18__unlearn__rand-500__salun__train1"
        ran = []
        item = WorkItem(rid, "unlearn", lambda: ran.append(rid), hparams_sha="same")
        store = self._store(present={rid}, shas={rid: "same"})
        _execute([item], dry_run=True, store=store, force=True)
        assert "[forced]" in capsys.readouterr().out
        _execute([item], dry_run=False, store=store, force=True)
        assert ran == [rid]
        assert "ran 1 (1 forced)" in capsys.readouterr().out

    def test_force_does_not_touch_items_that_are_absent_anyway(self):
        from forgetcheck.cli import _execute

        ran = []
        item = WorkItem("c10r18__unlearn__rand-500__salun__train1", "unlearn",
                        lambda: ran.append(1), hparams_sha="x")
        _execute([item], dry_run=False, store=self._store(), force=True)
        assert ran == [1]  # a todo item runs exactly once, forced or not

    def test_real_run_counts_what_ran(self, capsys):
        from forgetcheck.cli import _execute

        rc = _execute(self._items(4), dry_run=False, store=self._store())
        assert rc == 0
        assert "ran 4, skipped 0 already present, 0 failed" in capsys.readouterr().out

    def test_one_failure_does_not_stop_the_queue(self, capsys):
        # A session that dies on run 2 of 4 should still produce the other 3.
        from forgetcheck.cli import _execute

        rc = _execute(self._items(4, fail_at={1}), dry_run=False, store=self._store())
        assert rc == 1, "a failure must surface as a non-zero exit code"
        assert "ran 3, skipped 0 already present, 1 failed" in capsys.readouterr().out

    def test_counts_are_consistent(self, capsys):
        from forgetcheck.cli import _execute

        present = {"c10r18__base__full__none__train0"}
        _execute(self._items(5, fail_at={2}), dry_run=False, store=self._store(present))
        out = capsys.readouterr().out
        assert "ran 3, skipped 1 already present, 1 failed" in out  # 3 + 1 + 1 == 5


@pytest.mark.requires_data
class TestFiltering:
    """Narrowing a stage's work list.

    This is what makes a pilot one command instead of six, and the way to re-run a single method
    after changing it without touching the runs that are already correct.
    """

    def test_pilot_selects_one_condition_one_seed_all_methods(self, ctx):
        from forgetcheck.cli import filter_items
        from forgetcheck.unlearn import CORE_METHODS

        got = filter_items(plan_stage(ctx, 5), forget="mem-high-3000", seeds=[0])
        assert len(got) == len(CORE_METHODS) == 6
        keys = [parse_run_id(i.run_id) for i in got]
        assert {k.method for k in keys} == set(CORE_METHODS)
        assert {k.forget for k in keys} == {"mem-high-3000"}
        assert {k.seed for k in keys} == {0}

    def test_method_filter(self, ctx):
        from forgetcheck.cli import filter_items

        got = filter_items(plan_stage(ctx, 5), methods=["salun"])
        assert len(got) == 8 * 5  # conditions x seeds
        assert all(parse_run_id(i.run_id).method == "salun" for i in got)

    def test_filters_compose(self, ctx):
        from forgetcheck.cli import filter_items

        got = filter_items(
            plan_stage(ctx, 5), forget="rand-500", methods=["salun", "scrub"], seeds=[0, 1]
        )
        assert len(got) == 2 * 2

    def test_no_filter_is_identity(self, ctx):
        from forgetcheck.cli import filter_items

        items = plan_stage(ctx, 5)
        assert len(filter_items(items)) == len(items)

    def test_works_on_stages_without_methods(self, ctx):
        # Filtering parses the run id, so one implementation covers every stage.
        from forgetcheck.cli import filter_items

        got = filter_items(plan_stage(ctx, 3), forget="mem-low-3000")
        assert got and all(parse_run_id(i.run_id).forget == "mem-low-3000" for i in got)

    def test_empty_result_fails_loudly(self):
        # Silently running nothing would look like success; a typo must not do that.
        from forgetcheck.cli import main

        with pytest.raises(SystemExit, match="no stage-5 items in account 1's share match"):
            main(["--dry-run", "queue", "--stage", "5", "--methods", "typo"])
