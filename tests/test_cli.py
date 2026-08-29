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
