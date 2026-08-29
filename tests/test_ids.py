"""Stage 1 gate: identifiers are deterministic, and impossible combinations are rejected.

The point of these tests is not coverage. It is that four people on four Kaggle accounts
generate work queues independently, and the only thing preventing collisions or duplicated
compute is that run_id is a pure function everyone agrees on.
"""

import pytest

from forgetcheck.registry import ids
from forgetcheck.registry.ids import parse_run_id, run_id, tag_for


class TestTags:
    def test_known_pairs(self):
        assert tag_for("cifar10", "resnet18") == "c10r18"
        assert tag_for("cifar100", "resnet18") == "c100r18"
        assert tag_for("cifar10", "resnet9") == "c10r9"

    @pytest.mark.parametrize(
        "dataset,arch", [("svhn", "resnet18"), ("cifar10", "vit-tiny")]
    )
    def test_unregistered_fails_loudly(self, dataset, arch):
        # Silently minting a new identifier namespace is how two incompatible result sets end
        # up in one directory.
        with pytest.raises(KeyError, match="unregistered"):
            tag_for(dataset, arch)


class TestConstruction:
    def test_canonical_examples_from_the_plan(self):
        assert (
            run_id(role="base", forget="full", seed=0)
            == "c10r18__base__full__none__train0"
        )
        assert (
            run_id(role="oracle", forget="mem-high-3000", seed=205, seed_kind="oracle")
            == "c10r18__oracle__mem-high-3000__none__oracle205"
        )
        assert (
            run_id(role="unlearn", forget="mem-high-3000", method="scrub", seed=2)
            == "c10r18__unlearn__mem-high-3000__scrub__train2"
        )
        assert (
            run_id(role="shadow", forget="full", seed=7, seed_kind="shadow")
            == "c10r18__shadow__full__none__shadow7"
        )

    def test_is_a_pure_function(self):
        kw = dict(role="unlearn", forget="mem-low-3000", method="salun", seed=3)
        assert run_id(**kw) == run_id(**kw)

    def test_base_forget_slot_names_the_dataset_variant(self):
        # The canary condition needs its own original model: canaries must be present during
        # training for there to be anything to forget.
        clean = run_id(role="base", forget="full", seed=0)
        canary = run_id(role="base", forget="canary-500", seed=0)
        assert clean != canary

    def test_relearn_reference_arms_are_distinguishable(self):
        arms = {
            m: run_id(role="relearn", forget="mem-high-3000", method=m, seed=1)
            for m in ("scrub", "oracle", "original", "randinit")
        }
        assert len(set(arms.values())) == 4


class TestRejections:
    def test_unknown_role(self):
        with pytest.raises(ValueError, match="role must be one of"):
            run_id(role="finetune", forget="full", seed=0)

    def test_method_required_for_unlearn(self):
        with pytest.raises(ValueError, match="requires a method"):
            run_id(role="unlearn", forget="mem-high-3000", seed=0)

    def test_method_forbidden_for_base(self):
        with pytest.raises(ValueError, match="takes method='none'"):
            run_id(role="base", forget="full", method="scrub", seed=0)

    def test_reserved_method_cannot_label_an_unlearn_run(self):
        with pytest.raises(ValueError, match="reserved"):
            run_id(role="unlearn", forget="mem-high-3000", method="oracle", seed=0)

    def test_oracle_seed_kind_only_for_oracles(self):
        with pytest.raises(ValueError, match="only valid for role='oracle'"):
            run_id(role="base", forget="full", seed=205, seed_kind="oracle")

    def test_shadow_role_requires_shadow_seed_kind(self):
        with pytest.raises(ValueError, match="requires seed_kind='shadow'"):
            run_id(role="shadow", forget="full", seed=1, seed_kind="train")

    @pytest.mark.parametrize("kind", ["selection", "audit"])
    def test_non_run_seed_kinds_rejected(self, kind):
        # These parameterise forget-set selection and audit sampling; they never identify a run.
        with pytest.raises(ValueError, match="never identifies a run"):
            run_id(role="base", forget="full", seed=100, seed_kind=kind)

    @pytest.mark.parametrize(
        "forget", ["Mem-High", "mem_high", "mem high", "mem__high", "", "mem-high!"]
    )
    def test_malformed_segments(self, forget):
        with pytest.raises(ValueError):
            run_id(role="base", forget=forget, seed=0)

    def test_negative_seed(self):
        with pytest.raises(ValueError, match="non-negative"):
            run_id(role="base", forget="full", seed=-1)

    def test_bool_is_not_a_seed(self):
        with pytest.raises(TypeError):
            run_id(role="base", forget="full", seed=True)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "kw",
        [
            dict(role="base", forget="full", seed=0),
            dict(role="base", forget="canary-500", seed=4),
            dict(role="oracle", forget="rand-2500", seed=3),
            dict(role="oracle", forget="mem-high-3000", seed=211, seed_kind="oracle"),
            dict(role="shadow", forget="full", seed=31, seed_kind="shadow"),
            dict(role="unlearn", forget="mem-med-3000", method="l1sparse", seed=2),
            dict(role="relearn", forget="canary-500", method="original", seed=1),
        ],
    )
    def test_parse_inverts_construction(self, kw):
        rid = run_id(**kw)
        key = parse_run_id(rid)
        assert key.role == kw["role"]
        assert key.forget == kw["forget"]
        assert key.method == kw.get("method", "none")
        assert key.seed == kw["seed"]
        assert key.seed_kind == kw.get("seed_kind", "train")

    def test_reference_arm_flag(self):
        assert parse_run_id(
            run_id(role="relearn", forget="rand-3000", method="oracle", seed=0)
        ).is_reference_arm
        assert not parse_run_id(
            run_id(role="relearn", forget="rand-3000", method="scrub", seed=0)
        ).is_reference_arm

    @pytest.mark.parametrize(
        "bad",
        [
            "c10r18__base__full__none",           # too few segments
            "c10r18__base__full__none__train0__x",  # too many
            "c10r18__nope__full__none__train0",   # unknown role
            "c10r18__base__full__none__train",    # no digits
            "c10r18__base__full__none__wat0",     # unknown seed kind
        ],
    )
    def test_malformed_ids_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_run_id(bad)

    def test_keys_are_hashable_and_comparable(self):
        a = parse_run_id(run_id(role="base", forget="full", seed=0))
        b = parse_run_id(run_id(role="base", forget="full", seed=0))
        assert a == b and hash(a) == hash(b)
        assert len({a, b}) == 1


def test_vocabularies_are_closed():
    # A guard against someone adding a role in one place and not the other.
    assert ids.ROLES == {"base", "oracle", "shadow", "unlearn", "relearn"}
    assert "none" in ids.RESERVED_METHODS
