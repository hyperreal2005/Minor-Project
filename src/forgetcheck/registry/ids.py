"""Deterministic run and forget-set identifiers.

BINDING — implementation plan §4.1.

Every identifier is derivable from configuration *before the run exists*. That is what lets four
people generate work queues independently, on different Kaggle accounts, with no coordination and
no collisions: two members computing the ID for the same experiment always get the same string,
and two members computing IDs for different experiments never collide.

Identifier grammar (five segments, always):

    {tag}__{role}__{forget}__{method}__{seed_kind}{seed}

    tag        dataset+architecture, e.g. c10r18
    role       base | oracle | shadow | unlearn | relearn
    forget     forget-condition id, or 'full' for a model trained on everything
    method     unlearning method, or 'none' where not applicable
    seed_kind  which seed stream the trailing number belongs to
    seed       the integer

Examples:

    c10r18__base__full__none__train0                    M0, clean data, seed 0
    c10r18__base__canary-500__none__train0              M0 trained on the canary-corrupted set
    c10r18__oracle__mem-high-3000__none__train2         paired oracle for M0(seed 2)
    c10r18__oracle__mem-high-3000__none__oracle205      ensemble oracle, band member
    c10r18__shadow__full__none__shadow07                RMIA reference model 7
    c10r18__unlearn__mem-high-3000__scrub__train2       SCRUB applied to M0(seed 2)
    c10r18__relearn__mem-high-3000__scrub__train2       relearning the SCRUB arm
    c10r18__relearn__mem-high-3000__oracle__train2      relearning the oracle arm (lower anchor)
    c10r18__relearn__mem-high-3000__original__train2    relearning M0 (upper anchor)

Two conventions worth stating because they are easy to get wrong:

*   For ``base`` models the ``forget`` slot names the *dataset variant*, not a forget set.
    ``full`` is the clean training set; ``canary-500`` is the canary-corrupted one. The canary
    condition needs its own original model, because canaries must be present during training for
    there to be anything to forget.

*   For ``relearn`` runs the ``method`` slot names the *arm being relearned*. Real method names
    identify the unlearned-model arms; the reserved names ``oracle``, ``original`` and
    ``randinit`` identify the reference arms.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "ROLES",
    "SEED_KINDS",
    "RESERVED_METHODS",
    "NO_METHOD",
    "FULL_DATASET",
    "tag_for",
    "run_id",
    "parse_run_id",
    "RunKey",
]

# --------------------------------------------------------------------------- vocabularies

ROLES: Final = frozenset({"base", "oracle", "shadow", "unlearn", "relearn"})

SEED_KINDS: Final = frozenset({"train", "oracle", "shadow", "selection", "audit"})

#: Names usable in the ``method`` slot that are not unlearning algorithms.
RESERVED_METHODS: Final = frozenset({"none", "oracle", "original", "randinit"})

NO_METHOD: Final = "none"
FULL_DATASET: Final = "full"

# Abbreviations are registered explicitly rather than derived. An unknown dataset or architecture
# must fail loudly: silently minting a new identifier namespace is how two incompatible sets of
# results end up in one results directory.
_DATASET_TAGS: Final[dict[str, str]] = {
    "cifar10": "c10",
    "cifar100": "c100",
}

_ARCH_TAGS: Final[dict[str, str]] = {
    "resnet18": "r18",
    "resnet9": "r9",
}

#: Segments may contain lowercase letters, digits and single hyphens.
_SEGMENT_RE: Final = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

_SEP: Final = "__"


# --------------------------------------------------------------------------- helpers


def tag_for(dataset: str, arch: str) -> str:
    """Return the ``c10r18``-style tag for a dataset/architecture pair.

    Raises:
        KeyError: if either is unregistered. Register it in this module rather than working
            around it, so the identifier namespace stays enumerable.
    """
    try:
        d = _DATASET_TAGS[dataset]
    except KeyError:
        raise KeyError(
            f"unregistered dataset {dataset!r}; add it to ids._DATASET_TAGS "
            f"(known: {sorted(_DATASET_TAGS)})"
        ) from None
    try:
        a = _ARCH_TAGS[arch]
    except KeyError:
        raise KeyError(
            f"unregistered architecture {arch!r}; add it to ids._ARCH_TAGS "
            f"(known: {sorted(_ARCH_TAGS)})"
        ) from None
    return f"{d}{a}"


def _check_segment(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    if _SEP in value:
        raise ValueError(f"{field}={value!r} may not contain the separator {_SEP!r}")
    if not _SEGMENT_RE.match(value):
        raise ValueError(
            f"{field}={value!r} must be lowercase alphanumeric with single hyphens "
            "(no underscores, spaces, or uppercase)"
        )
    return value


# --------------------------------------------------------------------------- construction


def run_id(
    *,
    role: str,
    forget: str,
    seed: int,
    dataset: str = "cifar10",
    arch: str = "resnet18",
    method: str = NO_METHOD,
    seed_kind: str = "train",
) -> str:
    """Build a run identifier. Pure function of its arguments.

    Args:
        role: one of :data:`ROLES`.
        forget: forget-condition id, or ``'full'`` for a model trained on everything.
            For ``base`` runs this names the dataset variant (see module docstring).
        seed: the integer belonging to ``seed_kind``.
        dataset, arch: resolved to a tag via :func:`tag_for`.
        method: unlearning method name, or one of :data:`RESERVED_METHODS`.
        seed_kind: which seed stream ``seed`` belongs to.

    Raises:
        ValueError: on an unknown role or seed kind, a malformed segment, a negative seed,
            or a role/method/seed-kind combination that cannot occur.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}, got {role!r}")
    if seed_kind not in SEED_KINDS:
        raise ValueError(f"seed_kind must be one of {sorted(SEED_KINDS)}, got {seed_kind!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    _check_segment(forget, "forget")
    _check_segment(method, "method")

    _check_role_consistency(role=role, method=method, seed_kind=seed_kind)

    tag = tag_for(dataset, arch)
    return _SEP.join([tag, role, forget, method, f"{seed_kind}{seed}"])


def _check_role_consistency(*, role: str, method: str, seed_kind: str) -> None:
    """Reject combinations that are meaningless, so they fail here and not in analysis."""
    if role in {"base", "oracle", "shadow"} and method != NO_METHOD:
        raise ValueError(f"role={role!r} takes method='none', got {method!r}")

    if role in {"unlearn", "relearn"} and method == NO_METHOD:
        raise ValueError(f"role={role!r} requires a method name, got 'none'")

    if role == "unlearn" and method in RESERVED_METHODS:
        raise ValueError(
            f"method={method!r} is reserved for reference arms and cannot label an "
            "unlearn run; use the algorithm's own name"
        )

    if role == "shadow" and seed_kind != "shadow":
        raise ValueError(f"role='shadow' requires seed_kind='shadow', got {seed_kind!r}")

    if seed_kind == "shadow" and role != "shadow":
        raise ValueError(f"seed_kind='shadow' is only valid for role='shadow', got {role!r}")

    if seed_kind == "oracle" and role != "oracle":
        raise ValueError(
            f"seed_kind='oracle' identifies an ensemble oracle and is only valid for "
            f"role='oracle', got {role!r}"
        )

    if seed_kind in {"selection", "audit"}:
        raise ValueError(
            f"seed_kind={seed_kind!r} never identifies a run; it parameterises forget-set "
            "selection and audit sampling instead"
        )


# --------------------------------------------------------------------------- parsing


class RunKey:
    """The parsed coordinates of a run identifier.

    Deliberately not a NamedTuple: positional unpacking of six near-interchangeable strings is
    exactly the kind of thing that goes wrong silently.
    """

    __slots__ = ("tag", "role", "forget", "method", "seed_kind", "seed")

    def __init__(self, tag: str, role: str, forget: str, method: str, seed_kind: str, seed: int):
        self.tag = tag
        self.role = role
        self.forget = forget
        self.method = method
        self.seed_kind = seed_kind
        self.seed = seed

    def __repr__(self) -> str:
        return (
            f"RunKey(tag={self.tag!r}, role={self.role!r}, forget={self.forget!r}, "
            f"method={self.method!r}, seed_kind={self.seed_kind!r}, seed={self.seed})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunKey):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def as_tuple(self) -> tuple[str, str, str, str, str, int]:
        return (self.tag, self.role, self.forget, self.method, self.seed_kind, self.seed)

    @property
    def is_reference_arm(self) -> bool:
        """True for the relearning anchor arms, which are not unlearning methods."""
        return self.method in {"oracle", "original", "randinit"}


_SEED_SUFFIX_RE: Final = re.compile(r"^([a-z]+)(\d+)$")


def parse_run_id(rid: str) -> RunKey:
    """Inverse of :func:`run_id`. Raises ``ValueError`` on anything malformed."""
    parts = rid.split(_SEP)
    if len(parts) != 5:
        raise ValueError(f"run_id must have 5 {_SEP!r}-separated segments, got {len(parts)}: {rid!r}")

    tag, role, forget, method, seed_part = parts

    if role not in ROLES:
        raise ValueError(f"unknown role {role!r} in {rid!r}")

    m = _SEED_SUFFIX_RE.match(seed_part)
    if not m:
        raise ValueError(f"malformed seed segment {seed_part!r} in {rid!r}")
    seed_kind, seed_digits = m.group(1), m.group(2)
    if seed_kind not in SEED_KINDS:
        raise ValueError(f"unknown seed_kind {seed_kind!r} in {rid!r}")

    return RunKey(tag, role, forget, method, seed_kind, int(seed_digits))
