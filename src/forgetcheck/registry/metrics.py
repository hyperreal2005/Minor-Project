"""The metric registry: the project's advance commitment to what "better" means.

BINDING — implementation plan §4.4.

The registry is loaded from ``configs/metrics.yaml`` and consulted on every record write. Its
purpose is to make one specific failure impossible: an agreement analysis that ranks methods
using a metric whose direction nobody ever fixed, or that silently accumulates two spellings of
the same metric across four people's audit modules.

The direction vocabulary is small on purpose:

``higher_better`` / ``lower_better``
    The raw value is meaningful on its own.

``closer_to_oracle``
    Only ``|value - oracle|`` is meaningful. This is the right direction for anything the
    retrained oracle also exhibits, and getting it wrong is the classic unlearning evaluation
    error: forget-set accuracy is *not* lower_better, because a retrained model still classifies
    most forgotten examples correctly (master reference §3.2).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Literal

import yaml

__all__ = [
    "Direction",
    "MetricSpec",
    "MetricRegistry",
    "load_registry",
    "default_registry",
    "DIRECTIONS",
]

Direction = Literal["higher_better", "lower_better", "closer_to_oracle"]

DIRECTIONS: Final[frozenset[str]] = frozenset(
    {"higher_better", "lower_better", "closer_to_oracle"}
)

#: Families that participate in the cross-audit agreement analysis. ``meta`` does not: its
#: entries are bookkeeping and validity outputs, not verdicts about a method's quality.
RANKED_FAMILIES: Final[frozenset[str]] = frozenset(
    {"behavior", "privacy_weak", "privacy_strong", "representation", "reversibility"}
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One registered metric."""

    name: str
    family: str
    direction: Direction
    oracle_ref: bool
    probe_sets: tuple[str, ...]
    doc: str = ""

    @property
    def is_ranked(self) -> bool:
        """Whether this metric contributes to method rankings."""
        return self.family in RANKED_FAMILIES

    def accepts_probe_set(self, probe_set: str) -> bool:
        if "all" in self.probe_sets:
            return True
        return probe_set in self.probe_sets


class MetricRegistry:
    """Immutable lookup over the registered metrics."""

    __slots__ = ("_by_name", "_source")

    def __init__(self, specs: dict[str, MetricSpec], source: Path | None = None):
        self._by_name = dict(specs)
        self._source = source

    # -- lookup ---------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> MetricSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"metric {name!r} is not registered. Add it to configs/metrics.yaml — this is a "
                "four-person decision, not a convenience. "
                f"Nearest registered names: {self._suggest(name)}"
            ) from None

    def __iter__(self) -> Iterator[MetricSpec]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def by_family(self, family: str) -> tuple[MetricSpec, ...]:
        return tuple(s for s in self._by_name.values() if s.family == family)

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({s.family for s in self._by_name.values()}))

    def ranked(self) -> tuple[MetricSpec, ...]:
        return tuple(s for s in self._by_name.values() if s.is_ranked)

    def _suggest(self, name: str, k: int = 3) -> list[str]:
        import difflib

        return difflib.get_close_matches(name, self._by_name, n=k, cutoff=0.5)

    def __repr__(self) -> str:
        return (
            f"MetricRegistry({len(self._by_name)} metrics across "
            f"{len(self.families())} families, source={self._source})"
        )


def _parse(raw: dict, source: Path | None) -> MetricRegistry:
    """Flatten the family-nested YAML into a name -> MetricSpec mapping."""
    specs: dict[str, MetricSpec] = {}

    for family_key, entries in raw.items():
        if not isinstance(entries, dict):
            raise ValueError(
                f"metrics.yaml: top-level key {family_key!r} must map to a dict of metrics, "
                f"got {type(entries).__name__}"
            )
        for name, body in entries.items():
            if name in specs:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} defined twice "
                    f"(second occurrence under {family_key!r})"
                )
            if not isinstance(body, dict):
                raise ValueError(f"metrics.yaml: metric {name!r} must map to a dict")

            missing = {"family", "direction", "oracle_ref"} - body.keys()
            if missing:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} is missing required key(s) "
                    f"{sorted(missing)}"
                )

            direction = body["direction"]
            if direction not in DIRECTIONS:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} has direction {direction!r}; "
                    f"must be one of {sorted(DIRECTIONS)}"
                )

            family = body["family"]
            if family != family_key:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} sits under {family_key!r} but declares "
                    f"family={family!r}. The nesting and the field must agree."
                )

            oracle_ref = body["oracle_ref"]
            if not isinstance(oracle_ref, bool):
                raise ValueError(
                    f"metrics.yaml: metric {name!r} has oracle_ref={oracle_ref!r}; must be a bool"
                )

            if direction == "closer_to_oracle" and not oracle_ref:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} is closer_to_oracle but oracle_ref is false. "
                    "A metric interpreted relative to the oracle needs the oracle."
                )

            probe_sets = tuple(body.get("probe_sets", ()) or ())
            if not probe_sets:
                raise ValueError(
                    f"metrics.yaml: metric {name!r} declares no probe_sets. Use ['all'] if it "
                    "genuinely applies everywhere."
                )

            specs[name] = MetricSpec(
                name=name,
                family=family,
                direction=direction,
                oracle_ref=oracle_ref,
                probe_sets=probe_sets,
                doc=(body.get("doc") or "").strip(),
            )

    if not specs:
        raise ValueError("metrics.yaml defined no metrics")
    return MetricRegistry(specs, source)


def load_registry(path: str | Path) -> MetricRegistry:
    """Load and validate a metric registry from a YAML file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"metric registry not found at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return _parse(raw, p)


def _find_default_config() -> Path:
    """Locate ``configs/metrics.yaml`` by walking up from this file, then from the cwd.

    Walking up handles both an editable install inside the repo and a Kaggle session where the
    repo is checked out somewhere unexpected.
    """
    for start in (Path(__file__).resolve(), Path.cwd().resolve() / "_"):
        for parent in start.parents:
            candidate = parent / "configs" / "metrics.yaml"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "could not locate configs/metrics.yaml by walking up from "
        f"{Path(__file__).resolve()} or {Path.cwd().resolve()}. "
        "Pass an explicit path to load_registry()."
    )


@functools.lru_cache(maxsize=1)
def default_registry() -> MetricRegistry:
    """The project registry, loaded once per process."""
    return load_registry(_find_default_config())
