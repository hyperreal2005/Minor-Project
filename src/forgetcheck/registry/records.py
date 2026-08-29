"""RunRecord: the result schema every audit writes and the analysis reads.

BINDING — implementation plan §4.3. Changes require all four members.

Three design decisions, each guarding against a specific way this project could fail:

**Long format, not wide.** One row per (run, audit, metric, probe_set). Four people write four
audit modules producing heterogeneous outputs — scalars, per-layer vectors, curves. Long format
lets each append rows without anyone agreeing on a column set, and the analysis pivots at the
end. A wide schema would require a four-way negotiation every time an audit gains a metric.

**Append-only shards, one Parquet file per run.** Never a shared mutable CSV. Concurrent Kaggle
sessions across several accounts writing one file is guaranteed corruption, and it is the kind
of corruption that is discovered in week 14.

**Validated on write.** ``metric`` must exist in the registry and ``probe_set`` must be one the
metric declares. A typo fails at the moment it is made, in the session that made it, rather than
silently splitting one metric into two columns that never join.

Only pyarrow is needed to write. pandas is an analysis-time dependency, so the code path that
runs on every training session carries the smaller surface.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .ids import NO_METHOD, ROLES, parse_run_id
from .metrics import MetricRegistry, default_registry
from .provenance import NO_GIT, config_sha, env_sha, git_commit, utc_now

__all__ = [
    "RunRecord",
    "SCHEMA",
    "AUDITS",
    "validate",
    "write_records",
    "read_records",
    "shard_path",
    "RecordError",
]


class RecordError(ValueError):
    """A record violates the schema contract."""


#: The audit modules that may write records. ``meta`` covers training, calibration and
#: bookkeeping rows that no single audit owns.
AUDITS: Final[frozenset[str]] = frozenset(
    {
        "behavior",
        "privacy_pop",
        "privacy_rmia",
        "representation",
        "relearning",
        "calibration",
        "meta",
    }
)

_PROBE_RE: Final = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


# --------------------------------------------------------------------------- the record


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One measurement.

    The identity fields (``run_id`` .. ``shadow_idx``) are denormalised deliberately: they are
    all derivable from ``run_id``, but carrying them explicitly means the analysis can group
    without parsing strings, and a corrupted shard is still interpretable.
    """

    # --- identity
    run_id: str
    role: str
    dataset: str
    arch: str

    # --- experimental coordinates
    forget_id: str
    forget_kind: str
    forget_size: int
    method: str

    # --- the measurement
    audit: str
    metric: str
    probe_set: str
    value: float
    n_probe: int

    # --- optional coordinates
    forget_stratum: str | None = None
    train_seed: int | None = None
    selection_seed: int | None = None
    oracle_seed: int | None = None
    audit_seed: int | None = None
    shadow_idx: int | None = None

    # --- provenance (auto-filled)
    hparams_sha: str = ""
    checkpoint_sha: str | None = None
    runtime_s: float | None = None
    git_commit: str = field(default_factory=git_commit)
    env_sha: str = field(default_factory=env_sha)
    timestamp: str = field(default_factory=utc_now)

    #: Any deviation from the implementation plan goes here — in the data, not in memory.
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Explicit Arrow schema. Shards written by four people must be concatenable, which requires
#: identical types — letting pyarrow infer from Python objects produces int64 in one shard and
#: null in another the moment a column happens to be all-None.
SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("arch", pa.string(), nullable=False),
        pa.field("forget_id", pa.string(), nullable=False),
        pa.field("forget_kind", pa.string(), nullable=False),
        pa.field("forget_size", pa.int32(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("audit", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("probe_set", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("n_probe", pa.int64(), nullable=False),
        pa.field("forget_stratum", pa.string()),
        pa.field("train_seed", pa.int32()),
        pa.field("selection_seed", pa.int32()),
        pa.field("oracle_seed", pa.int32()),
        pa.field("audit_seed", pa.int32()),
        pa.field("shadow_idx", pa.int32()),
        pa.field("hparams_sha", pa.string(), nullable=False),
        pa.field("checkpoint_sha", pa.string()),
        pa.field("runtime_s", pa.float64()),
        pa.field("git_commit", pa.string(), nullable=False),
        pa.field("env_sha", pa.string(), nullable=False),
        pa.field("timestamp", pa.string(), nullable=False),
        pa.field("notes", pa.string(), nullable=False),
    ]
)

_FIELD_ORDER: Final[tuple[str, ...]] = tuple(SCHEMA.names)


def _check_field_parity() -> None:
    """Fail at import if the dataclass and the Arrow schema drift apart."""
    dc = {f.name for f in fields(RunRecord)}
    sc = set(_FIELD_ORDER)
    if dc != sc:
        raise RecordError(
            "RunRecord and SCHEMA disagree — "
            f"dataclass-only={sorted(dc - sc)}, schema-only={sorted(sc - dc)}"
        )


_check_field_parity()


# --------------------------------------------------------------------------- validation


def validate(rec: RunRecord, registry: MetricRegistry | None = None) -> None:
    """Raise :class:`RecordError` if ``rec`` violates the contract.

    Called by :func:`write_records` on every row. Call it directly in an audit's own tests to
    fail fast, before a 40-minute run.
    """
    reg = registry if registry is not None else default_registry()

    # -- identity is internally consistent
    try:
        key = parse_run_id(rec.run_id)
    except ValueError as exc:
        raise RecordError(f"{rec.run_id!r}: unparseable run_id — {exc}") from None

    if rec.role != key.role:
        raise RecordError(
            f"{rec.run_id!r}: role={rec.role!r} contradicts the run_id, which says {key.role!r}"
        )
    if rec.method != key.method:
        raise RecordError(
            f"{rec.run_id!r}: method={rec.method!r} contradicts the run_id, "
            f"which says {key.method!r}"
        )
    if rec.role not in ROLES:
        raise RecordError(f"{rec.run_id!r}: unknown role {rec.role!r}")

    # -- the measurement
    if rec.audit not in AUDITS:
        raise RecordError(
            f"{rec.run_id!r}: audit={rec.audit!r} is not one of {sorted(AUDITS)}"
        )

    spec = reg[rec.metric]  # raises a helpful KeyError with suggestions if unregistered

    if not _PROBE_RE.match(rec.probe_set):
        raise RecordError(
            f"{rec.run_id!r}: probe_set={rec.probe_set!r} must be lowercase alphanumeric "
            "with single underscores"
        )
    if not spec.accepts_probe_set(rec.probe_set):
        raise RecordError(
            f"{rec.run_id!r}: metric {rec.metric!r} does not declare probe_set "
            f"{rec.probe_set!r} (declared: {list(spec.probe_sets)}). Either the probe set is "
            "wrong or configs/metrics.yaml needs updating — decide which."
        )

    if not isinstance(rec.value, (int, float)) or isinstance(rec.value, bool):
        raise RecordError(f"{rec.run_id!r}: value must be a real number, got {rec.value!r}")
    if math.isnan(rec.value) or math.isinf(rec.value):
        raise RecordError(
            f"{rec.run_id!r}/{rec.metric}: value is {rec.value}. Write a real number or omit "
            "the row — a NaN here silently poisons every downstream aggregate."
        )

    if rec.n_probe < 0:
        raise RecordError(f"{rec.run_id!r}: n_probe must be non-negative, got {rec.n_probe}")
    if rec.n_probe == 0:
        raise RecordError(
            f"{rec.run_id!r}/{rec.metric}: n_probe=0 — a metric computed over no examples is "
            "not a measurement."
        )

    # -- role-specific requirements
    _validate_seeds(rec, key)

    if rec.forget_size < 0:
        raise RecordError(f"{rec.run_id!r}: forget_size must be non-negative")
    if rec.forget_kind == "memstratum" and not rec.forget_stratum:
        raise RecordError(
            f"{rec.run_id!r}: forget_kind='memstratum' requires forget_stratum "
            "(low/medium/high)"
        )
    if rec.forget_stratum is not None and rec.forget_stratum not in {"low", "medium", "high"}:
        raise RecordError(
            f"{rec.run_id!r}: forget_stratum={rec.forget_stratum!r} must be low, medium or high"
        )

    if rec.role in {"base", "oracle", "shadow"} and rec.method != NO_METHOD:
        raise RecordError(f"{rec.run_id!r}: role={rec.role!r} requires method='none'")

    if not rec.git_commit:
        raise RecordError(f"{rec.run_id!r}: git_commit is empty; expected a sha or {NO_GIT!r}")
    if not rec.env_sha:
        raise RecordError(f"{rec.run_id!r}: env_sha is empty")


def _validate_seeds(rec: RunRecord, key) -> None:
    """Exactly the seeds that identify this run must be present."""
    if key.seed_kind == "train" and rec.train_seed is None:
        raise RecordError(f"{rec.run_id!r}: run_id carries train{key.seed} but train_seed is None")
    if key.seed_kind == "train" and rec.train_seed != key.seed:
        raise RecordError(
            f"{rec.run_id!r}: train_seed={rec.train_seed} contradicts the run_id ({key.seed})"
        )

    if key.seed_kind == "oracle":
        if rec.oracle_seed is None:
            raise RecordError(
                f"{rec.run_id!r}: run_id carries oracle{key.seed} but oracle_seed is None"
            )
        if rec.oracle_seed != key.seed:
            raise RecordError(
                f"{rec.run_id!r}: oracle_seed={rec.oracle_seed} contradicts the run_id "
                f"({key.seed})"
            )

    if key.seed_kind == "shadow":
        if rec.shadow_idx is None:
            raise RecordError(
                f"{rec.run_id!r}: run_id carries shadow{key.seed} but shadow_idx is None"
            )
        if rec.shadow_idx != key.seed:
            raise RecordError(
                f"{rec.run_id!r}: shadow_idx={rec.shadow_idx} contradicts the run_id ({key.seed})"
            )

    # A forget set that is not the whole dataset must say which draw produced it, or the
    # experiment is not reproducible.
    if rec.forget_kind != "none" and rec.selection_seed is None:
        raise RecordError(
            f"{rec.run_id!r}: forget_kind={rec.forget_kind!r} requires selection_seed"
        )


# --------------------------------------------------------------------------- IO


def shard_path(out_dir: str | Path, run_id: str, *, suffix: str = "") -> Path:
    """Path of the shard for one run.

    ``suffix`` separates shards written by different audits for the same run, so that audits
    owned by different people never race for the same filename.
    """
    stem = run_id if not suffix else f"{run_id}--{suffix}"
    return Path(out_dir) / f"{stem}.parquet"


def write_records(
    rows: Sequence[RunRecord],
    out_dir: str | Path,
    *,
    suffix: str = "",
    registry: MetricRegistry | None = None,
    overwrite: bool = True,
) -> Path:
    """Validate and write one shard.

    All rows must share a ``run_id`` — a shard is the record of one run.

    Returns:
        Path of the written file.

    Raises:
        RecordError: on validation failure, mixed run_ids, or an empty batch. Nothing is
            written unless every row passes, so a shard is never half-valid.
    """
    if not rows:
        raise RecordError("refusing to write an empty shard")

    run_ids = {r.run_id for r in rows}
    if len(run_ids) != 1:
        raise RecordError(
            f"a shard holds one run; got {len(run_ids)}: {sorted(run_ids)[:4]}"
        )

    reg = registry if registry is not None else default_registry()
    for r in rows:
        validate(r, reg)

    seen: set[tuple[str, str, str]] = set()
    for r in rows:
        k = (r.audit, r.metric, r.probe_set)
        if k in seen:
            raise RecordError(
                f"{r.run_id!r}: duplicate (audit={k[0]}, metric={k[1]}, probe_set={k[2]}). "
                "Two values for one measurement cannot both be right."
            )
        seen.add(k)

    path = shard_path(out_dir, rows[0].run_id, suffix=suffix)
    if path.exists() and not overwrite:
        raise RecordError(f"{path} exists and overwrite=False")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = {name: [getattr(r, name) for r in rows] for name in _FIELD_ORDER}
    table = pa.Table.from_pydict(columns, schema=SCHEMA)

    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)  # atomic: a reader never sees a partial shard
    return path


def read_records(
    src: str | Path | Iterable[str | Path],
    *,
    as_pandas: bool = True,
):
    """Read shards back.

    Args:
        src: a directory of shards, or an iterable of shard paths.
        as_pandas: return a DataFrame (requires pandas) rather than an Arrow table.

    Raises:
        FileNotFoundError: if a directory contains no shards — silently returning an empty
            frame here would produce an empty analysis that looks like a real one.
    """
    if isinstance(src, (str, Path)):
        root = Path(src)
        paths = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
    else:
        paths = [Path(p) for p in src]

    if not paths:
        raise FileNotFoundError(f"no .parquet shards found under {src}")

    table = pa.concat_tables([pq.read_table(p, schema=SCHEMA) for p in paths])
    return table.to_pandas() if as_pandas else table


def make_record(
    *,
    run_id: str,
    audit: str,
    metric: str,
    probe_set: str,
    value: float,
    n_probe: int,
    forget_kind: str,
    forget_size: int,
    hparams: dict[str, Any] | str = "",
    **kwargs: Any,
) -> RunRecord:
    """Build a record, filling identity from ``run_id`` and hashing ``hparams``.

    Convenience over the dataclass: it keeps audits from re-deriving role/method by hand and
    getting them subtly wrong.
    """
    key = parse_run_id(run_id)
    hparams_sha = hparams if isinstance(hparams, str) else config_sha(hparams)

    seeds: dict[str, Any] = {}
    if key.seed_kind == "train":
        seeds["train_seed"] = key.seed
    elif key.seed_kind == "oracle":
        seeds["oracle_seed"] = key.seed
    elif key.seed_kind == "shadow":
        seeds["shadow_idx"] = key.seed
    seeds.update({k: v for k, v in kwargs.items() if k.endswith("_seed") or k == "shadow_idx"})
    rest = {k: v for k, v in kwargs.items() if k not in seeds}

    return RunRecord(
        run_id=run_id,
        role=key.role,
        dataset=rest.pop("dataset", "cifar10"),
        arch=rest.pop("arch", "resnet18"),
        forget_id=rest.pop("forget_id", key.forget),
        forget_kind=forget_kind,
        forget_size=forget_size,
        method=key.method,
        audit=audit,
        metric=metric,
        probe_set=probe_set,
        value=float(value),
        n_probe=int(n_probe),
        hparams_sha=hparams_sha,
        **seeds,
        **rest,
    )
