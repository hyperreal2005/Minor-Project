"""The registry: identifiers, result schema, metric contract, artifact store, provenance.

BINDING — implementation plan §4. Changes require all four members.

This package is built first and blocks everything else, because it is the interface between four
people working in parallel on separate audit modules. Everything downstream imports from here.

    from forgetcheck.registry import run_id, RunRecord, write_records, ArtifactStore
"""

from .ids import (
    FULL_DATASET,
    NO_METHOD,
    RESERVED_METHODS,
    ROLES,
    SEED_KINDS,
    RunKey,
    parse_run_id,
    run_id,
    tag_for,
)
from .metrics import (
    DIRECTIONS,
    RANKED_FAMILIES,
    MetricRegistry,
    MetricSpec,
    default_registry,
    load_registry,
)
from .provenance import (
    NO_GIT,
    bytes_sha,
    config_sha,
    env_details,
    env_sha,
    file_sha,
    git_commit,
    runtime_context,
    utc_now,
)
from .records import (
    AUDITS,
    SCHEMA,
    RecordError,
    RunRecord,
    make_record,
    read_records,
    shard_path,
    validate,
    write_records,
)
from .store import ArtifactStore, CheckpointMeta, StoreError

__all__ = [
    # ids
    "run_id",
    "parse_run_id",
    "RunKey",
    "tag_for",
    "ROLES",
    "SEED_KINDS",
    "RESERVED_METHODS",
    "NO_METHOD",
    "FULL_DATASET",
    # metrics
    "MetricRegistry",
    "MetricSpec",
    "default_registry",
    "load_registry",
    "DIRECTIONS",
    "RANKED_FAMILIES",
    # records
    "RunRecord",
    "SCHEMA",
    "AUDITS",
    "RecordError",
    "validate",
    "write_records",
    "read_records",
    "make_record",
    "shard_path",
    # store
    "ArtifactStore",
    "CheckpointMeta",
    "StoreError",
    # provenance
    "git_commit",
    "env_sha",
    "bytes_sha",
    "env_details",
    "file_sha",
    "config_sha",
    "utc_now",
    "runtime_context",
    "NO_GIT",
]
